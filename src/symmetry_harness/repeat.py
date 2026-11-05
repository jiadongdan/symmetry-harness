"""Repeated fine-tuning runs for measuring procedural stability.

A *repeat loop* answers one question: if the same k-shot fine-tuning procedure
is executed again and again, does it produce the same answer? Every round trains
a fresh adapter from the backbone with exactly ``points_per_class`` newly chosen
support points, so the support set never accumulates and each round is an
independent replicate of the same experiment.

Two accumulated fields decide where the next round samples:

* ``confidence`` - the arithmetic mean of every round's confidence grid. It is
  used only as a filter: candidates below ``tau`` are not sampled.
* ``votes`` - how often each grid cell was predicted as each class. It supplies
  the pseudo-label of every sampled point.

Points inside one round are drawn with a Poisson-disk constraint: no two support
points end up closer than ``min_point_distance`` pixels, so a round never spends
two of its support slots on nearly the same patch. The constraint spans both
classes, because two nearly co-located points of different classes make the
classifier memorise one patch with two contradictory labels.

Nothing in this module trains a model itself; each round delegates to
:func:`symmetry_harness.workflow.run_analysis` so the numerical path is exactly
the one the CLI ``run`` command uses.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any, Callable, Iterable, Sequence
from uuid import uuid4

import numpy as np

from .analysis import dense_prediction_from_arrays
from .annotations import (
    create_annotation_session,
    load_annotation_session,
    save_annotation_session,
    valid_center_bounds,
)
from .config import HarnessConfig
from .image_io import inspect_input, file_sha256
from .provider import doctor, run_provider_features
from .workflow import build_run_options, run_analysis


REPEAT_SCHEMA_VERSION = "symmetry-repeat-loop-v1"
DEFAULT_CLASS_NAMES = ("Class A", "Class B")
DEFAULT_CLASS_COLORS = ("#e41a1c", "#377eb8")

#: Artifacts copied out of a completed run before the run directory is removed.
KEPT_ARTIFACTS = (
    "prediction.npz",
    "run_record.json",
    "confidence.png",
    "prediction_overlay.png",
)
KEPT_ARTIFACTS_WITH_MODEL = KEPT_ARTIFACTS + ("fine_tuned_model.symmodel",)


class RepeatError(RuntimeError):
    """Raised when a repeat loop cannot start or cannot continue."""


@dataclass(frozen=True)
class RepeatOptions:
    """Everything that can change between two repeat loops."""

    rounds: int = 5
    points_per_class: int = 5
    tau: float = 0.95
    base_seed: int = 42
    alarm_drop: float = 3.0
    alarm_floor: float = 0.90
    interior_margin: int = 32
    label_codes: tuple[int, int] = (14, 17)
    class_names: tuple[str, str] = DEFAULT_CLASS_NAMES
    min_point_distance: float | None = None
    keep_checkpoint: bool = True
    stop_on_alarm: bool = True
    reuse_features: bool = True

    def validate(self) -> None:
        if self.rounds < 0:
            raise RepeatError("rounds must be nonnegative (0 keeps only the seed round).")
        if self.points_per_class < 1:
            raise RepeatError("points_per_class must be at least 1.")
        if not 0.5 <= self.tau <= 1.0:
            raise RepeatError("tau must lie in [0.5, 1.0]; confidence is max(p).")
        if self.base_seed < 0:
            raise RepeatError("base_seed must be nonnegative.")
        if self.alarm_floor > 1.0 or self.alarm_floor < 0.0:
            raise RepeatError("alarm_floor must lie in [0, 1].")
        if self.alarm_drop <= 0:
            raise RepeatError("alarm_drop must be positive.")
        if self.interior_margin < 0:
            raise RepeatError("interior_margin must be nonnegative.")
        if self.min_point_distance is not None and self.min_point_distance < 0:
            raise RepeatError(
                "min_point_distance must be nonnegative, or null for the patch-size default."
            )
        if len(set(self.label_codes)) != 2:
            raise RepeatError("label_codes must contain two distinct label codes.")
        if len(self.class_names) != 2 or len(set(self.class_names)) != 2:
            raise RepeatError("Exactly two distinct class names are required.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rounds": self.rounds,
            "points_per_class": self.points_per_class,
            "tau": self.tau,
            "base_seed": self.base_seed,
            "alarm_drop": self.alarm_drop,
            "alarm_floor": self.alarm_floor,
            "interior_margin": self.interior_margin,
            "min_point_distance": self.min_point_distance,
            "label_codes": list(self.label_codes),
            "class_names": list(self.class_names),
            "keep_checkpoint": self.keep_checkpoint,
            "stop_on_alarm": self.stop_on_alarm,
            "reuse_features": self.reuse_features,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RepeatOptions":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise RepeatError(f"Unknown repeat options in manifest: {unknown}")
        values = dict(payload)
        if "label_codes" in values:
            values["label_codes"] = tuple(int(code) for code in values["label_codes"])
        if "class_names" in values:
            values["class_names"] = tuple(str(name) for name in values["class_names"])
        options = cls(**values)
        options.validate()
        return options


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def load_label_map(path: str | Path) -> np.ndarray:
    """Load a two-dimensional integer label raster from ``.npy``."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise RepeatError(f"The ground-truth label map does not exist: {source}")
    if source.suffix.lower() != ".npy":
        raise RepeatError(
            "Ground truth must be a .npy integer raster; convert other formats first."
        )
    labels = np.load(source, allow_pickle=False)
    if labels.ndim != 2:
        raise RepeatError(
            f"Ground truth must be two-dimensional; received shape {labels.shape}."
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise RepeatError(
            f"Ground truth must be an integer raster; received dtype {labels.dtype}."
        )
    return labels


def class_map_from_labels(labels: np.ndarray, codes: Sequence[int]) -> np.ndarray:
    """Return a 0/1 class map, mapping ``codes`` positionally onto the two classes."""
    if len(codes) != 2:
        raise RepeatError("Exactly two label codes are required.")
    unique = np.unique(labels)
    unexpected = sorted(int(value) for value in unique if int(value) not in set(codes))
    if unexpected:
        raise RepeatError(
            "The label map contains codes that are not part of label_codes: "
            f"{unexpected}. Found {sorted(int(value) for value in unique)}."
        )
    class_map = np.full(labels.shape, -1, dtype=np.int8)
    for index, code in enumerate(codes):
        class_map[labels == int(code)] = index
    if (class_map < 0).any():
        raise RepeatError("Some pixels carry no class after applying label_codes.")
    return class_map


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Erode a boolean mask by ``radius`` pixels with a 4-connected kernel."""
    if radius <= 0:
        return mask.copy()
    current = mask.copy()
    for _ in range(int(radius)):
        shifted = current.copy()
        shifted[1:, :] &= current[:-1, :]
        shifted[:-1, :] &= current[1:, :]
        shifted[:, 1:] &= current[:, :-1]
        shifted[:, :-1] &= current[:, 1:]
        shifted[0, :] = False
        shifted[-1, :] = False
        shifted[:, 0] = False
        shifted[:, -1] = False
        current = shifted
    return current


def _interior_layers(base: np.ndarray, margin: int) -> list[np.ndarray]:
    """Return the mask eroded by 0, 1, ... ``margin`` pixels."""
    layers = [base]
    for _ in range(int(margin)):
        layers.append(_erode(layers[-1], 1))
    return layers


def _deepest_region(layers: list[np.ndarray], needed: int) -> tuple[np.ndarray, int]:
    """Pick the deepest erosion that still holds ``needed`` pixels.

    A phase narrower than twice the margin erodes to nothing, which is normal
    for a grain-boundary band: the default margin assumes a wide phase. Falling
    back to the deepest usable erosion keeps the loop running instead of
    stopping on a geometry the caller cannot control.
    """
    for radius in range(len(layers) - 1, -1, -1):
        if int(np.count_nonzero(layers[radius])) >= needed:
            return layers[radius], radius
    return layers[0], 0


def _interior_draw(
    class_map: np.ndarray,
    *,
    patch_size: int,
    points_per_class: int,
    margin: int,
    seed: int,
    min_distance: float = 0.0,
) -> tuple[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]], float]:
    """Draw the seed round's points and report the spacing that was used."""
    height, width = class_map.shape
    min_x, max_x, min_y, max_y = valid_center_bounds((height, width), patch_size)
    allowed = np.zeros(class_map.shape, dtype=bool)
    allowed[min_y : max_y + 1, min_x : max_x + 1] = True
    rng = np.random.default_rng(seed)
    point_groups: list[np.ndarray] = []
    for class_index in (0, 1):
        base = (class_map == class_index) & allowed
        region, _ = _deepest_region(_interior_layers(base, margin), points_per_class)
        rows, columns = np.nonzero(region)
        if len(rows) < points_per_class:
            raise RepeatError(
                f"Class {class_index} has only {len(rows)} interior pixels at "
                f"margin {margin} px; reduce interior_margin or points_per_class."
            )
        point_groups.append(np.stack([columns, rows], axis=1).astype(np.int64))
    chosen, used = poisson_disk_select(
        point_groups, points_per_class, min_distance, rng, floor=2.0
    )
    return (chosen[0], chosen[1]), used


def interior_points(
    class_map: np.ndarray,
    *,
    patch_size: int,
    points_per_class: int,
    margin: int,
    seed: int,
    min_distance: float = 0.0,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Draw the first round's support points from the interior of each phase.

    The margin keeps every ``patch_size`` patch inside one phase, so the initial
    support set never contains a mixed patch. ``min_distance`` additionally keeps
    the points of that first round apart, so a five-shot seed set does not spend
    four clicks on the same grain. This margin applies to the initial points
    only; later rounds sample on the confidence field alone.
    """
    groups, _ = _interior_draw(
        class_map,
        patch_size=patch_size,
        points_per_class=points_per_class,
        margin=margin,
        seed=seed,
        min_distance=min_distance,
    )
    return groups


# ---------------------------------------------------------------------------
# Sampling on the accumulated confidence field
# ---------------------------------------------------------------------------


@dataclass
class Accumulated:
    """Cross-round state used to choose the next support set."""

    confidence_sum: np.ndarray
    votes: np.ndarray
    rounds: int = 0

    @property
    def confidence(self) -> np.ndarray:
        if self.rounds == 0:
            raise RepeatError("No confidence map has been accumulated yet.")
        return self.confidence_sum / float(self.rounds)

    @property
    def labels(self) -> np.ndarray:
        if self.rounds == 0:
            raise RepeatError("No prediction has been accumulated yet.")
        return np.argmax(self.votes, axis=2).astype(np.int16)

    def update(self, prediction) -> None:
        confidence = np.asarray(prediction.confidence_grid, dtype=np.float64)
        grid = np.asarray(prediction.prediction_grid, dtype=np.int16)
        if self.rounds == 0:
            if self.confidence_sum.shape != confidence.shape:
                raise RepeatError("Confidence grid shape changed between rounds.")
            self.votes = np.zeros(confidence.shape + (2,), dtype=np.int32)
        elif self.confidence_sum.shape != confidence.shape:
            raise RepeatError("Confidence grid shape changed between rounds.")
        self.confidence_sum += confidence
        for class_index in (0, 1):
            self.votes[:, :, class_index] += (grid == class_index).astype(np.int32)
        self.rounds += 1

    @classmethod
    def empty(cls, shape: tuple[int, int]) -> "Accumulated":
        return cls(
            confidence_sum=np.zeros(shape, dtype=np.float64),
            votes=np.zeros(shape + (2,), dtype=np.int32),
        )

    def replay(self, predictions: Iterable[Any]) -> None:
        for prediction in predictions:
            self.update(prediction)


@dataclass(frozen=True)
class Sampling:
    """One round's sampled support points plus the pool they came from."""

    points_by_class: tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]
    pool_sizes: tuple[int, int]
    available: tuple[int, int]
    seed: int
    min_distance: float = 0.0
    requested_min_distance: float = 0.0

    @property
    def relaxed(self) -> bool:
        """True when the requested spacing did not fit and had to be shrunk."""
        return self.min_distance < self.requested_min_distance - 1e-9

    def as_points(self) -> list[tuple[int, int]]:
        return [*self.points_by_class[0], *self.points_by_class[1]]


def grid_step(coordinates: np.ndarray) -> float:
    """Return the spacing between neighbouring grid coordinates, in pixels."""
    unique = np.unique(np.asarray(coordinates, dtype=np.int64))
    if unique.size < 2:
        return 1.0
    return float(np.median(np.diff(unique)))


def _dart_throw(
    points: np.ndarray,
    count: int,
    radius: float,
    blocked: Sequence[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    """Greedily accept ``count`` points that sit at least ``radius`` from every
    point already taken.

    Points are visited in a seeded random order, which is exactly uniform
    without replacement when ``radius`` is zero and blue noise otherwise. A full
    scan that cannot place ``count`` points returns ``None``.
    """
    taken = np.asarray(blocked, dtype=np.float64).reshape(-1, 2)
    accepted: list[tuple[int, int]] = []
    radius_sq = float(radius) ** 2
    for index in rng.permutation(len(points)):
        candidate = points[index]
        if radius_sq > 0.0 and taken.size:
            deltas = taken - candidate
            if np.any((deltas * deltas).sum(axis=1) < radius_sq):
                continue
        point = (int(candidate[0]), int(candidate[1]))
        accepted.append(point)
        taken = np.vstack([taken, [[float(point[0]), float(point[1])]]])
        if len(accepted) == count:
            return accepted
    return None


def poisson_disk_select(
    groups: Sequence[np.ndarray],
    count: int,
    min_distance: float,
    rng: np.random.Generator,
    *,
    floor: float = 1.0,
    shrink: float = 0.85,
) -> tuple[tuple[tuple[tuple[int, int], ...], ...], float]:
    """Draw ``count`` well-spaced points from every group.

    Each group contributes ``count`` points, and no two accepted points - from
    the same group or from different groups - are closer than the spacing that
    was used. Groups are filled in order, so later groups also avoid the points
    already accepted.

    When the requested spacing cannot fit the pools, it is shrunk geometrically
    down to ``floor`` and the radius that actually fit is returned. The caller
    is expected to surface that number: a silently relaxed radius means the pool
    is too small for the requested separation, which is a property of the data,
    not a detail.
    """
    radius = float(min_distance)
    if radius <= 0.0:
        chosen: list[tuple[tuple[int, int], ...]] = []
        for points in groups:
            picked = _dart_throw(points, count, 0.0, [], rng)
            if picked is None:
                raise RepeatError(
                    f"A pool holds {len(points)} cells; {count} points are required."
                )
            chosen.append(tuple(picked))
        return tuple(chosen), 0.0

    while True:
        accepted: list[tuple[int, int]] = []
        chosen = []
        for points in groups:
            picked = _dart_throw(points, count, radius, accepted, rng)
            if picked is None:
                break
            chosen.append(tuple(picked))
            accepted.extend(picked)
        else:
            return tuple(chosen), radius
        if radius <= floor:
            raise RepeatError(
                f"Cannot place {count} points per class at least {radius:.1f} px "
                "apart; the sampling pool is too small. Lower min_point_distance "
                "or tau, or draw fewer points per class."
            )
        radius = max(floor, radius * shrink)


def select_points(
    accumulated: Accumulated,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    *,
    points_per_class: int,
    tau: float,
    seed: int,
    min_distance: float = 0.0,
) -> Sampling:
    """Draw ``points_per_class`` cells per class from the confident region.

    Candidates must have an accumulated confidence of at least ``tau``. Selection
    is uniform without replacement, with a Poisson-disk constraint of
    ``min_distance`` pixels across the whole support set: two points of the same
    class or of different classes never sit closer than that. Pass ``0`` to fall
    back to unconstrained uniform sampling.
    """
    confidence = accumulated.confidence
    labels = accumulated.labels
    pool = confidence >= float(tau)
    if pool.shape != labels.shape:
        raise RepeatError("The accumulated confidence and label grids differ in shape.")
    xs = np.asarray(x_coordinates, dtype=np.int64)
    ys = np.asarray(y_coordinates, dtype=np.int64)
    point_groups: list[np.ndarray] = []
    pool_sizes: list[int] = []
    available: list[int] = []
    rng = np.random.default_rng(seed)
    for class_index in (0, 1):
        candidates = np.argwhere(pool & (labels == class_index))
        pool_sizes.append(int(len(candidates)))
        class_total = int((labels == class_index).sum())
        available.append(class_total)
        if len(candidates) < points_per_class:
            raise RepeatError(
                f"Round pool for class {class_index} has {len(candidates)} confident "
                f"cells (tau={tau}); {points_per_class} are required. Lower tau or "
                "points_per_class."
            )
        point_groups.append(
            np.stack([xs[candidates[:, 1]], ys[candidates[:, 0]]], axis=1)
        )
    chosen, used = poisson_disk_select(
        point_groups,
        points_per_class,
        min_distance,
        rng,
        floor=max(2.0, 2.0 * grid_step(xs)),
    )
    return Sampling(
        points_by_class=(chosen[0], chosen[1]),
        pool_sizes=(pool_sizes[0], pool_sizes[1]),
        available=(available[0], available[1]),
        seed=seed,
        min_distance=used,
        requested_min_distance=float(min_distance),
    )


# ---------------------------------------------------------------------------
# Metrics and stop rules
# ---------------------------------------------------------------------------


def score_prediction(
    prediction,
    class_map: np.ndarray,
) -> dict[str, Any]:
    """Score one dense prediction against the ground-truth raster."""
    x_coordinates = np.asarray(prediction.x_coordinates, dtype=np.int64)
    y_coordinates = np.asarray(prediction.y_coordinates, dtype=np.int64)
    grid = np.asarray(prediction.prediction_grid, dtype=np.int16)
    truth = class_map[np.ix_(y_coordinates, x_coordinates)]
    if truth.shape != grid.shape:
        raise RepeatError("The ground-truth crop does not match the prediction grid.")
    correct = grid == truth
    metrics: dict[str, Any] = {
        "cells": int(grid.size),
        "acc": float(correct.mean()),
        "area_frac_pred": float((grid == 1).mean()),
        "area_frac_gt": float((truth == 1).mean()),
        "conf_mean": float(np.mean(prediction.confidence_grid)),
        "conf_median": float(np.median(prediction.confidence_grid)),
    }
    for class_index, tag in ((0, "acc_class_a"), (1, "acc_class_b")):
        mask = truth == class_index
        metrics[tag] = float(correct[mask].mean()) if mask.any() else float("nan")
        predicted = grid == class_index
        union = predicted | mask
        metrics["iou_class_" + ("a" if class_index == 0 else "b")] = (
            float((predicted & mask).sum() / union.sum()) if union.any() else float("nan")
        )
    return metrics


def agreement_with(previous_grid: np.ndarray | None, grid: np.ndarray) -> float:
    """Return the cell-wise agreement between two prediction grids."""
    if previous_grid is None:
        return float("nan")
    previous = np.asarray(previous_grid, dtype=np.int16)
    current = np.asarray(grid, dtype=np.int16)
    if previous.shape != current.shape:
        return float("nan")
    return float((previous == current).mean())


def history_statistics(values: Sequence[float]) -> dict[str, float]:
    """Summarise a series of per-round accuracy values."""
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "range": float("nan"),
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "range": float(array.max() - array.min()),
    }


@dataclass(frozen=True)
class StopDecision:
    """Whether the loop must stop, and why."""

    stop: bool
    status: str
    reason: str
    detail: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "stop": self.stop,
            "status": self.status,
            "reason": self.reason,
            "detail": dict(self.detail),
        }


def evaluate_round(
    metrics: dict[str, Any],
    previous_accuracies: Sequence[float],
    options: RepeatOptions,
) -> StopDecision:
    """Decide whether one completed round leaves the loop healthy."""
    accuracy = float(metrics["acc"])
    statistics = history_statistics(previous_accuracies)
    detail: dict[str, Any] = {
        "acc": accuracy,
        "history": statistics,
        "alarm_floor": options.alarm_floor,
        "alarm_drop": options.alarm_drop,
    }
    if accuracy < options.alarm_floor:
        return StopDecision(
            stop=options.stop_on_alarm,
            status="alarm",
            reason=(
                f"accuracy {accuracy:.4f} is below the absolute floor "
                f"{options.alarm_floor:.4f}"
            ),
            detail=detail,
        )
    if statistics["count"] >= 3 and statistics["std"] >= 0.0:
        bound = statistics["mean"] - options.alarm_drop * statistics["std"]
        detail["lower_bound"] = bound
        if accuracy < bound:
            return StopDecision(
                stop=options.stop_on_alarm,
                status="alarm",
                reason=(
                    f"accuracy {accuracy:.4f} is below mean-{options.alarm_drop:g}sigma "
                    f"({statistics['mean']:.4f} - {options.alarm_drop:g} x "
                    f"{statistics['std']:.4f} = {bound:.4f})"
                ),
                detail=detail,
            )
    for tag in ("iou_class_a", "iou_class_b"):
        value = float(metrics.get(tag, float("nan")))
        detail[tag] = value
        if np.isfinite(value) and value < 0.10:
            return StopDecision(
                stop=options.stop_on_alarm,
                status="alarm",
                reason=f"class collapse: {tag} is {value:.4f}",
                detail=detail,
            )
    return StopDecision(stop=False, status="ok", reason="within the historical range", detail=detail)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Convert non-finite floats and NumPy scalars into strict JSON values."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


CSV_COLUMNS = (
    "round",
    "points_a",
    "points_b",
    "spread_req",
    "spread",
    "pool_a",
    "pool_b",
    "acc",
    "acc_class_a",
    "acc_class_b",
    "iou_class_a",
    "iou_class_b",
    "area_frac_pred",
    "area_frac_gt",
    "agree_prev",
    "conf_mean",
    "conf_median",
    "support_acc",
    "support_loss",
    "train_seed",
    "sample_seed",
    "seconds",
    "points_xy",
)


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        if new_file:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in CSV_COLUMNS})


def _keep_artifacts(run_directory: Path, target: Path, *, keep_checkpoint: bool) -> list[str]:
    """Copy the retained artifacts out of a run directory."""
    names = KEPT_ARTIFACTS_WITH_MODEL if keep_checkpoint else KEPT_ARTIFACTS
    target.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    for name in names:
        source = run_directory / name
        if not source.is_file():
            raise RepeatError(f"The run did not produce {name!r} in {run_directory}.")
        shutil.copy2(source, target / name)
        kept.append(name)
    return kept


def _prune_run_directory(run_directory: Path) -> None:
    """Drop the two heavy duplicates that were already copied into ``kept/``.

    The run directory itself is kept. Removing it means deleting 60-plus files
    inside one round, which trips bulk-delete guards on locked-down machines and
    aborts the loop mid-round without an exception the loop can catch. The two
    ~46 MB duplicates are where all the space is, so deleting only those brings
    a round down to roughly 1 MB without ever issuing a bulk delete.
    """
    archive = run_directory.parent / f"{run_directory.name}-full-run.zip"
    for target in (run_directory / "fine_tuned_model.symmodel", archive):
        try:
            target.unlink()
        except FileNotFoundError:
            continue


def _slug(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "-" for char in value).strip("-")
    return cleaned.lower()[:48] or "image"


def _new_loop_directory(root: Path, slug: str) -> tuple[str, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    loop_id = f"repeat-{slug}-{timestamp}-{uuid4().hex[:6]}"
    directory = root.expanduser().resolve() / loop_id
    directory.mkdir(parents=True, exist_ok=False)
    return loop_id, directory


# ---------------------------------------------------------------------------
# One image, N rounds
# ---------------------------------------------------------------------------


def _features_artifacts(
    config: HarnessConfig,
    image: np.ndarray,
    options,
    directory: Path,
) -> tuple[Path, Path] | tuple[None, None]:
    """Precompute the reusable feature artifacts once per image."""
    try:
        computed = run_provider_features(config, image, options=options.to_dict())
    except Exception:  # noqa: BLE001 - reuse is an optimization, not a requirement
        return None, None
    directory.mkdir(parents=True, exist_ok=True)
    array_path = directory / "features.npz"
    record_path = directory / "features_record.json"
    np.savez_compressed(array_path, features=np.asarray(computed.features, dtype=np.float32))
    _write_json(record_path, computed.record)
    return array_path, record_path


def run_repeat(
    config: HarnessConfig,
    *,
    image_path: str | Path,
    label_path: str | Path,
    options: RepeatOptions,
    output_root: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    annotations: str | Path | None = None,
    resume: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run one repeat loop on one labeled image and return its summary."""
    options.validate()
    report = progress or (lambda message: None)
    readiness = doctor(config)
    if readiness["status"] != "ready":
        raise RepeatError(
            "The configured runtime is blocked: " + "; ".join(readiness["issues"])
        )

    image, input_record = inspect_input(image_path, config.features.input_normalization)
    labels = load_label_map(label_path)
    if labels.shape != image.shape:
        raise RepeatError(
            f"The label map {labels.shape} and image {image.shape} shapes differ."
        )
    class_map = class_map_from_labels(labels, options.label_codes)

    run_options = build_run_options(config, overrides)
    if options.points_per_class < run_options.minimum_shots_per_class:
        # The configured minimum is a UI guardrail for human annotation. A
        # stability loop legitimately replicates a 1-shot protocol, so the
        # floor is relaxed to the number of points the loop actually draws.
        run_options = replace(
            run_options, minimum_shots_per_class=options.points_per_class
        )
    patch_size = run_options.classifier_patch_size
    image_sha = input_record["sha256"]
    # Two support points closer than one patch would train on overlapping crops,
    # so the default spacing is exactly one patch size. Pools that cannot fit it
    # (a thin grain-boundary band, say) are reported after relaxation.
    requested_spread = (
        float(patch_size)
        if options.min_point_distance is None
        else float(options.min_point_distance)
    )

    if resume is not None:
        loop_dir = Path(resume).expanduser().resolve()
        if not loop_dir.is_dir():
            raise RepeatError(f"The resume directory does not exist: {loop_dir}")
        manifest = json.loads((loop_dir / "loop_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != REPEAT_SCHEMA_VERSION:
            raise RepeatError("The resume directory holds an incompatible loop schema.")
        if manifest["input"]["sha256"] != image_sha:
            raise RepeatError(
                "The resume directory was produced from a different image (SHA-256 differs)."
            )
        if manifest["ground_truth"]["sha256"] != file_sha256(label_path):
            raise RepeatError(
                "The resume directory was produced from a different ground truth."
            )
        recorded = RepeatOptions.from_dict(manifest["options"])
        # Only the horizon may grow on resume: changing the shot count or the
        # sampling rule would mix two protocols in one distribution. The
        # operational switches stay under the caller's control.
        options = replace(
            recorded,
            rounds=max(recorded.rounds, options.rounds),
            keep_checkpoint=options.keep_checkpoint,
            stop_on_alarm=options.stop_on_alarm,
            reuse_features=options.reuse_features,
        )
        loop_id = str(manifest["loop_id"])
        report(f"resuming {loop_id}")
    else:
        loop_id, loop_dir = _new_loop_directory(
            config.output_root if output_root is None else Path(output_root),
            _slug(Path(image_path).stem),
        )
        manifest = {}
        report(f"loop {loop_id}")

    rounds_dir = loop_dir / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    csv_path = loop_dir / "loop_rounds.csv"

    feature_path: Path | None = None
    feature_record: Path | None = None
    if options.reuse_features and resume is None:
        report("computing reusable features once for every round")
        feature_path, feature_record = _features_artifacts(
            config, image, run_options, loop_dir / "features"
        )
        if feature_path is None:
            report("feature reuse unavailable; each round recomputes features")
    elif options.reuse_features and resume is not None:
        candidate = loop_dir / "features" / "features.npz"
        candidate_record = loop_dir / "features" / "features_record.json"
        if candidate.is_file() and candidate_record.is_file():
            feature_path, feature_record = candidate, candidate_record

    if resume is None:
        manifest = {
            "schema_version": REPEAT_SCHEMA_VERSION,
            "loop_id": loop_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "input": input_record,
            "ground_truth": {
                "path": str(Path(label_path).expanduser().resolve()),
                "sha256": file_sha256(label_path),
                "shape": list(labels.shape),
                "dtype": str(labels.dtype),
                "label_codes": list(options.label_codes),
                "class_names": list(options.class_names),
            },
            "options": options.to_dict(),
            "run_options": run_options.to_dict(),
            "annotations": None if annotations is None else str(Path(annotations).resolve()),
            "model": {
                "identifier": config.model.identifier,
                "checkpoint_path": (
                    None
                    if config.model.checkpoint_path is None
                    else str(config.model.checkpoint_path)
                ),
                "checkpoint_sha256": config.model.checkpoint_sha256,
            },
        }
        manifest["features"] = {
            "reused": feature_path is not None,
            "path": None if feature_path is None else str(feature_path),
        }
        _write_json(loop_dir / "loop_manifest.json", manifest)

    # ---------------------------------------------------------------- rounds
    accumulated: Accumulated | None = None
    history: list[dict[str, Any]] = []
    previous_grid: np.ndarray | None = None
    x_coordinates: np.ndarray | None = None
    y_coordinates: np.ndarray | None = None
    completed = 0

    if resume is not None:
        for path in sorted(rounds_dir.glob("round-*/kept/prediction.npz")):
            with np.load(path, allow_pickle=False) as payload:
                arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
            prediction = dense_prediction_from_arrays(arrays)
            if accumulated is None:
                accumulated = Accumulated.empty(prediction.confidence_grid.shape)
                x_coordinates = np.asarray(prediction.x_coordinates, dtype=np.int64)
                y_coordinates = np.asarray(prediction.y_coordinates, dtype=np.int64)
            accumulated.update(prediction)
            previous_grid = np.asarray(prediction.prediction_grid, dtype=np.int16)
            completed += 1
        if csv_path.is_file():
            with csv_path.open(encoding="utf-8", newline="") as handle:
                history = list(csv.DictReader(handle))
        report(f"replayed {completed} completed round(s)")
        if completed == 0:
            raise RepeatError(
                "The resume directory holds a manifest but no completed round to replay."
            )
        if completed >= options.rounds + 1:
            return _finalize(loop_id, loop_dir, manifest, options, history, "completed")

    for round_index in range(completed, options.rounds + 1):
        round_started = perf_counter()
        train_seed = options.base_seed + round_index
        sample_seed = options.base_seed * 1000 + round_index
        round_dir = rounds_dir / f"round-{round_index:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        spread_used: float | None = None

        if round_index == 0:
            if annotations is not None:
                session = load_annotation_session(annotations)
                if tuple(session.image_shape) != tuple(image.shape):
                    raise RepeatError(
                        "The annotation session was drawn on a different image shape."
                    )
                if session.image_sha256 != image_sha:
                    raise RepeatError(
                        "The annotation session was drawn on a different image file."
                    )
                if len(session.classes) != 2:
                    raise RepeatError(
                        "A repeat loop needs exactly two annotation classes."
                    )
                points_by_class: tuple[
                    tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]
                ] = (
                    tuple(session.classes[0].points),
                    tuple(session.classes[1].points),
                )
            else:
                points_by_class, spread_used = _interior_draw(
                    class_map,
                    patch_size=patch_size,
                    points_per_class=options.points_per_class,
                    margin=options.interior_margin,
                    seed=options.base_seed,
                    min_distance=requested_spread,
                )
            pool_sizes = (0, 0)
            sample_seed = options.base_seed
        else:
            if accumulated is None or x_coordinates is None or y_coordinates is None:
                raise RepeatError("The loop has no accumulated prediction to sample from.")
            sampling = select_points(
                accumulated,
                x_coordinates,
                y_coordinates,
                points_per_class=options.points_per_class,
                tau=options.tau,
                seed=sample_seed,
                min_distance=requested_spread,
            )
            points_by_class = sampling.points_by_class
            pool_sizes = sampling.pool_sizes
            spread_used = sampling.min_distance
        if (
            spread_used is not None
            and requested_spread > 0
            and spread_used < requested_spread - 1e-9
        ):
            report(
                f"round {round_index}: spacing relaxed from {requested_spread:.1f} px "
                f"to {spread_used:.1f} px to fit the pool"
            )

        session = create_annotation_session(
            image_path=str(Path(image_path).expanduser().resolve()),
            image_sha256=image_sha,
            image_shape=image.shape,
            classifier_patch_size=patch_size,
            class_names=options.class_names,
            points_by_class=points_by_class,
            colors=DEFAULT_CLASS_COLORS,
        )
        session_path = round_dir / f"session_round{round_index:03d}.json"
        save_annotation_session(session_path, session)

        report(
            f"round {round_index}: fitting {len(points_by_class[0])}+"
            f"{len(points_by_class[1])} points"
        )
        result = run_analysis(
            config,
            image_path=image_path,
            annotation_session=session,
            output_root=rounds_dir,
            resolved_options=replace(run_options, seed=train_seed),
            features_path=feature_path,
            features_record_path=feature_record,
        )
        run_directory = Path(result["run_directory"])
        with np.load(run_directory / "prediction.npz", allow_pickle=False) as payload:
            arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
        prediction = dense_prediction_from_arrays(arrays)
        metrics: dict[str, Any] = score_prediction(prediction, class_map)
        metrics["agree_prev"] = agreement_with(previous_grid, prediction.prediction_grid)
        training_path = run_directory / "training_history.json"
        training = (
            json.loads(training_path.read_text(encoding="utf-8"))
            if training_path.is_file()
            else {}
        )
        metrics["support_acc"] = float(
            training.get("final_support_accuracy", float("nan"))
        )
        metrics["support_loss"] = float(training.get("final_support_loss", float("nan")))
        metrics["seconds"] = round(perf_counter() - round_started, 3)
        metrics["round"] = round_index
        metrics["train_seed"] = train_seed
        metrics["sample_seed"] = sample_seed
        metrics["run_id"] = result["run_id"]
        metrics["points_by_class"] = [list(group) for group in points_by_class]
        metrics["pool_sizes"] = list(pool_sizes)

        kept_dir = round_dir / "kept"
        _keep_artifacts(run_directory, kept_dir, keep_checkpoint=options.keep_checkpoint)
        _prune_run_directory(run_directory)
        shutil.copy2(session_path, kept_dir / "annotation_session.json")

        if accumulated is None:
            accumulated = Accumulated.empty(prediction.confidence_grid.shape)
            x_coordinates = np.asarray(prediction.x_coordinates, dtype=np.int64)
            y_coordinates = np.asarray(prediction.y_coordinates, dtype=np.int64)
        elif (
            x_coordinates is None
            or y_coordinates is None
            or len(x_coordinates) != len(prediction.x_coordinates)
            or not np.array_equal(x_coordinates, np.asarray(prediction.x_coordinates))
        ):
            raise RepeatError("The prediction grid changed between rounds.")
        accumulated.update(prediction)
        previous_grid = np.asarray(prediction.prediction_grid, dtype=np.int16)

        _write_json(round_dir / "metrics.json", metrics)
        row: dict[str, Any] = {
            **metrics,
            "points_a": len(points_by_class[0]),
            "points_b": len(points_by_class[1]),
            "spread_req": requested_spread,
            "spread": spread_used,
            "pool_a": pool_sizes[0],
            "pool_b": pool_sizes[1],
            "points_xy": json.dumps(metrics["points_by_class"]),
        }
        _append_csv(csv_path, row)
        history.append(row)

        decision = evaluate_round(
            metrics, _history_accuracies(history[:-1]), options
        )
        _write_json(round_dir / "stop_decision.json", decision.as_dict())
        if decision.stop:
            alarm = (
                f"round {round_index}: {decision.reason}. The loop stopped; support "
                f"points are in {session_path.name}."
            )
            report(alarm)
            return _finalize(
                loop_id, loop_dir, manifest, options, history, "alarm", alarm=alarm
            )

    return _finalize(loop_id, loop_dir, manifest, options, history, "completed")


def _history_accuracies(history: Sequence[dict[str, Any]]) -> list[float]:
    """Extract finite accuracies from rows that may have come from a CSV."""
    values: list[float] = []
    for entry in history:
        try:
            value = float(entry["acc"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def _finalize(
    loop_id: str,
    loop_dir: Path,
    manifest: dict[str, Any],
    options: RepeatOptions,
    history: Sequence[dict[str, Any]],
    status: str,
    *,
    alarm: str | None = None,
) -> dict[str, Any]:
    """Write the loop summary and figures, then return the CLI payload."""
    from .repeat_report import write_loop_summary

    accuracies = _history_accuracies(history)
    statistics = history_statistics(accuracies)
    summary_path = write_loop_summary(
        loop_dir,
        loop_id=loop_id,
        manifest=manifest,
        options=options,
        history=history,
        status=status,
        alarm=alarm,
    )
    return {
        "status": status,
        "loop_id": loop_id,
        "loop_directory": str(loop_dir),
        "rounds_completed": len(history),
        "planned_rounds": options.rounds + 1,
        "points_per_class": options.points_per_class,
        "tau": options.tau,
        "min_point_distance": options.min_point_distance,
        "accuracy": statistics,
        "alarm": alarm,
        "summary": str(summary_path),
        "rounds": [
            {
                "round": _as_int(entry.get("round")),
                "acc": _as_number(entry.get("acc")),
                "agree_prev": _as_number(entry.get("agree_prev")),
                "train_seed": _as_int(entry.get("train_seed")),
                "seconds": _as_number(entry.get("seconds")),
            }
            for entry in history
        ],
    }


def _as_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_number(value: Any) -> float | None:
    """Coerce a CSV cell or a float into a strict JSON number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def run_repeat_batch(
    config: HarnessConfig,
    entries: Sequence[dict[str, Any]],
    *,
    options: RepeatOptions,
    output_root: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the same repeat loop on several labeled images in sequence."""
    if not entries:
        raise RepeatError("A repeat batch needs at least one image/ground-truth pair.")
    report = progress or (lambda message: None)
    root = Path(config.output_root if output_root is None else output_root).expanduser()
    batch_id = (
        "repeat-batch-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid4().hex[:6]
    )
    batch_dir = root.resolve() / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        image_path = Path(entry["image"]).expanduser()
        label_path = Path(entry["gt"]).expanduser()
        report(f"[{index}/{len(entries)}] {image_path.name}")
        try:
            result = run_repeat(
                config,
                image_path=image_path,
                label_path=label_path,
                options=options,
                output_root=batch_dir,
                overrides=overrides,
                progress=progress,
            )
        except Exception as error:  # noqa: BLE001 - one image must not kill the batch
            result = {
                "status": "error",
                "image": str(image_path),
                "ground_truth": str(label_path),
                "error": f"{type(error).__name__}: {error}",
            }
            report(f"  failed: {error}")
        result["image"] = str(image_path)
        result["ground_truth"] = str(label_path)
        results.append(result)
    payload = {
        "status": "completed"
        if all(item["status"] in {"completed", "ok"} for item in results)
        else "partial",
        "batch_id": batch_id,
        "batch_directory": str(batch_dir),
        "loops": results,
    }
    _write_json(batch_dir / "repeat_batch.json", payload)
    return payload
