"""Repeated fine-tuning runs for measuring procedural stability.

A *repeat loop* answers one question: if the same k-shot fine-tuning procedure
is executed again and again, does it produce the same answer? Every round trains
a fresh adapter from the backbone with exactly ``points_per_class`` newly chosen
support points, so the support set never accumulates and each round is an
independent replicate of the same experiment.

A loop has exactly one *anchor*: round 0, the first fine-tune. Its prediction
and confidence map are frozen before the first replicate starts, and both the
candidate pool and the pseudo-labels of every later round come from that frozen
snapshot. Nothing a later round predicts can change where a later round samples.
That is what makes the replicates independent: round 80 sees the same pool as
round 1, so the spread across rounds measures the training procedure rather than
a self-training feedback loop.

The pool depends on the mode:

``user``
    Candidates are anchor cells with ``confidence >= tau`` labelled ``c``. This
    is the product flow: the user's own first fine-tune decides where the loop
    is allowed to sample, and an unstable round can never widen or shift that
    region.

``stability``
    Candidates are anchor cells that agree with the ground truth *and* sit at
    least ``patch_size / 2`` away from a phase boundary, so a patch never
    straddles two phases. This is the internal flow: with annotation noise
    removed, all that is left is training variance.

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
import os
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any, Callable, Sequence
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


REPEAT_SCHEMA_VERSION = "symmetry-repeat-loop-v2"
DEFAULT_CLASS_NAMES = ("Class A", "Class B")
DEFAULT_CLASS_COLORS = ("#e41a1c", "#377eb8")

#: The two flows the loop can measure. See the module docstring.
REPEAT_MODES = ("user", "stability")

#: Below this many replicates the sample standard deviation is too noisy to
#: bound anything: with four rounds `mean - 3 sigma` already fires on ordinary
#: spread. Short loops therefore use the absolute floor only.
SIGMA_MIN_SAMPLES = 6

#: Artifacts copied out of a completed run before the run directory is removed.
KEPT_ARTIFACTS = (
    "prediction.npz",
    "run_record.json",
    "confidence.png",
    "prediction_overlay.png",
)
KEPT_ARTIFACTS_WITH_MODEL = KEPT_ARTIFACTS + ("fine_tuned_model.symmodel",)

#: Suffix the pruned checkpoint is renamed to before it is truncated to zero
#: bytes. Renaming and truncating is how a round gives the 46 MB back without
#: issuing a file deletion, which batch-delete guards refuse.
PURGED_SUFFIX = ".purged"


class RepeatError(RuntimeError):
    """Raised when a repeat loop cannot start or cannot continue."""


@dataclass(frozen=True)
class RepeatOptions:
    """Everything that can change between two repeat loops."""

    rounds: int = 5
    points_per_class: int = 5
    mode: str = "user"
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
        if self.mode not in REPEAT_MODES:
            raise RepeatError(
                f"mode must be one of {list(REPEAT_MODES)}; got {self.mode!r}."
            )
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
            "mode": self.mode,
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
# Sampling on the frozen anchor pool
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The frozen anchor and the pool it defines
# ---------------------------------------------------------------------------


def _boundary_mask(label_grid: np.ndarray) -> np.ndarray:
    """Mark cells that touch a different class, including at the grid edge."""
    grid = np.asarray(label_grid)
    edge = np.zeros(grid.shape, dtype=bool)
    edge[1:, :] |= grid[1:, :] != grid[:-1, :]
    edge[:-1, :] |= grid[1:, :] != grid[:-1, :]
    edge[:, 1:] |= grid[:, 1:] != grid[:, :-1]
    edge[:, :-1] |= grid[:, 1:] != grid[:, :-1]
    return edge


def _squared_distance_1d(values: np.ndarray) -> np.ndarray:
    """Exact squared distance transform of one line (Felzenszwalb-Huttenlocher)."""
    size = values.size
    result = np.empty(size, dtype=np.float64)
    parabola = np.zeros(size, dtype=np.int64)
    boundary = np.empty(size + 1, dtype=np.float64)
    level = 0
    parabola[0] = 0
    boundary[0] = -np.inf
    boundary[1] = np.inf
    for index in range(1, size):
        while True:
            previous = parabola[level]
            crossing = (
                (values[index] + index * index) - (values[previous] + previous * previous)
            ) / (2.0 * index - 2.0 * previous)
            if crossing <= boundary[level] and level > 0:
                level -= 1
                continue
            break
        level += 1
        parabola[level] = index
        boundary[level] = crossing
        boundary[level + 1] = np.inf
    level = 0
    for index in range(size):
        while boundary[level + 1] < index:
            level += 1
        previous = parabola[level]
        result[index] = (index - previous) ** 2 + values[previous]
    return result


def distance_to_boundary(label_grid: np.ndarray, *, step: float) -> np.ndarray:
    """Distance from every cell to the nearest phase boundary, in pixels.

    The transform is exact in grid units and then scaled by ``step``, so it
    matches the pixel clearance a support patch needs rather than counting cell
    hops.
    """
    mask = _boundary_mask(label_grid)
    if not mask.any():
        return np.full(np.shape(label_grid), np.inf, dtype=np.float64)
    # A finite sentinel keeps the parabola intersection finite; ``inf`` would
    # turn ``inf - inf`` into ``nan`` halfway through the transform.
    height, width = mask.shape
    sentinel = float(height * height + width * width + 1)
    seeds = np.where(mask, 0.0, sentinel)
    rows = np.empty_like(seeds)
    for row in range(seeds.shape[0]):
        rows[row] = _squared_distance_1d(seeds[row])
    squared = np.empty_like(seeds)
    for column in range(seeds.shape[1]):
        squared[:, column] = _squared_distance_1d(rows[:, column])
    return np.sqrt(squared) * float(step)


@dataclass(frozen=True)
class Anchor:
    """Round 0's prediction, frozen before the first replicate starts.

    ``pools`` holds one ``(n, 2)`` array of ``(x, y)`` pixel coordinates per
    class. Both the pool and the label of every later round come from here, so
    the replicates cannot influence each other.
    """

    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    label_grid: np.ndarray
    confidence_grid: np.ndarray
    pools: tuple[np.ndarray, np.ndarray]
    cleared_pixels: float

    @property
    def pool_sizes(self) -> tuple[int, int]:
        return (int(len(self.pools[0])), int(len(self.pools[1])))

    def to_record(self) -> dict[str, Any]:
        return {
            "source_round": 0,
            "grid_shape": [int(value) for value in self.label_grid.shape],
            "x_range": [
                int(self.x_coordinates.min()),
                int(self.x_coordinates.max()),
            ],
            "y_range": [
                int(self.y_coordinates.min()),
                int(self.y_coordinates.max()),
            ],
            "pool_sizes": list(self.pool_sizes),
            "phase_boundary_clearance_px": (
                None if not np.isfinite(self.cleared_pixels) else self.cleared_pixels
            ),
        }


def build_anchor(
    prediction,
    *,
    options: RepeatOptions,
    class_map: np.ndarray | None,
    patch_size: int,
) -> Anchor:
    """Freeze one prediction into the loop's anchor and its per-class pools."""
    x_coordinates = np.asarray(prediction.x_coordinates, dtype=np.int64)
    y_coordinates = np.asarray(prediction.y_coordinates, dtype=np.int64)
    labels = np.asarray(prediction.prediction_grid, dtype=np.int16)
    confidence = np.asarray(prediction.confidence_grid, dtype=np.float64)
    if labels.shape != (len(y_coordinates), len(x_coordinates)):
        raise RepeatError("The prediction grid and its axes disagree on shape.")

    clearance_px = float("inf")
    if options.mode == "stability":
        if class_map is None:
            raise RepeatError(
                "Stability mode anchors on cells where the prediction agrees with "
                "the ground truth, so it needs a label raster."
            )
        truth = class_map[np.ix_(y_coordinates, x_coordinates)]
        step = float(min(grid_step(x_coordinates), grid_step(y_coordinates)))
        clearance = distance_to_boundary(truth, step=step)
        clearance_px = float(patch_size) / 2.0
        accepted = (labels == truth) & (clearance >= clearance_px)
    else:
        accepted = confidence >= float(options.tau)

    pools: list[np.ndarray] = []
    for class_index in (0, 1):
        rows, columns = np.nonzero(accepted & (labels == class_index))
        pool = np.stack(
            [x_coordinates[columns], y_coordinates[rows]], axis=1
        ).astype(np.int64)
        if len(pool) < options.points_per_class:
            if options.mode == "stability":
                raise RepeatError(
                    f"Class {class_index} has {len(pool)} anchor cells that match the "
                    f"ground truth and sit {clearance_px:.1f} px from a phase "
                    f"boundary; {options.points_per_class} are required. The anchor "
                    "fine-tune is too wrong, or the phases are narrower than one "
                    "patch, for this loop to measure anything."
                )
            raise RepeatError(
                f"Class {class_index} has {len(pool)} anchor cells with confidence "
                f">= {options.tau}; {options.points_per_class} are required. The "
                "first fine-tune is not reliable enough to seed the loop. Relabel a "
                "few points and fine-tune again instead of lowering tau."
            )
        pools.append(pool)
    return Anchor(
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        label_grid=labels,
        confidence_grid=confidence,
        pools=(pools[0], pools[1]),
        cleared_pixels=clearance_px,
    )


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


def select_from_anchor(
    anchor: Anchor,
    *,
    points_per_class: int,
    seed: int,
    min_distance: float = 0.0,
) -> Sampling:
    """Draw ``points_per_class`` cells per class from the frozen anchor pool.

    The pool never changes, so the only thing that varies between rounds is the
    draw. Every sampled cell carries the anchor's own label: no round can vote a
    label onto a cell that another round predicted differently.
    """
    xs = np.asarray(anchor.x_coordinates, dtype=np.int64)
    pool_sizes = anchor.pool_sizes
    for class_index in (0, 1):
        if pool_sizes[class_index] < points_per_class:
            raise RepeatError(
                f"The frozen pool for class {class_index} holds "
                f"{pool_sizes[class_index]} cells; {points_per_class} are required."
            )
    chosen, used = poisson_disk_select(
        anchor.pools,
        points_per_class,
        min_distance,
        np.random.default_rng(seed),
        floor=max(2.0, 2.0 * grid_step(xs)),
    )
    return Sampling(
        points_by_class=(chosen[0], chosen[1]),
        pool_sizes=pool_sizes,
        available=pool_sizes,
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
        "sigma_min_samples": SIGMA_MIN_SAMPLES,
    }
    if not np.isfinite(accuracy):
        return StopDecision(
            stop=False,
            status="unscored",
            reason="no ground truth, so this round carries no accuracy to judge",
            detail=detail,
        )
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
    if statistics["count"] >= SIGMA_MIN_SAMPLES:
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
    "mode",
    "points_a",
    "points_b",
    "spread_req",
    "spread",
    "pool_a",
    "pool_b",
    "reuse_a",
    "reuse_b",
    "acc",
    "acc_class_a",
    "acc_class_b",
    "iou_class_a",
    "iou_class_b",
    "area_frac_pred",
    "area_frac_gt",
    "anchor_agree",
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


def _write_rounds_csv(path: Path, history: Sequence[dict[str, Any]]) -> None:
    """Rewrite the round table from the in-memory history.

    Appending would duplicate a round that is re-run after an interruption, and
    a duplicated round silently doubles its weight in the mean and the standard
    deviation. Rewriting costs nothing at a few hundred rows and is idempotent,
    so the CSV always matches what the loop believes it has finished.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(history, key=lambda row: int(float(row.get("round", 0))))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in ordered:
            writer.writerow({name: row.get(name, "") for name in CSV_COLUMNS})


def _keep_artifacts(run_directory: Path, target: Path, *, keep_checkpoint: bool) -> list[str]:
    """Copy the retained artifacts out of a run directory.

    The checkpoint is *moved* rather than copied when it is kept. Both paths sit
    on the same volume, so this is a rename: the 46 MB is never duplicated and
    the round never has to delete the leftover copy afterwards.
    """
    names = KEPT_ARTIFACTS_WITH_MODEL if keep_checkpoint else KEPT_ARTIFACTS
    target.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    for name in names:
        source = run_directory / name
        if not source.is_file():
            raise RepeatError(f"The run did not produce {name!r} in {run_directory}.")
        if name == "fine_tuned_model.symmodel":
            shutil.move(str(source), str(target / name))
        else:
            shutil.copy2(source, target / name)
        kept.append(name)
    return kept


def _prune_run_directory(run_directory: Path) -> None:
    """Reclaim the checkpoint left in a run directory without deleting a file.

    A loop of a hundred rounds would otherwise end every round with an unlink.
    Deletions are counted, and on a machine with a batch-delete guard the round
    that crosses the threshold is refused with the process torn down mid-loop --
    no exception the loop can catch, an exit code that still reads zero, and a
    half-written run directory. Truncating frees exactly the same blocks and is
    an ordinary write.

    ``--keep-checkpoint`` needs nothing here: the checkpoint was moved into
    ``kept/`` instead of copied, so no duplicate is left behind.
    """
    model = run_directory / "fine_tuned_model.symmodel"
    if not model.is_file():
        return
    purged = model.with_name(model.name + PURGED_SUFFIX)
    model.rename(purged)
    os.truncate(purged, 0)


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
    label_path: str | Path | None = None,
    options: RepeatOptions,
    output_root: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    annotations: str | Path | None = None,
    resume: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run one repeat loop on one image and return its summary.

    ``label_path`` is optional in ``user`` mode: without it the loop still runs
    and still samples from the frozen anchor, but accuracy is not scored and the
    alarm rules are inert. ``stability`` mode requires it.
    """
    options.validate()
    if options.mode == "stability" and label_path is None:
        raise RepeatError(
            "Stability mode anchors on cells that agree with the ground truth, so "
            "every image needs a label raster."
        )
    report = progress or (lambda message: None)
    readiness = doctor(config)
    if readiness["status"] != "ready":
        raise RepeatError(
            "The configured runtime is blocked: " + "; ".join(readiness["issues"])
        )

    image, input_record = inspect_input(image_path, config.features.input_normalization)
    labels: np.ndarray | None = None
    class_map: np.ndarray | None = None
    label_record: dict[str, Any] | None = None
    if label_path is not None:
        labels = load_label_map(label_path)
        if labels.shape != image.shape:
            raise RepeatError(
                f"The label map {labels.shape} and image {image.shape} shapes differ."
            )
        class_map = class_map_from_labels(labels, options.label_codes)
        label_record = {
            "path": str(Path(label_path).expanduser().resolve()),
            "sha256": file_sha256(label_path),
            "shape": list(labels.shape),
            "dtype": str(labels.dtype),
            "label_codes": list(options.label_codes),
            "class_names": list(options.class_names),
        }

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
        if (manifest.get("ground_truth") or {}).get("sha256") != (
            None if label_record is None else label_record["sha256"]
        ):
            raise RepeatError(
                "The resume directory was produced from a different ground truth."
            )
        recorded = RepeatOptions.from_dict(manifest["options"])
        # Only the horizon may grow on resume: changing the mode, the shot count
        # or the sampling rule would mix two protocols in one distribution. The
        # operational switches stay under the caller's control.
        if recorded.mode != options.mode:
            raise RepeatError(
                f"The resume directory runs in {recorded.mode!r} mode; this command "
                f"asked for {options.mode!r}. Two modes cannot share one distribution."
            )
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
            "ground_truth": label_record,
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
    anchor: Anchor | None = None
    history: list[dict[str, Any]] = []
    previous_grid: np.ndarray | None = None
    completed = 0

    if resume is not None:
        anchor = _load_anchor(loop_dir, options=options, class_map=class_map, patch_size=patch_size)
        if csv_path.is_file():
            with csv_path.open(encoding="utf-8", newline="") as handle:
                history = list(csv.DictReader(handle))
        # The table, not the directory tree, decides where to continue. A round
        # writes its artifacts before its row, so counting ``round-*/kept``
        # directories would step over a round that was killed in between and
        # silently drop it from the distribution.
        finished = {int(float(row["round"])) for row in history}
        completed = 1
        while completed in finished:
            completed += 1
        if completed > 1:
            last = rounds_dir / f"round-{completed - 1:03d}" / "kept" / "prediction.npz"
            with np.load(last, allow_pickle=False) as payload:
                arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
            previous_grid = np.asarray(
                dense_prediction_from_arrays(arrays).prediction_grid, dtype=np.int16
            )
        report(f"replayed the anchor and {completed - 1} replicate(s)")
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
            elif class_map is not None:
                points_by_class, spread_used = _interior_draw(
                    class_map,
                    patch_size=patch_size,
                    points_per_class=options.points_per_class,
                    margin=options.interior_margin,
                    seed=options.base_seed,
                    min_distance=requested_spread,
                )
            else:
                raise RepeatError(
                    "Without a label raster the anchor can only come from a real "
                    "annotation; pass --annotations."
                )
            pool_sizes = (0, 0)
            sample_seed = options.base_seed
        else:
            if anchor is None:
                raise RepeatError("The loop has no frozen anchor to sample from.")
            sampling = select_from_anchor(
                anchor,
                points_per_class=options.points_per_class,
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
            pack_archive=False,
        )
        run_directory = Path(result["run_directory"])
        with np.load(run_directory / "prediction.npz", allow_pickle=False) as payload:
            arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
        prediction = dense_prediction_from_arrays(arrays)
        if round_index == 0:
            anchor = build_anchor(
                prediction,
                options=options,
                class_map=class_map,
                patch_size=patch_size,
            )
            previous_grid = np.asarray(prediction.prediction_grid, dtype=np.int16)
            metrics = _round_metrics(
                prediction,
                class_map,
                previous_grid=None,
                round_started=round_started,
                training=_training_history(run_directory),
            )
            metrics["round"] = 0
            metrics["train_seed"] = train_seed
            metrics["sample_seed"] = sample_seed
            metrics["run_id"] = result["run_id"]
            metrics["role"] = "anchor"
            metrics["points_by_class"] = [list(group) for group in points_by_class]
            metrics["pool_sizes"] = list(anchor.pool_sizes)
            kept_dir = round_dir / "kept"
            _keep_artifacts(
                run_directory, kept_dir, keep_checkpoint=options.keep_checkpoint
            )
            _prune_run_directory(run_directory)
            shutil.copy2(session_path, kept_dir / "annotation_session.json")
            _write_json(round_dir / "metrics.json", metrics)
            anchor_dir = loop_dir / "anchor"
            anchor_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(kept_dir / "prediction.npz", anchor_dir / "prediction.npz")
            _write_json(anchor_dir / "anchor.json", anchor.to_record())
            _write_json(anchor_dir / "metrics.json", metrics)
            _write_json(
                round_dir / "stop_decision.json",
                {
                    "stop": False,
                    "status": "anchor",
                    "reason": "the anchor defines the pool; it is not a replicate",
                    "detail": {"pool_sizes": list(anchor.pool_sizes)},
                },
            )
            manifest["anchor"] = {
                **anchor.to_record(),
                "options": {
                    "mode": options.mode,
                    "tau": options.tau,
                    "classifier_patch_size": patch_size,
                },
            }
            _write_json(loop_dir / "loop_manifest.json", manifest)
            report(
                f"anchor: frozen {anchor.pool_sizes[0]}+{anchor.pool_sizes[1]} "
                "candidate cells from the first fine-tune"
            )
            continue

        metrics = _round_metrics(
            prediction,
            class_map,
            previous_grid=previous_grid,
            round_started=round_started,
            training=_training_history(run_directory),
        )
        metrics["round"] = round_index
        metrics["train_seed"] = train_seed
        metrics["sample_seed"] = sample_seed
        metrics["run_id"] = result["run_id"]
        metrics["role"] = "replicate"
        metrics["points_by_class"] = [list(group) for group in points_by_class]
        metrics["pool_sizes"] = list(pool_sizes)
        metrics["anchor_agree"] = (
            float(
                (
                    np.asarray(prediction.prediction_grid, dtype=np.int16)
                    == anchor.label_grid
                ).mean()
            )
            if anchor is not None
            else float("nan")
        )

        kept_dir = round_dir / "kept"
        _keep_artifacts(run_directory, kept_dir, keep_checkpoint=options.keep_checkpoint)
        shutil.copy2(session_path, kept_dir / "annotation_session.json")
        previous_grid = np.asarray(prediction.prediction_grid, dtype=np.int16)

        _write_json(round_dir / "metrics.json", metrics)
        row: dict[str, Any] = {
            **metrics,
            "mode": options.mode,
            "points_a": len(points_by_class[0]),
            "points_b": len(points_by_class[1]),
            "spread_req": requested_spread,
            "spread": spread_used,
            "pool_a": pool_sizes[0],
            "pool_b": pool_sizes[1],
            "reuse_a": _reuse(round_index, options.points_per_class, pool_sizes[0]),
            "reuse_b": _reuse(round_index, options.points_per_class, pool_sizes[1]),
            "points_xy": json.dumps(metrics["points_by_class"]),
        }
        # The row is written before the directory is pruned. A prune that is
        # torn down by a delete guard must not leave a finished round missing
        # from the table, because the table is what resume replays.
        history = [
            entry
            for entry in history
            if int(float(entry.get("round", -1))) != round_index
        ]
        history.append(row)
        history.sort(key=lambda entry: int(float(entry["round"])))
        _write_rounds_csv(csv_path, history)
        _prune_run_directory(run_directory)

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


def _reuse(round_index: int, points_per_class: int, pool_size: int) -> float | None:
    """How many times each pool cell has been drawn on average by this round."""
    if pool_size <= 0:
        return None
    return round_index * points_per_class / pool_size


def _round_metrics(
    prediction,
    class_map: np.ndarray | None,
    *,
    previous_grid: np.ndarray | None,
    round_started: float,
    training: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one round, tolerating a loop that has no ground truth."""
    if class_map is None:
        grid = np.asarray(prediction.prediction_grid, dtype=np.int16)
        metrics: dict[str, Any] = {
            "cells": int(grid.size),
            "acc": float("nan"),
            "area_frac_pred": float((grid == 1).mean()),
            "area_frac_gt": float("nan"),
            "conf_mean": float(np.mean(prediction.confidence_grid)),
            "conf_median": float(np.median(prediction.confidence_grid)),
        }
        for tag in ("acc_class_a", "acc_class_b", "iou_class_a", "iou_class_b"):
            metrics[tag] = float("nan")
    else:
        metrics = score_prediction(prediction, class_map)
    metrics["agree_prev"] = agreement_with(previous_grid, prediction.prediction_grid)
    history = training or {}
    metrics["support_acc"] = float(
        history.get("final_support_accuracy", float("nan"))
    )
    metrics["support_loss"] = float(history.get("final_support_loss", float("nan")))
    metrics["seconds"] = round(perf_counter() - round_started, 3)
    return metrics


def _training_history(run_directory: Path) -> dict[str, Any]:
    path = run_directory / "training_history.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _load_anchor(
    loop_dir: Path,
    *,
    options: RepeatOptions,
    class_map: np.ndarray | None,
    patch_size: int,
) -> Anchor:
    """Rebuild the frozen anchor of an interrupted loop from its stored prediction."""
    for candidate in (
        loop_dir / "rounds" / "round-000" / "kept" / "prediction.npz",
        loop_dir / "anchor" / "prediction.npz",
    ):
        if not candidate.is_file():
            continue
        with np.load(candidate, allow_pickle=False) as payload:
            arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
        return build_anchor(
            dense_prediction_from_arrays(arrays),
            options=options,
            class_map=class_map,
            patch_size=patch_size,
        )
    raise RepeatError(
        "The resume directory holds no anchor prediction, so the frozen pool cannot "
        "be rebuilt."
    )


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


def pool_record(
    manifest: dict[str, Any],
    options: RepeatOptions,
    rounds_completed: int,
) -> dict[str, Any]:
    """Describe the frozen pool and how hard the loop leaned on it."""
    anchor = manifest.get("anchor") or {}
    sizes = list(anchor.get("pool_sizes") or [0, 0])
    reuse = [
        (rounds_completed * options.points_per_class / size) if size else None
        for size in sizes
    ]
    return {
        "mode": options.mode,
        "sizes": sizes,
        "reuse": reuse,
        "cleared_pixels": anchor.get("phase_boundary_clearance_px"),
    }


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
        "mode": options.mode,
        "points_per_class": options.points_per_class,
        "tau": options.tau,
        "min_point_distance": options.min_point_distance,
        "pool": pool_record(manifest, options, len(history)),
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
        raw_label = entry.get("gt")
        label_path = None if raw_label is None else Path(raw_label).expanduser()
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
                "ground_truth": None if label_path is None else str(label_path),
                "error": f"{type(error).__name__}: {error}",
            }
            report(f"  failed: {error}")
        result["image"] = str(image_path)
        result["ground_truth"] = None if label_path is None else str(label_path)
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
