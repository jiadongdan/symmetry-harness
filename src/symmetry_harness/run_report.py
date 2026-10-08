"""Fixed-template Markdown reports for completed Harness runs.

This module turns a *completed* run directory into two durable, self-contained
artifacts:

* ``report_summary.md`` -- a fixed-template human report, and
* ``figures/`` -- the PNG figures the report embeds.

It is deliberately a presentation layer. It only reads artifacts that already
exist on disk (``run_record.json`` / ``batch_record.json`` / ``prediction.npz`` /
``features.npz`` / ``annotation_session.json`` / ``training_history.json``) and
it never imports ``gradio``, ``torch`` or ``symmlearn``; it never computes
features, fine-tunes, restores a model or runs dense prediction. Re-running it
can therefore only re-render pixels and re-emit prose.

Two run kinds are supported:

``fine-tune``
    A ``symmetry-<utc>-<hash>`` directory written by
    :func:`symmetry_harness.workflow.run_analysis`. It carries the support
    annotations, the eight-channel representation, the adapter training history
    and one dense prediction.

``prediction-batch``
    A ``prediction-<utc>-<hash>`` directory written by
    :mod:`symmetry_harness.prediction_workflow`. It carries a batch record plus
    one self-contained ``items/<item-id>/`` directory per predicted image.

The report is intentionally *fixed*: the section order, the figure set and the
caveats do not vary with the data. Only the values and the figures do. The one
free-form slot is ``Analysis Context``, which the caller supplies (typically an
agent summarising the scientific context of the interactive session).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import resources
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import __version__
from .analysis import DensePrediction, dense_prediction_from_arrays
from .image_io import unit_to_uint8
from .result_exports import (
    DEFAULT_OVERLAY_ALPHA,
    build_categorical_legend,
    compose_with_legend,
    render_result_variants,
    write_png,
)

REPORT_FILENAME = "report_summary.md"
FIGURES_DIRNAME = "figures"

FINE_TUNE_PREFIX = "symmetry-"
PREDICTION_PREFIX = "prediction-"

KIND_FINE_TUNE = "fine-tune"
KIND_FINE_TUNE_SESSION = "fine-tune-session"
KIND_PREDICTION = "prediction-batch"

#: Orientation and rotation-response channels are displayed on ``[-1, 1]``;
#: the source image and reflection strength use ``[0, 1]``. This mirrors the
#: annotation UI so a report figure and the live UI never disagree.
_SIGNED_FEATURE_CHANNELS = frozenset(
    {
        "reflection_sin_2theta",
        "reflection_cos_2theta",
        "rotation_2_fold",
        "rotation_3_fold",
        "rotation_4_fold",
        "rotation_6_fold",
    }
)

_CONFIDENCE_COLORMAP = "viridis"
_ENTROPY_COLORMAP = "magma"

_TEMPLATE_IMAGE = "run_report_image.md"
_TEMPLATE_ITEM = "run_report_item.md"
_TEMPLATE_BATCH = "run_report_batch.md"
_TEMPLATE_SESSION = "run_report_session.md"

# Figure basenames. ``00`` is the human-facing overview; the stable ``01`` to
# ``08`` files remain the inspectable source figures used to build it.
FIG_OVERVIEW = "00_results_overview.png"
FIG_INPUT = "01_input.png"
FIG_SUPPORT = "02_support_points.png"
FIG_SYMMETRY = "03_symmetry_maps.png"
FIG_TRAINING = "04_training.png"
FIG_MASK = "05_prediction_mask.png"
FIG_OVERLAY = "06_prediction_overlay.png"
FIG_CONFIDENCE = "07_confidence.png"
FIG_ENTROPY = "08_entropy.png"

_MONTAGE_TILE = 260
_MONTAGE_LABEL_HEIGHT = 30
_MONTAGE_COLUMNS = 4
_MONTAGE_ROWS = 2

_TRAINING_WIDTH = 960
_TRAINING_PANEL_HEIGHT = 300
_TRAINING_GAP = 56
_TRAINING_MARGIN = 72

_SUPPORT_MARKER_RADIUS = 7

_OVERVIEW_PANEL_WIDTH = 520
_OVERVIEW_PANEL_HEIGHT = 520
_OVERVIEW_LABEL_HEIGHT = 34
_OVERVIEW_GAP = 18


class ReportError(ValueError):
    """Raised when a run directory cannot produce a report."""


@dataclass(frozen=True)
class ReportResult:
    """Paths and identity of one generated report."""

    run_id: str
    kind: str
    report_path: Path
    figure_paths: tuple[Path, ...] = ()
    item_reports: tuple[Path, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the CLI contract."""
        return {
            "status": "ok",
            "run_id": self.run_id,
            "kind": self.kind,
            "report": str(self.report_path),
            "figures": [str(path) for path in self.figure_paths],
            "item_reports": [str(path) for path in self.item_reports],
        }


@dataclass
class _ClassEntry:
    index: int
    name: str
    color: str
    points_xy: list[tuple[int, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReportError(f"Required record is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ReportError(f"Record is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ReportError(f"Record is not a JSON object: {path}")
    return payload


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    return _read_json(path) if path.is_file() else None


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ReportError(f"Required array bundle is missing: {path}")
    with np.load(path, allow_pickle=False) as bundle:
        return {name: bundle[name] for name in bundle.files}


def _fmt(value: Any, *, digits: int = 6) -> str:
    """Format one value for a Markdown table cell."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            return str(value)
        if value != 0 and abs(value) < 1e-4:
            return f"{value:.{digits}g}"
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def _fmt_any(value: Any, *, digits: int = 6) -> str:
    """Format a value for a table cell, expanding sequences element-wise."""
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt_any(item, digits=digits) for item in value) + "]"
    if value is None:
        return "-"
    return _fmt(value, digits=digits)


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _md_table(
    header: Sequence[str], rows: Sequence[Sequence[Any]]
) -> str:
    """Render a GitHub-flavoured Markdown table with aligned pipes.

    Cells are formatted through :func:`_fmt_any`, so callers may pass raw record
    values (floats, lists, ``None``) without pre-formatting them.
    """

    def cell(value: Any) -> str:
        return _escape_cell(_fmt_any(value, digits=4))

    columns = len(header)
    lines = [
        "| " + " | ".join(cell(value) for value in header) + " |",
        "| " + " | ".join("---" for _ in range(columns)) + " |",
    ]
    for row in rows:
        padded = list(row) + [""] * (columns - len(row))
        lines.append("| " + " | ".join(cell(value) for value in padded) + " |")
    return "\n".join(lines)


def _bullet_list(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- (none)"


def _font(size: int = 13) -> Any:
    """Return a usable font, preferring a sized default face."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - Pillow < 10.1
        return ImageFont.load_default()


def _template(name: str) -> str:
    return (
        resources.files(__package__) / "templates" / name
    ).read_text(encoding="utf-8")


def _rgb(image: np.ndarray) -> np.ndarray:
    """Promote a unit-scaled or uint8 2-D image to ``uint8`` RGB."""
    values = np.asarray(image)
    if values.dtype != np.uint8:
        values = unit_to_uint8(values)
    if values.ndim == 2:
        values = np.repeat(values[..., None], 3, axis=2)
    return np.ascontiguousarray(values, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------
def discover_runs(
    output_root: str | Path, *, prefix: str | None = None
) -> list[Path]:
    """Return run directories under ``output_root``, newest first.

    Ordering is by directory name. Run names embed the UTC creation timestamp
    (``<prefix><YYYYMMDD>T<HHMMSS>Z-<hash>``), so a name sort is a time sort and
    needs no extra I/O or clock agreement.
    """
    root = Path(output_root).expanduser()
    if not root.is_dir():
        return []
    candidates = [
        entry
        for entry in root.iterdir()
        if entry.is_dir() and (prefix is None or entry.name.startswith(prefix))
    ]
    return sorted(candidates, key=lambda entry: entry.name, reverse=True)


def resolve_run(
    value: str | Path | None,
    output_root: str | Path,
    *,
    prefix: str | None = FINE_TUNE_PREFIX,
) -> Path:
    """Resolve a run directory from an explicit path, a run id, or ``latest``.

    ``None`` and ``"latest"`` select the newest run matching ``prefix``.
    """
    if value is None or str(value).strip().lower() == "latest":
        runs = discover_runs(output_root, prefix=prefix)
        if not runs:
            searched = Path(output_root).expanduser()
            raise ReportError(
                f"No run directories matching {prefix!r} under {searched}."
            )
        return runs[0]

    candidate = Path(value).expanduser()
    if candidate.is_dir():
        return candidate

    named = Path(output_root).expanduser() / str(value)
    if named.is_dir():
        return named

    for run in discover_runs(output_root, prefix=None):
        if run.name == str(value) or run.name.startswith(str(value)):
            return run

    raise ReportError(f"Run directory not found: {value}")


def resolve_fine_tune_session(
    output_root: str | Path, session: str = "latest"
) -> tuple[str, list[Path]]:
    """Resolve all fine-tune runs belonging to one recorded UI session.

    Older runs have no ``session_id``. For ``latest`` they degrade to a
    one-run session so the agent command remains useful across upgrades.
    """
    runs = discover_runs(output_root, prefix=FINE_TUNE_PREFIX)
    records: list[tuple[Path, Mapping[str, Any]]] = []
    for run in runs:
        try:
            record = _read_json(run / "run_record.json")
        except ReportError:
            continue
        records.append((run, record))
    if not records:
        raise ReportError(f"No fine-tune runs found under {Path(output_root).expanduser()}.")

    requested = str(session).strip()
    if requested.casefold() == "latest":
        selected_id = next(
            (
                str(record.get("session_id"))
                for _, record in records
                if record.get("session_id")
            ),
            None,
        )
        if selected_id is None:
            newest = records[0][0]
            return f"legacy-{newest.name}", [newest]
    else:
        selected_id = requested

    matching = [
        run
        for run, record in records
        if str(record.get("session_id") or "") == selected_id
    ]
    if not matching:
        raise ReportError(f"Fine-tuning session not found: {session}")
    return selected_id, sorted(matching, key=lambda path: path.name)


def detect_kind(run_dir: str | Path) -> str:
    """Classify a run directory by its record files."""
    directory = Path(run_dir)
    if (directory / "run_record.json").is_file():
        return KIND_FINE_TUNE
    if (directory / "batch_record.json").is_file():
        return KIND_PREDICTION
    raise ReportError(
        f"Not a completed Harness run (no run_record.json or batch_record.json): "
        f"{directory}"
    )


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def _feature_display_range(name: str) -> tuple[float, float]:
    return (-1.0, 1.0) if name in _SIGNED_FEATURE_CHANNELS else (0.0, 1.0)


def _feature_tile(channel: np.ndarray, name: str) -> np.ndarray:
    """Render one feature channel as a fixed-range ``uint8`` grayscale tile."""
    values = np.asarray(channel, dtype=np.float32)
    minimum, maximum = _feature_display_range(name)
    normalized = (np.clip(values, minimum, maximum) - minimum) / (
        maximum - minimum
    )
    return np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)


def build_input_figure(image: np.ndarray) -> np.ndarray:
    """Render the normalized input image as ``uint8`` RGB."""
    return _rgb(image)


def build_symmetry_maps_figure(
    features: np.ndarray, channel_names: Sequence[Any]
) -> np.ndarray:
    """Render the 2x4 montage of the eight engineering channels.

    Each tile carries its own fixed cross-image colour range, exactly as the
    annotation UI displays it, so a signed channel and an unsigned channel are
    never compared on a shared scale.
    """
    channels = np.asarray(features)
    names = [str(name) for name in channel_names]
    if channels.ndim != 3 or channels.shape[0] != len(names):
        raise ReportError(
            "Feature bundle does not match its channel names: "
            f"{channels.shape} vs {len(names)} names."
        )

    columns = min(_MONTAGE_COLUMNS, max(1, len(names)))
    rows = (len(names) + columns - 1) // columns
    tile = _MONTAGE_TILE
    canvas = Image.new(
        "RGB",
        (tile * columns, (tile + _MONTAGE_LABEL_HEIGHT) * rows),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    font = _font(13)
    for position, (channel, name) in enumerate(zip(channels, names)):
        row, column = divmod(position, columns)
        tile_image = Image.fromarray(_feature_tile(channel, name)).convert("RGB")
        fitted = tile_image.resize((tile, tile), Image.Resampling.LANCZOS)
        x = column * tile
        y = row * (tile + _MONTAGE_LABEL_HEIGHT)
        canvas.paste(fitted, (x, y))
        minimum, maximum = _feature_display_range(name)
        caption = f"{name}  [{minimum:g}, {maximum:g}]"
        draw.text(
            (x + 6, y + tile + 6), caption, fill=(0, 0, 0), font=font
        )
        draw.rectangle(
            [x, y, x + tile - 1, y + tile - 1], outline=(160, 160, 160)
        )
    return np.asarray(canvas, dtype=np.uint8)


def build_support_overlay_figure(
    image: np.ndarray, classes: Sequence[_ClassEntry]
) -> np.ndarray:
    """Overlay every support point on the source image, coloured by class."""
    canvas = Image.fromarray(_rgb(image)).copy()
    draw = ImageDraw.Draw(canvas)
    for entry in sorted(classes, key=lambda item: item.index):
        color = entry.color
        for x, y in entry.points_xy:
            cx, cy = int(x), int(y)
            radius = _SUPPORT_MARKER_RADIUS
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=color,
                outline=(255, 255, 255),
                width=2,
            )
    rendered = np.asarray(canvas, dtype=np.uint8)
    legend = build_categorical_legend(
        [
            {
                "index": entry.index,
                "name": f"{entry.name} ({len(entry.points_xy)} points)",
                "color": entry.color,
            }
            for entry in sorted(classes, key=lambda item: item.index)
        ]
    )
    return compose_with_legend(rendered, legend)


def _log_axis(values: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Return ``[0, 1]`` fractions and tick pairs for a base-10 log axis."""
    transformed = np.log10(np.clip(np.asarray(values, dtype=np.float64), 1e-12, None))
    low = float(transformed.min())
    high = float(transformed.max())
    if high - low < 1e-9:
        high = low + 1.0
    ticks = [
        (0.0, float(10.0**low)),
        (0.5, float(10.0 ** ((low + high) / 2.0))),
        (1.0, float(10.0**high)),
    ]
    return (transformed - low) / (high - low), ticks


def _linear_axis(
    values: np.ndarray, *, low: float | None = None, high: float | None = None
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Return ``[0, 1]`` fractions and tick pairs for a linear axis."""
    array = np.asarray(values, dtype=np.float64)
    resolved_low = float(array.min()) if low is None else float(low)
    resolved_high = float(array.max()) if high is None else float(high)
    if resolved_high - resolved_low < 1e-9:
        resolved_high = resolved_low + 1.0
    ticks = [
        (0.0, resolved_low),
        (0.5, (resolved_low + resolved_high) / 2.0),
        (1.0, resolved_high),
    ]
    return (array - resolved_low) / (resolved_high - resolved_low), ticks


def _draw_line_panel(
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    fractions: np.ndarray,
    color: tuple[int, int, int],
    title: str,
    ticks: Sequence[tuple[float, float]],
    x_note: str,
    font: Any,
) -> None:
    left, top, right, bottom = box
    draw.rectangle([left, top, right, bottom], outline=(120, 120, 120))
    draw.text((left, top - 22), title, fill=(0, 0, 0), font=font)
    for fraction, value in ticks:
        y = int(round(bottom - float(fraction) * (bottom - top)))
        draw.line([(left - 4, y), (left, y)], fill=(120, 120, 120))
        draw.text(
            (left - 80, y - 8), _fmt(float(value), digits=3), fill=(0, 0, 0), font=font
        )
    count = len(fractions)
    if count == 0:
        return
    width = max(right - left, 1)
    points: list[tuple[float, float]] = []
    for index, fraction in enumerate(fractions):
        x = left + (index / (count - 1) if count > 1 else 0.0) * width
        y = bottom - float(np.clip(fraction, 0.0, 1.0)) * (bottom - top)
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=color, width=2)
    else:
        draw.ellipse(
            [
                points[0][0] - 2,
                points[0][1] - 2,
                points[0][0] + 2,
                points[0][1] + 2,
            ],
            fill=color,
        )
    draw.text((right - 110, bottom + 6), x_note, fill=(0, 0, 0), font=font)


def build_training_figure(
    loss: Sequence[float], accuracy: Sequence[float]
) -> np.ndarray:
    """Render the loss and support-accuracy curves as one stacked figure.

    The loss panel uses a base-10 log axis: an adapter loss routinely falls from
    O(1) to O(1e-6) within the first epochs, and a linear axis would collapse the
    entire useful range onto the baseline.
    """
    font = _font(13)
    panel = _TRAINING_PANEL_HEIGHT
    canvas = Image.new(
        "RGB",
        (_TRAINING_WIDTH, _TRAINING_MARGIN + 2 * (panel + _TRAINING_GAP)),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    left = 108
    right = _TRAINING_WIDTH - 40

    loss_values = np.asarray([float(value) for value in loss], dtype=np.float64)
    if loss_values.size:
        fractions, ticks = _log_axis(loss_values)
        _draw_line_panel(
            draw,
            box=(left, _TRAINING_MARGIN, right, _TRAINING_MARGIN + panel),
            fractions=fractions,
            color=(214, 39, 40),
            title="Support loss (log10, training diagnostic)",
            ticks=ticks,
            x_note=f"epochs: {loss_values.size}",
            font=font,
        )

    second_top = _TRAINING_MARGIN + panel + _TRAINING_GAP
    accuracy_values = np.asarray(
        [float(value) for value in accuracy], dtype=np.float64
    )
    if accuracy_values.size:
        fractions, ticks = _linear_axis(accuracy_values, low=0.0, high=1.0)
        _draw_line_panel(
            draw,
            box=(left, second_top, right, second_top + panel),
            fractions=fractions,
            color=(31, 119, 180),
            title="Support accuracy (training diagnostic)",
            ticks=ticks,
            x_note=f"epochs: {accuracy_values.size}",
            font=font,
        )
    return np.asarray(canvas, dtype=np.uint8)


def build_results_overview_figure(
    panels: Sequence[tuple[str, np.ndarray]],
) -> np.ndarray:
    """Compose the four most useful result views into a compact 2x2 summary."""
    selected = list(panels[:4])
    if not selected:
        raise ReportError("A results overview requires at least one panel.")
    columns = 2
    rows = (len(selected) + columns - 1) // columns
    cell_width = _OVERVIEW_PANEL_WIDTH
    cell_height = _OVERVIEW_PANEL_HEIGHT + _OVERVIEW_LABEL_HEIGHT
    canvas = Image.new(
        "RGB",
        (
            columns * cell_width + (columns - 1) * _OVERVIEW_GAP,
            rows * cell_height + (rows - 1) * _OVERVIEW_GAP,
        ),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    for position, (title, array) in enumerate(selected):
        row, column = divmod(position, columns)
        x = column * (cell_width + _OVERVIEW_GAP)
        y = row * (cell_height + _OVERVIEW_GAP)
        draw.text((x + 6, y + 6), title, fill=(0, 0, 0), font=font)
        image = Image.fromarray(_rgb(array))
        image.thumbnail(
            (cell_width, _OVERVIEW_PANEL_HEIGHT), Image.Resampling.LANCZOS
        )
        image_x = x + (cell_width - image.width) // 2
        image_y = y + _OVERVIEW_LABEL_HEIGHT + (
            _OVERVIEW_PANEL_HEIGHT - image.height
        ) // 2
        canvas.paste(image, (image_x, image_y))
        draw.rectangle(
            [x, y + _OVERVIEW_LABEL_HEIGHT, x + cell_width - 1, y + cell_height - 1],
            outline=(190, 190, 190),
        )
    return np.asarray(canvas, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------
def _classes_from_annotations(session: Mapping[str, Any]) -> list[_ClassEntry]:
    entries: list[_ClassEntry] = []
    for raw in session.get("classes", []) or []:
        if not isinstance(raw, Mapping):
            continue
        points = [
            (int(point[0]), int(point[1]))
            for point in raw.get("points_xy", []) or []
            if isinstance(point, Sequence) and len(point) >= 2
        ]
        entries.append(
            _ClassEntry(
                index=int(raw.get("index", len(entries))),
                name=str(raw.get("name", f"Class {len(entries) + 1}")),
                color=str(raw.get("color", "#ffffff")),
                points_xy=points,
            )
        )
    return sorted(entries, key=lambda entry: entry.index)


def _classes_from_names(
    names: Sequence[Any], colors: Sequence[Any]
) -> list[_ClassEntry]:
    entries: list[_ClassEntry] = []
    for index, name in enumerate(names):
        color = str(colors[index]) if index < len(colors) else "#ffffff"
        entries.append(_ClassEntry(index=index, name=str(name), color=color))
    return entries


def _prediction_from_item(item_dir: Path) -> DensePrediction:
    return dense_prediction_from_arrays(_load_npz(item_dir / "prediction.npz"))


def _png_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _item_figures(
    item_dir: Path,
    *,
    figures_dir: Path,
    classes: Sequence[_ClassEntry],
    stride: int,
    include_training: bool,
    include_support: bool,
) -> dict[str, Path]:
    """Render every figure one image-level report can embed.

    Returns a mapping of template placeholder -> written PNG path. Missing
    optional inputs (features, prediction, training history) simply omit the
    corresponding key so the template can degrade gracefully.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    image = np.load(item_dir / "input.npy")
    written["fig_input"] = write_png(
        build_input_figure(image), figures_dir / FIG_INPUT
    )

    if include_support and classes and any(entry.points_xy for entry in classes):
        written["fig_support"] = write_png(
            build_support_overlay_figure(image, classes),
            figures_dir / FIG_SUPPORT,
        )

    features_path = item_dir / "features.npz"
    if features_path.is_file():
        bundle = _load_npz(features_path)
        features = bundle["features"]
        names = [str(name) for name in bundle["channel_names"]]
        written["fig_symmetry"] = write_png(
            build_symmetry_maps_figure(features, names),
            figures_dir / FIG_SYMMETRY,
        )

    if include_training:
        history = _read_json_optional(item_dir / "training_history.json") or {}
        record = _read_json(item_dir / "run_record.json")
        training = record.get("training", {}) if isinstance(record, dict) else {}
        loss = history.get("loss") or training.get("loss") or []
        accuracy = (
            history.get("support_accuracy")
            or training.get("support_accuracy")
            or []
        )
        if loss or accuracy:
            written["fig_training"] = write_png(
                build_training_figure(loss, accuracy),
                figures_dir / FIG_TRAINING,
            )

    if (item_dir / "prediction.npz").is_file():
        prediction = _prediction_from_item(item_dir)
        variants = render_result_variants(
            image,
            prediction,
            class_names=[entry.name for entry in classes],
            colors=[entry.color for entry in classes],
            alpha=DEFAULT_OVERLAY_ALPHA,
            stride=stride,
        )
        entry_maps = [
            {
                "index": entry.index,
                "name": entry.name,
                "color": entry.color,
            }
            for entry in classes
        ]
        legend = build_categorical_legend(entry_maps)
        written["fig_mask"] = write_png(
            compose_with_legend(variants["mask"], legend),
            figures_dir / FIG_MASK,
        )
        written["fig_overlay"] = write_png(
            variants["overlay_legend"], figures_dir / FIG_OVERLAY
        )
        written["fig_confidence"] = write_png(
            variants["confidence_colorbar"], figures_dir / FIG_CONFIDENCE
        )
        written["fig_entropy"] = write_png(
            variants["entropy_colorbar"], figures_dir / FIG_ENTROPY
        )

        source_key = "fig_support" if "fig_support" in written else "fig_input"
        overview_panels = [
            (
                "Support annotations"
                if source_key == "fig_support"
                else "Normalized input",
                _png_array(written[source_key]),
            ),
            ("Prediction overlay", _png_array(written["fig_overlay"])),
            ("Model confidence", _png_array(written["fig_confidence"])),
            ("Prediction entropy", _png_array(written["fig_entropy"])),
        ]
        written["fig_overview"] = write_png(
            build_results_overview_figure(overview_panels),
            figures_dir / FIG_OVERVIEW,
        )

    return written


def _figure_markdown(
    written: Mapping[str, Path],
    key: str,
    caption: str,
    *,
    report_dir: Path,
) -> str:
    """Render one embedded figure as Markdown.

    The link is relative to the directory holding the report, because figures
    live in a ``figures/`` sub-directory beside it. A bare basename would
    resolve against the report directory and miss that sub-directory, leaving
    every image broken. ``--output`` may place the report outside the run
    directory, so the offset is computed rather than hard-coded.
    """
    path = written.get(key)
    if path is None:
        return ""
    link = Path(os.path.relpath(path, report_dir)).as_posix()
    return f"![{caption}]({link})"


def _figure_block(
    written: Mapping[str, Path],
    lines: Sequence[tuple[str, str]],
    *,
    report_dir: Path,
) -> str:
    blocks = [
        _figure_markdown(written, key, caption, report_dir=report_dir)
        for key, caption in lines
    ]
    return "\n\n".join(block for block in blocks if block)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def _write_report(
    path: Path,
    *,
    template: str,
    context: Mapping[str, str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.format(**context), encoding="utf-8")
    return path


def _annotation_table(classes: Sequence[_ClassEntry]) -> str:
    rows = [
        [
            entry.index,
            entry.name,
            entry.color,
            len(entry.points_xy),
        ]
        for entry in classes
    ]
    return _md_table(
        ["Index", "Class", "Color", "Support points"],
        rows,
    )


def _annotation_details(classes: Sequence[_ClassEntry]) -> str:
    rows = [
        [
            entry.name,
            ", ".join(f"({x}, {y})" for x, y in entry.points_xy) or "(none)",
        ]
        for entry in classes
    ]
    return (
        "<details>\n<summary>Support-point coordinates</summary>\n\n"
        + _md_table(["Class", "Coordinates (x, y)"], rows)
        + "\n\n</details>"
    )


def _percent(value: float, *, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%"


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _prediction_readout(
    prediction: DensePrediction | None,
    classes: Sequence[_ClassEntry],
    *,
    include_support: bool,
) -> tuple[str, str, str]:
    """Return class distribution, uncertainty table and compact interpretation."""
    if prediction is None:
        unavailable = "_(prediction bundle not persisted)_"
        return unavailable, unavailable, unavailable

    labels = np.asarray(prediction.predictions, dtype=np.int64)
    total = max(int(labels.size), 1)
    class_rows: list[list[Any]] = []
    known_indices = {entry.index for entry in classes}
    for entry in classes:
        count = int(np.count_nonzero(labels == entry.index))
        row: list[Any] = [entry.name, entry.color]
        if include_support:
            row.append(len(entry.points_xy))
        row.extend([count, _percent(count / total)])
        class_rows.append(row)
    for index in sorted(set(int(value) for value in np.unique(labels)) - known_indices):
        count = int(np.count_nonzero(labels == index))
        row = [f"Unmapped class {index}", "-"]
        if include_support:
            row.append(0)
        row.extend([count, _percent(count / total)])
        class_rows.append(row)
    class_header = ["Class", "Color"]
    if include_support:
        class_header.append("Support points")
    class_header.extend(["Predicted grid points", "Grid share"])

    confidence = _finite(prediction.confidence)
    entropy = _finite(prediction.entropy)
    class_count = max(len(classes), int(prediction.probabilities.shape[1]), 2)
    entropy_scale = float(np.log(class_count))
    normalized_entropy = entropy / entropy_scale if entropy.size else entropy

    low_confidence_fraction = (
        float(np.mean(confidence < 0.75)) if confidence.size else float("nan")
    )
    high_entropy_fraction = (
        float(np.mean(normalized_entropy > 0.5))
        if normalized_entropy.size
        else float("nan")
    )
    median_confidence = (
        float(np.quantile(confidence, 0.50)) if confidence.size else float("nan")
    )
    uncertainty_table = _md_table(
        ["What was checked", "Result", "Practical meaning"],
        [
            [
                "At a typical sampled location",
                f"{_percent(median_confidence, digits=3)} model confidence"
                if np.isfinite(median_confidence)
                else "-",
                "The model strongly favors one class.",
            ],
            [
                "Locations with weaker predictions",
                _percent(low_confidence_fraction)
                if np.isfinite(low_confidence_fraction)
                else "-",
                "These locations fall below 75% model confidence and deserve closer visual review.",
            ],
            [
                "Locations with substantial class ambiguity",
                _percent(high_entropy_fraction)
                if np.isfinite(high_entropy_fraction)
                else "-",
                "The model does not clearly favor a single class at these locations.",
            ],
        ],
    )
    uncertainty_table += (
        "\n\n_The two cutoffs are practical flags for reviewing the maps, not "
        "measurements of prediction accuracy or physical correctness._\n\n"
        "<details>\n<summary>Technical definition of these checks</summary>\n\n"
        "A weaker prediction has maximum class probability below 0.75. "
        "Substantial class ambiguity means normalized prediction entropy "
        "`H / ln(N)` is above 0.5.\n\n</details>"
    )
    readout = (
        "Most sampled locations received a decisive class assignment. "
        f"{_percent(low_confidence_fraction)} had weaker predictions and "
        f"{_percent(high_entropy_fraction)} showed substantial ambiguity between "
        "classes. Those locations are the most useful places to inspect in the "
        "confidence and ambiguity maps; they are not automatically incorrect."
    )
    return _md_table(class_header, class_rows), uncertainty_table, readout


def _compact_prediction_summary(
    prediction: DensePrediction | None,
    classes: Sequence[_ClassEntry],
) -> tuple[str, str]:
    """Return short class-share and review summaries for a session index."""
    if prediction is None:
        return "not available", "not available"
    labels = np.asarray(prediction.predictions, dtype=np.int64)
    total = max(int(labels.size), 1)
    distribution = "; ".join(
        f"{entry.name} {_percent(int(np.count_nonzero(labels == entry.index)) / total)}"
        for entry in classes
    )
    confidence = _finite(prediction.confidence)
    entropy = _finite(prediction.entropy)
    class_count = max(len(classes), int(prediction.probabilities.shape[1]), 2)
    normalized_entropy = entropy / float(np.log(class_count)) if entropy.size else entropy
    weaker = float(np.mean(confidence < 0.75)) if confidence.size else float("nan")
    ambiguous = (
        float(np.mean(normalized_entropy > 0.5))
        if normalized_entropy.size
        else float("nan")
    )
    review = (
        f"{_percent(weaker)} lower confidence; "
        f"{_percent(ambiguous)} mixed class evidence"
        if np.isfinite(weaker) and np.isfinite(ambiguous)
        else "not available"
    )
    return distribution or "not available", review


def _evidence_status(
    classes: Sequence[_ClassEntry],
    *,
    recommended_shots: int | None = None,
) -> str:
    paragraphs = [
        "> **Evidence status:** exploratory model output. This run contains no "
        "independent ground truth or held-out evaluation, so support accuracy, "
        "confidence and entropy must not be read as physical validation."
    ]
    if classes and all(entry.name.strip().casefold().startswith("class ") for entry in classes):
        paragraphs.append(
            "> **Class semantics:** the classes still use generic names. Rename them "
            "before sharing if they correspond to phases, domains, defects or other "
            "material categories."
        )
    if recommended_shots is not None:
        below = [
            entry.name
            for entry in classes
            if len(entry.points_xy) < recommended_shots
        ]
        if below:
            paragraphs.append(
                "> **Annotation coverage:** "
                + ", ".join(below)
                + f" use fewer than the configured recommendation of {recommended_shots} "
                "support points per class."
            )
    return "\n\n".join(paragraphs)


def _training_interpretation(
    loss: Sequence[Any], accuracy: Sequence[Any]
) -> str:
    values = np.asarray([float(value) for value in accuracy], dtype=np.float64)
    perfect = np.flatnonzero(values >= 1.0 - 1e-12)
    if perfect.size:
        fit = (
            "The model matched all selected support points by epoch "
            f"{int(perfect[0]) + 1}."
        )
    elif values.size:
        fit = (
            "At the end of fine-tuning, the model matched "
            f"{_percent(float(values[-1]))} of the selected support points."
        )
    else:
        fit = "The point-by-point fitting history was not persisted."
    return (
        fit
        + " This shows that it learned the selected examples; it does not measure "
        "performance elsewhere in the image or on independent images."
    )


def _feature_table(bundle: Mapping[str, np.ndarray]) -> str:
    names = [str(name) for name in bundle["channel_names"]]
    features = np.asarray(bundle["features"])
    rows = []
    for index, name in enumerate(names):
        channel = features[index]
        low, high = _feature_display_range(name)
        rows.append(
            [
                index + 1,
                name,
                f"[{low:g}, {high:g}]",
                _fmt(float(np.min(channel)), digits=4),
                _fmt(float(np.max(channel)), digits=4),
            ]
        )
    return _md_table(
        ["#", "Channel", "Display range", "Observed min", "Observed max"], rows
    )


def _caveats(record: Mapping[str, Any]) -> str:
    """Return the run's own warnings plus the fixed presentation addendum."""
    recorded = [str(item) for item in record.get("warnings", []) or []]
    if not recorded:
        recorded = [
            "Support loss and support accuracy describe the training examples "
            "only; they are not independent estimates of generalization accuracy.",
            "Confidence and entropy describe model output, not physical "
            "correctness.",
        ]
    addendum = (
        "Dense prediction is sampled on a grid at the configured stride; cells "
        "between samples are filled for display only."
    )
    if addendum not in recorded:
        recorded.append(addendum)
    return _bullet_list(recorded)


def _artifact_list(
    record: Mapping[str, Any],
    *,
    run_dir: Path,
    report_dir: Path,
    extras: Mapping[str, Path] | None = None,
) -> str:
    artifacts = record.get("artifacts", {})
    merged: dict[str, Any] = dict(artifacts) if isinstance(artifacts, Mapping) else {}
    if extras:
        merged.update(extras)
    if not merged:
        return "- (no artifact map recorded)"
    rows: list[str] = []
    for name, value in sorted(merged.items()):
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = run_dir / path
        try:
            link = Path(os.path.relpath(path, report_dir)).as_posix()
        except ValueError:
            rows.append(f"`{name}`: `{path.name}`")
            continue
        rows.append(f"[`{name}`](<{link}>) - `{path.name}`")
    return _bullet_list(rows)


def _latest(record: Mapping[str, Any]) -> float:
    values = record.get("training", {}).get("loss") or []
    if values:
        return float(min(values))
    return float("nan")


def _build_fine_tune_report(
    run_dir: Path,
    *,
    notes: str | None,
    figures: bool,
    output: Path | None,
) -> ReportResult:
    record = _read_json(run_dir / "run_record.json")
    session = _read_json(run_dir / "annotation_session.json")
    classes = _classes_from_annotations(session)
    options = record.get("options", {})
    stride = int(options.get("stride", 1))

    figures_dir = run_dir / FIGURES_DIRNAME
    written: dict[str, Path] = {}
    if figures:
        written = _item_figures(
            run_dir,
            figures_dir=figures_dir,
            classes=classes,
            stride=stride,
            include_training=True,
            include_support=True,
        )

    prediction = record.get("prediction", {})
    training = record.get("training", {})
    support_points = sum(len(entry.points_xy) for entry in classes)
    image_shape = record.get("input", {}).get("shape", [None, None])
    input_path = record.get("input", {}).get("path")
    input_label = Path(str(input_path)).name if input_path else "input image"
    grid_shape = prediction.get("grid_shape", ["?", "?"]) or ["?", "?"]
    summary = (
        f"This run used {support_points} user-selected support points across "
        f"{len(classes)} classes to adapt `{record.get('model', {}).get('identifier', 'unknown')}` "
        f"for **{input_label}** ({image_shape[0]} x {image_shape[1]} pixels). "
        f"The fitted model classified {prediction.get('sample_count', '?')} sampled "
        f"locations on a {grid_shape[0]} x {grid_shape[1]} grid at stride {stride}. "
        "The table below reports the distribution of model-assigned classes; it "
        "does not by itself establish their physical meaning or correctness."
    )

    workflow = _bullet_list(
        [
            f"Loaded and normalized one single-channel image "
            f"({image_shape[0]}x{image_shape[1]}; "
            f"{record.get('input', {}).get('normalization', {}).get('policy', 'minmax_0_1')}).",
            f"Computed the eight-channel symmetry representation "
            f"(n_max={options.get('n_max')}, "
            f"patch={options.get('symmetry_patch_size')}, "
            f"folds={options.get('rotation_folds')}).",
            f"Defined {len(classes)} classes and placed {support_points} support points.",
            f"Fine-tuned the residual adapters and task-specific classification "
            f"head while the pretrained trunk stayed frozen "
            f"({options.get('epochs')} epochs, lr={options.get('learning_rate')}, "
            f"bottleneck={options.get('adapter_bottleneck')}, "
            f"seed={options.get('seed')}).",
            f"Ran dense prediction at stride {stride} and rendered the class mask, "
            f"confidence and entropy maps.",
            "Persisted the run record and the fitted-model artifacts available for this run.",
        ]
    )

    history = _read_json_optional(run_dir / "training_history.json") or {}
    loss_history = history.get("loss") or training.get("loss") or []
    accuracy_history = (
        history.get("support_accuracy") or training.get("support_accuracy") or []
    )
    persisted_prediction = (
        _prediction_from_item(run_dir)
        if (run_dir / "prediction.npz").is_file()
        else None
    )
    class_distribution_table, uncertainty_table, result_readout = (
        _prediction_readout(persisted_prediction, classes, include_support=True)
    )
    recommended_shots_raw = (
        record.get("configuration", {})
        .get("fine_tuning", {})
        .get("recommended_shots_per_class")
    )
    recommended_shots = (
        int(recommended_shots_raw)
        if isinstance(recommended_shots_raw, (int, float))
        else None
    )

    training_table = _md_table(
        ["Metric", "Value"],
        [
            ["Epochs", training.get("epochs", options.get("epochs"))],
            ["Best support loss", training.get("best_support_loss")],
            ["Final support loss", training.get("final_support_loss")],
            ["Final support accuracy", training.get("final_support_accuracy")],
            ["Trainable parameters", training.get("trainable_parameters")],
            ["Adapter bottleneck", options.get("adapter_bottleneck")],
            ["Learning rate", options.get("learning_rate")],
            ["Weight decay", options.get("weight_decay")],
            ["Seed", options.get("seed")],
            ["Training runtime (s)", training.get("runtime_seconds")],
        ],
    )

    prediction_table = _md_table(
        ["Metric", "Value"],
        [
            ["Grid shape", prediction.get("grid_shape")],
            ["Sample count", prediction.get("sample_count")],
            ["Stride", stride],
            ["Batch size", options.get("batch_size")],
            ["Confidence range", prediction.get("confidence_range")],
            ["Entropy range", prediction.get("entropy_range")],
        ],
    )

    configuration_table = _md_table(
        ["Key", "Value"],
        [
            ["Contract version", record.get("contract_version")],
            ["Run ID", record.get("run_id")],
            ["Created (UTC)", record.get("created_utc")],
            ["Status", record.get("status")],
            ["Input", input_label],
            ["Classifier patch size", options.get("classifier_patch_size")],
            ["Symmetry patch size", options.get("symmetry_patch_size")],
            ["Rotation folds", options.get("rotation_folds")],
            ["Reflection p", options.get("reflection_p")],
            ["N max", options.get("n_max")],
            ["Input normalization", options.get("input_normalization")],
            ["Base weight", record.get("model", {}).get("checkpoint", {}).get("weight_identifier")],
            ["Checkpoint source", record.get("model", {}).get("checkpoint", {}).get("source")],
            [
                "Checkpoint SHA-256",
                record.get("model", {}).get("checkpoint", {}).get("sha256"),
            ],
            ["Run harness version", record.get("symmetry_harness_version", "not recorded")],
            [
                "Provider",
                f"{record.get('provider', {}).get('name')} "
                f"{record.get('provider', {}).get('version')}",
            ],
            [
                "Provider contract",
                record.get("provider", {}).get("contract_version"),
            ],
            ["Device", record.get("runtime", {}).get("resolved_device")],
            ["Torch", record.get("runtime", {}).get("torch_version")],
            ["Python", record.get("runtime", {}).get("python_version")],
            ["Platform", record.get("runtime", {}).get("platform")],
            [
                "Total seconds",
                record.get("runtime", {}).get("harness_total_seconds"),
            ],
        ],
    )

    # The report sits beside ``figures/``, so its own directory is the base the
    # image links must be relative to.
    target = output or (run_dir / REPORT_FILENAME)
    report_dir = target.parent
    record_path = Path(os.path.relpath(run_dir / "run_record.json", report_dir)).as_posix()
    context = {
        "run_id": str(record.get("run_id", run_dir.name)),
        "kind": KIND_FINE_TUNE,
        "status": str(record.get("status", "unknown")),
        "generated_utc": _utc_now_iso(),
        "harness_version": __version__,
        "summary": summary,
        "workflow": workflow,
        "notes": notes.strip() if notes and notes.strip() else "_Not provided._",
        "evidence_status": _evidence_status(
            classes, recommended_shots=recommended_shots
        ),
        "class_distribution_table": class_distribution_table,
        "result_readout": result_readout,
        "uncertainty_table": uncertainty_table,
        "training_interpretation": _training_interpretation(
            loss_history, accuracy_history
        ),
        "fig_overview": _figure_markdown(
            written,
            "fig_overview",
            "Annotations, prediction and uncertainty overview",
            report_dir=report_dir,
        ),
        "fig_input": _figure_markdown(
            written, "fig_input", "Normalized input image", report_dir=report_dir
        ),
        "fig_support": _figure_markdown(
            written, "fig_support", "Support points by class", report_dir=report_dir
        ),
        "fig_symmetry": _figure_markdown(
            written,
            "fig_symmetry",
            "Eight-channel symmetry representation",
            report_dir=report_dir,
        ),
        "fig_training": _figure_markdown(
            written, "fig_training", "Training diagnostics", report_dir=report_dir
        ),
        "fig_mask": _figure_markdown(
            written, "fig_mask", "Predicted class mask", report_dir=report_dir
        ),
        "fig_overlay": _figure_markdown(
            written,
            "fig_overlay",
            "Prediction overlay on the source image",
            report_dir=report_dir,
        ),
        "fig_confidence": _figure_markdown(
            written,
            "fig_confidence",
            "Confidence map (viridis, 0-1)",
            report_dir=report_dir,
        ),
        "fig_entropy": _figure_markdown(
            written, "fig_entropy", "Entropy map (magma)", report_dir=report_dir
        ),
        "annotation_table": _annotation_table(classes),
        "annotation_details": _annotation_details(classes),
        "feature_table": _feature_table(_load_npz(run_dir / "features.npz"))
        if (run_dir / "features.npz").is_file()
        else "(feature bundle not persisted)",
        "training_table": training_table,
        "prediction_table": prediction_table,
        "configuration_table": configuration_table,
        "caveats": _caveats(record),
        "artifact_list": _artifact_list(
            record,
            run_dir=run_dir,
            report_dir=report_dir,
            extras={
                "report_summary": target,
                **({"report_figures": figures_dir} if figures else {}),
            },
        ),
        "record_path": str(record_path),
    }

    report_path = _write_report(
        target, template=_template(_TEMPLATE_IMAGE), context=context
    )
    return ReportResult(
        run_id=context["run_id"],
        kind=KIND_FINE_TUNE,
        report_path=report_path,
        figure_paths=tuple(sorted(written.values(), key=lambda p: p.name)),
    )


def _build_prediction_item_report(
    item_dir: Path,
    *,
    batch: Mapping[str, Any],
    classes: Sequence[_ClassEntry],
    stride: int,
    notes: str | None,
    figures: bool,
    item_label: str,
) -> ReportResult:
    record = _read_json(item_dir / "prediction_record.json")
    figures_dir = item_dir / FIGURES_DIRNAME
    written: dict[str, Path] = {}
    if figures:
        written = _item_figures(
            item_dir,
            figures_dir=figures_dir,
            classes=classes,
            stride=stride,
            include_training=False,
            include_support=False,
        )

    output = record.get("output", {})
    image_shape = record.get("input", {}).get("shape", [None, None])
    summary = (
        f"The saved fine-tuned model "
        f"`{record.get('model', {}).get('identifier', 'unknown')}` classified "
        f"{item_label}, a {image_shape[0]} x {image_shape[1]} single-channel image. "
        f"The result covers {output.get('sample_count')} sampled locations on a "
        f"{output.get('grid_shape', ['?', '?'])[0]} x "
        f"{output.get('grid_shape', ['?', '?'])[1]} grid at stride {stride}. "
        "The class distribution below describes model assignments, not verified "
        "material fractions."
    )

    workflow = _bullet_list(
        [
            f"Imported the fine-tuned package "
            f"`{batch.get('package', {}).get('schema_version', 'symmodel')}` "
            f"(training run "
            f"`{batch.get('package', {}).get('training_run_id', 'unknown')}`).",
            f"Loaded and normalized {item_label} "
            f"({image_shape[0]}x{image_shape[1]}) and computed its eight-channel "
            f"representation.",
            f"Ran dense prediction at stride {stride}.",
        ]
    )

    persisted_prediction = (
        _prediction_from_item(item_dir)
        if (item_dir / "prediction.npz").is_file()
        else None
    )
    class_distribution_table, uncertainty_table, result_readout = (
        _prediction_readout(persisted_prediction, classes, include_support=False)
    )

    prediction_table = _md_table(
        ["Metric", "Value"],
        [
            ["Grid shape", output.get("grid_shape")],
            ["Sample count", output.get("sample_count")],
            ["Stride", stride],
            ["Confidence range", output.get("confidence_range")],
            ["Entropy range", output.get("entropy_range")],
        ],
    )

    configuration_table = _md_table(
        ["Key", "Value"],
        [
            ["Contract version", record.get("contract_version")],
            ["Run ID", record.get("run_id")],
            ["Item ID", record.get("item_id")],
            ["Created (UTC)", record.get("created_utc")],
            ["Status", record.get("status")],
            ["Class names", [entry.name for entry in classes]],
            ["Class colors", [entry.color for entry in classes]],
            [
                "Model package",
                Path(str(batch.get("package", {}).get("path"))).name
                if batch.get("package", {}).get("path")
                else None,
            ],
            ["Package SHA-256", batch.get("package", {}).get("package_sha256")],
            [
                "Model state SHA-256",
                batch.get("package", {}).get("model_state_sha256"),
            ],
            ["Input normalization", batch.get("input_normalization")],
            [
                "Provider",
                f"{record.get('provider', {}).get('name')} "
                f"{record.get('provider', {}).get('version')}",
            ],
            ["Feature n_max", batch.get("features", {}).get("n_max")],
            [
                "Symmetry patch size",
                batch.get("features", {}).get("symmetry_patch_size"),
            ],
        ],
    )

    context = {
        "run_id": str(batch.get("run_id", item_dir.parent.parent.name)),
        "item_label": item_label,
        "kind": KIND_PREDICTION,
        "status": str(record.get("status", "unknown")),
        "generated_utc": _utc_now_iso(),
        "harness_version": __version__,
        "summary": summary,
        "workflow": workflow,
        "notes": notes.strip() if notes and notes.strip() else "_Not provided._",
        "evidence_status": _evidence_status(classes),
        "class_distribution_table": class_distribution_table,
        "result_readout": result_readout,
        "uncertainty_table": uncertainty_table,
        "fig_overview": _figure_markdown(
            written,
            "fig_overview",
            "Input, prediction and uncertainty overview",
            report_dir=item_dir,
        ),
        "fig_input": _figure_markdown(
            written, "fig_input", "Normalized input image", report_dir=item_dir
        ),
        "fig_symmetry": _figure_markdown(
            written,
            "fig_symmetry",
            "Eight-channel symmetry representation",
            report_dir=item_dir,
        ),
        "fig_mask": _figure_markdown(
            written, "fig_mask", "Predicted class mask", report_dir=item_dir
        ),
        "fig_overlay": _figure_markdown(
            written,
            "fig_overlay",
            "Prediction overlay on the source image",
            report_dir=item_dir,
        ),
        "fig_confidence": _figure_markdown(
            written,
            "fig_confidence",
            "Confidence map (viridis, 0-1)",
            report_dir=item_dir,
        ),
        "fig_entropy": _figure_markdown(
            written, "fig_entropy", "Entropy map (magma)", report_dir=item_dir
        ),
        "class_table": _md_table(
            ["Index", "Class", "Color"],
            [[entry.index, entry.name, entry.color] for entry in classes],
        ),
        "feature_table": _feature_table(_load_npz(item_dir / "features.npz"))
        if (item_dir / "features.npz").is_file()
        else "(feature bundle not persisted)",
        "prediction_table": prediction_table,
        "configuration_table": configuration_table,
        "caveats": _caveats(
            {"warnings": list(batch.get("warnings", []) or []) + list(
                record.get("warnings", []) or []
            )}
        ),
        "artifact_list": _artifact_list(
            record,
            run_dir=item_dir,
            report_dir=item_dir,
            extras={
                "report_summary": item_dir / REPORT_FILENAME,
                **({"report_figures": figures_dir} if figures else {}),
            },
        ),
        "record_path": str((item_dir / "prediction_record.json").resolve()),
    }

    report_path = _write_report(
        item_dir / REPORT_FILENAME,
        template=_template(_TEMPLATE_ITEM),
        context=context,
    )
    return ReportResult(
        run_id=context["run_id"],
        kind=KIND_PREDICTION,
        report_path=report_path,
        figure_paths=tuple(sorted(written.values(), key=lambda p: p.name)),
    )


def _build_prediction_batch_report(
    run_dir: Path,
    *,
    notes: str | None,
    figures: bool,
    output: Path | None,
) -> ReportResult:
    batch = _read_json(run_dir / "batch_record.json")
    target = output or (run_dir / REPORT_FILENAME)
    model = batch.get("model", {})
    classes = _classes_from_names(
        model.get("class_names", []) or [], model.get("class_colors", []) or []
    )
    stride = int(batch.get("prediction_options", {}).get("stride", 1))

    item_reports: list[ReportResult] = []
    for index, raw in enumerate(batch.get("items", []) or []):
        item_id = str(raw.get("item_id", f"image-{index + 1:04d}"))
        item_dir = run_dir / "items" / item_id
        if not item_dir.is_dir():
            continue
        item_reports.append(
            _build_prediction_item_report(
                item_dir,
                batch=batch,
                classes=classes,
                stride=stride,
                notes=None,
                figures=figures,
                item_label=f"`{item_id}`",
            )
        )

    counts = batch.get("counts", {})
    summary = (
        f"The saved fine-tuned model `{model.get('identifier', 'unknown')}` was "
        f"applied to {counts.get('requested', len(item_reports))} image(s) "
        f"({counts.get('completed', 0)} completed, {counts.get('failed', 0)} failed) "
        f"using {len(classes)} classes at stride {stride}. Per-image reports are "
        f"listed below."
    )

    item_rows = [
        [
            report.report_path.parent.name,
            "[report](<"
            + Path(os.path.relpath(report.report_path, target.parent)).as_posix()
            + ">)",
            len(report.figure_paths),
        ]
        for report in item_reports
    ]
    context = {
        "run_id": str(batch.get("run_id", run_dir.name)),
        "status": str(batch.get("status", "unknown")),
        "generated_utc": _utc_now_iso(),
        "harness_version": __version__,
        "summary": summary,
        "evidence_status": _evidence_status(classes),
        "notes": notes.strip() if notes and notes.strip() else "_Not provided._",
        "class_table": _md_table(
            ["Index", "Class", "Color"],
            [[entry.index, entry.name, entry.color] for entry in classes],
        ),
        "item_table": _md_table(
            ["Item", "Report", "Figures"], item_rows
        )
        if item_rows
        else "(no items persisted)",
        "configuration_table": _md_table(
            ["Key", "Value"],
            [
                ["Contract version", batch.get("contract_version")],
                ["Run ID", batch.get("run_id")],
                ["Created (UTC)", batch.get("created_utc")],
                ["Status", batch.get("status")],
                ["Harness version", batch.get("symmetry_harness_version")],
                ["Runtime (s)", batch.get("runtime_seconds")],
                ["Input normalization", batch.get("input_normalization")],
                ["Class names", model.get("class_names")],
                ["Task classes", model.get("task_classes")],
                ["Stride", stride],
                ["Batch size", batch.get("prediction_options", {}).get("batch_size")],
                [
                    "Model package",
                    Path(str(batch.get("package", {}).get("path"))).name
                    if batch.get("package", {}).get("path")
                    else None,
                ],
                ["Package SHA-256", batch.get("package", {}).get("package_sha256")],
                [
                    "Training run",
                    batch.get("package", {}).get("training_run_id"),
                ],
                ["Feature n_max", batch.get("features", {}).get("n_max")],
                [
                    "Symmetry patch size",
                    batch.get("features", {}).get("symmetry_patch_size"),
                ],
                [
                    "Provider",
                    f"{batch.get('provider', {}).get('name')} "
                    f"{batch.get('provider', {}).get('version')}",
                ],
            ],
        ),
        "caveats": _caveats(batch),
        "artifact_list": _artifact_list(
            batch,
            run_dir=run_dir,
            report_dir=target.parent,
            extras={"report_summary": target},
        ),
        "record_path": str((run_dir / "batch_record.json").resolve()),
    }

    report_path = _write_report(
        target, template=_template(_TEMPLATE_BATCH), context=context
    )
    return ReportResult(
        run_id=context["run_id"],
        kind=KIND_PREDICTION,
        report_path=report_path,
        figure_paths=tuple(
            path for report in item_reports for path in report.figure_paths
        ),
        item_reports=tuple(report.report_path for report in item_reports),
    )


def build_run_report(
    run_dir: str | Path,
    *,
    notes: str | None = None,
    figures: bool = True,
    output: str | Path | None = None,
) -> ReportResult:
    """Generate the fixed-template report for one completed run directory.

    ``notes`` is the free-form ``Analysis Context`` body. ``figures=False`` skips
    PNG rendering and the report embeds no images. ``output`` overrides the
    default ``<run>/report_summary.md`` location.
    """
    directory = Path(run_dir).expanduser()
    if not directory.is_dir():
        raise ReportError(f"Run directory does not exist: {directory}")
    kind = detect_kind(directory)
    target = Path(output).expanduser() if output is not None else None
    if kind == KIND_FINE_TUNE:
        return _build_fine_tune_report(
            directory, notes=notes, figures=figures, output=target
        )
    return _build_prediction_batch_report(
        directory, notes=notes, figures=figures, output=target
    )


def build_fine_tune_session_report(
    output_root: str | Path,
    *,
    session: str = "latest",
    notes: str | None = None,
    figures: bool = True,
    output: str | Path | None = None,
) -> ReportResult:
    """Build one compact index for all fine-tune runs from a UI session.

    A one-run session returns the ordinary image report directly. Multi-run
    sessions keep those reports independent and add only a lightweight index.
    """
    root = Path(output_root).expanduser().resolve()
    session_id, run_dirs = resolve_fine_tune_session(root, session)
    if len(run_dirs) == 1:
        return build_run_report(
            run_dirs[0], notes=notes, figures=figures, output=output
        )

    safe_session_id = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in session_id
    )[:120]
    target = (
        Path(output).expanduser()
        if output is not None
        else root / "sessions" / safe_session_id / REPORT_FILENAME
    )
    report_dir = target.parent

    child_results: list[ReportResult] = []
    run_rows: list[list[Any]] = []
    run_sections: list[str] = []
    reproduction_rows: list[list[Any]] = []
    class_signatures: list[tuple[str, ...]] = []
    setting_signatures: list[tuple[Any, ...]] = []

    for index, run_dir in enumerate(run_dirs, start=1):
        existing_report = run_dir / REPORT_FILENAME
        existing_overview = run_dir / FIGURES_DIRNAME / FIG_OVERVIEW
        if existing_report.is_file() and (not figures or existing_overview.is_file()):
            figure_paths = (
                tuple(sorted((run_dir / FIGURES_DIRNAME).glob("*.png")))
                if figures
                else ()
            )
            child = ReportResult(
                run_id=run_dir.name,
                kind=KIND_FINE_TUNE,
                report_path=existing_report,
                figure_paths=figure_paths,
            )
        else:
            child = build_run_report(run_dir, figures=figures)
        child_results.append(child)

        record = _read_json(run_dir / "run_record.json")
        annotation = _read_json(run_dir / "annotation_session.json")
        classes = _classes_from_annotations(annotation)
        prediction = (
            _prediction_from_item(run_dir)
            if (run_dir / "prediction.npz").is_file()
            else None
        )
        distribution, review = _compact_prediction_summary(prediction, classes)
        input_path = record.get("input", {}).get("path")
        input_label = Path(str(input_path)).name if input_path else run_dir.name
        support_count = sum(len(entry.points_xy) for entry in classes)
        report_link = Path(
            os.path.relpath(child.report_path, report_dir)
        ).as_posix()
        overview_path = run_dir / FIGURES_DIRNAME / FIG_OVERVIEW
        overview_link = Path(
            os.path.relpath(overview_path, report_dir)
        ).as_posix()
        record_link = Path(
            os.path.relpath(run_dir / "run_record.json", report_dir)
        ).as_posix()

        run_rows.append(
            [
                input_label,
                ", ".join(entry.name for entry in classes),
                support_count,
                distribution,
                review,
                f"[open](<{report_link}>)",
            ]
        )
        section_parts = [
            "<details>",
            f"<summary>{index}. {input_label} - {run_dir.name}</summary>",
            "",
        ]
        if figures and overview_path.is_file():
            section_parts.extend(
                [f"![Results overview for {input_label}]({overview_link})", ""]
            )
        section_parts.extend(
            [f"[Open the complete image report](<{report_link}>)", "", "</details>"]
        )
        run_sections.append("\n".join(section_parts))
        reproduction_rows.append(
            [
                run_dir.name,
                f"[`run_record.json`](<{record_link}>)",
                f"[image report](<{report_link}>)",
            ]
        )
        class_signatures.append(tuple(entry.name for entry in classes))
        options = record.get("options", {})
        setting_signatures.append(
            (
                record.get("model", {}).get("identifier"),
                options.get("classifier_patch_size"),
                options.get("symmetry_patch_size"),
                tuple(options.get("rotation_folds", []) or []),
                options.get("n_max"),
                options.get("stride"),
            )
        )

    structurally_comparable = (
        len(set(class_signatures)) == 1 and len(set(setting_signatures)) == 1
    )
    comparison_status = (
        "> **Comparison note:** the runs use the same class names and core analysis "
        "settings, so side-by-side review is structurally consistent. Confirm that "
        "the class names carry the same material meaning before interpreting changes."
        if structurally_comparable
        else "> **Comparison note:** class names or analysis settings differ across "
        "these runs. Treat them as separate experiments rather than comparing their "
        "class proportions directly."
    )
    context = {
        "session_id": session_id,
        "run_count": str(len(run_dirs)),
        "generated_utc": _utc_now_iso(),
        "harness_version": __version__,
        "comparison_status": comparison_status,
        "run_table": _md_table(
            [
                "Image",
                "Labels used",
                "Examples used for fine-tuning",
                "Predicted area by label",
                "Areas worth checking",
                "Full report",
            ],
            run_rows,
        ),
        "run_sections": "\n\n".join(run_sections),
        "notes": notes.strip() if notes and notes.strip() else "_Not provided._",
        "reproduction_table": _md_table(
            ["Run", "Versioned record", "Detailed report"], reproduction_rows
        ),
    }
    report_path = _write_report(
        target, template=_template(_TEMPLATE_SESSION), context=context
    )
    return ReportResult(
        run_id=session_id,
        kind=KIND_FINE_TUNE_SESSION,
        report_path=report_path,
        figure_paths=tuple(
            path for child in child_results for path in child.figure_paths
        ),
        item_reports=tuple(child.report_path for child in child_results),
    )
