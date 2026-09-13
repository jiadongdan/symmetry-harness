"""Presentation-only rendering and export helpers for completed results.

This module is intentionally pure: it only imports ``numpy``, ``PIL`` and the
Harness-internal :mod:`analysis` / :mod:`colormaps` modules. It never imports
``gradio``, ``torch`` or ``symmlearn`` and it never runs a numerical job. Every
function here operates on arrays that already exist on disk (a completed
``prediction.npz`` plus the normalized input image), so changing colors or alpha
can only rerender pixels -- it can never trigger feature computation,
fine-tuning, model restoration or dense prediction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .analysis import (
    DensePrediction,
    TraditionalPrediction,
    prediction_display_bounds,
    render_prediction_mask,
    render_prediction_overlay,
    render_scalar_map,
)

GridPrediction = DensePrediction | TraditionalPrediction

DEFAULT_OVERLAY_ALPHA: float = 0.48
ALPHA_MIN: float = 0.0
ALPHA_MAX: float = 1.0

_CONFIDENCE_COLORMAP: str = "viridis"
_ENTROPY_COLORMAP: str = "magma"
_CONFIDENCE_RANGE: tuple[float, float] = (0.0, 1.0)

_HEX_COLOR_RE = re.compile(r"\A#[0-9a-fA-F]{6}\Z")

LEGEND_MARGIN_WIDTH: int = 220
LEGEND_SWATCH_SIZE: int = 16
LEGEND_ROW_HEIGHT: int = 28
LEGEND_PADDING: int = 12
LEGEND_MAX_COLUMNS: int = 3
LEGEND_COLUMN_GAP: int = 18

COLORBAR_MARGIN_WIDTH: int = 112
COLORBAR_BAR_WIDTH: int = 24
COLORBAR_PADDING: int = 12
COLORBAR_TICK_COUNT: int = 5

_BACKGROUND_RGB: tuple[int, int, int] = (255, 255, 255)
_FOREGROUND_RGB: tuple[int, int, int] = (0, 0, 0)

RESULT_EXPORT_NAMES: tuple[str, ...] = (
    "overlay",
    "overlay_legend",
    "mask",
    "confidence",
    "confidence_colorbar",
    "entropy",
    "entropy_colorbar",
)


class PresentationError(ValueError):
    """Raised when transient presentation colors, alpha or geometry are invalid."""


# ---------------------------------------------------------------------------
# Validation and normalization
# ---------------------------------------------------------------------------
def validate_hex_color(value: str) -> str:
    """Validate and normalize one strict ``#RRGGBB`` class color.

    Exactly six hexadecimal digits are accepted; three-digit shorthand and
    named colors are rejected. The normalized value is lowercase ``#rrggbb``.
    """
    text = str(value).strip()
    if not _HEX_COLOR_RE.match(text):
        raise PresentationError(
            f"Invalid class color {value!r}: expected #RRGGBB."
        )
    return text.lower()


def normalize_palette(
    colors: Sequence[str],
    *,
    class_count: int,
) -> tuple[str, ...]:
    """Normalize a palette and require one color per task class.

    Duplicate, identical or visually similar colors are allowed: the protocol
    treats them as a valid (if unusual) user choice.
    """
    count = int(class_count)
    if count <= 0:
        raise PresentationError("The class count must be positive.")
    values = list(colors)
    if len(values) != count:
        raise PresentationError(
            f"Expected {count} class colors but received {len(values)}."
        )
    return tuple(validate_hex_color(color) for color in values)


def validate_alpha(value: float) -> float:
    """Validate one overlay alpha within the inclusive ``[0.0, 1.0]`` range."""
    try:
        resolved = float(value)
    except (TypeError, ValueError) as error:
        raise PresentationError("Overlay alpha must be a number.") from error
    if not np.isfinite(resolved) or resolved < ALPHA_MIN or resolved > ALPHA_MAX:
        raise PresentationError(
            f"Overlay alpha must be between {ALPHA_MIN} and {ALPHA_MAX}."
        )
    return resolved


def _rgb_tuple(color: str) -> tuple[int, int, int]:
    normalized = validate_hex_color(color)
    return (
        int(normalized[1:3], 16),
        int(normalized[3:5], 16),
        int(normalized[5:7], 16),
    )


# ---------------------------------------------------------------------------
# Mask / overlay / scalar rendering
# ---------------------------------------------------------------------------
def render_overlay(
    image: np.ndarray,
    prediction: GridPrediction,
    *,
    colors: Sequence[str],
    alpha: float,
    stride: int,
) -> np.ndarray:
    """Blend the normalized source image with the categorical prediction grid.

    Geometry and blending logic are reused from
    :func:`analysis.render_prediction_overlay`; this function never re-derives
    the prediction region.
    """
    resolved_alpha = validate_alpha(alpha)
    return render_prediction_overlay(
        image, prediction, list(colors), stride=stride, alpha=resolved_alpha
    )


def render_categorical_mask(
    prediction: GridPrediction,
    image_shape: tuple[int, int],
    *,
    colors: Sequence[str],
    stride: int,
) -> np.ndarray:
    """Render the categorical class mask with nearest-neighbour colors only.

    The mask carries no source-image background and no alpha; its region matches
    :func:`render_overlay` exactly.
    """
    return render_prediction_mask(
        prediction, image_shape, colors=list(colors), stride=stride
    )


# ---------------------------------------------------------------------------
# Legends and colorbars
# ---------------------------------------------------------------------------
def _fit_height(block: np.ndarray, height: int) -> np.ndarray:
    """Return ``block`` padded with background to exactly ``height`` rows."""
    if block.shape[0] == height:
        return block
    if block.shape[0] > height:
        return block[:height]
    fill = np.full(
        (height - block.shape[0], block.shape[1], 3),
        _BACKGROUND_RGB,
        dtype=np.uint8,
    )
    return np.concatenate([block, fill], axis=0)


def _fit_width(block: np.ndarray, width: int) -> np.ndarray:
    """Return ``block`` padded with background to exactly ``width`` columns."""
    if block.shape[1] == width:
        return block
    if block.shape[1] > width:
        return block[:, :width]
    fill = np.full(
        (block.shape[0], width - block.shape[1], 3),
        _BACKGROUND_RGB,
        dtype=np.uint8,
    )
    return np.concatenate([block, fill], axis=1)


def build_categorical_legend(
    classes: Sequence[Mapping[str, object]],
    *,
    height: int | None = None,
) -> np.ndarray:
    """Render an ordered categorical legend: color swatch plus exact class name.

    The legend has a white background and no title. Rows follow the model's
    ordered ``index`` mapping.

    Pass ``height`` to fill a fixed-height margin (the historical side-by-side
    layout). Leave it ``None`` to shrink to the rows actually needed, which is
    what the below-the-image composition uses.
    """
    entries = sorted(classes, key=lambda entry: int(entry.get("index", 0)))
    font = ImageFont.load_default()
    measurement = ImageDraw.Draw(Image.new("RGB", (1, 1), _BACKGROUND_RGB))

    def text_width(value: object) -> int:
        left, _, right, _ = measurement.textbbox((0, 0), str(value), font=font)
        return int(right - left)

    column_count = min(LEGEND_MAX_COLUMNS, max(1, len(entries)))
    row_count = max(1, (len(entries) + column_count - 1) // column_count)
    column_widths = []
    for column in range(column_count):
        labels = [
            text_width(entries[index].get("name", ""))
            for index in range(column, len(entries), column_count)
        ]
        column_widths.append(LEGEND_SWATCH_SIZE + 8 + max(labels, default=0))
    resolved_width = max(
        LEGEND_MARGIN_WIDTH,
        LEGEND_PADDING * 2
        + sum(column_widths)
        + LEGEND_COLUMN_GAP * max(0, column_count - 1),
    )
    if height is None:
        resolved_height = max(
            1, LEGEND_PADDING * 2 + row_count * LEGEND_ROW_HEIGHT
        )
    else:
        resolved_height = max(1, int(height))
    canvas = np.full(
        (resolved_height, resolved_width, 3), _BACKGROUND_RGB, dtype=np.uint8
    )
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    column_offsets = [LEGEND_PADDING]
    for width in column_widths[:-1]:
        column_offsets.append(column_offsets[-1] + width + LEGEND_COLUMN_GAP)
    for index, entry in enumerate(entries):
        row, column = divmod(index, column_count)
        x = column_offsets[column]
        y = LEGEND_PADDING + row * LEGEND_ROW_HEIGHT
        if y + LEGEND_SWATCH_SIZE > resolved_height:
            break
        color = _rgb_tuple(str(entry.get("color", "#ffffff")))
        draw.rectangle(
            [
                (x, y),
                (x + LEGEND_SWATCH_SIZE - 1, y + LEGEND_SWATCH_SIZE - 1),
            ],
            fill=color,
            outline=_FOREGROUND_RGB,
        )
        label = str(entry.get("name", ""))
        draw.text(
            (x + LEGEND_SWATCH_SIZE + 8, y + 2),
            label,
            fill=_FOREGROUND_RGB,
            font=font,
        )
    return np.asarray(pil, dtype=np.uint8)


def _format_tick(value: float) -> str:
    """Format one scalar tick label deterministically."""
    return f"{float(value):.3f}"


def build_scalar_colorbar(
    value_range: tuple[float, float],
    *,
    colormap: str,
    height: int,
    tick_labels: Sequence[str] | None = None,
) -> np.ndarray:
    """Render a vertical colorbar for a fixed scalar value range.

    The gradient runs from the range maximum at the top to the minimum at the
    bottom, and the tick labels reflect the declared fixed bounds. The colorbar
    has a white background and no title.
    """
    minimum, maximum = (float(value) for value in value_range)
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise PresentationError("Scalar colorbar value_range must be finite and increasing.")
    resolved_height = max(1, int(height))
    canvas = np.full(
        (resolved_height, COLORBAR_MARGIN_WIDTH, 3), _BACKGROUND_RGB, dtype=np.uint8
    )
    bar_top = min(COLORBAR_PADDING, max(0, resolved_height - 1))
    bar_bottom = max(bar_top + 1, resolved_height - COLORBAR_PADDING)
    bar_height = bar_bottom - bar_top
    values = np.linspace(maximum, minimum, bar_height, dtype=np.float32)
    column = render_scalar_map(
        values.reshape(bar_height, 1),
        (1, bar_height),
        value_range=(minimum, maximum),
        colormap=colormap,
    ).reshape(bar_height, 3)
    bar_right = COLORBAR_PADDING + COLORBAR_BAR_WIDTH
    canvas[
        bar_top:bar_bottom, COLORBAR_PADDING:bar_right
    ] = np.repeat(column[:, None, :], COLORBAR_BAR_WIDTH, axis=1)

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()
    labels = list(tick_labels) if tick_labels is not None else None
    for index in range(COLORBAR_TICK_COUNT):
        fraction = index / (COLORBAR_TICK_COUNT - 1)
        value = minimum + fraction * (maximum - minimum)
        y = int(round(bar_bottom - fraction * bar_height))
        y = min(max(bar_top, y), bar_bottom - 1)
        draw.line(
            [(bar_right, y), (bar_right + 4, y)], fill=_FOREGROUND_RGB
        )
        if labels is not None:
            label = labels[index] if index < len(labels) else _format_tick(value)
        else:
            label = _format_tick(value)
        draw.text(
            (bar_right + 6, y - 4), label, fill=_FOREGROUND_RGB, font=font
        )
    return np.asarray(pil, dtype=np.uint8)


def compose_with_legend(image: np.ndarray, legend: np.ndarray) -> np.ndarray:
    """Place a legend *below* the image without rescaling the image.

    The output expands only when the legend needs more width for complete class
    names. The scientific image stays byte-for-byte at its native size in the
    top-left; neither the image nor the legend is ever cropped or rescaled.
    """
    width = max(int(image.shape[1]), int(legend.shape[1]))
    fitted_image = _fit_width(image, width)
    fitted_legend = _fit_width(legend, width)
    return np.concatenate([fitted_image, fitted_legend], axis=0)


def compose_with_colorbar(image: np.ndarray, colorbar: np.ndarray) -> np.ndarray:
    """Place a colorbar in a right-hand margin without rescaling the image."""
    fitted = _fit_height(colorbar, image.shape[0])
    return np.concatenate([image, fitted], axis=1)


# ---------------------------------------------------------------------------
# Variant orchestration
# ---------------------------------------------------------------------------
def _entropy_tick_labels(upper: float) -> list[str]:
    labels: list[str] = []
    for index in range(COLORBAR_TICK_COUNT):
        fraction = index / (COLORBAR_TICK_COUNT - 1)
        value = fraction * upper
        labels.append(_format_tick(value))
    labels[-1] = f"ln(N)={upper:.3f}"
    return labels


def render_result_variants(
    image: np.ndarray,
    prediction: GridPrediction,
    *,
    class_names: Sequence[str],
    colors: Sequence[str],
    alpha: float,
    stride: int,
    class_count: int | None = None,
) -> dict[str, np.ndarray]:
    """Produce every downloadable presentation PNG as a ``uint8`` RGB array.

    Returns the seven keys in :data:`RESULT_EXPORT_NAMES`. This is a pure
    function: it performs no I/O and never mutates its inputs.
    """
    count = int(class_count) if class_count is not None else len(colors)
    palette = list(normalize_palette(colors, class_count=count))
    resolved_alpha = validate_alpha(alpha)

    overlay = render_overlay(
        image, prediction, colors=palette, alpha=resolved_alpha, stride=stride
    )
    mask = render_categorical_mask(
        prediction, image.shape, colors=palette, stride=stride
    )

    classes = [
        {
            "index": index,
            "name": str(class_names[index]) if index < len(class_names) else f"Class {index + 1}",
            "color": palette[index],
        }
        for index in range(count)
    ]
    legend = build_categorical_legend(classes)
    overlay_legend = compose_with_legend(overlay, legend)

    left, top, right, bottom = prediction_display_bounds(
        prediction, image.shape, stride=stride
    )
    map_size = (right - left, bottom - top)

    confidence = render_scalar_map(
        prediction.confidence_grid,
        map_size,
        value_range=_CONFIDENCE_RANGE,
        colormap=_CONFIDENCE_COLORMAP,
    )
    confidence_colorbar = compose_with_colorbar(
        confidence,
        build_scalar_colorbar(
            _CONFIDENCE_RANGE,
            colormap=_CONFIDENCE_COLORMAP,
            height=confidence.shape[0],
        ),
    )

    entropy_upper = float(np.log(max(count, 2)))
    entropy = render_scalar_map(
        prediction.entropy_grid,
        map_size,
        value_range=(0.0, entropy_upper),
        colormap=_ENTROPY_COLORMAP,
    )
    entropy_colorbar = compose_with_colorbar(
        entropy,
        build_scalar_colorbar(
            (0.0, entropy_upper),
            colormap=_ENTROPY_COLORMAP,
            height=entropy.shape[0],
            tick_labels=_entropy_tick_labels(entropy_upper),
        ),
    )

    return {
        "overlay": overlay,
        "overlay_legend": overlay_legend,
        "mask": mask,
        "confidence": confidence,
        "confidence_colorbar": confidence_colorbar,
        "entropy": entropy,
        "entropy_colorbar": entropy_colorbar,
    }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def write_png(array: np.ndarray, path: str | Path) -> Path:
    """Write a ``uint8`` RGB array as a lossless PNG and return its path."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(target)
    return target


def prepare_download_copy(
    source: str | Path,
    *,
    run_id: str,
    filename: str,
) -> Path:
    """Copy an artifact into the system temporary directory for browser download.

    The durable source artifact is only read, never moved or deleted. The
    temporary copy is what Gradio can serve; it is never the durable record.
    """
    origin = Path(source).expanduser().resolve()
    if not origin.is_file():
        raise PresentationError(f"The artifact to download does not exist: {origin}")
    download_directory = Path(tempfile.gettempdir()) / "symmetry-harness-downloads"
    download_directory.mkdir(parents=True, exist_ok=True)
    safe_name = Path(str(filename)).name
    destination = download_directory / f"{str(run_id)}-{safe_name}"
    shutil.copyfile(origin, destination)
    return destination
