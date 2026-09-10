"""Model-independent point-annotation helpers shared by the Harness workspaces.

Both the fine-tuning workspace and the standalone traditional-ML validation page
let a user define named classes and click support points on one single-channel
image. Only the shot-count vocabulary and the fallback patch size differ between
them, so the state machine and rendering live here and every caller supplies its
own numeric policy. No pretrained-model concept appears in this module.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .annotations import DEFAULT_CLASS_COLORS, valid_center_bounds
from .image_io import unit_to_uint8
from .ui_shared import (
    ANNOTATION_DISPLAY_MAX_EDGE,
    INVALID_SUPPORT_POINT_MESSAGE,
    parse_class_names,
)


def copy_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-enough copy of the annotation UI state."""
    copied = dict(state)
    copied["class_names"] = list(state.get("class_names", []))
    copied["colors"] = list(state.get("colors", []))
    copied["points"] = [list(group) for group in state.get("points", [])]
    return copied


def state_patch_size(state: dict[str, Any], fallback: int) -> int:
    """Return the classifier patch size retained by the UI state."""
    patch_size = int(state.get("classifier_patch_size", fallback))
    if patch_size <= 0:
        raise ValueError("The model input patch size must be positive.")
    return patch_size


def draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    *,
    fill: str,
    width: int,
    dash_length: int,
    gap_length: int,
) -> None:
    """Draw a dashed rectangle with a consistent display-space line width."""
    left, top, right, bottom = bounds
    step = dash_length + gap_length
    for start in range(left, right + 1, step):
        end = min(start + dash_length, right)
        draw.line((start, top, end, top), fill=fill, width=width)
        draw.line((start, bottom, end, bottom), fill=fill, width=width)
    for start in range(top, bottom + 1, step):
        end = min(start + dash_length, bottom)
        draw.line((left, start, left, end), fill=fill, width=width)
        draw.line((right, start, right, end), fill=fill, width=width)


def annotation_rows(state: dict[str, Any]) -> list[list[Any]]:
    """Return compact class counts and source-image coordinates for display."""
    rows = []
    for name, points in zip(state.get("class_names", []), state.get("points", [])):
        rows.append([name, len(points), ", ".join(f"({x}, {y})" for x, y in points)])
    return rows


def annotation_display_shape(
    image_shape: tuple[int, int],
    maximum_edge: int = ANNOTATION_DISPLAY_MAX_EDGE,
) -> tuple[int, int]:
    """Return an aspect-preserving display raster with a bounded longest edge."""
    height, width = (int(value) for value in image_shape)
    if height <= 0 or width <= 0 or maximum_edge <= 0:
        raise ValueError("Annotation display dimensions must be positive.")
    scale = float(maximum_edge) / float(max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))


def display_to_source_point(
    point: tuple[int, int], image_shape: tuple[int, int]
) -> tuple[int, int]:
    """Map a click in the display raster back to an exact source-array index."""
    display_height, display_width = annotation_display_shape(image_shape)
    source_height, source_width = (int(value) for value in image_shape)
    display_x, display_y = (int(value) for value in point)
    if not 0 <= display_x < display_width or not 0 <= display_y < display_height:
        raise ValueError("The image click falls outside the annotation display raster.")

    def map_index(index: int, display_length: int, source_length: int) -> int:
        if display_length == 1 or source_length == 1:
            return 0
        return int(round(index * (source_length - 1) / (display_length - 1)))

    return (
        map_index(display_x, display_width, source_width),
        map_index(display_y, display_height, source_height),
    )


def render_annotations(state: dict[str, Any], patch_size: int) -> np.ndarray | None:
    """Render support points and patch outlines over a display-only image copy."""
    image = state.get("image")
    if image is None:
        return None
    patch_size = state_patch_size(state, patch_size)
    gray = unit_to_uint8(np.asarray(image, dtype=np.float32))
    base = Image.fromarray(gray).convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    shade = ImageDraw.Draw(overlay)
    before = patch_size // 2
    after = patch_size - before
    width, height = base.size
    shade.rectangle((0, 0, width, before - 1), fill=(50, 50, 50, 95))
    shade.rectangle((0, height - after + 1, width, height), fill=(50, 50, 50, 95))
    shade.rectangle((0, 0, before - 1, height), fill=(50, 50, 50, 95))
    shade.rectangle((width - after + 1, 0, width, height), fill=(50, 50, 50, 95))
    base = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(base)
    for color, points in zip(state.get("colors", []), state.get("points", [])):
        for x, y in points:
            draw.rectangle(
                (x - before, y - before, x + after - 1, y + after - 1),
                outline=color,
                width=2,
            )
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="white")
    display_height, display_width = annotation_display_shape((height, width))
    if base.size != (display_width, display_height):
        base = base.resize(
            (display_width, display_height), resample=Image.Resampling.BILINEAR
        )
    if state.get("show_valid_region", False):
        min_x, max_x, min_y, max_y = valid_center_bounds(
            (height, width), patch_size
        )

        def display_index(index: int, source_length: int, display_length: int) -> int:
            if source_length == 1 or display_length == 1:
                return 0
            return int(round(index * (display_length - 1) / (source_length - 1)))

        valid_bounds = (
            display_index(min_x, width, display_width),
            display_index(min_y, height, display_height),
            display_index(max_x, width, display_width),
            display_index(max_y, height, display_height),
        )
        line_width = max(4, round(max(display_width, display_height) / 160))
        draw_dashed_rectangle(
            ImageDraw.Draw(base),
            valid_bounds,
            fill="white",
            width=line_width,
            dash_length=line_width * 3,
            gap_length=line_width * 2,
        )
    return np.asarray(base)


def source_patch_preview(
    state: dict[str, Any], point: tuple[int, int] | None, patch_size: int
) -> np.ndarray | None:
    """Return the raw normalized source patch for immediate annotation review."""
    if point is None or state.get("image") is None:
        return None
    x, y = point
    before = patch_size // 2
    after = patch_size - before
    patch = np.asarray(state["image"])[
        y - before : y + after, x - before : x + after
    ]
    return unit_to_uint8(patch)


def array_sha256(values: np.ndarray) -> str:
    """Return a content digest for a normalized array and its geometry."""
    array = np.ascontiguousarray(values, dtype=np.float32)
    digest = sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def image_shape_text(record: dict[str, Any] | None) -> str:
    """Render the only image metadata needed in the interactive layout."""
    if not record:
        return "**Image shape:** \u2014"
    height, width = (int(value) for value in record["shape"])
    return f"**Image shape:** {height} \u00d7 {width} (H \u00d7 W)"


def annotation_state(
    image: np.ndarray,
    record: dict[str, Any],
    *,
    classifier_patch_size: int,
    previous_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Build a fresh annotation state for one loaded image.

    Class names and colors may be retained by the caller, but every support
    point is cleared because the image geometry changed.
    """
    previous = previous_state or {}
    class_names = list(previous.get("class_names", []))
    colors = list(previous.get("colors", []))
    state = {
        "image": image,
        "image_path": record["path"],
        "image_sha256": record["sha256"],
        "normalized_image_sha256": array_sha256(image),
        "image_shape": record["shape"],
        "class_names": class_names,
        "colors": colors,
        "points": [[] for _ in class_names],
        "classifier_patch_size": int(classifier_patch_size),
        "show_valid_region": False,
    }
    next_step = (
        "Class definitions were retained; select new support points."
        if class_names
        else "Configure local classes next."
    )
    height, width = (int(value) for value in record["shape"])
    status = (
        f"Loaded {Path(record['path']).name}: shape {height} \u00d7 {width}. {next_step}"
    )
    return state, status


def configure_class_state(
    value: str,
    state: dict[str, Any],
    *,
    classifier_patch_size: int,
) -> tuple[dict[str, Any], int]:
    """Parse class names, reset support points, and validate patch geometry.

    Returns the updated state and its resolved classifier patch size. Callers
    own the human-readable status message because the shot-count vocabulary is
    workspace-specific.
    """
    if state.get("image") is None:
        raise ValueError("Load an input image before configuring classes.")
    names = parse_class_names(value)
    updated = copy_state(state)
    updated["class_names"] = names
    updated["colors"] = list(DEFAULT_CLASS_COLORS[: len(names)])
    updated["points"] = [[] for _ in names]
    patch_size = int(classifier_patch_size)
    if patch_size <= 0:
        raise ValueError("The classifier patch size must be positive.")
    valid_center_bounds(tuple(updated["image_shape"]), patch_size)
    updated["classifier_patch_size"] = patch_size
    updated["show_valid_region"] = True
    return updated, patch_size


def add_point_state(
    state: dict[str, Any],
    active_class: str | None,
    point: tuple[int, int],
    *,
    fallback_patch_size: int,
    maximum_points: int,
) -> tuple[dict[str, Any], int, str, bool]:
    """Add one support point.

    Returns ``(state, patch_size, status, invalid)``. ``invalid`` is ``True``
    when the click falls outside the valid center region; the state is then
    unchanged and the status carries the shared invalid-point message.
    """
    if not state.get("class_names"):
        raise ValueError("Configure local classes before selecting points.")
    if active_class not in state["class_names"]:
        raise ValueError("Choose an active class before selecting points.")
    x, y = int(point[0]), int(point[1])
    patch_size = state_patch_size(state, fallback_patch_size)
    min_x, max_x, min_y, max_y = valid_center_bounds(
        tuple(state["image_shape"]), patch_size
    )
    if not min_x <= x <= max_x or not min_y <= y <= max_y:
        return state, patch_size, INVALID_SUPPORT_POINT_MESSAGE, True
    updated = copy_state(state)
    class_index = updated["class_names"].index(active_class)
    if any((x, y) in group for group in updated["points"]):
        raise ValueError("This source-image point has already been assigned.")
    if len(updated["points"][class_index]) >= maximum_points:
        raise ValueError("This class has reached the configured support-point limit.")
    updated["points"][class_index].append((x, y))
    count = len(updated["points"][class_index])
    status = f"Added ({x}, {y}) to {active_class}. This class now has {count} support points."
    return updated, patch_size, status, False


def undo_point_state(
    state: dict[str, Any],
    active_class: str | None,
    *,
    fallback_patch_size: int,
) -> tuple[dict[str, Any], int, tuple[int, int], str]:
    """Remove the most recent support point of the active class."""
    if active_class not in state.get("class_names", []):
        raise ValueError("Choose an active class first.")
    updated = copy_state(state)
    class_index = updated["class_names"].index(active_class)
    if not updated["points"][class_index]:
        raise ValueError("The active class has no point to undo.")
    removed = updated["points"][class_index].pop()
    patch_size = state_patch_size(updated, fallback_patch_size)
    return updated, patch_size, tuple(removed), f"Removed {tuple(removed)} from {active_class}."


def clear_class_state(
    state: dict[str, Any],
    active_class: str | None,
    *,
    fallback_patch_size: int,
) -> tuple[dict[str, Any], int, str]:
    """Remove every support point of the active class."""
    if active_class not in state.get("class_names", []):
        raise ValueError("Choose an active class first.")
    updated = copy_state(state)
    class_index = updated["class_names"].index(active_class)
    updated["points"][class_index] = []
    patch_size = state_patch_size(updated, fallback_patch_size)
    return updated, patch_size, f"Cleared all support points for {active_class}."


__all__ = [
    "add_point_state",
    "annotation_display_shape",
    "annotation_rows",
    "annotation_state",
    "array_sha256",
    "clear_class_state",
    "configure_class_state",
    "copy_state",
    "display_to_source_point",
    "draw_dashed_rectangle",
    "image_shape_text",
    "render_annotations",
    "source_patch_preview",
    "state_patch_size",
    "undo_point_state",
]
