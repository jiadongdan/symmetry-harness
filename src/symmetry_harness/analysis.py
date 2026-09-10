"""Provider result validation and display-only rendering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageColor

from .image_io import unit_to_uint8


@dataclass(frozen=True)
class DensePrediction:
    """Dense grid outputs in source-image coordinates."""

    coordinates_xy: np.ndarray
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    confidence: np.ndarray
    entropy: np.ndarray
    prediction_grid: np.ndarray
    confidence_grid: np.ndarray
    entropy_grid: np.ndarray


@dataclass(frozen=True)
class TraditionalPrediction:
    """Dense traditional-ML outputs in source-image coordinates.

    This mirrors :class:`DensePrediction` without neural-network logits. The
    traditional workflow never fabricates logits, so its contract is validated
    separately and the existing DL contract stays strict.
    """

    coordinates_xy: np.ndarray
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    confidence: np.ndarray
    entropy: np.ndarray
    prediction_grid: np.ndarray
    confidence_grid: np.ndarray
    entropy_grid: np.ndarray


TRADITIONAL_REQUIRED_ARRAYS = (
    "coordinates_xy",
    "x_coordinates",
    "y_coordinates",
    "probabilities",
    "predictions",
    "confidence",
    "entropy",
    "prediction_grid",
    "confidence_grid",
    "entropy_grid",
)


def dense_prediction_from_arrays(arrays: dict[str, np.ndarray]) -> DensePrediction:
    """Validate and materialize the dense output contract returned by the provider."""
    required = {
        "coordinates_xy",
        "x_coordinates",
        "y_coordinates",
        "logits",
        "probabilities",
        "predictions",
        "confidence",
        "entropy",
        "prediction_grid",
        "confidence_grid",
        "entropy_grid",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise RuntimeError(f"Provider output is missing arrays: {missing}")
    prediction = DensePrediction(
        coordinates_xy=np.asarray(arrays["coordinates_xy"], dtype=np.int32),
        x_coordinates=np.asarray(arrays["x_coordinates"], dtype=np.int32),
        y_coordinates=np.asarray(arrays["y_coordinates"], dtype=np.int32),
        logits=np.asarray(arrays["logits"], dtype=np.float32),
        probabilities=np.asarray(arrays["probabilities"], dtype=np.float32),
        predictions=np.asarray(arrays["predictions"], dtype=np.int16),
        confidence=np.asarray(arrays["confidence"], dtype=np.float32),
        entropy=np.asarray(arrays["entropy"], dtype=np.float32),
        prediction_grid=np.asarray(arrays["prediction_grid"], dtype=np.int16),
        confidence_grid=np.asarray(arrays["confidence_grid"], dtype=np.float32),
        entropy_grid=np.asarray(arrays["entropy_grid"], dtype=np.float32),
    )
    sample_count = len(prediction.coordinates_xy)
    grid_shape = (len(prediction.y_coordinates), len(prediction.x_coordinates))
    if prediction.coordinates_xy.shape != (sample_count, 2):
        raise RuntimeError("Provider coordinates have an invalid shape.")
    if prediction.logits.shape[0] != sample_count:
        raise RuntimeError("Provider logits do not align with coordinates.")
    if prediction.probabilities.shape != prediction.logits.shape:
        raise RuntimeError("Provider probabilities do not align with logits.")
    for values in (
        prediction.predictions,
        prediction.confidence,
        prediction.entropy,
    ):
        if values.shape != (sample_count,):
            raise RuntimeError("Provider vector outputs do not align with coordinates.")
    for grid in (
        prediction.prediction_grid,
        prediction.confidence_grid,
        prediction.entropy_grid,
    ):
        if grid.shape != grid_shape:
            raise RuntimeError("Provider grid outputs do not match the coordinate grid.")
    return prediction


def traditional_prediction_from_arrays(
    arrays: dict[str, np.ndarray],
    *,
    class_count: int | None = None,
) -> TraditionalPrediction:
    """Validate and materialize a traditional-ML dense output contract.

    Unlike the DL contract this validator requires no ``logits`` array, and it
    additionally checks that the returned probability matrix is a real
    probability distribution aligned with contiguous class indices.
    """
    missing = sorted(set(TRADITIONAL_REQUIRED_ARRAYS).difference(arrays))
    if missing:
        raise RuntimeError(f"Traditional Provider output is missing arrays: {missing}")
    prediction = TraditionalPrediction(
        coordinates_xy=np.asarray(arrays["coordinates_xy"], dtype=np.int32),
        x_coordinates=np.asarray(arrays["x_coordinates"], dtype=np.int32),
        y_coordinates=np.asarray(arrays["y_coordinates"], dtype=np.int32),
        probabilities=np.asarray(arrays["probabilities"], dtype=np.float32),
        predictions=np.asarray(arrays["predictions"], dtype=np.int16),
        confidence=np.asarray(arrays["confidence"], dtype=np.float32),
        entropy=np.asarray(arrays["entropy"], dtype=np.float32),
        prediction_grid=np.asarray(arrays["prediction_grid"], dtype=np.int16),
        confidence_grid=np.asarray(arrays["confidence_grid"], dtype=np.float32),
        entropy_grid=np.asarray(arrays["entropy_grid"], dtype=np.float32),
    )
    sample_count = len(prediction.coordinates_xy)
    grid_shape = (len(prediction.y_coordinates), len(prediction.x_coordinates))
    if prediction.coordinates_xy.shape != (sample_count, 2):
        raise RuntimeError("Traditional Provider coordinates have an invalid shape.")
    if prediction.probabilities.ndim != 2 or prediction.probabilities.shape[0] != sample_count:
        raise RuntimeError(
            "Traditional Provider probabilities do not align with coordinates."
        )
    if prediction.probabilities.shape[1] < 2:
        raise RuntimeError("Traditional Provider probabilities require two classes.")
    if class_count is not None and prediction.probabilities.shape[1] != int(class_count):
        raise RuntimeError(
            "Traditional Provider probabilities do not match the requested classes."
        )
    if not np.isfinite(prediction.probabilities).all():
        raise RuntimeError("Traditional Provider probabilities contain nonfinite values.")
    if float(prediction.probabilities.min()) < -1e-6:
        raise RuntimeError("Traditional Provider probabilities contain negative values.")
    row_sums = prediction.probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise RuntimeError("Traditional Provider probability rows must sum to one.")
    for values in (
        prediction.predictions,
        prediction.confidence,
        prediction.entropy,
    ):
        if values.shape != (sample_count,):
            raise RuntimeError(
                "Traditional Provider vector outputs do not align with coordinates."
            )
    for grid in (
        prediction.prediction_grid,
        prediction.confidence_grid,
        prediction.entropy_grid,
    ):
        if grid.shape != grid_shape:
            raise RuntimeError(
                "Traditional Provider grid outputs do not match the coordinate grid."
            )
    if not np.isfinite(prediction.confidence).all() or not np.isfinite(
        prediction.entropy
    ).all():
        raise RuntimeError("Traditional Provider confidence or entropy is nonfinite.")
    return prediction


GridPrediction = DensePrediction | TraditionalPrediction


def _resize_grid(grid: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(grid)
    return np.asarray(image.resize(size, resample=Image.Resampling.NEAREST))


def prediction_display_bounds(
    prediction: GridPrediction,
    image_shape: tuple[int, int],
    *,
    stride: int,
) -> tuple[int, int, int, int]:
    """Return the source-image rectangle covered by dense prediction cells."""
    if stride <= 0:
        raise ValueError("Prediction stride must be positive.")
    height, width = (int(value) for value in image_shape)
    if height <= 0 or width <= 0:
        raise ValueError("Prediction image dimensions must be positive.")
    if not len(prediction.x_coordinates) or not len(prediction.y_coordinates):
        raise ValueError("Dense prediction coordinates cannot be empty.")
    before = stride // 2
    after = stride - before
    left = max(0, int(prediction.x_coordinates[0]) - before)
    top = max(0, int(prediction.y_coordinates[0]) - before)
    right = min(width, int(prediction.x_coordinates[-1]) + after)
    bottom = min(height, int(prediction.y_coordinates[-1]) + after)
    if right <= left or bottom <= top:
        raise ValueError("Dense prediction has no displayable region.")
    return left, top, right, bottom


def render_prediction_overlay(
    image: np.ndarray,
    prediction: GridPrediction,
    class_colors: list[str],
    *,
    stride: int,
    alpha: float = 0.48,
) -> np.ndarray:
    """Render only the source-image region covered by dense predictions."""
    base = np.repeat(unit_to_uint8(image)[..., None], 3, axis=2).astype(np.float32)
    palette = np.asarray(
        [ImageColor.getrgb(color) for color in class_colors], dtype=np.uint8
    )
    color_grid = palette[prediction.prediction_grid]
    left, top, right, bottom = prediction_display_bounds(
        prediction, image.shape, stride=stride
    )
    overlay = base[top:bottom, left:right].copy()
    colors = _resize_grid(
        color_grid, (right - left, bottom - top)
    ).astype(np.float32)
    overlay[:] = (1.0 - alpha) * overlay + alpha * colors
    return np.rint(np.clip(overlay, 0, 255)).astype(np.uint8)



def render_scalar_map(
    grid: np.ndarray,
    size: tuple[int, int],
    *,
    value_range: tuple[float, float],
) -> np.ndarray:
    """Render a scalar grid against a fixed, interpretable value range."""
    values = np.asarray(grid, dtype=np.float32)
    minimum, maximum = (float(value) for value in value_range)
    if not np.isfinite(values).all():
        raise ValueError("Scalar maps must contain only finite values.")
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError("Scalar map value_range must be finite and increasing.")
    normalized = np.clip((values - minimum) / (maximum - minimum), 0.0, 1.0)
    red = normalized
    green = 1.0 - np.abs(2.0 * normalized - 1.0)
    blue = 1.0 - normalized
    rgb = np.stack((red, green, blue), axis=2)
    pixels = np.rint(rgb * 255.0).astype(np.uint8)
    return _resize_grid(pixels, size).astype(np.uint8)
