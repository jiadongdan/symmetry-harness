"""Strict single-channel image loading, normalization, and previews."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import NORMALIZATION_POLICIES


def file_sha256(path: str | Path) -> str:
    """Calculate a file checksum without loading the complete file into memory."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_single_channel(path: str | Path) -> np.ndarray:
    """Load one numeric two-dimensional image from a local file."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input image does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".npy":
        image = np.load(source, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            if len(archive.files) != 1:
                raise ValueError("An NPZ input must contain exactly one array.")
            image = archive[archive.files[0]]
    else:
        with Image.open(source) as loaded:
            image = np.asarray(loaded)
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(
            f"Expected a single-channel two-dimensional image, got shape {image.shape}."
        )
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("Input image must have a numeric dtype.")
    if not np.isfinite(image).all():
        raise ValueError("Input image contains nonfinite values.")
    if min(image.shape) < 64:
        raise ValueError("Both input dimensions must be at least 64 pixels.")
    return image


def normalize_image(
    image: np.ndarray, policy: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert a numeric image into the explicit float32 unit interval."""
    if policy not in NORMALIZATION_POLICIES:
        raise ValueError(f"Unsupported normalization policy: {policy}")
    original = np.asarray(image)
    values = original.astype(np.float32, copy=False)
    minimum = float(values.min())
    maximum = float(values.max())
    parameters: dict[str, Any]
    if policy == "minmax_0_1":
        if maximum == minimum:
            raise ValueError("A constant image cannot use min-max normalization.")
        unit = (values - minimum) / (maximum - minimum)
        parameters = {"minimum": minimum, "maximum": maximum}
    elif policy == "already_0_1":
        if minimum < -1e-6 or maximum > 1.000001:
            raise ValueError("already_0_1 input lies outside [0, 1].")
        unit = values.copy()
        parameters = {}
    else:
        if np.issubdtype(original.dtype, np.integer):
            dtype_info = np.iinfo(original.dtype)
            if dtype_info.min < 0:
                raise ValueError("dtype_unit does not support signed integer inputs.")
            unit = values / float(dtype_info.max)
            parameters = {"dtype_maximum": int(dtype_info.max)}
        else:
            if minimum < -1e-6 or maximum > 1.000001:
                raise ValueError("Floating dtype_unit input lies outside [0, 1].")
            unit = values.copy()
            parameters = {}
    unit = np.asarray(unit, dtype=np.float32)
    if not np.isfinite(unit).all():
        raise RuntimeError("Normalization produced nonfinite values.")
    return unit, {
        "policy": policy,
        "original_dtype": str(original.dtype),
        "original_range": [minimum, maximum],
        "output_dtype": "float32",
        "output_range": [float(unit.min()), float(unit.max())],
        "parameters": parameters,
    }


def inspect_input(
    path: str | Path, normalization_policy: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load, normalize, checksum, and describe one input image."""
    source = Path(path).expanduser().resolve()
    raw = load_single_channel(source)
    unit, transform = normalize_image(raw, normalization_policy)
    return unit, {
        "path": str(source),
        "sha256": file_sha256(source),
        "shape": list(raw.shape),
        "dtype": str(raw.dtype),
        "finite": True,
        "normalization": transform,
    }


def unit_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a unit-range image to an eight-bit array for display only."""
    values = np.asarray(image, dtype=np.float32)
    return np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_preview(path: str | Path, image: np.ndarray) -> None:
    """Save a display-only grayscale PNG preview."""
    Image.fromarray(unit_to_uint8(image), mode="L").save(Path(path))

