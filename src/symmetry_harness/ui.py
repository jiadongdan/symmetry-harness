"""Local Gradio interface for user-selected few-shot support points."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import os
import socket
import tempfile
from typing import Any, Callable
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .annotations import (
    DEFAULT_CLASS_COLORS,
    create_annotation_session,
    valid_center_bounds,
)
from .catalog import model_capability, model_weights, provider_models, weight_capability
from .config import HarnessConfig, config_for_model_selection
from .image_io import file_sha256, inspect_input, unit_to_uint8
from .provider import install_registered_weight, run_provider_features
from .workflow import build_run_options, run_analysis


REGISTERED_WEIGHT_SOURCE = "Registered weight"
CUSTOM_CHECKPOINT_SOURCE = "Custom checkpoint"
ANNOTATION_DISPLAY_MAX_EDGE = 720
APP_CSS = "#annotation-image {max-width: 760px; margin: 0 auto;}"


def parse_class_names(value: str) -> list[str]:
    """Parse unique comma-separated local class names."""
    names = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(names) < 2:
        raise ValueError("Enter at least two comma-separated class names.")
    if len(set(names)) != len(names):
        raise ValueError("Class names must be unique.")
    if len(names) > len(DEFAULT_CLASS_COLORS):
        raise ValueError(f"The interface supports at most {len(DEFAULT_CLASS_COLORS)} classes.")
    return names


def _copy_state(state: dict[str, Any]) -> dict[str, Any]:
    copied = dict(state)
    copied["class_names"] = list(state.get("class_names", []))
    copied["colors"] = list(state.get("colors", []))
    copied["points"] = [list(group) for group in state.get("points", [])]
    return copied


def annotation_rows(state: dict[str, Any]) -> list[list[Any]]:
    """Return compact class counts and source-image coordinates for display."""
    rows = []
    for name, points in zip(state.get("class_names", []), state.get("points", [])):
        rows.append([name, len(points), ", ".join(f"({x}, {y})" for x, y in points)])
    return rows


def _annotation_display_shape(
    image_shape: tuple[int, int],
    maximum_edge: int = ANNOTATION_DISPLAY_MAX_EDGE,
) -> tuple[int, int]:
    """Return an aspect-preserving display raster with a bounded longest edge."""
    height, width = (int(value) for value in image_shape)
    if height <= 0 or width <= 0 or maximum_edge <= 0:
        raise ValueError("Annotation display dimensions must be positive.")
    scale = float(maximum_edge) / float(max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))


def _display_to_source_point(
    point: tuple[int, int], image_shape: tuple[int, int]
) -> tuple[int, int]:
    """Map a click in the display raster back to an exact source-array index."""
    display_height, display_width = _annotation_display_shape(image_shape)
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
    display_height, display_width = _annotation_display_shape((height, width))
    if base.size != (display_width, display_height):
        base = base.resize(
            (display_width, display_height), resample=Image.Resampling.BILINEAR
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


def _array_sha256(values: np.ndarray) -> str:
    """Return a content digest for a normalized array and its geometry."""
    array = np.ascontiguousarray(values, dtype=np.float32)
    digest = sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _image_shape_text(record: dict[str, Any] | None) -> str:
    """Render the only image metadata needed in the interactive layout."""
    if not record:
        return "**Image shape:** —"
    height, width = (int(value) for value in record["shape"])
    return f"**Image shape:** {height} × {width} (H × W)"


def _loaded_image_outputs(
    image: np.ndarray,
    record: dict[str, Any],
    config: HarnessConfig,
    previous_state: dict[str, Any] | None = None,
):
    previous = previous_state or {}
    class_names = list(previous.get("class_names", []))
    colors = list(previous.get("colors", []))
    state = {
        "image": image,
        "image_path": record["path"],
        "image_sha256": record["sha256"],
        "normalized_image_sha256": _array_sha256(image),
        "image_shape": record["shape"],
        "class_names": class_names,
        "colors": colors,
        "points": [[] for _ in class_names],
    }
    next_step = (
        "Class definitions were retained; select new support points."
        if class_names
        else "Configure local classes next."
    )
    height, width = (int(value) for value in record["shape"])
    status = (
        f"Loaded {Path(record['path']).name}: shape {height} × {width}. {next_step}"
    )
    return render_annotations(state, config.model.classifier_patch_size), state, status


def _load_image(
    path: str | None,
    config: HarnessConfig,
    previous_state: dict[str, Any] | None = None,
):
    if not path:
        raise ValueError("Choose an input image first.")
    image, record = inspect_input(path, config.features.input_normalization)
    return _loaded_image_outputs(image, record, config, previous_state)


def _configure_classes(value: str, state: dict[str, Any], config: HarnessConfig):
    import gradio as gr

    if state.get("image") is None:
        raise ValueError("Load an input image before configuring classes.")
    names = parse_class_names(value)
    updated = _copy_state(state)
    updated["class_names"] = names
    updated["colors"] = list(DEFAULT_CLASS_COLORS[: len(names)])
    updated["points"] = [[] for _ in names]
    status = (
        f"Configured {len(names)} classes. Select an active class and click at least "
        f"{config.fine_tuning.minimum_shots_per_class} support points per class; "
        f"{config.fine_tuning.recommended_shots_per_class} are recommended."
    )
    return (
        updated,
        gr.update(choices=names, value=names[0]),
        render_annotations(updated, config.model.classifier_patch_size),
        annotation_rows(updated),
        status,
    )


def _add_point(
    state: dict[str, Any], active_class: str | None, point: tuple[int, int], config: HarnessConfig
):
    if not state.get("class_names"):
        raise ValueError("Configure local classes before selecting points.")
    if active_class not in state["class_names"]:
        raise ValueError("Choose an active class before selecting points.")
    x, y = int(point[0]), int(point[1])
    min_x, max_x, min_y, max_y = valid_center_bounds(
        tuple(state["image_shape"]), config.model.classifier_patch_size
    )
    if not min_x <= x <= max_x or not min_y <= y <= max_y:
        raise ValueError(
            "This point is too close to the image boundary for a complete classifier patch."
        )
    updated = _copy_state(state)
    class_index = updated["class_names"].index(active_class)
    if any((x, y) in group for group in updated["points"]):
        raise ValueError("This source-image point has already been assigned.")
    if len(updated["points"][class_index]) >= config.fine_tuning.maximum_shots_per_class:
        raise ValueError("This class has reached the configured support-point limit.")
    updated["points"][class_index].append((x, y))
    count = len(updated["points"][class_index])
    status = f"Added ({x}, {y}) to {active_class}. This class now has {count} support points."
    return (
        updated,
        render_annotations(updated, config.model.classifier_patch_size),
        source_patch_preview(updated, (x, y), config.model.classifier_patch_size),
        annotation_rows(updated),
        status,
    )


def _undo_point(state: dict[str, Any], active_class: str | None, config: HarnessConfig):
    if active_class not in state.get("class_names", []):
        raise ValueError("Choose an active class first.")
    updated = _copy_state(state)
    class_index = updated["class_names"].index(active_class)
    if not updated["points"][class_index]:
        raise ValueError("The active class has no point to undo.")
    removed = updated["points"][class_index].pop()
    return (
        updated,
        render_annotations(updated, config.model.classifier_patch_size),
        annotation_rows(updated),
        f"Removed {removed} from {active_class}.",
    )


def _clear_class(state: dict[str, Any], active_class: str | None, config: HarnessConfig):
    if active_class not in state.get("class_names", []):
        raise ValueError("Choose an active class first.")
    updated = _copy_state(state)
    class_index = updated["class_names"].index(active_class)
    updated["points"][class_index] = []
    return (
        updated,
        render_annotations(updated, config.model.classifier_patch_size),
        annotation_rows(updated),
        f"Cleared all support points for {active_class}.",
    )


def _configured_capabilities(config: HarnessConfig) -> dict[str, Any]:
    """Build an offline catalog fallback for UI construction tests."""
    weight = None
    if config.model.weight_identifier is not None:
        weight = {
            "identifier": config.model.weight_identifier,
            "status": "configured",
            "default": True,
            "bundled": False,
        }
    model = {
        "identifier": config.model.identifier,
        "display_name": config.model.identifier,
        "available": True,
        "input_channels": config.model.input_channels,
        "pretrained_classes": config.model.pretrained_classes,
        "classifier_patch_size": config.model.classifier_patch_size,
        "feature_channels": [],
        "fine_tuning_strategy": "Provider-managed fine-tuning",
        "minimum_shots_per_class": config.fine_tuning.minimum_shots_per_class,
        "recommended_shots_per_class": config.fine_tuning.recommended_shots_per_class,
        "maximum_shots_per_class": config.fine_tuning.maximum_shots_per_class,
        "defaults": {
            "n_max": config.features.n_max,
            "symmetry_patch_size": config.features.symmetry_patch_size,
            "rotation_folds": list(config.features.rotation_folds),
            "reflection_p": config.features.reflection_p,
            "normalize_rotation_maps": config.features.normalize_rotation_maps,
            "adapter_bottleneck": config.fine_tuning.adapter_bottleneck,
            "epochs": config.fine_tuning.epochs,
            "learning_rate": config.fine_tuning.learning_rate,
            "weight_decay": config.fine_tuning.weight_decay,
            "seed": config.fine_tuning.seed,
            "stride": config.prediction.stride,
            "batch_size": config.prediction.batch_size,
        },
        "weights": [] if weight is None else [weight],
        "default_weight": weight,
    }
    return {"models": [model]}


def _model_choices(capabilities: dict[str, Any]) -> list[tuple[str, str]]:
    choices = []
    for model in provider_models(capabilities):
        identifier = str(model["identifier"])
        display = str(model.get("display_name", identifier))
        choices.append((f"{display} ({identifier})", identifier))
    return choices


def _weight_choices(model: dict[str, Any]) -> list[tuple[str, str]]:
    choices = []
    for weight in model_weights(model):
        identifier = str(weight["identifier"])
        status = str(weight.get("status", "unknown"))
        version = str(weight.get("version", "")).strip()
        suffix = f", v{version}" if version else ""
        choices.append((f"{identifier} ({status}{suffix})", identifier))
    return choices


def _selection_details(
    capability: dict[str, Any], weight_identifier: str | None
) -> str:
    channels = ", ".join(str(item) for item in capability.get("feature_channels", []))
    strategy = str(
        capability.get("fine_tuning_strategy", "Provider-managed fine-tuning")
    )
    minimum = int(capability.get("minimum_shots_per_class", 3))
    recommended = int(capability.get("recommended_shots_per_class", 5))
    maximum = int(capability.get("maximum_shots_per_class", 50))
    weight_text = "Custom checkpoint"
    if weight_identifier is not None:
        selected_weight = weight_capability(capability, weight_identifier)
        weight_text = (
            f"`{selected_weight['identifier']}` — "
            f"{selected_weight.get('status', 'unknown')}"
        )
    return (
        f"**Model:** `{capability['identifier']}`  \n"
        f"**Weight:** {weight_text}  \n"
        f"**Input:** {capability.get('input_channels')} channels, "
        f"{capability.get('classifier_patch_size')} px classifier patch  \n"
        f"**Feature channels:** {channels or 'reported at runtime'}  \n"
        f"**Fine-tuning:** {strategy}  \n"
        f"**Support points per class:** minimum {minimum}, "
        f"recommended {recommended}, maximum {maximum}"
    )


def _reset_annotations_for_model(
    state: dict[str, Any], patch_size: int
) -> tuple[dict[str, Any], np.ndarray | None]:
    updated = _copy_state(state)
    updated["points"] = [[] for _ in updated.get("class_names", [])]
    return updated, render_annotations(updated, patch_size)


def _support_is_ready(state: dict[str, Any], config: HarnessConfig) -> bool:
    """Return whether every configured class has enough support points to run."""
    groups = state.get("points", [])
    return bool(state.get("image") is not None and len(groups) >= 2) and all(
        config.fine_tuning.minimum_shots_per_class <= len(group)
        <= config.fine_tuning.maximum_shots_per_class
        for group in groups
    )


def _runtime_config(
    config: HarnessConfig,
    capabilities: dict[str, Any],
    *,
    model_identifier: str,
    source: str,
    weight_identifier: str | None,
    checkpoint_path: str | None,
) -> HarnessConfig:
    capability = model_capability(capabilities, model_identifier)
    if source == CUSTOM_CHECKPOINT_SOURCE:
        if not checkpoint_path:
            raise ValueError("Choose a custom checkpoint before running.")
        resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
        return config_for_model_selection(
            config,
            capability,
            weight_identifier=None,
            checkpoint_path=resolved_checkpoint,
            checkpoint_sha256=file_sha256(resolved_checkpoint),
        )
    if source != REGISTERED_WEIGHT_SOURCE:
        raise ValueError("Choose a supported model-weight source.")
    selected_weight = weight_capability(capability, weight_identifier)
    return config_for_model_selection(
        config,
        capability,
        weight_identifier=str(selected_weight["identifier"]),
    )


def _model_contract_config(
    config: HarnessConfig,
    capabilities: dict[str, Any],
    model_identifier: str,
) -> HarnessConfig:
    """Apply model-dependent UI contracts without resolving a weight file."""
    return config_for_model_selection(
        config,
        model_capability(capabilities, model_identifier),
        weight_identifier=None,
    )


_FEATURE_OPTION_NAMES = (
    "n_max",
    "symmetry_patch_size",
    "rotation_folds",
    "reflection_p",
    "normalize_rotation_maps",
    "device",
)


def _feature_request(
    config: HarnessConfig,
    capabilities: dict[str, Any],
    state: dict[str, Any],
    *,
    model_identifier: str,
    symmetry_patch_size: int,
    device: str,
) -> tuple[HarnessConfig, dict[str, Any], dict[str, Any], str]:
    """Resolve a deterministic feature request and its cache key."""
    if state.get("image") is None:
        raise ValueError("Load an input image before computing symmetry maps.")
    contract = _model_contract_config(config, capabilities, model_identifier)
    options = build_run_options(
        contract,
        {
            "symmetry_patch_size": int(symmetry_patch_size),
            "device": str(device),
        },
    ).to_dict()
    capability = model_capability(capabilities, model_identifier)
    fingerprint = {
        "input_sha256": state["image_sha256"],
        "normalized_image_sha256": state["normalized_image_sha256"],
        "image_shape": list(state["image_shape"]),
        "input_normalization": options["input_normalization"],
        "provider_contract_version": capabilities.get("contract_version"),
        "provider_version": capabilities.get("provider_version"),
        "model_identifier": model_identifier,
        "feature_pipeline": capability.get("feature_pipeline"),
        "feature_options": {name: options[name] for name in _FEATURE_OPTION_NAMES},
    }
    # Canonicalize tuples and other JSON-compatible containers before retaining
    # the fingerprint. The on-disk record is JSON, so comparing it with a
    # non-canonical in-memory tuple would otherwise turn every lookup into a
    # false cache miss.
    serialized_text = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    fingerprint = json.loads(serialized_text)
    serialized = serialized_text.encode("utf-8")
    return contract, options, fingerprint, sha256(serialized).hexdigest()


def _feature_cache_paths(cache_root: Path, key: str) -> tuple[Path, Path]:
    directory = cache_root / key
    return directory / "features.npz", directory / "feature_record.json"


def _load_cached_features(
    features_path: Path,
    record_path: Path,
    *,
    expected_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(features_path, allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float32).copy()
        channel_names = np.asarray(archive["channel_names"]).copy()
    if features.shape != expected_shape or not np.isfinite(features).all():
        raise RuntimeError("Cached features do not match the current image contract.")
    if channel_names.shape != (expected_shape[0],):
        raise RuntimeError("Cached feature channel names are incomplete.")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    return features, channel_names, record


def _feature_preview(channel: np.ndarray, name: str) -> np.ndarray:
    """Convert one float feature channel into a display-only grayscale image."""
    values = np.asarray(channel, dtype=np.float32)
    if name == "image":
        normalized = np.clip(values, 0.0, 1.0)
    elif name in {"reflection_sin_2theta", "reflection_cos_2theta"}:
        normalized = (np.clip(values, -1.0, 1.0) + 1.0) / 2.0
    else:
        minimum = float(values.min())
        maximum = float(values.max())
        normalized = (
            np.zeros_like(values)
            if maximum == minimum
            else (values - minimum) / (maximum - minimum)
        )
    return np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)


def feature_gallery(
    features: np.ndarray, channel_names: np.ndarray
) -> list[tuple[np.ndarray, str]]:
    """Build the fixed two-by-four display payload for the eight channels."""
    items: list[tuple[np.ndarray, str]] = []
    for channel, raw_name in zip(features, channel_names):
        name = str(raw_name)
        caption = (
            f"{name}  [min={float(channel.min()):.4g}, "
            f"max={float(channel.max()):.4g}]"
        )
        items.append((_feature_preview(channel, name), caption))
    return items


def _write_feature_png_bundle(
    directory: Path, features: np.ndarray, channel_names: np.ndarray
) -> Path:
    """Create lossless display PNGs and a labeled two-by-four montage."""
    png_directory = directory / "png"
    png_directory.mkdir(parents=True, exist_ok=True)
    previews: list[tuple[Image.Image, str]] = []
    paths: list[Path] = []
    for index, (channel, raw_name) in enumerate(zip(features, channel_names), start=1):
        name = str(raw_name)
        preview = Image.fromarray(_feature_preview(channel, name))
        path = png_directory / f"{index:02d}_{name}.png"
        preview.save(path)
        paths.append(path)
        previews.append((preview, name))

    tile_size = 320
    label_height = 28
    montage = Image.new("L", (tile_size * 4, (tile_size + label_height) * 2), 255)
    draw = ImageDraw.Draw(montage)
    for index, (preview, name) in enumerate(previews):
        column = index % 4
        row = index // 4
        fitted = ImageOps.contain(
            preview, (tile_size, tile_size), method=Image.Resampling.LANCZOS
        )
        left = column * tile_size + (tile_size - fitted.width) // 2
        top = row * (tile_size + label_height) + label_height
        montage.paste(fitted, (left, top))
        draw.text((column * tile_size + 6, row * (tile_size + label_height) + 7), name, fill=0)
    montage_path = png_directory / "symmetry_features_montage.png"
    montage.save(montage_path)
    paths.append(montage_path)

    archive_path = directory / "symmetry_features_png.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, arcname=path.name)
    return archive_path


def prepare_feature_exports(
    feature_state: dict[str, Any], formats: list[str] | None
) -> tuple[list[str], str]:
    """Materialize the user-selected browser-download feature artifacts."""
    if not feature_state or not feature_state.get("features_path"):
        raise ValueError("Compute the eight-channel features before exporting them.")
    selected = list(formats or [])
    if not selected:
        raise ValueError("Select PNG, NPY, or both before preparing an export.")
    features_path = Path(feature_state["features_path"])
    record_path = Path(feature_state["record_path"])
    expected = tuple(int(value) for value in feature_state["feature_shape"])
    features, channel_names, _ = _load_cached_features(
        features_path, record_path, expected_shape=expected
    )
    export_directory = features_path.parent / "exports"
    export_directory.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    if "NPY" in selected:
        npy_path = export_directory / "symmetry_features.npy"
        np.save(npy_path, features)
        outputs.append(str(npy_path))
    if "PNG" in selected:
        outputs.append(
            str(_write_feature_png_bundle(export_directory, features, channel_names))
        )
    return outputs, "Feature export is ready. Use the download control to choose its destination."


def build_app(
    config: HarnessConfig,
    *,
    prepared_input: tuple[np.ndarray, dict[str, Any]] | None = None,
    capabilities: dict[str, Any] | None = None,
):
    """Construct the local point-annotation and few-shot analysis interface."""
    import gradio as gr

    catalog = capabilities or _configured_capabilities(config)
    initial_model = model_capability(catalog, config.model.identifier)
    registered_choices = _weight_choices(initial_model)
    initial_source = (
        CUSTOM_CHECKPOINT_SOURCE
        if config.model.checkpoint_path is not None
        else REGISTERED_WEIGHT_SOURCE
    )
    initial_weight = config.model.weight_identifier
    if initial_source == REGISTERED_WEIGHT_SOURCE:
        initial_weight = str(weight_capability(initial_model, initial_weight)["identifier"])
    initial_details = _selection_details(
        initial_model,
        initial_weight if initial_source == REGISTERED_WEIGHT_SOURCE else None,
    )
    initial_state: dict[str, Any] = {}
    initial_annotation = None
    initial_status = "Load a single-channel image to begin."
    initial_file = None
    initial_input_record = None
    if prepared_input is not None:
        initial_image, initial_record = prepared_input
        initial_annotation, initial_state, initial_status = _loaded_image_outputs(
            initial_image, initial_record, config
        )
        initial_file = initial_record["path"]
        initial_input_record = initial_record

    feature_cache_directory = tempfile.TemporaryDirectory(
        prefix="symmetry-harness-ui-features-"
    )
    feature_cache_root = Path(feature_cache_directory.name)

    with gr.Blocks(title="Symmetry Harness") as app:
        state = gr.State(initial_state)
        catalog_state = gr.State(catalog)
        feature_state = gr.State({})
        gr.Markdown(
            "# Symmetry Harness\n"
            "Choose a local model and image, select three to five representative "
            "points per class, and run adapter-plus-head fine-tuning locally."
        )
        status = gr.Markdown(initial_status)
        gr.Markdown("## Model and weight")
        with gr.Row():
            model_selector = gr.Dropdown(
                label="Model",
                choices=_model_choices(catalog),
                value=config.model.identifier,
                interactive=True,
            )
            weight_source = gr.Radio(
                label="Weight source",
                choices=[REGISTERED_WEIGHT_SOURCE, CUSTOM_CHECKPOINT_SOURCE],
                value=initial_source,
            )
            weight_selector = gr.Dropdown(
                label="Registered weight",
                choices=registered_choices,
                value=initial_weight,
                interactive=True,
            )
            custom_checkpoint = gr.File(
                label="Custom checkpoint",
                type="filepath",
                value=(
                    None
                    if config.model.checkpoint_path is None
                    or not config.model.checkpoint_path.is_file()
                    else str(config.model.checkpoint_path)
                ),
            )
        install_button = gr.Button("Install selected weight", variant="secondary")
        model_details = gr.Markdown(initial_details)

        gr.Markdown("## Input image")
        with gr.Row():
            input_file = gr.File(
                label="Input image",
                type="filepath",
                value=initial_file,
                file_types=[
                    ".npy",
                    ".npz",
                    ".tif",
                    ".tiff",
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".bmp",
                ],
                scale=2,
            )
            image_details = gr.Markdown(
                _image_shape_text(initial_input_record), min_width=220, scale=1
            )

        gr.Markdown("## Compute 8-channel symmetry maps")
        with gr.Row():
            symmetry_patch_size = gr.Number(
                label="Symmetry patch size",
                value=config.features.symmetry_patch_size,
                precision=0,
                info="Odd local-neighborhood size used to calculate the symmetry maps.",
            )
            device = gr.Dropdown(
                label="Device",
                choices=["auto", "cuda", "cpu"],
                value=config.model.device,
                info="Used for feature extraction, fine-tuning, and prediction.",
            )
            compute_features_button = gr.Button(
                "Compute / update symmetry maps",
                variant="primary",
                interactive=prepared_input is not None,
            )
        feature_status = gr.Markdown(
            "Load an image, then compute its eight-channel representation."
        )
        feature_maps = gr.Gallery(
            label="Eight-channel representation",
            columns=4,
            rows=2,
            height=650,
            object_fit="contain",
            allow_preview=True,
            buttons=["fullscreen"],
            type="numpy",
            visible=False,
        )
        with gr.Row():
            export_formats = gr.CheckboxGroup(
                label="Feature export formats",
                choices=["PNG", "NPY"],
                value=["PNG", "NPY"],
            )
            export_features_button = gr.Button(
                "Prepare selected export", variant="secondary"
            )
        with gr.Row():
            png_download = gr.DownloadButton(
                "Download PNG bundle", value=None, visible=False
            )
            npy_download = gr.DownloadButton(
                "Download NPY array", value=None, visible=False
            )

        gr.Markdown("## Select support points")
        with gr.Row():
            class_names = gr.Textbox(
                label="Local class names",
                value="Class A, Class B",
                info="Enter comma-separated names in the intended local-label order.",
                scale=3,
            )
            configure_button = gr.Button(
                "Configure classes", variant="secondary", scale=1
            )
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                active_class = gr.Radio(
                    label="Active class",
                    choices=[],
                    interactive=True,
                )
                annotation_image = gr.Image(
                    label="Click support points",
                    type="numpy",
                    format="png",
                    interactive=True,
                    buttons=["fullscreen"],
                    value=initial_annotation,
                    height=720,
                    elem_id="annotation-image",
                )
                gr.Markdown(
                    f"Classifier patch: {config.model.classifier_patch_size} × "
                    f"{config.model.classifier_patch_size} px — fixed by the selected model."
                )
            with gr.Column(scale=1, min_width=280):
                patch_preview = gr.Image(
                    label="Selected source patch",
                    type="numpy",
                    interactive=False,
                    height=260,
                )
                support_table = gr.Dataframe(
                    headers=["Class", "Count", "Source coordinates (x, y)"],
                    datatype=["str", "number", "str"],
                    interactive=False,
                    label="Support summary",
                )
                with gr.Row():
                    undo_button = gr.Button("Undo last point")
                    clear_button = gr.Button("Clear active class")

        gr.Markdown("## Fine-tuning")
        with gr.Row():
            epochs = gr.Number(
                label="Fine-tuning epochs",
                value=config.fine_tuning.epochs,
                precision=0,
            )
            learning_rate = gr.Number(
                label="Learning rate", value=config.fine_tuning.learning_rate
            )
        gr.Markdown("## Dense prediction")
        with gr.Row():
            stride = gr.Number(
                label="Prediction stride",
                value=config.prediction.stride,
                precision=0,
            )
            batch_size = gr.Number(
                label="Prediction batch size",
                value=config.prediction.batch_size,
                precision=0,
            )
        run_button = gr.Button(
            "Fine-tune and predict", variant="primary", interactive=False
        )
        with gr.Row():
            prediction_overlay = gr.Image(
                label="Prediction overlay", type="numpy", interactive=False
            )
            confidence_image = gr.Image(
                label="Confidence — maximum class probability",
                type="numpy",
                interactive=False,
            )
            entropy_image = gr.Image(
                label="Predictive entropy — higher means more ambiguous",
                type="numpy",
                interactive=False,
            )
        result_json = gr.JSON(label="Run result")

        def on_model_change(model_id, source, current_catalog, current):
            selected_model = model_capability(current_catalog, model_id)
            choices = _weight_choices(selected_model)
            selected_weight = None
            if source == REGISTERED_WEIGHT_SOURCE:
                selected_weight = str(weight_capability(selected_model, None)["identifier"])
            contract = _model_contract_config(config, current_catalog, model_id)
            updated, annotated = _reset_annotations_for_model(
                current, contract.model.classifier_patch_size
            )
            defaults = dict(selected_model.get("defaults", {}))
            message = f"Selected model `{model_id}`. " + (
                "Class definitions were retained; select new support points."
                if updated.get("class_names")
                else "Configure local classes before selecting support points."
            )
            return (
                gr.update(choices=choices, value=selected_weight),
                _selection_details(selected_model, selected_weight),
                updated,
                annotated,
                gr.update(
                    choices=updated.get("class_names", []),
                    value=(
                        updated["class_names"][0]
                        if updated.get("class_names")
                        else None
                    ),
                ),
                annotation_rows(updated),
                None,
                None,
                None,
                None,
                None,
                message,
                int(defaults.get("symmetry_patch_size", contract.features.symmetry_patch_size)),
                int(defaults.get("epochs", contract.fine_tuning.epochs)),
                float(defaults.get("learning_rate", contract.fine_tuning.learning_rate)),
                int(defaults.get("stride", contract.prediction.stride)),
                int(defaults.get("batch_size", contract.prediction.batch_size)),
                gr.update(interactive=False),
                {},
                gr.update(value=None, visible=False),
                "Model changed; recompute the eight-channel symmetry maps.",
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(interactive=current.get("image") is not None),
            )

        model_selector.change(
            on_model_change,
            inputs=[model_selector, weight_source, catalog_state, state],
            outputs=[
                weight_selector,
                model_details,
                state,
                annotation_image,
                active_class,
                support_table,
                patch_preview,
                prediction_overlay,
                confidence_image,
                entropy_image,
                result_json,
                status,
                symmetry_patch_size,
                epochs,
                learning_rate,
                stride,
                batch_size,
                run_button,
                feature_state,
                feature_maps,
                feature_status,
                png_download,
                npy_download,
                compute_features_button,
            ],
        )

        def on_weight_change(model_id, source, weight_id, current_catalog):
            selected_model = model_capability(current_catalog, model_id)
            displayed_weight = (
                weight_id if source == REGISTERED_WEIGHT_SOURCE else None
            )
            return (
                _selection_details(selected_model, displayed_weight),
                None,
                None,
                None,
                None,
                "Weight selection changed. Existing support points remain available; "
                "run fine-tuning again to update predictions.",
            )

        weight_selector.change(
            on_weight_change,
            inputs=[model_selector, weight_source, weight_selector, catalog_state],
            outputs=[
                model_details,
                prediction_overlay,
                confidence_image,
                entropy_image,
                result_json,
                status,
            ],
        )
        weight_source.change(
            on_weight_change,
            inputs=[model_selector, weight_source, weight_selector, catalog_state],
            outputs=[
                model_details,
                prediction_overlay,
                confidence_image,
                entropy_image,
                result_json,
                status,
            ],
        )

        def on_install(model_id, source, weight_id, current_catalog):
            if source != REGISTERED_WEIGHT_SOURCE:
                raise ValueError("Select Registered weight before installing a package.")
            if not weight_id:
                raise ValueError("Choose a registered weight to install.")
            result = install_registered_weight(
                config,
                model_identifier=model_id,
                weight_identifier=weight_id,
            )
            refreshed = result["capabilities"]
            selected_model = model_capability(refreshed, model_id)
            return (
                refreshed,
                gr.update(choices=_weight_choices(selected_model), value=weight_id),
                _selection_details(selected_model, weight_id),
                f"Installed `{result['requirement']}` and refreshed Provider capabilities.",
            )

        install_button.click(
            on_install,
            inputs=[model_selector, weight_source, weight_selector, catalog_state],
            outputs=[catalog_state, weight_selector, model_details, status],
        )

        def on_image_change(path, current, model_id, current_catalog):
            try:
                if not path:
                    raise ValueError("Choose an input image first.")
                image, record = inspect_input(
                    path, config.features.input_normalization
                )
                contract = _model_contract_config(config, current_catalog, model_id)
                annotated, updated, message = _loaded_image_outputs(
                    image, record, contract, current
                )
            except Exception as error:
                return (
                    None,
                    {},
                    gr.update(choices=[], value=None),
                    [],
                    None,
                    None,
                    None,
                    None,
                    None,
                    _image_shape_text(None),
                    f"Image validation failed: {error}",
                    gr.update(interactive=False),
                    {},
                    gr.update(value=None, visible=False),
                    "Load a valid image before computing symmetry maps.",
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                    gr.update(interactive=False),
                )
            return (
                annotated,
                updated,
                gr.update(
                    choices=updated["class_names"],
                    value=(
                        updated["class_names"][0]
                        if updated["class_names"]
                        else None
                    ),
                ),
                annotation_rows(updated),
                None,
                None,
                None,
                None,
                None,
                _image_shape_text(record),
                message
                + " Selecting another image clears support points and predictions.",
                gr.update(interactive=False),
                {},
                gr.update(value=None, visible=False),
                "Input changed; compute the eight-channel symmetry maps.",
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(interactive=True),
            )

        input_file.change(
            on_image_change,
            inputs=[input_file, state, model_selector, catalog_state],
            outputs=[
                annotation_image,
                state,
                active_class,
                support_table,
                patch_preview,
                prediction_overlay,
                confidence_image,
                entropy_image,
                result_json,
                image_details,
                status,
                run_button,
                feature_state,
                feature_maps,
                feature_status,
                png_download,
                npy_download,
                compute_features_button,
            ],
        )

        def on_feature_setting_change(current):
            message = (
                "Feature settings changed; compute the eight-channel symmetry maps."
                if current.get("image") is not None
                else "Load an image, then compute its eight-channel representation."
            )
            return (
                {},
                gr.update(value=None, visible=False),
                message,
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(interactive=False),
            )

        symmetry_patch_size.change(
            on_feature_setting_change,
            inputs=[state],
            outputs=[
                feature_state,
                feature_maps,
                feature_status,
                png_download,
                npy_download,
                run_button,
            ],
        )
        device.change(
            on_feature_setting_change,
            inputs=[state],
            outputs=[
                feature_state,
                feature_maps,
                feature_status,
                png_download,
                npy_download,
                run_button,
            ],
        )

        def on_compute_features(
            current, model_id, current_catalog, sym_size, run_device
        ):
            contract, options, fingerprint, cache_key = _feature_request(
                config,
                current_catalog,
                current,
                model_identifier=model_id,
                symmetry_patch_size=int(sym_size),
                device=str(run_device),
            )
            features_path, record_path = _feature_cache_paths(
                feature_cache_root, cache_key
            )
            expected_shape = (
                contract.model.input_channels,
                int(current["image_shape"][0]),
                int(current["image_shape"][1]),
            )
            cache_hit = False
            try:
                features, channel_names, feature_record = _load_cached_features(
                    features_path,
                    record_path,
                    expected_shape=expected_shape,
                )
                if feature_record.get("harness_cache_fingerprint") != fingerprint:
                    raise RuntimeError("Cached feature fingerprint mismatch.")
                cache_hit = True
            except (FileNotFoundError, KeyError, OSError, ValueError, RuntimeError):
                result = run_provider_features(
                    contract,
                    np.asarray(current["image"], dtype=np.float32),
                    options=options,
                )
                features = result.features
                channel_names = result.channel_names
                if features.shape != expected_shape:
                    raise RuntimeError(
                        "Provider features do not match the selected model contract."
                    )
                feature_record = {
                    **result.record,
                    "harness_cache_fingerprint": fingerprint,
                }
                features_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    features_path,
                    features=features,
                    channel_names=channel_names,
                )
                record_path.write_text(
                    json.dumps(feature_record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            current_features = {
                "cache_key": cache_key,
                "features_path": str(features_path),
                "record_path": str(record_path),
                "fingerprint": fingerprint,
                "feature_shape": list(features.shape),
            }
            source = "Reused cached" if cache_hit else "Computed"
            message = (
                f"{source} eight-channel symmetry maps with shape "
                f"{tuple(features.shape)}. Changing only classes or support points "
                "will reuse these maps."
            )
            ready = _support_is_ready(current, contract)
            return (
                gr.update(
                    value=feature_gallery(features, channel_names), visible=True
                ),
                current_features,
                message,
                message,
                gr.update(interactive=ready),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
            )

        compute_features_button.click(
            on_compute_features,
            inputs=[
                state,
                model_selector,
                catalog_state,
                symmetry_patch_size,
                device,
            ],
            outputs=[
                feature_maps,
                feature_state,
                status,
                feature_status,
                run_button,
                png_download,
                npy_download,
            ],
        )

        def on_export_features(current_features, formats):
            outputs, message = prepare_feature_exports(current_features, formats)
            png_path = next((path for path in outputs if path.endswith(".zip")), None)
            npy_path = next((path for path in outputs if path.endswith(".npy")), None)
            return (
                gr.update(value=png_path, visible=png_path is not None),
                gr.update(value=npy_path, visible=npy_path is not None),
                message,
            )

        export_features_button.click(
            on_export_features,
            inputs=[feature_state, export_formats],
            outputs=[png_download, npy_download, feature_status],
        )

        def features_are_current(
            current,
            current_features,
            model_id,
            current_catalog,
            sym_size,
            run_device,
        ):
            if not current_features:
                return False
            try:
                _, _, _, expected_key = _feature_request(
                    config,
                    current_catalog,
                    current,
                    model_identifier=model_id,
                    symmetry_patch_size=int(sym_size),
                    device=str(run_device),
                )
            except (KeyError, TypeError, ValueError):
                return False
            return bool(
                current_features.get("cache_key") == expected_key
                and Path(str(current_features.get("features_path", ""))).is_file()
                and Path(str(current_features.get("record_path", ""))).is_file()
            )

        def on_configure_classes(value, current, model_id, current_catalog):
            contract = _model_contract_config(config, current_catalog, model_id)
            return (*_configure_classes(value, current, contract), gr.update(interactive=False))

        configure_button.click(
            on_configure_classes,
            inputs=[class_names, state, model_selector, catalog_state],
            outputs=[
                state,
                active_class,
                annotation_image,
                support_table,
                status,
                run_button,
            ],
        )

        def on_select(
            current,
            active,
            model_id,
            current_catalog,
            current_features,
            sym_size,
            run_device,
            event: gr.SelectData,
        ):
            index = event.index
            if not isinstance(index, (list, tuple)) or len(index) != 2:
                raise ValueError("The image click did not provide a valid pixel coordinate.")
            contract = _model_contract_config(config, current_catalog, model_id)
            source_point = _display_to_source_point(
                (int(index[0]), int(index[1])), tuple(current["image_shape"])
            )
            result = _add_point(
                current, active, source_point, contract
            )
            ready = _support_is_ready(result[0], contract) and features_are_current(
                result[0],
                current_features,
                model_id,
                current_catalog,
                sym_size,
                run_device,
            )
            return (*result, gr.update(interactive=ready))

        annotation_image.select(
            on_select,
            inputs=[
                state,
                active_class,
                model_selector,
                catalog_state,
                feature_state,
                symmetry_patch_size,
                device,
            ],
            outputs=[
                state,
                annotation_image,
                patch_preview,
                support_table,
                status,
                run_button,
            ],
        )

        def on_undo(
            current,
            active,
            model_id,
            current_catalog,
            current_features,
            sym_size,
            run_device,
        ):
            contract = _model_contract_config(config, current_catalog, model_id)
            result = _undo_point(current, active, contract)
            ready = _support_is_ready(result[0], contract) and features_are_current(
                result[0],
                current_features,
                model_id,
                current_catalog,
                sym_size,
                run_device,
            )
            return (*result, gr.update(interactive=ready))

        undo_button.click(
            on_undo,
            inputs=[
                state,
                active_class,
                model_selector,
                catalog_state,
                feature_state,
                symmetry_patch_size,
                device,
            ],
            outputs=[state, annotation_image, support_table, status, run_button],
        )

        def on_clear(current, active, model_id, current_catalog):
            contract = _model_contract_config(config, current_catalog, model_id)
            result = _clear_class(current, active, contract)
            return (*result, gr.update(interactive=False))

        clear_button.click(
            on_clear,
            inputs=[state, active_class, model_selector, catalog_state],
            outputs=[state, annotation_image, support_table, status, run_button],
        )

        def on_run(
            current,
            current_features,
            model_id,
            source,
            weight_id,
            checkpoint,
            current_catalog,
            sym_size,
            epoch_count,
            lr,
            pred_stride,
            pred_batch,
            run_device,
        ):
            if current.get("image") is None or not current.get("class_names"):
                raise ValueError("Load an image and configure classes before running.")
            _, _, _, expected_feature_key = _feature_request(
                config,
                current_catalog,
                current,
                model_identifier=model_id,
                symmetry_patch_size=int(sym_size),
                device=str(run_device),
            )
            if current_features.get("cache_key") != expected_feature_key:
                raise ValueError(
                    "The symmetry maps are missing or stale. Compute them before fine-tuning."
                )
            features_path = Path(str(current_features.get("features_path", "")))
            features_record_path = Path(
                str(current_features.get("record_path", ""))
            )
            if not features_path.is_file() or not features_record_path.is_file():
                raise ValueError(
                    "The cached symmetry maps are unavailable. Compute them again."
                )
            runtime_config = _runtime_config(
                config,
                current_catalog,
                model_identifier=model_id,
                source=source,
                weight_identifier=weight_id,
                checkpoint_path=checkpoint,
            )
            runtime_config = replace(
                runtime_config,
                model=replace(runtime_config.model, device=str(run_device)),
            )
            session = create_annotation_session(
                image_path=current["image_path"],
                image_sha256=current["image_sha256"],
                image_shape=tuple(current["image_shape"]),
                classifier_patch_size=runtime_config.model.classifier_patch_size,
                class_names=current["class_names"],
                points_by_class=current["points"],
                colors=current["colors"],
            )
            result = run_analysis(
                runtime_config,
                image_path=current["image_path"],
                annotation_session=session,
                overrides={
                    "symmetry_patch_size": int(sym_size),
                    "epochs": int(epoch_count),
                    "learning_rate": float(lr),
                    "stride": int(pred_stride),
                    "batch_size": int(pred_batch),
                    "device": str(run_device),
                },
                features_path=features_path,
                features_record_path=features_record_path,
            )
            overlay = np.asarray(Image.open(result["prediction_overlay"]).convert("RGB"))
            confidence = np.asarray(Image.open(result["confidence"]).convert("RGB"))
            entropy = np.asarray(Image.open(result["entropy"]).convert("RGB"))
            message = (
                f"Completed run `{result['run_id']}`. Artifacts were saved to "
                f"`{result['run_directory']}`."
            )
            return overlay, confidence, entropy, result, message

        run_button.click(
            on_run,
            inputs=[
                state,
                feature_state,
                model_selector,
                weight_source,
                weight_selector,
                custom_checkpoint,
                catalog_state,
                symmetry_patch_size,
                epochs,
                learning_rate,
                stride,
                batch_size,
                device,
            ],
            outputs=[
                prediction_overlay,
                confidence_image,
                entropy_image,
                result_json,
                status,
            ],
        )
    app._symmetry_feature_cache_directory = feature_cache_directory
    return app.queue(default_concurrency_limit=1)


def _resolve_server_port(server_name: str, server_port: int | None) -> int | None:
    if server_port is None:
        return None
    if not 0 <= server_port <= 65535:
        raise ValueError("server_port must be between 0 and 65535.")
    if server_port:
        return server_port
    host = "127.0.0.1" if server_name == "localhost" else server_name
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        return int(listener.getsockname()[1])


def _protect_localhost_from_proxies() -> None:
    """Ensure Gradio startup checks connect directly to the local-only server."""
    required = ("127.0.0.1", "localhost")
    existing = []
    for name in ("NO_PROXY", "no_proxy"):
        existing.extend(
            item.strip()
            for item in os.environ.get(name, "").split(",")
            if item.strip()
        )
    unique = []
    lowered: set[str] = set()
    for item in (*existing, *required):
        if item.lower() not in lowered:
            unique.append(item)
            lowered.add(item.lower())
    value = ",".join(unique)
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def launch_ui(
    config: HarnessConfig,
    *,
    server_name: str,
    server_port: int | None,
    inbrowser: bool,
    prepared_input: tuple[np.ndarray, dict[str, Any]] | None = None,
    capabilities: dict[str, Any] | None = None,
    on_ready: Callable[[str], None] | None = None,
) -> None:
    """Launch a local-only interface without public sharing."""
    if server_name not in {"127.0.0.1", "localhost"}:
        raise ValueError("The v1 interface only binds to localhost.")
    _protect_localhost_from_proxies()
    resolved_port = _resolve_server_port(server_name, server_port)
    app = build_app(
        config,
        prepared_input=prepared_input,
        capabilities=capabilities,
    )
    _, local_url, _ = app.launch(
        server_name=server_name,
        server_port=resolved_port,
        inbrowser=inbrowser,
        share=False,
        show_error=True,
        prevent_thread_lock=True,
        quiet=on_ready is not None,
        css=APP_CSS,
    )
    if on_ready is not None:
        on_ready(local_url)
    app.block_thread()
