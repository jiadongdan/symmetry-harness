"""Fine-tuning workspace for the local symmetry interface.

The workspace is organised into four stable, initially folded stages:

``1. Model and image`` / ``2. Review symmetry maps`` / ``3. Select support
points`` / ``4. Fine-tune and review``.

All stage blocks remain present so a front-end layout refresh can never make the
next step unreachable. Readiness evaluation and the primary action converge in
a single refresh step. Presentation-only state (result class colors and the
overlay alpha) is kept in an isolated :class:`PresentationState`; editing it
rerenders pixels from the already-completed ``prediction.npz`` and never
triggers feature computation, fine-tuning, model restoration or dense
prediction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
from pathlib import Path
from queue import Empty, Queue
import tempfile
from threading import Thread
from typing import Any
import zipfile

try:  # pragma: no cover - depends on the optional UI extra
    # Gradio injects event data by resolving this annotation at runtime, so the
    # class must be bound to a module-level name. Import lazily to keep Gradio
    # optional, but bind it here so `get_type_hints` can resolve it.
    from gradio.events import SelectData
except ImportError:  # pragma: no cover - exercised without the UI extra
    SelectData = None  # type: ignore[assignment,misc]

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .analysis import DensePrediction, dense_prediction_from_arrays
from .annotations import create_annotation_session
from .catalog import model_capability, model_weights, provider_models, weight_capability
from .config import HarnessConfig, config_for_model_selection
from .image_io import file_sha256, inspect_input
from .numeric import require_whole_number
from .provider import install_registered_weight, run_provider_features
from .result_exports import (
    DEFAULT_OVERLAY_ALPHA,
    RESULT_EXPORT_NAMES,
    PresentationError,
    normalize_palette,
    prepare_download_copy,
    render_result_variants,
    validate_alpha,
    write_png,
)
from .ui_annotation import (
    add_point_state,
    annotation_display_shape as _annotation_display_shape,
    annotation_rows,
    annotation_state,
    array_sha256,
    clear_class_state,
    configure_class_state,
    copy_state as _copy_state,
    display_to_source_point as _display_to_source_point,
    image_shape_text as _image_shape_text,
    render_annotations,
    source_patch_preview,
    state_patch_size as _state_classifier_patch_size,
    undo_point_state,
)
from .ui_readiness import (
    evaluate_fine_tune_readiness,
    has_blocker,
    readiness_panel_html,
)
from .ui_shared import (
    INVALID_SUPPORT_POINT_MESSAGE,
    class_statistics_html,
    progress_bar_html,
)
from .workflow import build_run_options, run_analysis


# ---------------------------------------------------------------------------
# Transient presentation state (protocol sections 3.4 / 3.5 / 7.4)
# ---------------------------------------------------------------------------
@dataclass
class PresentationState:
    """Isolated, presentation-only state for one workspace.

    Nothing in here is ever persisted to ``.symmodel`` files, annotation
    sessions, run/prediction records, reports or run-archive metadata. It only
    drives the on-screen overlay and the on-demand PNG exports.
    """

    class_names: list[str] = field(default_factory=list)
    display_colors: list[str] = field(default_factory=list)
    alpha: float = DEFAULT_OVERLAY_ALPHA
    selected_result_item: str | None = None
    numerical_result_reference: dict[str, str] = field(default_factory=dict)

    @classmethod
    def defaults(
        cls,
        class_names: Sequence[str] = (),
        colors: Sequence[str] = (),
        *,
        alpha: float = DEFAULT_OVERLAY_ALPHA,
        result_reference: Mapping[str, str] | None = None,
    ) -> "PresentationState":
        """Build a fresh state from system/package default names and colors.

        ``result_reference`` records the persisted artifacts of a completed run so
        the rerender path can reload them without retraining. Without it the
        presentation controls stay disabled, because there is nothing to redraw.
        """
        return cls(
            class_names=[str(name) for name in class_names],
            display_colors=[str(color) for color in colors],
            alpha=float(alpha),
            numerical_result_reference=dict(result_reference or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_names": list(self.class_names),
            "display_colors": list(self.display_colors),
            "alpha": float(self.alpha),
            "selected_result_item": self.selected_result_item,
            "numerical_result_reference": dict(self.numerical_result_reference),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "PresentationState":
        payload = payload or {}
        return cls(
            class_names=[str(name) for name in payload.get("class_names", [])],
            display_colors=[str(color) for color in payload.get("display_colors", [])],
            alpha=float(payload.get("alpha", DEFAULT_OVERLAY_ALPHA)),
            selected_result_item=payload.get("selected_result_item"),
            numerical_result_reference=dict(
                payload.get("numerical_result_reference", {})
            ),
        )


# ---------------------------------------------------------------------------
# Result statistics and presentation rendering (no Provider calls)
# ---------------------------------------------------------------------------
def _fine_tune_class_statistics(
    prediction_grid: np.ndarray, class_count: int
) -> dict[str, Any]:
    """Return per-class predicted-grid counts and fractions for one result.

    Counts are computed over the predicted sampling grid. The numerical layer
    owns the grid; the Harness only presents it.
    """
    grid = np.asarray(prediction_grid)
    total = int(grid.size)
    entries = []
    for index in range(int(class_count)):
        count = int(np.count_nonzero(grid == index))
        entries.append(
            {
                "index": index,
                "count": count,
                "fraction": (count / total) if total else 0.0,
            }
        )
    return {"grid_cell_count": total, "classes": entries}


def load_fine_tune_result(
    reference: Mapping[str, str],
) -> tuple[np.ndarray, DensePrediction]:
    """Load the normalized source image and the dense prediction from disk.

    Only already-persisted artifacts are read; no numerical job is invoked.
    """
    image = np.asarray(
        np.load(Path(str(reference["input_array"])), allow_pickle=False),
        dtype=np.float32,
    )
    with np.load(Path(str(reference["prediction"])), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return image, dense_prediction_from_arrays(arrays)


def render_fine_tune_variants(
    reference: Mapping[str, str],
    class_names: Sequence[str],
    colors: Sequence[str],
    alpha: float,
) -> dict[str, np.ndarray]:
    """Render every presentation PNG variant from the completed numerical result."""
    image, prediction = load_fine_tune_result(reference)
    return render_result_variants(
        image,
        prediction,
        class_names=class_names,
        colors=colors,
        alpha=alpha,
        stride=int(reference["stride"]),
    )


def prepare_presentation_pngs(
    reference: Mapping[str, str],
    class_names: Sequence[str],
    colors: Sequence[str],
    alpha: float,
    names: Sequence[str],
) -> list[str]:
    """Write the selected presentation PNGs into a temporary download directory.

    The durable run artifacts (canonical overlay/confidence/entropy PNGs) are
    never overwritten or mutated.
    """
    variants = render_fine_tune_variants(reference, class_names, colors, alpha)
    selected = [name for name in names if name in variants]
    download_directory = Path(tempfile.gettempdir()) / "symmetry-harness-downloads"
    download_directory.mkdir(parents=True, exist_ok=True)
    run_id = str(reference.get("run_id", "result"))
    outputs: list[str] = []
    for name in selected:
        target = download_directory / f"{run_id}-{name}.png"
        write_png(variants[name], target)
        outputs.append(str(target))
    return outputs


def prepare_model_download(reference: Mapping[str, str]) -> str:
    """Serve a browser copy of the durable ``.symmodel`` (identical bytes)."""
    source = Path(str(reference["fine_tuned_model"]))
    return str(
        prepare_download_copy(
            source, run_id=str(reference["run_id"]), filename=source.name
        )
    )


def prepare_run_zip_download(reference: Mapping[str, str]) -> str:
    """Serve a browser copy of the durable complete-run ZIP archive."""
    source = Path(str(reference["full_run_zip"]))
    return str(
        prepare_download_copy(
            source, run_id=str(reference["run_id"]), filename=source.name
        )
    )


def parse_presentation_colors(text: str, class_count: int) -> list[str]:
    """Parse comma-separated ``#RRGGBB`` colors, one per task class."""
    colors = [item.strip() for item in str(text or "").split(",")]
    return list(normalize_palette(colors, class_count=int(class_count)))


# ---------------------------------------------------------------------------
# Checkpoint presentation
# ---------------------------------------------------------------------------
def _checkpoint_display(
    capability: Mapping[str, Any], checkpoint_path: str | None
) -> tuple[bool, str, bool]:
    """Return ``(ready, detail, install_visible)`` for the selected checkpoint.

    A custom checkpoint hides the install action; a registered checkpoint that
    is already installed hides it too. The install action is only shown when an
    explicit installation is actually required.
    """
    if checkpoint_path:
        path = Path(str(checkpoint_path))
        if path.is_file():
            return True, f"Custom checkpoint `{path.name}` — ready.", False
        return False, f"Custom checkpoint `{path.name}` is missing.", False
    try:
        weight = weight_capability(dict(capability or {}), None)
    except (TypeError, ValueError) as error:
        return False, f"No registered checkpoint available: {error}", False
    identifier = str(weight.get("identifier", ""))
    status = str(weight.get("status", "unknown"))
    if status == "installed":
        return True, f"Registered checkpoint `{identifier}` — installed.", False
    return (
        False,
        f"Registered checkpoint `{identifier}` — {status}. Install it explicitly "
        "before fine-tuning.",
        True,
    )


# ---------------------------------------------------------------------------
# Readiness snapshot / stage visibility / checklist HTML
# ---------------------------------------------------------------------------
def build_fine_tune_snapshot(
    config: HarnessConfig,
    capabilities: Mapping[str, Any],
    *,
    state: Mapping[str, Any] | None,
    feature_state: Mapping[str, Any] | None,
    model_identifier: str,
    checkpoint_path: str | None,
    symmetry_patch_size: Any,
    epochs: Any,
    learning_rate: Any,
    stride: Any,
    batch_size: Any,
    device: Any,
) -> dict[str, Any]:
    """Assemble a plain readiness snapshot from UI values (no Gradio needed)."""
    state = state or {}
    feature_state = feature_state or {}
    try:
        capability = model_capability(dict(capabilities), str(model_identifier))
        model_supported = bool(capability.get("available", True))
        model_name = str(capability.get("display_name", model_identifier))
    except (TypeError, ValueError):
        capability = {}
        model_supported = False
        model_name = str(model_identifier or "the selected model")

    checkpoint_ready, checkpoint_detail, _ = _checkpoint_display(
        capability, checkpoint_path
    )

    image = state.get("image")
    image_valid = (
        image is not None
        and bool(state.get("image_path"))
        and bool(state.get("image_sha256"))
    )
    image_name = (
        Path(str(state.get("image_path"))).name if state.get("image_path") else "the source image"
    )

    features_current = False
    features_detail = "Symmetry maps are missing or stale."
    if image_valid and feature_state:
        try:
            _, _, _, expected_key = _feature_request(
                config,
                dict(capabilities),
                dict(state),
                model_identifier=str(model_identifier),
                symmetry_patch_size=symmetry_patch_size,
                device=str(device),
            )
            if (
                feature_state.get("cache_key") == expected_key
                and Path(str(feature_state.get("features_path", ""))).is_file()
                and Path(str(feature_state.get("record_path", ""))).is_file()
            ):
                features_current = True
                features_detail = (
                    "Symmetry maps match the current image, model and settings."
                )
        except (KeyError, TypeError, ValueError):
            features_current = False

    names = [str(name) for name in state.get("class_names", [])]
    counts = [len(group) for group in state.get("points", [])]

    minimum = int(
        capability.get(
            "minimum_shots_per_class", config.fine_tuning.minimum_shots_per_class
        )
    )
    recommended = int(
        capability.get(
            "recommended_shots_per_class", config.fine_tuning.recommended_shots_per_class
        )
    )
    maximum = int(
        capability.get(
            "maximum_shots_per_class", config.fine_tuning.maximum_shots_per_class
        )
    )

    return {
        "model_supported": model_supported,
        "model_name": model_name,
        "checkpoint_ready": checkpoint_ready,
        "checkpoint_detail": checkpoint_detail,
        "image_valid": image_valid,
        "image_name": image_name,
        "features_current": features_current,
        "features_detail": features_detail,
        "class_names": names,
        "class_counts": counts,
        "minimum_shots": minimum,
        "recommended_shots": recommended,
        "maximum_shots": maximum,
        "symmetry_patch_size": symmetry_patch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "stride": stride,
        "batch_size": batch_size,
        "device": device,
        "stale_state": False,
        "stale_detail": "",
        "has_result": False,
    }


def _stage_visibility(snapshot: Mapping[str, Any]) -> dict[str, bool]:
    """Keep every workflow block present regardless of current readiness.

    Dynamically mounting parent Accordions proved unreliable in real Gradio
    sessions: a newly eligible stage could remain absent until another UI
    interaction forced a layout refresh. Readiness still controls whether the
    numerical action can run, while the four stable, initially folded blocks
    keep every next step reachable.
    """
    del snapshot
    return {
        "symmetry": True,
        "support": True,
        "run": True,
    }


def _current_stage(snapshot: Mapping[str, Any]) -> str:
    """Return the first revealed-but-not-yet-complete stage for a snapshot.

    Stage completion is defined by the frozen protocol (section 3.10):

    1. ``model_and_image`` is complete once the source image is valid;
    2. ``symmetry`` is complete once the symmetry maps are current;
    3. ``support`` is complete once readiness reports no blocker (a run is
       possible);
    4. ``run`` is complete once a numerical result exists (``has_result``);
       before that it is the current stage, and once complete the current stage
       intentionally remains ``run`` so the result stays in view.
    """
    items = evaluate_fine_tune_readiness(snapshot)
    if not snapshot.get("image_valid"):
        return "model_and_image"
    if not snapshot.get("features_current"):
        return "symmetry"
    if has_blocker(items):
        return "support"
    return "run"


# ---------------------------------------------------------------------------
# Shared annotation wrappers (kept stable for existing tests / re-exports)
# ---------------------------------------------------------------------------
def _loaded_image_outputs(
    image: np.ndarray,
    record: dict[str, Any],
    config: HarnessConfig,
    previous_state: dict[str, Any] | None = None,
):
    state, status = annotation_state(
        image,
        record,
        classifier_patch_size=config.model.classifier_patch_size,
        previous_state=previous_state,
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


def _configure_classes(
    value: str,
    state: dict[str, Any],
    config: HarnessConfig,
    *,
    classifier_patch_size: int | None = None,
):
    """Thin fine-tuning wrapper over the shared annotation state machine."""
    import gradio as gr

    patch_size = int(
        config.model.classifier_patch_size
        if classifier_patch_size is None
        else classifier_patch_size
    )
    if patch_size <= 0:
        raise ValueError("The model input patch size must be positive.")
    updated, patch_size = configure_class_state(
        value, state, classifier_patch_size=patch_size
    )
    names = list(updated["class_names"])
    status = (
        f"Configured {len(names)} classes. Select an active class and click at least "
        f"{config.fine_tuning.minimum_shots_per_class} support points per class; "
        f"{config.fine_tuning.recommended_shots_per_class} are recommended."
    )
    return (
        updated,
        gr.update(choices=names, value=names[0]),
        render_annotations(updated, patch_size),
        annotation_rows(updated),
        status,
    )


def _add_point(
    state: dict[str, Any], active_class: str | None, point: tuple[int, int], config: HarnessConfig
):
    """Thin fine-tuning wrapper over the shared annotation state machine."""
    updated, patch_size, status, invalid = add_point_state(
        state,
        active_class,
        point,
        fallback_patch_size=config.model.classifier_patch_size,
        maximum_points=config.fine_tuning.maximum_shots_per_class,
    )
    if invalid:
        return (
            updated,
            render_annotations(updated, patch_size),
            None,
            annotation_rows(updated),
            status,
        )
    return (
        updated,
        render_annotations(updated, patch_size),
        source_patch_preview(updated, (int(point[0]), int(point[1])), patch_size),
        annotation_rows(updated),
        status,
    )


def _undo_point(state: dict[str, Any], active_class: str | None, config: HarnessConfig):
    """Thin fine-tuning wrapper over the shared annotation state machine."""
    updated, patch_size, _removed, status = undo_point_state(
        state,
        active_class,
        fallback_patch_size=config.model.classifier_patch_size,
    )
    return (
        updated,
        render_annotations(updated, patch_size),
        annotation_rows(updated),
        status,
    )


def _clear_class(state: dict[str, Any], active_class: str | None, config: HarnessConfig):
    """Thin fine-tuning wrapper over the shared annotation state machine."""
    updated, patch_size, status = clear_class_state(
        state,
        active_class,
        fallback_patch_size=config.model.classifier_patch_size,
    )
    return (
        updated,
        render_annotations(updated, patch_size),
        annotation_rows(updated),
        status,
    )


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------
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
            "device": config.model.device,
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


def _default_weight_identifier(capability: Mapping[str, Any]) -> str | None:
    try:
        return str(weight_capability(dict(capability), None)["identifier"])
    except (TypeError, ValueError):
        return None


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
    weight_text = "Registered default" if weight_identifier is None else None
    if weight_identifier is not None:
        selected_weight = weight_capability(capability, weight_identifier)
        weight_text = (
            f"`{selected_weight['identifier']}` — "
            f"{selected_weight.get('status', 'unknown')}"
        )
    return (
        f"**Model:** `{capability['identifier']}`  \n"
        f"**Checkpoint:** {weight_text}  \n"
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
    updated["classifier_patch_size"] = int(patch_size)
    updated["show_valid_region"] = False
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
    checkpoint_path: str | None,
) -> HarnessConfig:
    """Resolve a runtime config from the selected model and checkpoint.

    The selected model's sole registered checkpoint is used unless the user
    supplies a compatible custom checkpoint in the Advanced area.
    """
    capability = model_capability(capabilities, model_identifier)
    if checkpoint_path:
        resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
        return config_for_model_selection(
            config,
            capability,
            weight_identifier=None,
            checkpoint_path=resolved_checkpoint,
            checkpoint_sha256=file_sha256(resolved_checkpoint),
        )
    selected_weight = weight_capability(capability, None)
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
    resolved_patch_size = require_whole_number(
        symmetry_patch_size, "The symmetry patch size", minimum=1
    )
    contract = _model_contract_config(config, capabilities, model_identifier)
    options = build_run_options(
        contract,
        {
            "symmetry_patch_size": resolved_patch_size,
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
    provider_features = record.get("features")
    if not isinstance(provider_features, dict):
        raise RuntimeError("Cached feature provenance is missing; compute again.")
    recorded_image_sha256 = str(
        provider_features.get("normalized_image_sha256", "")
    ).lower()
    recorded_feature_sha256 = str(provider_features.get("feature_sha256", "")).lower()
    if len(recorded_image_sha256) != 64 or len(recorded_feature_sha256) != 64:
        raise RuntimeError("Cached feature provenance is missing; compute again.")
    if recorded_feature_sha256 != array_sha256(features):
        raise RuntimeError("Cached feature contents do not match their checksum.")
    fingerprint = record.get("harness_cache_fingerprint")
    if isinstance(fingerprint, dict):
        expected_image_sha256 = str(
            fingerprint.get("normalized_image_sha256", "")
        ).lower()
        if expected_image_sha256 and recorded_image_sha256 != expected_image_sha256:
            raise RuntimeError("Cached features belong to a different normalized image.")
    return features, channel_names, record


_SIGNED_FEATURE_CHANNELS = {
    "reflection_sin_2theta",
    "reflection_cos_2theta",
}


def _feature_display_range(name: str) -> tuple[float, float]:
    """Return the fixed value range used for a feature-channel preview."""
    return (-1.0, 1.0) if name in _SIGNED_FEATURE_CHANNELS else (0.0, 1.0)


def _feature_preview(channel: np.ndarray, name: str) -> np.ndarray:
    """Render a feature channel with its fixed cross-image value range."""
    values = np.asarray(channel, dtype=np.float32)
    minimum, maximum = _feature_display_range(name)
    normalized = (np.clip(values, minimum, maximum) - minimum) / (
        maximum - minimum
    )
    return np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)


def feature_gallery(
    features: np.ndarray, channel_names: np.ndarray
) -> list[tuple[np.ndarray, str]]:
    """Build the fixed two-by-four display payload for the eight channels."""
    items: list[tuple[np.ndarray, str]] = []
    for channel, raw_name in zip(features, channel_names):
        name = str(raw_name)
        minimum, maximum = _feature_display_range(name)
        caption = f"{name}  [color range {minimum:g} to {maximum:g}]"
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


# ---------------------------------------------------------------------------
# Workspace construction
# ---------------------------------------------------------------------------
def build_fine_tune_workspace(
    config: HarnessConfig,
    *,
    prepared_input: tuple[np.ndarray, dict[str, Any]] | None = None,
    capabilities: dict[str, Any] | None = None,
):
    """Construct the fine-tuning workspace inside the caller's Blocks."""
    import gradio as gr

    catalog = capabilities or _configured_capabilities(config)
    initial_model = model_capability(catalog, config.model.identifier)
    initial_details = _selection_details(
        initial_model, _default_weight_identifier(initial_model)
    )
    initial_custom = (
        str(config.model.checkpoint_path)
        if config.model.checkpoint_path is not None
        and Path(config.model.checkpoint_path).is_file()
        else None
    )
    _, initial_checkpoint_status, initial_install_visible = _checkpoint_display(
        initial_model, initial_custom
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

    # Initial readiness snapshot / stage visibility (no Gradio needed).
    initial_snapshot = build_fine_tune_snapshot(
        config,
        catalog,
        state=initial_state,
        feature_state={},
        model_identifier=str(config.model.identifier),
        checkpoint_path=initial_custom,
        symmetry_patch_size=config.features.symmetry_patch_size,
        epochs=config.fine_tuning.epochs,
        learning_rate=config.fine_tuning.learning_rate,
        stride=config.prediction.stride,
        batch_size=config.prediction.batch_size,
        device=config.model.device,
    )
    initial_items = evaluate_fine_tune_readiness(initial_snapshot)
    initial_visibility = _stage_visibility(initial_snapshot)

    state = gr.State(initial_state)
    catalog_state = gr.State(catalog)
    feature_state = gr.State({})
    presentation_state = gr.State(PresentationState())
    # Remembers the last current stage that was announced so open/closed state
    # is only pushed when the current stage actually changes (manual expansion
    # between refreshes is preserved).
    last_stage = gr.State(None)

    def _cleared_result_updates() -> tuple[Any, ...]:
        """Every result-derived output, hidden and cleared (no stale downloads)."""
        return (
            None,  # prediction_overlay
            None,  # confidence_image
            None,  # entropy_image
            "",  # class_statistics
            "",  # run_path
            "",  # copy_path
            gr.update(value=None, visible=False),  # model_download
            gr.update(value=None, visible=False),  # run_zip_download
            "",  # color_text
            DEFAULT_OVERLAY_ALPHA,  # alpha_slider
            gr.update(value=None, visible=False),  # result_exports_files
            None,  # result_json
            gr.update(visible=False),  # results_group
            PresentationState(),  # presentation_state
        )

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(
                "Work through the four stages below. Stage 4 contains the readiness "
                "checklist and fine-tuning action, so the action stays next to the "
                "settings and results it controls."
            )
            status = gr.Markdown(initial_status)

            # ----- Stage 1: model and image --------------------------------
            with gr.Accordion(
                "1. Model and image", open=True
            ) as model_image_stage:
                model_selector = gr.Dropdown(
                    label="Model",
                    choices=_model_choices(catalog),
                    value=config.model.identifier,
                    interactive=True,
                )
                checkpoint_status = gr.Markdown(initial_checkpoint_status)
                install_button = gr.Button(
                    "Install selected weight",
                    variant="secondary",
                    visible=initial_install_visible,
                )
                model_details = gr.Markdown(initial_details)
                with gr.Accordion(
                    "Advanced: custom checkpoint", open=False
                ):
                    gr.Markdown(
                        "A custom checkpoint must be compatible with the selected "
                        "registered model. A checkpoint never defines new model "
                        "code or a new feature pipeline."
                    )
                    custom_checkpoint = gr.File(
                        label="Custom checkpoint",
                        type="filepath",
                        value=initial_custom,
                    )
                    custom_checkpoint_sha = gr.Markdown(
                        "No custom checkpoint selected."
                    )
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
                )
                image_details = gr.Markdown(
                    _image_shape_text(initial_input_record)
                )

            # ----- Stage 2: symmetry maps ----------------------------------
            with gr.Accordion(
                "2. Review symmetry maps",
                open=False,
                visible=initial_visibility["symmetry"],
            ) as symmetry_stage:
                # The symmetry patch size is the one setting every STEM image
                # forces you to revisit: resolution differs from image to image,
                # so this block is expanded by default and sits *above* the
                # compute action instead of hiding in a collapsed "Advanced"
                # section at the bottom.
                with gr.Accordion("Symmetry map settings", open=True):
                    symmetry_patch_size = gr.Number(
                        label="Symmetry patch size",
                        value=config.features.symmetry_patch_size,
                        precision=0,
                        info=(
                            "Odd local-neighborhood size used to calculate the "
                            "symmetry maps. Different STEM images have different "
                            "resolutions, so this value usually needs adjusting."
                        ),
                    )
                    device = gr.Dropdown(
                        label="Device",
                        choices=["auto", "cuda", "cpu"],
                        value=config.model.device,
                        info=(
                            "Used for feature extraction, fine-tuning, and "
                            "prediction."
                        ),
                    )
                with gr.Row():
                    compute_features_button = gr.Button(
                        "Compute / update symmetry maps",
                        variant="primary",
                        interactive=prepared_input is not None,
                    )
                    feature_status = gr.Markdown(
                        (
                            "Ready to compute symmetry maps."
                            if prepared_input is not None
                            else "Load an image to compute symmetry maps."
                        ),
                        elem_id="symmetry-compute-status",
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
                        "Prepare selected export",
                        variant="secondary",
                        interactive=False,
                    )
                with gr.Row():
                    png_download = gr.DownloadButton(
                        "Download PNG bundle", value=None, visible=False
                    )
                    npy_download = gr.DownloadButton(
                        "Download NPY array", value=None, visible=False
                    )

            # ----- Stage 3: support points ---------------------------------
            with gr.Accordion(
                "3. Select support points",
                open=False,
                visible=initial_visibility["support"],
            ) as support_stage:
                with gr.Row():
                    class_names = gr.Textbox(
                        label="Local class names",
                        value="Class A, Class B",
                        info=(
                            "Enter comma-separated names in the intended local-label "
                            "order."
                        ),
                        scale=2,
                    )
                    classifier_patch_size = gr.Number(
                        label="Model input patch size",
                        value=int(
                            initial_model.get(
                                "classifier_patch_size",
                                config.model.classifier_patch_size,
                            )
                        ),
                        precision=0,
                        interactive=False,
                        info="Fixed by the selected model.",
                        scale=1,
                    )
                    configure_button = gr.Button(
                        "Configure classes and patch size",
                        variant="secondary",
                        scale=1,
                        interactive=bool(initial_snapshot["features_current"]),
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
                    with gr.Column(scale=1, min_width=260):
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

            # ----- Stage 4: fine-tune and review ---------------------------
            with gr.Accordion(
                "4. Fine-tune and review",
                open=False,
                visible=initial_visibility["run"],
            ) as run_stage:
                with gr.Accordion(
                    "Advanced: training and prediction settings", open=False
                ):
                    with gr.Row():
                        epochs = gr.Number(
                            label="Fine-tuning epochs",
                            value=config.fine_tuning.epochs,
                            precision=0,
                        )
                        learning_rate = gr.Number(
                            label="Learning rate",
                            value=config.fine_tuning.learning_rate,
                        )
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
                gr.Markdown("### Readiness")
                readiness_html = gr.HTML(
                    readiness_panel_html(initial_items, elem_id=None)
                )
                run_button = gr.Button(
                    "Fine-tune and predict",
                    variant="primary",
                    interactive=not has_blocker(initial_items),
                )
                fine_tuning_progress = gr.HTML(
                    progress_bar_html("Fine-tuning", 0, 1, "Waiting to start.")
                )
                prediction_progress = gr.HTML(
                    progress_bar_html("Prediction", 0, 1, "Waiting for fine-tuning.")
                )

                with gr.Column(visible=False) as results_group:
                    gr.Markdown("### Prediction results")
                    with gr.Row():
                        prediction_overlay = gr.Image(
                            label="Prediction overlay (class legend below)",
                            type="numpy",
                            interactive=False,
                            height=480,
                        )
                        confidence_image = gr.Image(
                            label="Confidence — maximum class probability (0 to 1)",
                            type="numpy",
                            interactive=False,
                            height=480,
                        )
                        entropy_image = gr.Image(
                            label="Predictive entropy — higher means more ambiguous (0 to ln N)",
                            type="numpy",
                            interactive=False,
                            height=480,
                        )
                    # The colour and alpha controls sit directly beneath the maps
                    # and above every result artefact, so a rerender happens next
                    # to the images it changes instead of further down the page.
                    gr.Markdown(
                        "### Presentation (rerenders only — never retrains or "
                        "repredicts)"
                    )
                    color_text = gr.Textbox(
                        label="Result class colors (#RRGGBB, comma-separated)",
                        interactive=True,
                    )
                    alpha_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=DEFAULT_OVERLAY_ALPHA,
                        step=0.01,
                        label="Overlay alpha",
                    )
                    update_presentation_button = gr.Button(
                        "Update colors and alpha (rerender)", variant="secondary"
                    )
                    gr.Markdown("### Per-class results")
                    class_statistics = gr.HTML()
                    run_path = gr.Markdown()
                    copy_path = gr.Textbox(
                        label="Run directory",
                        interactive=False,
                        buttons=["copy"],
                    )
                    with gr.Row():
                        model_download = gr.DownloadButton(
                            "Download model (.symmodel)",
                            value=None,
                            variant="primary",
                            visible=False,
                        )
                        run_zip_download = gr.DownloadButton(
                            "Download complete run (.zip)",
                            value=None,
                            visible=False,
                        )
                    export_names = gr.CheckboxGroup(
                        label="PNG export variants",
                        choices=list(RESULT_EXPORT_NAMES),
                        value=list(RESULT_EXPORT_NAMES),
                    )
                    prepare_exports_button = gr.Button(
                        "Prepare PNG exports", variant="secondary"
                    )
                    result_exports_files = gr.File(
                        label="Prepared PNG exports",
                        file_count="multiple",
                        visible=False,
                    )
                    with gr.Accordion("Technical details", open=False):
                        result_json = gr.JSON(label="Run result")

    RESULT_OUTPUTS = [
        prediction_overlay,
        confidence_image,
        entropy_image,
        class_statistics,
        run_path,
        copy_path,
        model_download,
        run_zip_download,
        color_text,
        alpha_slider,
        result_exports_files,
        result_json,
        results_group,
        presentation_state,
    ]
    REFRESH_INPUTS = [
        state,
        feature_state,
        model_selector,
        custom_checkpoint,
        catalog_state,
        symmetry_patch_size,
        epochs,
        learning_rate,
        stride,
        batch_size,
        device,
        presentation_state,
        last_stage,
    ]
    REFRESH_OUTPUTS = [
        readiness_html,
        run_button,
        model_image_stage,
        symmetry_stage,
        support_stage,
        run_stage,
        last_stage,
        configure_button,
        export_features_button,
    ]

    # ------------------------------------------------------------------
    # Centralized readiness / stage-open refresh
    # ------------------------------------------------------------------
    def _refresh_updates(snapshot, announced_stage):
        """Return the shared readiness + stable-stage updates.

        ``snapshot["has_result"]`` must already be set. This is shared by
        ``on_refresh`` and by the symmetry-map computation, because chaining a
        separate refresh after that computation reads the previous
        ``feature_state`` and would only reveal stage 3 on a second click.
        """
        items = evaluate_fine_tune_readiness(snapshot)
        visibility = _stage_visibility(snapshot)
        current_stage = _current_stage(snapshot)
        stage_changed = current_stage != announced_stage
        # Parent stages are always visible. The current-stage assignment is kept
        # only for best-effort automatic opening and never controls reachability.
        if current_stage in visibility:
            visibility[current_stage] = True

        def _stage_update(visible: bool, is_current: bool):
            # Stable disclosure is additive: a stage is never collapsed by a
            # refresh. ``open`` is pushed as ``True`` only for the stage the
            # workflow has just advanced to, so a manual collapse of the current
            # stage survives repeated refreshes, and -- crucially -- running a
            # command inside a stage no longer closes the block that was used.
            if visible and is_current and stage_changed:
                return gr.update(visible=visible, open=True)
            return gr.update(visible=visible)

        return (
            gr.update(value=readiness_panel_html(items, elem_id=None)),
            gr.update(interactive=not has_blocker(items)),
            _stage_update(True, current_stage == "model_and_image"),
            _stage_update(visibility["symmetry"], current_stage == "symmetry"),
            _stage_update(visibility["support"], current_stage == "support"),
            _stage_update(visibility["run"], current_stage == "run"),
            current_stage,
            gr.update(interactive=bool(snapshot.get("features_current"))),
            gr.update(interactive=bool(snapshot.get("features_current"))),
        )

    def _snapshot_from(
        current,
        current_features,
        model_id,
        checkpoint_path,
        current_catalog,
        sym_size,
        epoch_count,
        lr,
        pred_stride,
        pred_batch,
        run_device,
    ):
        """Build a readiness snapshot from the shared widget inputs."""
        return build_fine_tune_snapshot(
            config,
            current_catalog,
            state=current,
            feature_state=current_features,
            model_identifier=model_id,
            checkpoint_path=checkpoint_path,
            symmetry_patch_size=sym_size,
            epochs=epoch_count,
            learning_rate=lr,
            stride=pred_stride,
            batch_size=pred_batch,
            device=run_device,
        )

    def on_refresh(
        current,
        current_features,
        model_id,
        checkpoint_path,
        current_catalog,
        sym_size,
        epoch_count,
        lr,
        pred_stride,
        pred_batch,
        run_device,
        current_presentation,
        announced_stage,
    ):
        snapshot = _snapshot_from(
            current,
            current_features,
            model_id,
            checkpoint_path,
            current_catalog,
            sym_size,
            epoch_count,
            lr,
            pred_stride,
            pred_batch,
            run_device,
        )
        snapshot["has_result"] = bool(
            current_presentation is not None
            and getattr(current_presentation, "numerical_result_reference", None)
        )
        return _refresh_updates(snapshot, announced_stage)

    # Training/prediction settings are upstream inputs to a numerical result.
    # A user edit must therefore clear every old result/download and refresh
    # readiness from the values that are currently visible in the controls.
    def on_run_setting_change(
        current,
        current_features,
        model_id,
        checkpoint_path,
        current_catalog,
        sym_size,
        epoch_count,
        lr,
        pred_stride,
        pred_batch,
        run_device,
        announced_stage,
    ):
        snapshot = _snapshot_from(
            current,
            current_features,
            model_id,
            checkpoint_path,
            current_catalog,
            sym_size,
            epoch_count,
            lr,
            pred_stride,
            pred_batch,
            run_device,
        )
        snapshot["has_result"] = False
        return (
            *_cleared_result_updates(),
            "Run settings changed; any previous result and downloads were cleared.",
            *_refresh_updates(snapshot, announced_stage),
        )

    run_setting_inputs = [
        state,
        feature_state,
        model_selector,
        custom_checkpoint,
        catalog_state,
        symmetry_patch_size,
        epochs,
        learning_rate,
        stride,
        batch_size,
        device,
        last_stage,
    ]
    run_setting_outputs = [*RESULT_OUTPUTS, status, *REFRESH_OUTPUTS]
    for control in (epochs, learning_rate, stride, batch_size):
        control.input(
            on_run_setting_change,
            inputs=run_setting_inputs,
            outputs=run_setting_outputs,
        )


    # ------------------------------------------------------------------
    # Stage 1 callbacks
    # ------------------------------------------------------------------
    def on_model_change(model_id, current_catalog, current, checkpoint_path):
        try:
            selected_model = model_capability(current_catalog, model_id)
        except (TypeError, ValueError) as error:
            failure = f"Model selection failed: {error}"
            return (
                "",  # model_details
                failure,  # checkpoint_status
                gr.update(visible=False),  # install_button
                gr.update(value=None),  # custom_checkpoint
                "No custom checkpoint selected.",  # custom_checkpoint_sha
                {},  # state
                None,  # annotation_image
                gr.update(choices=[], value=None),  # active_class
                gr.update(value="Class A, Class B"),  # class_names
                config.model.classifier_patch_size,  # classifier_patch_size
                [],  # support_table
                None,  # patch_preview
                _image_shape_text(None),  # image_details
                {},  # feature_state
                gr.update(value=None, visible=False),  # feature_maps
                "Load a valid model to continue.",  # feature_status
                gr.update(value=None, visible=False),  # png_download
                gr.update(value=None, visible=False),  # npy_download
                gr.update(interactive=False),  # compute_features_button
                config.features.symmetry_patch_size,  # symmetry_patch_size
                config.model.device,  # device
                config.fine_tuning.epochs,  # epochs
                config.fine_tuning.learning_rate,  # learning_rate
                config.prediction.stride,  # stride
                config.prediction.batch_size,  # batch_size
                *_cleared_result_updates(),
                failure,  # status
            )

        contract = _model_contract_config(config, current_catalog, model_id)
        defaults = dict(selected_model.get("defaults", {}))

        # Only the source path is retained; every model-dependent state is
        # rebuilt from the new model contract (protocol section 3.3).
        updated_state: dict[str, Any] = {}
        annotated = None
        image_details_text = _image_shape_text(None)
        status_text = f"Selected model `{model_id}`. Load an image to begin."
        has_image = False
        source_path = (current or {}).get("image_path")
        if source_path and Path(source_path).is_file():
            try:
                image, record = inspect_input(
                    source_path, contract.features.input_normalization
                )
                annotated, updated_state, message = _loaded_image_outputs(
                    image, record, contract, None
                )
                image_details_text = _image_shape_text(record)
                status_text = (
                    f"Selected model `{model_id}`. Re-read and revalidated the "
                    f"retained image. {message}"
                )
                has_image = True
            except Exception as error:  # pragma: no cover - defensive
                updated_state = {}
                annotated = None
                status_text = (
                    f"Selected model `{model_id}`. The retained image could not be "
                    f"revalidated: {error}"
                )

        checkpoint_ready_text, checkpoint_text, install_visible = _checkpoint_display(
            selected_model, None
        )
        del checkpoint_ready_text
        feature_status_text = (
            "Model changed; recompute the eight-channel symmetry maps."
            if has_image
            else "Load an image to compute symmetry maps."
        )
        return (
            _selection_details(
                selected_model, _default_weight_identifier(selected_model)
            ),  # model_details
            checkpoint_text,  # checkpoint_status
            gr.update(visible=install_visible),  # install_button
            gr.update(value=None),  # custom_checkpoint
            "No custom checkpoint selected.",  # custom_checkpoint_sha
            updated_state,  # state
            annotated,  # annotation_image
            gr.update(choices=[], value=None),  # active_class
            gr.update(value="Class A, Class B"),  # class_names
            int(selected_model["classifier_patch_size"]),  # classifier_patch_size
            [],  # support_table
            None,  # patch_preview
            image_details_text,  # image_details
            {},  # feature_state
            gr.update(value=None, visible=False),  # feature_maps
            feature_status_text,  # feature_status
            gr.update(value=None, visible=False),  # png_download
            gr.update(value=None, visible=False),  # npy_download
            gr.update(interactive=has_image),  # compute_features_button
            int(defaults.get("symmetry_patch_size", contract.features.symmetry_patch_size)),  # symmetry_patch_size
            str(defaults.get("device", contract.model.device)),  # device
            int(defaults.get("epochs", contract.fine_tuning.epochs)),  # epochs
            float(defaults.get("learning_rate", contract.fine_tuning.learning_rate)),  # learning_rate
            int(defaults.get("stride", contract.prediction.stride)),  # stride
            int(defaults.get("batch_size", contract.prediction.batch_size)),  # batch_size
            *_cleared_result_updates(),
            status_text,  # status
        )

    model_change_outputs = [
        model_details,
        checkpoint_status,
        install_button,
        custom_checkpoint,
        custom_checkpoint_sha,
        state,
        annotation_image,
        active_class,
        class_names,
        classifier_patch_size,
        support_table,
        patch_preview,
        image_details,
        feature_state,
        feature_maps,
        feature_status,
        png_download,
        npy_download,
        compute_features_button,
        symmetry_patch_size,
        device,
        epochs,
        learning_rate,
        stride,
        batch_size,
        *RESULT_OUTPUTS,
        status,
    ]
    model_selector.change(
        on_model_change,
        inputs=[model_selector, catalog_state, state, custom_checkpoint],
        outputs=model_change_outputs,
    ).then(on_refresh, inputs=REFRESH_INPUTS, outputs=REFRESH_OUTPUTS)

    def on_install(model_id, current_catalog, checkpoint_path):
        if checkpoint_path:
            raise ValueError(
                "Clear the custom checkpoint before installing a registered weight."
            )
        capability = model_capability(current_catalog, model_id)
        weight = weight_capability(capability, None)
        result = install_registered_weight(
            config,
            model_identifier=model_id,
            weight_identifier=str(weight["identifier"]),
        )
        refreshed = result["capabilities"]
        selected_model = model_capability(refreshed, model_id)
        _, checkpoint_text, install_visible = _checkpoint_display(selected_model, None)
        return (
            refreshed,
            checkpoint_text,
            gr.update(visible=install_visible),
            f"Installed `{result['requirement']}` and refreshed Provider capabilities.",
        )

    install_button.click(
        on_install,
        inputs=[model_selector, catalog_state, custom_checkpoint],
        outputs=[catalog_state, checkpoint_status, install_button, status],
    ).then(on_refresh, inputs=REFRESH_INPUTS, outputs=REFRESH_OUTPUTS)

    def on_custom_checkpoint_change(path, model_id, current_catalog):
        capability = model_capability(current_catalog, model_id)
        _, checkpoint_text, install_visible = _checkpoint_display(
            capability, path or None
        )
        if not path:
            return (
                "No custom checkpoint selected.",
                checkpoint_text,
                gr.update(visible=install_visible),
            )
        resolved = Path(str(path))
        if resolved.is_file():
            digest = file_sha256(resolved)
            return (
                f"**Custom checkpoint SHA-256:** `{digest}`",
                checkpoint_text,
                gr.update(visible=install_visible),
            )
        return (
            f"Custom checkpoint `{resolved.name}` is missing.",
            checkpoint_text,
            gr.update(visible=install_visible),
        )

    custom_checkpoint.change(
        on_custom_checkpoint_change,
        inputs=[custom_checkpoint, model_selector, catalog_state],
        outputs=[custom_checkpoint_sha, checkpoint_status, install_button],
    ).then(on_refresh, inputs=REFRESH_INPUTS, outputs=REFRESH_OUTPUTS)

    image_change_outputs = [
        annotation_image,
        state,
        active_class,
        support_table,
        patch_preview,
        image_details,
        status,
        feature_state,
        feature_maps,
        feature_status,
        png_download,
        npy_download,
        compute_features_button,
        *RESULT_OUTPUTS,
    ]

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
                _image_shape_text(None),
                f"Image validation failed: {error}",
                {},
                gr.update(value=None, visible=False),
                "Load a valid image before computing symmetry maps.",
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(interactive=False),
                *_cleared_result_updates(),
            )
        return (
            annotated,
            updated,
            gr.update(
                choices=updated["class_names"],
                value=(
                    updated["class_names"][0] if updated["class_names"] else None
                ),
            ),
            annotation_rows(updated),
            None,
            _image_shape_text(record),
            message + " Selecting another image clears support points and predictions.",
            {},
            gr.update(value=None, visible=False),
            "Input changed; compute the eight-channel symmetry maps.",
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(interactive=True),
            *_cleared_result_updates(),
        )

    input_file.change(
        on_image_change,
        inputs=[input_file, state, model_selector, catalog_state],
        outputs=image_change_outputs,
    ).then(on_refresh, inputs=REFRESH_INPUTS, outputs=REFRESH_OUTPUTS)

    # ------------------------------------------------------------------
    # Stage 2 callbacks
    # ------------------------------------------------------------------
    feature_setting_outputs = [
        feature_state,
        feature_maps,
        feature_status,
        png_download,
        npy_download,
        *RESULT_OUTPUTS,
    ]

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
            *_cleared_result_updates(),
        )

    symmetry_patch_size.change(
        on_feature_setting_change,
        inputs=[state],
        outputs=feature_setting_outputs,
    ).then(on_refresh, inputs=REFRESH_INPUTS, outputs=REFRESH_OUTPUTS)
    device.change(
        on_feature_setting_change,
        inputs=[state],
        outputs=feature_setting_outputs,
    ).then(on_refresh, inputs=REFRESH_INPUTS, outputs=REFRESH_OUTPUTS)

    compute_outputs = [
        feature_maps,
        feature_state,
        status,
        feature_status,
        png_download,
        npy_download,
        *RESULT_OUTPUTS,
        *REFRESH_OUTPUTS,
    ]

    def on_compute_features(
        current,
        model_id,
        current_catalog,
        sym_size,
        run_device,
        checkpoint_path,
        epoch_count,
        lr,
        pred_stride,
        pred_batch,
        announced_stage,
    ):
        contract, options, fingerprint, cache_key = _feature_request(
            config,
            current_catalog,
            current,
            model_identifier=model_id,
            symmetry_patch_size=sym_size,
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
        feature_message = (
            "Symmetry maps are ready (reused cached result)."
            if cache_hit
            else "Symmetry maps have been computed."
        )
        # Refresh the stages here rather than chaining a separate refresh: a
        # chained call reads the *previous* feature_state. Using the freshly
        # computed state immediately enables stage 3 and attempts to open it.
        snapshot = _snapshot_from(
            current,
            current_features,
            model_id,
            checkpoint_path,
            current_catalog,
            sym_size,
            epoch_count,
            lr,
            pred_stride,
            pred_batch,
            run_device,
        )
        # The computation invalidates every result, so nothing is loaded.
        snapshot["has_result"] = False
        return (
            gr.update(value=feature_gallery(features, channel_names), visible=True),
            current_features,
            message,
            feature_message,
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            *_cleared_result_updates(),
            *_refresh_updates(snapshot, announced_stage),
        )

    compute_started = compute_features_button.click(
        lambda: "Computing symmetry maps...",
        inputs=[],
        outputs=[feature_status],
        queue=False,
        show_progress="hidden",
    )
    compute_finished = compute_started.then(
        on_compute_features,
        inputs=[
            state,
            model_selector,
            catalog_state,
            symmetry_patch_size,
            device,
            custom_checkpoint,
            epochs,
            learning_rate,
            stride,
            batch_size,
            last_stage,
        ],
        outputs=compute_outputs,
    )
    compute_finished.failure(
        lambda: "Symmetry-map computation failed. Review the error message above.",
        inputs=[],
        outputs=[feature_status],
        queue=False,
        show_progress="hidden",
    )
    # No chained refresh: ``on_compute_features`` already updates readiness,
    # action availability and the stage accordions from the fresh feature state.

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
                symmetry_patch_size=sym_size,
                device=str(run_device),
            )
        except (KeyError, TypeError, ValueError):
            return False
        return bool(
            current_features.get("cache_key") == expected_key
            and Path(str(current_features.get("features_path", ""))).is_file()
            and Path(str(current_features.get("record_path", ""))).is_file()
        )

    # ------------------------------------------------------------------
    # Stage 3 callbacks
    # ------------------------------------------------------------------
    support_point_outputs = [
        state,
        annotation_image,
        patch_preview,
        support_table,
        status,
        *RESULT_OUTPUTS,
    ]

    def on_configure_classes(
        value, selected_patch_size, current, model_id, current_catalog
    ):
        contract = _model_contract_config(config, current_catalog, model_id)
        if int(selected_patch_size) != contract.model.classifier_patch_size:
            raise ValueError(
                "The selected model requires a "
                f"{contract.model.classifier_patch_size}-pixel input patch."
            )
        return (
            *_configure_classes(
                value,
                current,
                contract,
                classifier_patch_size=int(selected_patch_size),
            ),
            *_cleared_result_updates(),
        )

    configure_button.click(
        on_configure_classes,
        inputs=[
            class_names,
            classifier_patch_size,
            state,
            model_selector,
            catalog_state,
        ],
        outputs=[
            state,
            active_class,
            annotation_image,
            support_table,
            status,
            *RESULT_OUTPUTS,
        ],
    ).then(on_refresh, inputs=REFRESH_INPUTS, outputs=REFRESH_OUTPUTS)

    def on_select(
        current,
        active,
        model_id,
        current_catalog,
        current_features,
        sym_size,
        run_device,
        event: SelectData,  # type: ignore[valid-type]
    ):
        index = event.index
        if not isinstance(index, (list, tuple)) or len(index) != 2:
            raise ValueError("The image click did not provide a valid pixel coordinate.")
        contract = _model_contract_config(config, current_catalog, model_id)
        source_point = _display_to_source_point(
            (int(index[0]), int(index[1])), tuple(current["image_shape"])
        )
        result = _add_point(current, active, source_point, contract)
        if result[4] == INVALID_SUPPORT_POINT_MESSAGE:
            gr.Info(
                INVALID_SUPPORT_POINT_MESSAGE,
                duration=1.5,
                title="Valid region",
            )
            return (
                result[0],
                result[1],
                gr.update(),
                result[3],
                gr.update(),
                *_cleared_result_updates(),
            )
        return (
            result[0],
            result[1],
            result[2],
            result[3],
            result[4],
            *_cleared_result_updates(),
        )

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
        outputs=support_point_outputs,
    ).then(on_refresh, inputs=REFRESH_INPUTS, outputs=REFRESH_OUTPUTS)

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
        return (
            result[0],
            result[1],
            gr.update(),
            result[2],
            result[3],
            *_cleared_result_updates(),
        )

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
        outputs=support_point_outputs,
    ).then(on_refresh, inputs=REFRESH_INPUTS, outputs=REFRESH_OUTPUTS)

    def on_clear(current, active, model_id, current_catalog):
        contract = _model_contract_config(config, current_catalog, model_id)
        result = _clear_class(current, active, contract)
        return (
            result[0],
            result[1],
            gr.update(),
            result[2],
            result[3],
            *_cleared_result_updates(),
        )

    clear_button.click(
        on_clear,
        inputs=[state, active_class, model_selector, catalog_state],
        outputs=support_point_outputs,
    ).then(on_refresh, inputs=REFRESH_INPUTS, outputs=REFRESH_OUTPUTS)

    # ------------------------------------------------------------------
    # Stage 4: run
    # ------------------------------------------------------------------
    run_outputs = [
        *RESULT_OUTPUTS,
        status,
        fine_tuning_progress,
        prediction_progress,
    ]

    def on_run(
        current,
        current_features,
        model_id,
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
        resolved_symmetry_patch_size = require_whole_number(
            sym_size, "The symmetry patch size", minimum=1
        )
        resolved_epochs = require_whole_number(
            epoch_count, "Fine-tuning epochs", minimum=1
        )
        resolved_stride = require_whole_number(
            pred_stride, "Prediction stride", minimum=1
        )
        resolved_batch_size = require_whole_number(
            pred_batch, "Prediction batch size", minimum=1
        )
        _, _, _, expected_feature_key = _feature_request(
            config,
            current_catalog,
            current,
            model_identifier=model_id,
            symmetry_patch_size=resolved_symmetry_patch_size,
            device=str(run_device),
        )
        if current_features.get("cache_key") != expected_feature_key:
            raise ValueError(
                "The symmetry maps are missing or stale. Compute them before fine-tuning."
            )
        features_path = Path(str(current_features.get("features_path", "")))
        features_record_path = Path(str(current_features.get("record_path", "")))
        if not features_path.is_file() or not features_record_path.is_file():
            raise ValueError(
                "The cached symmetry maps are unavailable. Compute them again."
            )
        runtime_config = _runtime_config(
            config,
            current_catalog,
            model_identifier=model_id,
            checkpoint_path=checkpoint,
        )
        runtime_config = replace(
            runtime_config,
            model=replace(runtime_config.model, device=str(run_device)),
        )
        selected_classifier_patch_size = _state_classifier_patch_size(
            current, runtime_config.model.classifier_patch_size
        )
        session = create_annotation_session(
            image_path=current["image_path"],
            image_sha256=current["image_sha256"],
            image_shape=tuple(current["image_shape"]),
            classifier_patch_size=selected_classifier_patch_size,
            class_names=current["class_names"],
            points_by_class=current["points"],
            colors=current["colors"],
        )
        progress_events: Queue[tuple[str, Any]] = Queue()

        def report_progress(phase: str, completed: int, total: int) -> None:
            progress_events.put(("progress", (phase, completed, total)))

        def execute_analysis() -> None:
            try:
                result = run_analysis(
                    runtime_config,
                    image_path=current["image_path"],
                    annotation_session=session,
                    overrides={
                        "symmetry_patch_size": resolved_symmetry_patch_size,
                        "classifier_patch_size": selected_classifier_patch_size,
                        "epochs": resolved_epochs,
                        "learning_rate": float(lr),
                        "stride": resolved_stride,
                        "batch_size": resolved_batch_size,
                        "device": str(run_device),
                    },
                    features_path=features_path,
                    features_record_path=features_record_path,
                    progress_callback=report_progress,
                )
            except Exception as error:
                progress_events.put(("error", error))
            else:
                progress_events.put(("result", result))

        fine_html = progress_bar_html(
            "Fine-tuning", 0, resolved_epochs, "Starting fine-tuning..."
        )
        prediction_html = progress_bar_html(
            "Prediction", 0, 1, "Waiting for fine-tuning."
        )
        yield (
            *[gr.skip()] * len(RESULT_OUTPUTS),
            "Fine-tuning started.",
            fine_html,
            prediction_html,
        )

        worker = Thread(target=execute_analysis, daemon=True)
        worker.start()
        result = None
        while result is None:
            try:
                event, payload = progress_events.get(timeout=0.25)
            except Empty:
                continue
            if event == "progress":
                phase, completed, total = payload
                if phase == "fine_tuning":
                    phase_status = (
                        "Fine-tuning complete."
                        if completed >= total
                        else f"Training epoch {completed} of {total}."
                    )
                    fine_html = progress_bar_html(
                        "Fine-tuning", completed, total, phase_status
                    )
                elif phase == "prediction":
                    fine_html = progress_bar_html(
                        "Fine-tuning",
                        resolved_epochs,
                        resolved_epochs,
                        "Fine-tuning complete.",
                    )
                    phase_status = (
                        "Prediction complete."
                        if completed >= total
                        else f"Predicting batch {completed} of {total}."
                    )
                    prediction_html = progress_bar_html(
                        "Prediction", completed, total, phase_status
                    )
                yield (
                    *[gr.skip()] * (len(RESULT_OUTPUTS) + 1),
                    fine_html,
                    prediction_html,
                )
            elif event == "error":
                fine_html = progress_bar_html("Fine-tuning", 0, 1, "Run failed.")
                prediction_html = progress_bar_html("Prediction", 0, 1, "Run failed.")
                yield (
                    *[gr.skip()] * (len(RESULT_OUTPUTS) + 1),
                    fine_html,
                    prediction_html,
                )
                raise payload
            elif event == "result":
                result = payload

        reference = {
            "run_id": str(result["run_id"]),
            "run_directory": str(result["run_directory"]),
            "prediction": str(result["prediction"]),
            "input_array": str(result["input_array"]),
            "input_preview": str(result["input_preview"]),
            "fine_tuned_model": str(result["fine_tuned_model"]),
            "stride": str(result["stride"]),
        }
        if result.get("full_run_zip"):
            reference["full_run_zip"] = str(result["full_run_zip"])
        class_names_list = [str(entry["name"]) for entry in result["classes"]]
        colors = [str(entry["color"]) for entry in result["classes"]]
        presentation = PresentationState.defaults(
            class_names_list,
            colors,
            alpha=DEFAULT_OVERLAY_ALPHA,
            result_reference=reference,
        )
        image, prediction = load_fine_tune_result(reference)
        variants = render_result_variants(
            image,
            prediction,
            class_names=class_names_list,
            colors=colors,
            alpha=presentation.alpha,
            stride=int(reference["stride"]),
        )
        statistics = _fine_tune_class_statistics(
            prediction.prediction_grid, len(class_names_list)
        )
        stats_html = class_statistics_html(
            statistics, class_names=class_names_list, class_colors=colors
        )
        model_download_value = prepare_model_download(reference)
        zip_download_value = (
            prepare_run_zip_download(reference)
            if reference.get("full_run_zip")
            else None
        )
        message = (
            f"Completed run `{result['run_id']}`. Saved locally to "
            f"`{result['run_directory']}`."
        )
        support_summary = (
            ", ".join(
                f"{name}: {len(group)}"
                for name, group in zip(
                    current.get("class_names", []), current.get("points", [])
                )
            )
            or "—"
        )
        run_summary = (
            f"**Run:** `{result['run_id']}` — completed  \n"
            f"**Model:** `{model_id}`  \n"
            f"**Classes:** {', '.join(class_names_list)}  \n"
            f"**Support points:** {support_summary}  \n"
            f"**Saved locally:** `{result['run_directory']}`"
        )
        result_warnings = [str(item) for item in result.get("warnings", [])]
        if result_warnings:
            run_summary += "\n\n**Warnings:** " + "; ".join(result_warnings)
        fine_html = progress_bar_html(
            "Fine-tuning",
            resolved_epochs,
            resolved_epochs,
            "Fine-tuning complete.",
        )
        prediction_html = progress_bar_html("Prediction", 1, 1, "Prediction complete.")
        yield (
            variants["overlay_legend"],  # prediction_overlay
            variants["confidence_colorbar"],  # confidence_image
            variants["entropy_colorbar"],  # entropy_image
            stats_html,  # class_statistics
            run_summary,  # run_path
            str(result["run_directory"]),  # copy_path
            gr.update(value=model_download_value, visible=True),  # model_download
            gr.update(
                value=zip_download_value,
                visible=zip_download_value is not None,
            ),  # run_zip_download
            ", ".join(colors),  # color_text
            DEFAULT_OVERLAY_ALPHA,  # alpha_slider
            gr.update(value=None, visible=False),  # result_exports_files
            result,  # result_json
            gr.update(visible=True),  # results_group
            presentation,  # presentation_state
            message,  # status
            fine_html,
            prediction_html,
        )

    run_button.click(
        on_run,
        inputs=[
            state,
            feature_state,
            model_selector,
            custom_checkpoint,
            catalog_state,
            symmetry_patch_size,
            epochs,
            learning_rate,
            stride,
            batch_size,
            device,
        ],
        outputs=run_outputs,
        show_progress="hidden",
    ).then(on_refresh, inputs=REFRESH_INPUTS, outputs=REFRESH_OUTPUTS)

    # ------------------------------------------------------------------
    # Presentation rerender / PNG exports (never touch the Provider)
    # ------------------------------------------------------------------
    def on_update_presentation(current_presentation, colors_text, alpha_value):
        if (
            current_presentation is None
            or not current_presentation.numerical_result_reference
        ):
            raise ValueError("Complete a fine-tuning run before editing colors.")
        reference = current_presentation.numerical_result_reference
        class_count = len(current_presentation.class_names)
        try:
            colors = parse_presentation_colors(colors_text, class_count)
            resolved_alpha = validate_alpha(alpha_value)
        except PresentationError as error:
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                f"Invalid presentation input: {error} The completed result is unchanged.",
                gr.update(),
            )
        updated = PresentationState(
            class_names=list(current_presentation.class_names),
            display_colors=colors,
            alpha=resolved_alpha,
            selected_result_item=current_presentation.selected_result_item,
            numerical_result_reference=dict(reference),
        )
        image, prediction = load_fine_tune_result(reference)
        variants = render_result_variants(
            image,
            prediction,
            class_names=updated.class_names,
            colors=updated.display_colors,
            alpha=updated.alpha,
            stride=int(reference["stride"]),
        )
        statistics = _fine_tune_class_statistics(
            prediction.prediction_grid, class_count
        )
        stats_html = class_statistics_html(
            statistics,
            class_names=updated.class_names,
            class_colors=updated.display_colors,
        )
        return (
            variants["overlay_legend"],
            variants["confidence_colorbar"],
            variants["entropy_colorbar"],
            stats_html,
            updated,
            "Presentation updated; the numerical result is unchanged.",
            gr.update(value=None, visible=False),
        )

    update_presentation_button.click(
        on_update_presentation,
        inputs=[presentation_state, color_text, alpha_slider],
        outputs=[
            prediction_overlay,
            confidence_image,
            entropy_image,
            class_statistics,
            presentation_state,
            status,
            result_exports_files,
        ],
    )

    def on_prepare_exports(current_presentation, names):
        if (
            current_presentation is None
            or not current_presentation.numerical_result_reference
        ):
            raise ValueError("Complete a fine-tuning run before exporting PNGs.")
        selected = [name for name in (names or []) if name in RESULT_EXPORT_NAMES]
        if not selected:
            raise ValueError("Select at least one PNG variant to export.")
        reference = current_presentation.numerical_result_reference
        try:
            parse_presentation_colors(
                ", ".join(current_presentation.display_colors),
                len(current_presentation.class_names),
            )
            validate_alpha(current_presentation.alpha)
        except PresentationError as error:
            raise ValueError(
                f"Fix the presentation colors before exporting: {error}"
            ) from error
        paths = prepare_presentation_pngs(
            reference,
            current_presentation.class_names,
            current_presentation.display_colors,
            current_presentation.alpha,
            selected,
        )
        return (
            gr.update(value=paths, visible=bool(paths)),
            f"Prepared {len(paths)} PNG export(s). The durable run artifacts are unchanged.",
        )

    prepare_exports_button.click(
        on_prepare_exports,
        inputs=[presentation_state, export_names],
        outputs=[result_exports_files, status],
    )

    return feature_cache_directory
