"""Saved-model prediction workspace for the local symmetry interface.

This workspace has its own state. It never reuses fine-tuning state, and it
never exposes feature parameters as editable controls.

Design notes (Phase 1 usability revision):

* The file input accepts ``.symmodel`` only, and the user-visible workflow
  rejects any other suffix (:func:`require_symmodel_path`). A ``.symmodel`` is
  internally a ZIP container, but it is a validated portable saved model -- a
  renamed generic ZIP or a full run ZIP is still rejected by the complete
  content validation performed by :func:`model_package.inspect_model_package`.
* The visible stride and batch-size controls are hydrated from the package's
  saved ``prediction_defaults`` (:func:`_saved_prediction_defaults`). The
  controls always hold the value that will actually be used, so what the user
  sees is what runs. They remain editable as runtime-safe overrides.
* Transient presentation state (colors and overlay alpha) lives in its own
  :class:`gr.State` and is never persisted to the package, records, reports or
  the batch archive.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from queue import Empty, Queue
import tempfile
from threading import Thread
from typing import Any

import numpy as np

from .config import HarnessConfig
from .image_io import unit_to_uint8
from .model_package import (
    FineTunedModelPackage,
    ModelPackageError,
    inspect_model_package,
)
from .numeric import require_whole_number
from .prediction_workflow import (
    PREDICTION_OPERATION,
    PredictionError,
    _check_provider_compatibility,
    run_saved_model_prediction_batch,
)
from .result_exports import (
    DEFAULT_OVERLAY_ALPHA,
    RESULT_EXPORT_NAMES,
    PresentationError,
    normalize_palette,
    render_result_variants,
    validate_alpha,
    write_png,
)
from .ui_readiness import (
    evaluate_predict_readiness,
    has_blocker,
    readiness_panel_html,
)
from .ui_shared import (
    class_legend_html,
    class_statistics_html,
    progress_bar_html,
    result_item_label,
)


LOGGER = logging.getLogger(__name__)


SUPPORTED_IMAGE_SUFFIXES = (
    ".npy",
    ".npz",
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
)

SYMMODEL_SUFFIX = ".symmodel"

# Application fallback defaults, used only until a package is loaded. The saved
# package defaults always win once a package validates.
DEFAULT_STRIDE = 4
DEFAULT_BATCH_SIZE = 512

# Every downloaded presentation PNG variant, in RESULT_EXPORT_NAMES order.
_VARIANT_FILENAMES: dict[str, str] = {
    "overlay": "overlay.png",
    "overlay_legend": "overlay_with_legend.png",
    "mask": "categorical_mask.png",
    "confidence": "confidence.png",
    "confidence_colorbar": "confidence_with_colorbar.png",
    "entropy": "entropy.png",
    "entropy_colorbar": "entropy_with_colorbar.png",
}


# ---------------------------------------------------------------------------
# Pure helpers (module level, importable without Gradio for focused tests)
# ---------------------------------------------------------------------------
def require_symmodel_path(path: str | None) -> str:
    """Return the resolved path if it ends with ``.symmodel``; else raise.

    The ``.symmodel`` suffix is a user-facing policy enforced here at the UI
    layer. The numerical workflow keeps its own semantics unchanged so the CLI
    path is unaffected.
    """
    if path is None or not str(path).strip():
        raise PredictionError("Select a .symmodel model package.")
    resolved = str(Path(str(path)).expanduser().resolve())
    if not resolved.lower().endswith(SYMMODEL_SUFFIX):
        name = Path(resolved).name
        raise PredictionError(
            f"Only .symmodel model packages are accepted; got {name!r}."
        )
    return resolved


def _provider_supports_prediction(capabilities: dict[str, Any] | None) -> bool:
    if not capabilities:
        return False
    return PREDICTION_OPERATION in capabilities.get("operations", [])


def _overrides_are_valid(stride: Any, batch_size: Any) -> bool:
    try:
        require_whole_number(stride, "Prediction stride", minimum=1)
        require_whole_number(batch_size, "Prediction batch size", minimum=1)
        return True
    except ValueError:
        return False


def _saved_prediction_defaults(
    package: FineTunedModelPackage | None,
) -> dict[str, int]:
    """Return the runtime defaults saved inside ``package``.

    On a successful load the visible stride and batch-size controls are set from
    these values so the workspace does not mask the saved feature options with
    the current application defaults.
    """
    if package is None:
        return {"stride": DEFAULT_STRIDE, "batch_size": DEFAULT_BATCH_SIZE}
    defaults = dict(getattr(package, "prediction_defaults", {}) or {})
    stride = _coerce_positive_int(defaults.get("stride"), DEFAULT_STRIDE)
    batch_size = _coerce_positive_int(
        defaults.get("batch_size"), DEFAULT_BATCH_SIZE
    )
    return {"stride": stride, "batch_size": batch_size}


def _coerce_positive_int(value: Any, fallback: int) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return int(fallback)
    return resolved if resolved > 0 else int(fallback)


def _package_details(package: FineTunedModelPackage, compatibility: str) -> str:
    """Render saved-model metadata without loading PyTorch weights."""
    manifest = package.manifest
    provenance = manifest.get("provenance", {})
    software = manifest.get("software", {})
    features = manifest["features"]
    lines = [
        f"- Package: `{Path(package.source_path).name}`",
        f"- Package SHA-256: `{package.package_sha256[:16]}...`",
        f"- Model: `{package.model_identifier}`",
        f"- Task classes: `{package.task_classes}`",
        f"- Classifier patch size: `{manifest['model']['classifier_patch_size']}`",
        f"- Feature pipeline: `{features['pipeline']}`",
        f"- Symmetry patch size: `{features['symmetry_patch_size']}`",
        f"- Input normalization: `{features['input_normalization']}`",
        f"- Training run: `{provenance.get('training_run_id', 'unknown')}`",
        (
            "- Recorded versions: "
            f"symmetry-harness `{software.get('symmetry_harness_version', 'unknown')}`, "
            f"symmetry-learn `{software.get('symmetry_learn_version', 'unknown')}`, "
            f"torch `{software.get('torch_version', 'unknown')}`"
        ),
        f"- Compatibility: **{compatibility}**",
    ]
    return "\n".join(lines)


def _load_package(
    path: str | None,
    *,
    config: HarnessConfig | None = None,
    capabilities: dict[str, Any] | None = None,
):
    """Return the validated package and a compatibility message.

    This function keeps its historical three-tuple contract
    ``(package, compatibility, message)``; the saved-defaults hydration is done
    separately by :func:`_saved_prediction_defaults`.
    """
    if not path:
        return None, "Waiting", "Import a `.symmodel` package to begin."
    try:
        package = inspect_model_package(path)
    except ModelPackageError as error:
        return None, "Blocked", f"Package validation failed: {error}"
    except Exception as error:  # pragma: no cover - defensive UI boundary
        return None, "Blocked", f"Package could not be inspected: {error}"
    if capabilities is not None and config is not None:
        try:
            _, warnings = _check_provider_compatibility(
                config, capabilities, package
            )
        except PredictionError as error:
            return package, "Blocked", f"Package is incompatible: {error}"
        if warnings:
            return package, "Warning", " ".join(warnings)
    return package, "Ready", "Package validated. Choose one or more images."


def _validate_inputs(
    package: FineTunedModelPackage | None,
    paths: list[str] | None,
) -> list[dict[str, Any]]:
    from .image_io import inspect_input

    rows: list[dict[str, Any]] = []
    for raw in paths or []:
        candidate = Path(str(raw)).expanduser().resolve()
        row = {"file": candidate.name, "status": "invalid", "message": ""}
        if package is None:
            row["message"] = "Import a valid model package first."
            rows.append(row)
            continue
        try:
            image, _ = inspect_input(candidate, package.input_normalization)
        except Exception as error:
            row["message"] = str(error)
            rows.append(row)
            continue
        minimum = int(package.manifest["model"]["classifier_patch_size"])
        if min(image.shape) < minimum:
            row["message"] = f"Smaller than the saved {minimum}-pixel patch."
            rows.append(row)
            continue
        row["status"] = "valid"
        row["message"] = f"{image.shape[0]} x {image.shape[1]}"
        rows.append(row)
    return rows


def _package_status(
    *,
    selected_path: str | None,
    extension_ok: bool,
    package: FineTunedModelPackage | None,
    compatibility: str,
    message: str,
    supported: bool,
) -> dict[str, Any]:
    """Collect the package-level facts readiness needs into one plain dict."""
    warnings: list[str] = []
    if compatibility == "Warning":
        warnings = [message]
    return {
        "supported": bool(supported),
        "path": selected_path,
        "extension_ok": bool(extension_ok),
        "package_valid": package is not None,
        "compatible": compatibility in {"Ready", "Warning"},
        "warnings": warnings,
        "detail": message,
    }


def _readiness_snapshot(
    status: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    *,
    device: Any,
    stride: Any,
    batch_size: Any,
) -> dict[str, Any]:
    """Assemble a plain readiness snapshot from UI-provided values."""
    status = dict(status or {})
    valid_images = sum(1 for row in rows if row.get("status") == "valid")
    invalid_images = sum(1 for row in rows if row.get("status") == "invalid")
    return {
        "provider_supported": bool(status.get("supported")),
        "package_path": status.get("path"),
        "extension_ok": bool(status.get("extension_ok")),
        "package_valid": bool(status.get("package_valid")),
        "package_compatible": bool(status.get("compatible")),
        "compatibility_warnings": list(status.get("warnings", []) or []),
        "package_detail": str(status.get("detail", "") or ""),
        "valid_images": valid_images,
        "invalid_images": invalid_images,
        "device": device,
        "stride": stride,
        "batch_size": batch_size,
    }


def _input_rows_for_table(rows: list[dict[str, Any]]) -> list[list[str]]:
    """Convert validation rows into the Dataframe's list-of-rows shape."""
    return [
        [str(row.get("file", "")), str(row.get("status", "")), str(row.get("message", ""))]
        for row in rows
    ]


def _item_choices(result: dict[str, Any] | None) -> list[tuple[str, str]]:
    """Return ``(label, item_id)`` pairs with deterministic disambiguation.

    Labels use the source filename and status; the stable internal ``item_id``
    (``image-0001`` ...) is retained as the selector value. Duplicate basenames
    receive a deterministic ``(2)``, ``(3)`` ... suffix in encounter order.
    """
    if not result:
        return []
    seen: dict[str, int] = {}
    choices: list[tuple[str, str]] = []
    for entry in result.get("items", []):
        item_id = str(entry.get("item_id", ""))
        source = entry.get("source_path") or entry.get("file") or item_id
        label = result_item_label(
            str(source), str(entry.get("status", "")), item_id=item_id, seen=seen
        )
        choices.append((label, item_id))
    return choices


# ---------------------------------------------------------------------------
# Presentation state
# ---------------------------------------------------------------------------
def new_presentation_state(
    class_names: list[str] | tuple[str, ...],
    class_colors: list[str] | tuple[str, ...],
    *,
    alpha: float = DEFAULT_OVERLAY_ALPHA,
    selected_result_item: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Create a fresh transient presentation state for one batch/package."""
    names = [str(name) for name in class_names]
    colors = [str(color) for color in class_colors]
    return {
        "class_names": names,
        "saved_colors": list(colors),
        "display_colors": list(colors),
        "alpha": float(alpha),
        "selected_result_item": selected_result_item,
        "run_id": run_id,
    }


def split_colors_text(colors_text: Any) -> list[str]:
    """Split a color widget value into stripped hex-color candidates."""
    if colors_text is None:
        return []
    raw = str(colors_text).replace(",", "\n")
    return [line.strip() for line in raw.splitlines() if line.strip()]


def presentation_colors_text(presentation: dict[str, Any] | None) -> str:
    if not presentation:
        return ""
    colors = presentation.get("display_colors") or presentation.get("saved_colors") or []
    return "\n".join(str(color) for color in colors)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _empty_render() -> dict[str, Any]:
    return {
        "source": None,
        "overlay": None,
        "confidence": None,
        "entropy": None,
        "legend": "",
        "statistics": "",
        "metadata": None,
        "message": "",
        "downloads": {},
        "blocked": False,
    }


def _load_npz_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _stride_for_item(result: dict[str, Any] | None, entry: dict[str, Any]) -> int:
    """Resolve the stride used for one prediction item.

    The UI always records the stride it actually ran with on the batch result;
    reading the persisted record is only a defensive fallback.
    """
    result_payload = result or {}
    if "stride" in result_payload:
        return require_whole_number(
            result_payload["stride"], "Recorded prediction stride", minimum=1
        )

    record_path = entry.get("prediction_record")
    if not record_path and entry.get("item_directory"):
        record_path = str(
            Path(str(entry["item_directory"])) / "prediction_record.json"
        )
    if not record_path:
        LOGGER.warning(
            "Prediction stride metadata is absent; using legacy default %d.",
            DEFAULT_STRIDE,
        )
        return DEFAULT_STRIDE

    try:
        payload = json.loads(Path(str(record_path)).read_text(encoding="utf-8"))
    except FileNotFoundError:
        LOGGER.warning(
            "Prediction stride metadata is absent; using legacy default %d.",
            DEFAULT_STRIDE,
        )
        return DEFAULT_STRIDE
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Prediction stride metadata could not be read from {record_path}."
        ) from error

    if not isinstance(payload, dict):
        raise ValueError("Prediction record must contain a JSON object.")
    prediction_options = payload.get("prediction_options")
    if prediction_options is None:
        LOGGER.warning(
            "Prediction stride metadata is absent; using legacy default %d.",
            DEFAULT_STRIDE,
        )
        return DEFAULT_STRIDE
    if not isinstance(prediction_options, dict):
        raise ValueError("Recorded prediction options must be an object.")
    if "stride" not in prediction_options:
        LOGGER.warning(
            "Prediction stride metadata is absent; using legacy default %d.",
            DEFAULT_STRIDE,
        )
        return DEFAULT_STRIDE
    return require_whole_number(
        prediction_options["stride"], "Recorded prediction stride", minimum=1
    )


def _write_variant_downloads(
    variants: dict[str, np.ndarray], *, run_id: str, item_id: str
) -> dict[str, str]:
    """Write the presentation PNGs into a temporary download directory."""
    directory = (
        Path(tempfile.gettempdir())
        / "symmetry-harness-presentation"
        / f"{run_id}-{item_id}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    return {
        name: str(write_png(array, directory / _VARIANT_FILENAMES.get(name, f"{name}.png")))
        for name, array in variants.items()
    }


def render_selected_item(
    item_id: str | None,
    result: dict[str, Any] | None,
    *,
    colors_text: Any,
    alpha: Any,
    class_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Render every presentation variant for one selected batch item.

    Pure with respect to the numerical result: it only reads the completed
    per-item ``prediction.npz`` and normalized input, so it can never restore the
    model, recompute features or rerun prediction. Invalid presentation colors
    block rerendering/downloads (``blocked=True``) without invalidating the
    completed numerical result.
    """
    render = _empty_render()
    if not result or not item_id:
        return render
    entry = next(
        (item for item in result.get("items", []) if item.get("item_id") == item_id),
        None,
    )
    if entry is None:
        return render

    render["metadata"] = entry
    if entry.get("status") != "completed":
        render["message"] = (
            f"`{item_id}` failed: {entry.get('error', 'no error record')}."
        )
        return render

    names = list(class_names or result.get("class_names") or [])
    try:
        palette = list(
            normalize_palette(split_colors_text(colors_text), class_count=len(names))
        )
        resolved_alpha = validate_alpha(alpha)
    except PresentationError as error:
        render["blocked"] = True
        render["message"] = (
            f"Presentation colors are invalid: {error} Rerendering and downloads "
            "are blocked until they are fixed; the completed prediction is "
            "unchanged."
        )
        return render

    arrays = _load_npz_arrays(entry["artifact_prediction"])
    from .analysis import dense_prediction_from_arrays

    prediction = dense_prediction_from_arrays(arrays)
    image = np.asarray(
        np.load(str(entry["artifact_input"]), allow_pickle=False), dtype=np.float32
    )
    stride = _stride_for_item(result, entry)
    variants = render_result_variants(
        image,
        prediction,
        class_names=names,
        colors=palette,
        alpha=resolved_alpha,
        stride=stride,
        class_count=len(names),
    )
    classes = [
        {"index": index, "name": name, "color": palette[index]}
        for index, name in enumerate(names)
    ]
    render["source"] = unit_to_uint8(image)
    # The legend already lives beside the result (``render["legend"]``), so the
    # overlay here is the pure variant: an annotated margin would make the
    # overlay wider than the confidence and entropy maps and therefore render
    # smaller than them.
    render["overlay"] = variants["overlay"]
    render["confidence"] = variants["confidence_colorbar"]
    render["entropy"] = variants["entropy_colorbar"]
    render["legend"] = class_legend_html(classes)
    render["statistics"] = class_statistics_html(
        entry.get("class_statistics", {}),
        class_names=names,
        class_colors=palette,
    )
    render["downloads"] = _write_variant_downloads(
        variants,
        run_id=str((result or {}).get("run_id", "prediction")),
        item_id=str(item_id),
    )
    render["message"] = ""
    return render


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
def build_prediction_workspace(
    config: HarnessConfig,
    *,
    capabilities: dict[str, Any] | None = None,
):
    """Construct the saved-model prediction workspace inside the caller's Blocks."""
    import gradio as gr

    supported = _provider_supports_prediction(capabilities)
    initial_status = (
        "Import a saved `.symmodel` package to begin."
        if supported
        else (
            "The installed symmetry Provider does not support saved-model "
            "prediction. Update symmetry-learn to use this workspace."
        )
    )

    package_state = gr.State(None)
    package_compatible_state = gr.State(False)
    package_status_state = gr.State({})
    result_state = gr.State(None)
    presentation_state = gr.State(None)

    gr.Markdown("## Saved model")
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            package_file = gr.File(
                label="Fine-tuned model package",
                type="filepath",
                file_types=[SYMMODEL_SUFFIX],
            )
            gr.Markdown(
                "A `.symmodel` package is a portable, validated saved model. "
                "Internally it is a ZIP container, but only verified `.symmodel` "
                "packages are accepted -- a renamed generic ZIP or a full run ZIP "
                "is rejected by content validation."
            )
            package_details = gr.Markdown(initial_status)
            feature_contract = gr.Markdown("")

            gr.Markdown("## Prediction images")
            image_files = gr.File(
                label="Prediction images",
                type="filepath",
                file_count="multiple",
                file_types=list(SUPPORTED_IMAGE_SUFFIXES),
            )
            input_table = gr.Dataframe(
                headers=["File", "Status", "Detail"],
                datatype=["str", "str", "str"],
                interactive=False,
                label="Input validation",
            )
            with gr.Row():
                predict_device = gr.Dropdown(
                    label="Device",
                    choices=["auto", "cuda", "cpu"],
                    value="auto",
                    info="Use auto unless a specific execution device is required.",
                )
                predict_stride = gr.Number(
                    label="Prediction stride (saved default)",
                    value=DEFAULT_STRIDE,
                    precision=0,
                    info="Loaded from the package's saved prediction defaults.",
                )
                predict_batch = gr.Number(
                    label="Prediction batch size (saved default)",
                    value=DEFAULT_BATCH_SIZE,
                    precision=0,
                    info="Loaded from the package's saved prediction defaults.",
                )
        with gr.Column(scale=1, min_width=280, elem_classes=["readiness-sticky"]):
            # The sticky hook lives on the outer column as a *class*, never an
            # id: the Fine-tune workspace already owns id="readiness-panel", and
            # Gradio renders both tab bodies into one document, so a second
            # identical id would be invalid HTML.
            readiness_panel = gr.HTML(
                readiness_panel_html(
                    evaluate_predict_readiness(
                        _readiness_snapshot(
                            None,
                            [],
                            device="auto",
                            stride=DEFAULT_STRIDE,
                            batch_size=DEFAULT_BATCH_SIZE,
                        )
                    ),
                    elem_id=None,
                )
            )
            predict_button = gr.Button(
                "Predict", variant="primary", interactive=False
            )

    prediction_progress = gr.HTML(
        progress_bar_html("Prediction", 0, 1, "Waiting to start.")
    )
    prediction_status = gr.Markdown("")

    with gr.Column(visible=False) as result_section:
        gr.Markdown("## Results")
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                result_selector = gr.Dropdown(
                    label="Predicted image", choices=[], interactive=True
                )
                with gr.Row():
                    result_source = gr.Image(
                        label="Source image",
                        type="numpy",
                        interactive=False,
                        height=480,
                    )
                    result_overlay = gr.Image(
                        label="Prediction overlay",
                        type="numpy",
                        interactive=False,
                        height=480,
                    )
                with gr.Row():
                    result_confidence = gr.Image(
                        label="Confidence (fixed 0-1 colorbar)",
                        type="numpy",
                        interactive=False,
                        height=480,
                    )
                    result_entropy = gr.Image(
                        label="Predictive entropy (fixed 0-ln(N) colorbar)",
                        type="numpy",
                        interactive=False,
                        height=480,
                    )
            with gr.Column(scale=1, min_width=260):
                presentation_message = gr.Markdown("")
                class_legend = gr.HTML()
                class_stats = gr.HTML()
                presentation_colors = gr.Textbox(
                    label="Class colors (one #RRGGBB per line, in class order)",
                    lines=3,
                    max_lines=12,
                )
                presentation_alpha = gr.Slider(
                    label="Overlay alpha (global)",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=DEFAULT_OVERLAY_ALPHA,
                )
                with gr.Accordion("Technical details", open=False):
                    result_metadata = gr.JSON(label="Prediction record")
        gr.Markdown("### PNG downloads")
        with gr.Row():
            download_overlay = gr.DownloadButton(
                "Overlay", visible=False, size="sm"
            )
            download_overlay_legend = gr.DownloadButton(
                "Overlay + legend", visible=False, size="sm"
            )
            download_mask = gr.DownloadButton(
                "Categorical mask", visible=False, size="sm"
            )
            download_confidence = gr.DownloadButton(
                "Confidence", visible=False, size="sm"
            )
            download_confidence_colorbar = gr.DownloadButton(
                "Confidence + colorbar", visible=False, size="sm"
            )
            download_entropy = gr.DownloadButton(
                "Entropy", visible=False, size="sm"
            )
            download_entropy_colorbar = gr.DownloadButton(
                "Entropy + colorbar", visible=False, size="sm"
            )
        batch_download = gr.DownloadButton(
            "Download batch archive", visible=False
        )

    render_components = [
        result_source,
        result_overlay,
        result_confidence,
        result_entropy,
        class_legend,
        class_stats,
        result_metadata,
        presentation_message,
        download_overlay,
        download_overlay_legend,
        download_mask,
        download_confidence,
        download_confidence_colorbar,
        download_entropy,
        download_entropy_colorbar,
    ]

    def _render_tuple(render: dict[str, Any]):
        values: list[Any] = [
            render.get("source"),
            render.get("overlay"),
            render.get("confidence"),
            render.get("entropy"),
            render.get("legend", ""),
            render.get("statistics", ""),
            render.get("metadata"),
            render.get("message", ""),
        ]
        downloads = render.get("downloads") or {}
        for name in RESULT_EXPORT_NAMES:
            path = downloads.get(name)
            values.append(gr.update(value=path, visible=path is not None))
        return tuple(values)

    def _blocked_tuple(render: dict[str, Any]):
        # Keep the last good images/legend/stats/metadata, surface the error and
        # clear every presentation download derived from the current colors.
        values: list[Any] = [gr.skip() for _ in range(7)]
        values.append(render.get("message", ""))
        for _ in RESULT_EXPORT_NAMES:
            values.append(gr.update(value=None, visible=False))
        return tuple(values)

    def _cleared_render_tuple():
        return _render_tuple(_empty_render())

    def _readiness_outputs(status, rows, device, stride, batch_size):
        snapshot = _readiness_snapshot(
            status, rows, device=device, stride=stride, batch_size=batch_size
        )
        items = evaluate_predict_readiness(snapshot)
        return items, bool(has_blocker(items))

    # -- Package change ----------------------------------------------------
    def on_package_change(path):
        selected_path: str | None = None
        extension_ok = False
        try:
            selected_path = require_symmodel_path(path)
        except PredictionError as error:
            package, compatibility, message = None, "Blocked", str(error)
        else:
            extension_ok = True
            package, compatibility, message = _load_package(
                selected_path, config=config, capabilities=capabilities
            )

        status = _package_status(
            selected_path=selected_path,
            extension_ok=extension_ok,
            package=package,
            compatibility=compatibility,
            message=message,
            supported=supported,
        )
        compatible = package is not None and compatibility in {"Ready", "Warning"}
        defaults = _saved_prediction_defaults(package)
        details = (
            _package_details(package, compatibility) if package is not None else message
        )
        contract = (
            (
                "Feature parameters are restored from the package and are "
                "read-only: "
                f"`n_max={package.feature_options['n_max']}`, "
                f"`symmetry_patch_size={package.feature_options['symmetry_patch_size']}`, "
                f"`rotation_folds={list(package.feature_options['rotation_folds'])}`, "
                f"`normalization={package.input_normalization}`."
            )
            if package is not None
            else ""
        )
        presentation = (
            new_presentation_state(package.class_names, package.class_colors)
            if package is not None
            else None
        )
        colors_text = presentation_colors_text(presentation)
        _, blocked = _readiness_outputs(
            status, [], "auto", defaults["stride"], defaults["batch_size"]
        )
        return (
            package,
            compatible,
            status,
            details,
            contract,
            [],
            gr.update(value=defaults["stride"]),
            gr.update(value=defaults["batch_size"]),
            gr.update(interactive=not blocked),
            readiness_panel_html(
                evaluate_predict_readiness(
                    _readiness_snapshot(
                        status,
                        [],
                        device="auto",
                        stride=defaults["stride"],
                        batch_size=defaults["batch_size"],
                    )
                ),
                elem_id=None,
            ),
            message if package is None else "",
            None,
            gr.update(choices=[], value=None),
            gr.update(value=None, visible=False),
            presentation,
            gr.update(value=colors_text),
            gr.update(value=DEFAULT_OVERLAY_ALPHA),
            gr.update(visible=False),
            gr.update(value=None),
            *_cleared_render_tuple(),
        )

    package_file.change(
        on_package_change,
        inputs=[package_file],
        outputs=[
            package_state,
            package_compatible_state,
            package_status_state,
            package_details,
            feature_contract,
            input_table,
            predict_stride,
            predict_batch,
            predict_button,
            readiness_panel,
            prediction_status,
            result_state,
            result_selector,
            batch_download,
            presentation_state,
            presentation_colors,
            presentation_alpha,
            result_section,
            image_files,
            *render_components,
        ],
    )

    # -- Input / override change ------------------------------------------
    def on_inputs_change(
        paths, package, package_compatible, status, device, stride, batch_size
    ):
        rows = _validate_inputs(package, paths)
        items, blocked = _readiness_outputs(status, rows, device, stride, batch_size)
        return (
            _input_rows_for_table(rows),
            gr.update(interactive=not blocked),
            readiness_panel_html(items, elem_id=None),
            "",
            None,
            gr.update(choices=[], value=None),
            gr.update(value=None, visible=False),
            gr.update(visible=False),
            *_cleared_render_tuple(),
        )

    _input_inputs = [
        image_files,
        package_state,
        package_compatible_state,
        package_status_state,
        predict_device,
        predict_stride,
        predict_batch,
    ]
    _input_outputs = [
        input_table,
        predict_button,
        readiness_panel,
        prediction_status,
        result_state,
        result_selector,
        batch_download,
        result_section,
        *render_components,
    ]
    image_files.change(
        on_inputs_change, inputs=_input_inputs, outputs=_input_outputs
    )
    for control in (predict_device, predict_stride, predict_batch):
        control.change(
            on_inputs_change, inputs=_input_inputs, outputs=_input_outputs
        )

    # -- Render (selector / color / alpha change) -------------------------
    def on_render(item_id, result, presentation, colors_text, alpha):
        base = presentation or new_presentation_state(
            (result or {}).get("class_names", []),
            (result or {}).get("class_colors", []),
        )
        updated = dict(base)
        if result:
            updated["class_names"] = list(
                result.get("class_names", base.get("class_names", []))
            )
            updated["run_id"] = result.get("run_id")
        updated["display_colors"] = split_colors_text(colors_text)
        try:
            updated["alpha"] = validate_alpha(alpha)
        except PresentationError:
            updated["alpha"] = alpha
        updated["selected_result_item"] = item_id

        render = render_selected_item(
            item_id,
            result,
            colors_text=colors_text,
            alpha=alpha,
            class_names=updated["class_names"],
        )
        if render.get("blocked"):
            return (updated, gr.update(visible=True), *_blocked_tuple(render))
        if not result or not item_id:
            return (updated, gr.update(visible=False), *_cleared_render_tuple())
        return (updated, gr.update(visible=True), *_render_tuple(render))

    def on_presentation_render(item_id, result, presentation, colors_text, alpha):
        """Refresh only presentation-dependent UI after color/alpha edits.

        Source, confidence, entropy and metadata are immutable for a completed
        numerical result. Leaving them out of this callback prevents Gradio from
        needlessly reloading all four image components on every presentation edit.
        """
        full = on_render(item_id, result, presentation, colors_text, alpha)
        updated = full[0]
        overlay = full[3]
        legend = full[6]
        statistics = full[7]
        message = full[9]
        presentation_downloads = full[10:13]
        return (
            updated,
            overlay,
            legend,
            statistics,
            message,
            *presentation_downloads,
        )

    result_selector.change(
        on_render,
        inputs=[
            result_selector,
            result_state,
            presentation_state,
            presentation_colors,
            presentation_alpha,
        ],
        outputs=[presentation_state, result_section, *render_components],
    )
    presentation_colors.change(
        on_presentation_render,
        inputs=[
            result_selector,
            result_state,
            presentation_state,
            presentation_colors,
            presentation_alpha,
        ],
        outputs=[
            presentation_state,
            result_overlay,
            class_legend,
            class_stats,
            presentation_message,
            download_overlay,
            download_overlay_legend,
            download_mask,
        ],
    )
    presentation_alpha.change(
        on_presentation_render,
        inputs=[
            result_selector,
            result_state,
            presentation_state,
            presentation_colors,
            presentation_alpha,
        ],
        outputs=[
            presentation_state,
            result_overlay,
            class_legend,
            class_stats,
            presentation_message,
            download_overlay,
            download_overlay_legend,
            download_mask,
        ],
    )

    # -- Prediction --------------------------------------------------------
    def on_predict(
        paths, package, package_compatible, device, stride, batch_size
    ):
        selected_paths = [
            str(Path(str(raw)).expanduser().resolve()) for raw in (paths or [])
        ]
        if package is None:
            raise gr.Error("Import a valid `.symmodel` package before predicting.")
        if not package_compatible:
            raise gr.Error(
                "The selected model package is incompatible with the Provider."
            )
        if not selected_paths:
            raise gr.Error("Select at least one valid prediction image.")

        events: Queue[tuple[str, Any]] = Queue()

        def report_progress(phase, current, total):
            events.put(("progress", (phase, current, total)))

        def execute() -> None:
            try:
                resolved_stride = require_whole_number(
                    stride, "Prediction stride", minimum=1
                )
                resolved_batch_size = require_whole_number(
                    batch_size, "Prediction batch size", minimum=1
                )
                batch = run_saved_model_prediction_batch(
                    config,
                    model_package=package.source_path,
                    image_paths=selected_paths,
                    device=None if device in {None, "auto"} else str(device),
                    stride=resolved_stride,
                    batch_size=resolved_batch_size,
                    progress_callback=report_progress,
                )
            except (PredictionError, ModelPackageError, ValueError) as error:
                events.put(("error", error))
            else:
                events.put(("result", batch))

        total_outputs = 10 + len(render_components)

        def progress_only(html: str):
            return (html,) + tuple(gr.skip() for _ in range(total_outputs - 1))

        html = progress_bar_html("Prediction", 0, 1, "Starting prediction...")
        yield (
            html,
            "Prediction started.",
            gr.update(interactive=False),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            *[gr.skip() for _ in render_components],
        )

        worker = Thread(target=execute, daemon=True)
        worker.start()
        result = None
        while result is None:
            try:
                event, payload = events.get(timeout=0.25)
            except Empty:
                continue
            if event == "progress":
                phase, current, total = payload
                html = progress_bar_html(
                    phase, current, max(total, 1), f"{current} of {max(total, 1)}"
                )
                yield progress_only(html)
            elif event == "error":
                html = progress_bar_html("Prediction", 0, 1, "Prediction failed.")
                yield (
                    html,
                    f"Prediction failed: {payload}",
                    gr.update(interactive=True),
                    None,
                    gr.update(choices=[], value=None),
                    gr.update(value=None, visible=False),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.update(visible=False),
                    *[gr.skip() for _ in render_components],
                )
                raise payload
            elif event == "result":
                result = payload

        enriched = dict(result)
        enriched["stride"] = int(stride)
        enriched["batch_size"] = int(batch_size)
        enriched["device"] = None if device in {None, "auto"} else str(device)
        enriched["classes"] = [
            {"index": index, "name": name, "color": color}
            for index, (name, color) in enumerate(
                zip(result.get("class_names", []), result.get("class_colors", []))
            )
        ]

        class_names = list(result.get("class_names", []))
        presentation = new_presentation_state(
            class_names,
            result.get("class_colors", []),
            alpha=DEFAULT_OVERLAY_ALPHA,
            run_id=result.get("run_id"),
        )
        colors_text = presentation_colors_text(presentation)

        choices = _item_choices(enriched)
        completed = [
            entry for entry in result["items"] if entry.get("status") == "completed"
        ]
        first = (
            completed[0]["item_id"]
            if completed
            else (choices[0][1] if choices else None)
        )
        presentation["selected_result_item"] = first

        render = render_selected_item(
            first,
            enriched,
            colors_text=colors_text,
            alpha=DEFAULT_OVERLAY_ALPHA,
            class_names=class_names,
        )
        html = progress_bar_html("Prediction", 1, 1, "Prediction complete.")
        status = (
            f"Completed `{result['run_id']}`: "
            f"{result['counts']['completed']} succeeded, "
            f"{result['counts']['failed']} failed."
        )
        yield (
            html,
            status,
            gr.update(interactive=True),
            enriched,
            gr.update(choices=choices, value=first),
            gr.update(value=result["download_archive"], visible=True),
            presentation,
            gr.update(value=colors_text),
            gr.update(value=DEFAULT_OVERLAY_ALPHA),
            gr.update(visible=True),
            *_render_tuple(render),
        )

    predict_button.click(
        on_predict,
        inputs=[
            image_files,
            package_state,
            package_compatible_state,
            predict_device,
            predict_stride,
            predict_batch,
        ],
        outputs=[
            prediction_progress,
            prediction_status,
            predict_button,
            result_state,
            result_selector,
            batch_download,
            presentation_state,
            presentation_colors,
            presentation_alpha,
            result_section,
            *render_components,
        ],
        show_progress="hidden",
    )
    return package_state, result_state, presentation_state
