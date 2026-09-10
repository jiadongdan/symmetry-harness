"""Saved-model prediction workspace for the local symmetry interface.

This workspace has its own state. It never reuses fine-tuning state, and it
never exposes feature parameters as editable controls.
"""

from __future__ import annotations

from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

import numpy as np
from PIL import Image

from .config import HarnessConfig
from .model_package import (
    FineTunedModelPackage,
    ModelPackageError,
    inspect_model_package,
)
from .prediction_workflow import (
    PREDICTION_OPERATION,
    PredictionError,
    _check_provider_compatibility,
    run_saved_model_prediction_batch,
)
from .provider import provider_capabilities
from .ui_shared import class_legend_html, class_statistics_html, progress_bar_html


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


def _provider_supports_prediction(capabilities: dict[str, Any] | None) -> bool:
    if not capabilities:
        return False
    return PREDICTION_OPERATION in capabilities.get("operations", [])


def _overrides_are_valid(stride: Any, batch_size: Any) -> bool:
    try:
        return int(stride) > 0 and int(batch_size) > 0
    except (TypeError, ValueError):
        return False


def _load_package(
    path: str | None,
    *,
    config: HarnessConfig | None = None,
    capabilities: dict[str, Any] | None = None,
):
    """Return the validated package and a compatibility message."""
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
    result_state = gr.State(None)

    gr.Markdown("## Saved model")
    with gr.Row(equal_height=False):
        package_file = gr.File(
            label="Fine-tuned model package",
            type="filepath",
            file_types=[".symmodel"],
            scale=1,
        )
        package_details = gr.Markdown(initial_status, scale=2)

    gr.Markdown("## Prediction images")
    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
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
        with gr.Column(scale=1, min_width=260):
            predict_device = gr.Dropdown(
                label="Device",
                choices=["auto", "cuda", "cpu"],
                value="auto",
            )
            predict_stride = gr.Number(
                label="Prediction stride",
                value=4,
                precision=0,
            )
            predict_batch = gr.Number(
                label="Prediction batch size",
                value=512,
                precision=0,
            )

    predict_button = gr.Button(
        "Predict", variant="primary", interactive=False
    )
    prediction_progress = gr.HTML(
        progress_bar_html("Prediction", 0, 1, "Waiting to start.")
    )
    prediction_status = gr.Markdown("")
    feature_contract = gr.Markdown("")

    gr.Markdown("## Results")
    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            result_selector = gr.Dropdown(
                label="Predicted image", choices=[], interactive=True
            )
            with gr.Row():
                result_source = gr.Image(
                    label="Source image", type="numpy", interactive=False
                )
                result_overlay = gr.Image(
                    label="Prediction overlay", type="numpy", interactive=False
                )
            with gr.Row():
                result_confidence = gr.Image(
                    label="Confidence", type="numpy", interactive=False
                )
                result_entropy = gr.Image(
                    label="Predictive entropy", type="numpy", interactive=False
                )
        with gr.Column(scale=1, min_width=240):
            class_legend = gr.HTML()
            class_stats = gr.HTML()
            result_metadata = gr.JSON(label="Prediction record")
    batch_download = gr.DownloadButton("Download batch archive", visible=False)

    def on_package_change(path):
        package, compatibility, message = _load_package(
            path, config=config, capabilities=capabilities
        )
        compatible = compatibility in {"Ready", "Warning"}
        rows = _validate_inputs(package, None)
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
        return (
            package,
            compatible,
            details,
            rows,
            contract,
            gr.update(interactive=False),
            None,
            gr.update(choices=[], value=None),
            gr.update(visible=False),
            message,
        )

    package_file.change(
        on_package_change,
        inputs=[package_file],
        outputs=[
            package_state,
            package_compatible_state,
            package_details,
            input_table,
            feature_contract,
            predict_button,
            result_state,
            result_selector,
            batch_download,
            prediction_status,
        ],
    )

    def on_images_change(
        paths, package, package_compatible, device, stride, batch_size
    ):
        rows = _validate_inputs(package, paths)
        ready = (
            package is not None
            and bool(package_compatible)
            and supported
            and any(row["status"] == "valid" for row in rows)
            and _overrides_are_valid(stride, batch_size)
        )
        return rows, gr.update(interactive=bool(ready))

    image_files.change(
        on_images_change,
        inputs=[
            image_files,
            package_state,
            package_compatible_state,
            predict_device,
            predict_stride,
            predict_batch,
        ],
        outputs=[input_table, predict_button],
    )
    for control in (predict_device, predict_stride, predict_batch):
        control.change(
            on_images_change,
            inputs=[
                image_files,
                package_state,
                package_compatible_state,
                predict_device,
                predict_stride,
                predict_batch,
            ],
            outputs=[input_table, predict_button],
        )

    def on_select_result(item_id, result):
        if not result:
            return None, None, None, None, "", "", None
        entry = next(
            (item for item in result["items"] if item["item_id"] == item_id), None
        )
        if entry is None:
            return None, None, None, None, "", "", None
        if entry.get("status") != "completed":
            return (
                None,
                None,
                None,
                None,
                "",
                "",
                {"item_id": item_id, "status": "failed", "error": entry.get("error")},
            )
        overlay = np.asarray(
            Image.open(entry["artifact_prediction_overlay"]).convert("RGB")
        )
        confidence = np.asarray(Image.open(entry["artifact_confidence"]).convert("RGB"))
        entropy = np.asarray(Image.open(entry["artifact_entropy"]).convert("RGB"))
        source = np.asarray(
            Image.open(entry["artifact_input_preview"]).convert("RGB")
        )
        return (
            source,
            overlay,
            confidence,
            entropy,
            _legend_html(result),
            class_statistics_html(entry["class_statistics"]),
            entry,
        )

    def _legend_html(result) -> str:
        classes = result.get("classes", [])
        return class_legend_html(classes)

    result_selector.change(
        on_select_result,
        inputs=[result_selector, result_state],
        outputs=[
            result_source,
            result_overlay,
            result_confidence,
            result_entropy,
            class_legend,
            class_stats,
            result_metadata,
        ],
    )

    def on_predict(
        paths, package, package_compatible, device, stride, batch_size
    ):
        selected_paths = [
            str(Path(str(raw)).expanduser().resolve())
            for raw in (paths or [])
        ]
        if package is None:
            raise gr.Error("Import a valid `.symmodel` package before predicting.")
        if not package_compatible:
            raise gr.Error("The selected model package is incompatible with the Provider.")
        if not selected_paths:
            raise gr.Error("Select at least one valid prediction image.")
        events: Queue[tuple[str, Any]] = Queue()

        def report_progress(phase, current, total):
            events.put(("progress", (phase, current, total)))

        def execute() -> None:
            try:
                batch = run_saved_model_prediction_batch(
                    config,
                    model_package=package.source_path,
                    image_paths=selected_paths,
                    device=None if device in {None, "auto"} else str(device),
                    stride=int(stride),
                    batch_size=int(batch_size),
                    progress_callback=report_progress,
                )
            except (PredictionError, ModelPackageError, ValueError) as error:
                events.put(("error", error))
            else:
                events.put(("result", batch))

        html = progress_bar_html("Prediction", 0, 1, "Starting prediction...")
        yield (
            html,
            "Prediction started.",
            gr.update(interactive=False),
            gr.skip(),
            gr.skip(),
            gr.skip(),
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
                yield html, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
            elif event == "error":
                html = progress_bar_html("Prediction", 0, 1, "Prediction failed.")
                yield (
                    html,
                    f"Prediction failed: {payload}",
                    gr.update(interactive=False),
                    None,
                    gr.update(choices=[], value=None),
                    gr.update(visible=False),
                )
                raise payload
            elif event == "result":
                result = payload

        completed = [
            entry for entry in result["items"] if entry["status"] == "completed"
        ]
        choices = [entry["item_id"] for entry in result["items"]]
        first = completed[0]["item_id"] if completed else (
            choices[0] if choices else None
        )
        html = progress_bar_html("Prediction", 1, 1, "Prediction complete.")
        status = (
            f"Completed `{result['run_id']}`: "
            f"{result['counts']['completed']} succeeded, "
            f"{result['counts']['failed']} failed."
        )
        enriched = dict(result)
        enriched["classes"] = [
            {"index": index, "name": name, "color": color}
            for index, (name, color) in enumerate(
                zip(
                    result.get("class_names", []),
                    result.get("class_colors", []),
                )
            )
        ]
        yield (
            html,
            status,
            gr.update(interactive=bool(completed)),
            enriched,
            gr.update(choices=choices, value=first),
            gr.update(value=result["download_archive"], visible=True),
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
        ],
        show_progress="hidden",
    )
    return package_state, result_state
