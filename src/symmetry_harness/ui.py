"""Local Gradio interface for user-selected few-shot support points."""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .annotations import (
    DEFAULT_CLASS_COLORS,
    create_annotation_session,
    valid_center_bounds,
)
from .config import HarnessConfig
from .image_io import inspect_input, unit_to_uint8
from .workflow import run_analysis


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


def render_annotations(state: dict[str, Any], patch_size: int) -> np.ndarray | None:
    """Render support points and patch outlines over a display-only image copy."""
    image = state.get("image")
    if image is None:
        return None
    gray = unit_to_uint8(np.asarray(image, dtype=np.float32))
    base = Image.fromarray(gray, mode="L").convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    shade = ImageDraw.Draw(overlay)
    half = patch_size // 2
    width, height = base.size
    shade.rectangle((0, 0, width, half - 1), fill=(50, 50, 50, 95))
    shade.rectangle((0, height - half + 1, width, height), fill=(50, 50, 50, 95))
    shade.rectangle((0, 0, half - 1, height), fill=(50, 50, 50, 95))
    shade.rectangle((width - half + 1, 0, width, height), fill=(50, 50, 50, 95))
    base = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(base)
    for color, points in zip(state.get("colors", []), state.get("points", [])):
        for x, y in points:
            draw.rectangle(
                (x - half, y - half, x + half - 1, y + half - 1),
                outline=color,
                width=2,
            )
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="white")
    return np.asarray(base)


def source_patch_preview(
    state: dict[str, Any], point: tuple[int, int] | None, patch_size: int
) -> np.ndarray | None:
    """Return the raw normalized source patch for immediate annotation review."""
    if point is None or state.get("image") is None:
        return None
    x, y = point
    half = patch_size // 2
    patch = np.asarray(state["image"])[y - half : y + half, x - half : x + half]
    return unit_to_uint8(patch)


def _load_image(path: str | None, config: HarnessConfig):
    if not path:
        raise ValueError("Choose an input image first.")
    image, record = inspect_input(path, config.features.input_normalization)
    state = {
        "image": image,
        "image_path": record["path"],
        "image_sha256": record["sha256"],
        "image_shape": record["shape"],
        "class_names": [],
        "colors": [],
        "points": [],
    }
    status = (
        f"Loaded {Path(record['path']).name}: shape {tuple(record['shape'])}, "
        f"dtype {record['dtype']}. Configure local classes next."
    )
    return render_annotations(state, config.model.classifier_patch_size), state, status


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


def build_app(config: HarnessConfig):
    """Construct the local point-annotation and few-shot analysis interface."""
    import gradio as gr

    with gr.Blocks(title="Symmetry Harness") as app:
        state = gr.State({})
        gr.Markdown(
            "# Symmetry Harness\n"
            "Select three to five representative points per local class, review the "
            "full patch outlines, and run adapter-plus-head fine-tuning locally."
        )
        status = gr.Markdown("Load a single-channel image to begin.")
        with gr.Row():
            with gr.Column(scale=1):
                input_file = gr.File(label="Input image", type="filepath")
                class_names = gr.Textbox(
                    label="Local class names",
                    value="Class A, Class B",
                    info="Enter comma-separated names in the intended local-label order.",
                )
                configure_button = gr.Button("Configure classes", variant="secondary")
                active_class = gr.Dropdown(
                    label="Active class", choices=[], interactive=True
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
            with gr.Column(scale=2):
                annotation_image = gr.Image(
                    label="Click support points",
                    type="numpy",
                    interactive=True,
                    buttons=["fullscreen"],
                )
                patch_preview = gr.Image(
                    label="Selected source patch", type="numpy", interactive=False
                )

        gr.Markdown("## Fine-tuning and dense prediction")
        with gr.Row():
            symmetry_patch_size = gr.Number(
                label="Symmetry patch size",
                value=config.features.symmetry_patch_size,
                precision=0,
            )
            epochs = gr.Number(
                label="Fine-tuning epochs",
                value=config.fine_tuning.epochs,
                precision=0,
            )
            learning_rate = gr.Number(
                label="Learning rate", value=config.fine_tuning.learning_rate
            )
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
            device = gr.Dropdown(
                label="Device",
                choices=["auto", "cuda", "cpu"],
                value=config.model.device,
            )
        run_button = gr.Button("Fine-tune and predict", variant="primary")
        with gr.Row():
            prediction_overlay = gr.Image(
                label="Prediction overlay", type="numpy", interactive=False
            )
            confidence_image = gr.Image(
                label="Confidence diagnostic", type="numpy", interactive=False
            )
            entropy_image = gr.Image(
                label="Entropy diagnostic", type="numpy", interactive=False
            )
        result_json = gr.JSON(label="Run result")

        input_file.change(
            lambda path: _load_image(path, config),
            inputs=[input_file],
            outputs=[annotation_image, state, status],
        )
        configure_button.click(
            lambda value, current: _configure_classes(value, current, config),
            inputs=[class_names, state],
            outputs=[state, active_class, annotation_image, support_table, status],
        )

        def on_select(current, active, event: gr.SelectData):
            index = event.index
            if not isinstance(index, (list, tuple)) or len(index) != 2:
                raise ValueError("The image click did not provide a valid pixel coordinate.")
            return _add_point(current, active, (int(index[0]), int(index[1])), config)

        annotation_image.select(
            on_select,
            inputs=[state, active_class],
            outputs=[state, annotation_image, patch_preview, support_table, status],
        )
        undo_button.click(
            lambda current, active: _undo_point(current, active, config),
            inputs=[state, active_class],
            outputs=[state, annotation_image, support_table, status],
        )
        clear_button.click(
            lambda current, active: _clear_class(current, active, config),
            inputs=[state, active_class],
            outputs=[state, annotation_image, support_table, status],
        )

        def on_run(current, sym_size, epoch_count, lr, pred_stride, pred_batch, run_device):
            if current.get("image") is None or not current.get("class_names"):
                raise ValueError("Load an image and configure classes before running.")
            session = create_annotation_session(
                image_path=current["image_path"],
                image_sha256=current["image_sha256"],
                image_shape=tuple(current["image_shape"]),
                classifier_patch_size=config.model.classifier_patch_size,
                class_names=current["class_names"],
                points_by_class=current["points"],
                colors=current["colors"],
            )
            result = run_analysis(
                config,
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
    return app.queue(default_concurrency_limit=1)


def launch_ui(
    config: HarnessConfig,
    *,
    server_name: str,
    server_port: int | None,
    inbrowser: bool,
) -> None:
    """Launch a local-only interface without public sharing."""
    if server_name not in {"127.0.0.1", "localhost"}:
        raise ValueError("The v1 interface only binds to localhost.")
    app = build_app(config)
    app.launch(
        server_name=server_name,
        server_port=server_port,
        inbrowser=inbrowser,
        share=False,
        show_error=True,
    )
