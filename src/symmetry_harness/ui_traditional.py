"""Standalone traditional-ML validation page.

This module owns its own ``gr.Blocks`` root and is never inserted into the
normal two-workspace application. It is reachable only through the explicit
``symmetry validate-traditional`` command or its bundled launcher script.

The page always builds the existing eight-channel symmetry representation
through the Provider's ``compute_features`` operation, then trains one
conventional scikit-learn classifier on user-selected patches and densely
predicts the full image. No pretrained checkpoint is required or accessed.
"""

from __future__ import annotations

import json
from pathlib import Path
from queue import Empty, Queue
import shutil
import tempfile
from threading import Thread
from typing import Any, Callable

try:  # pragma: no cover - depends on the optional UI extra
    # Gradio injects event data by resolving this annotation at runtime, so the
    # class must be bound to a module-level name.
    from gradio.events import SelectData
except ImportError:  # pragma: no cover - exercised without the UI extra
    SelectData = None  # type: ignore[assignment,misc]

import numpy as np
from PIL import Image

from .annotations import create_annotation_session
from .config import HarnessConfig
from .contracts import (
    TRADITIONAL_CLASSIFIER_IDENTIFIERS,
    TRADITIONAL_DEFAULT_CLASSIFIER_PATCH_SIZE,
    TRADITIONAL_DEFAULT_SEED,
    TRADITIONAL_FEATURE_MODES,
    TRADITIONAL_MAXIMUM_PATCHES_PER_CLASS,
    TRADITIONAL_MINIMUM_PATCHES_PER_CLASS,
    TraditionalMLSettings,
)
from .image_io import inspect_input
from .provider import (
    run_provider_features,
    traditional_classifier_defaults,
)
from .traditional_workflow import run_traditional_validation
from .ui_annotation import (
    add_point_state,
    annotation_rows,
    annotation_state,
    clear_class_state,
    configure_class_state,
    display_to_source_point,
    image_shape_text,
    render_annotations,
    source_patch_preview,
    state_patch_size,
    undo_point_state,
)
from .ui_shared import (
    APP_CSS,
    INVALID_SUPPORT_POINT_MESSAGE,
    progress_bar_html,
)
from .ui_fine_tune import (
    _feature_cache_paths,
    _feature_request,
    _load_cached_features,
    feature_gallery,
)


LOGISTIC_REGRESSION = "logistic_regression"
RANDOM_FOREST = "random_forest"

CLASSIFIER_LABELS = {
    LOGISTIC_REGRESSION: "Logistic Regression",
    RANDOM_FOREST: "Random Forest",
}
CLASSIFIER_CHOICES = [
    (CLASSIFIER_LABELS[identifier], identifier)
    for identifier in TRADITIONAL_CLASSIFIER_IDENTIFIERS
]

FEATURE_MODE_LABELS = {
    "raw_image": "Raw image only",
    "image_plus_symmetry_maps": "Image + symmetry maps (8 channels)",
}
FEATURE_MODE_CHOICES = [
    (FEATURE_MODE_LABELS[mode], mode) for mode in TRADITIONAL_FEATURE_MODES
]

# Fixed parameters are shown for transparency but never sent as overrides; the
# Provider registry fills them from its own contract.
CLASSIFIER_FIXED_PARAMETERS: dict[str, dict[str, Any]] = {
    LOGISTIC_REGRESSION: {"solver": "lbfgs"},
    RANDOM_FOREST: {
        "criterion": "gini",
        "bootstrap": True,
        "n_jobs": 1,
    },
}
CLASSIFIER_PARAMETER_KEYS: dict[str, tuple[str, ...]] = {
    LOGISTIC_REGRESSION: ("C", "class_weight", "max_iter"),
    RANDOM_FOREST: (
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "class_weight",
    ),
}

PROGRESS_PHASE_LABELS = {
    "training": "Training",
    "prediction": "Dense prediction",
}

SECTION_TITLES = (
    "1. Input image",
    "2. Compute and inspect symmetry maps",
    "3. Choose traditional ML input features",
    "4. Choose classifier and adjust its parameters",
    "5. Define classes",
    "6. Select training patches",
    "7. Configure dense prediction",
    "8. Train and predict",
    "9. Review and export results",
)


def classifier_panel_visibility(identifier: str) -> tuple[bool, bool]:
    """Return the (logistic regression, random forest) panel visibility."""
    key = str(identifier).strip()
    if key == LOGISTIC_REGRESSION:
        return True, False
    if key == RANDOM_FOREST:
        return False, True
    raise ValueError(
        f"Unsupported classifier {identifier!r}. "
        f"Supported classifiers are {list(TRADITIONAL_CLASSIFIER_IDENTIFIERS)}."
    )


def feature_mode_value(label: str) -> str:
    """Return the contract feature-mode value for one display label."""
    key = str(label).strip()
    if key in FEATURE_MODE_LABELS:
        return key
    for mode, text in FEATURE_MODE_LABELS.items():
        if key == text:
            return mode
    raise ValueError(f"Unsupported traditional ML feature mode label: {label!r}")


def classifier_parameters(identifier: str, values: dict[str, Any]) -> dict[str, Any]:
    """Return the explicit parameter mapping for one classifier."""
    key = str(identifier).strip()
    if key not in CLASSIFIER_PARAMETER_KEYS:
        raise ValueError(
            f"Unsupported classifier {identifier!r}. "
            f"Supported classifiers are {list(TRADITIONAL_CLASSIFIER_IDENTIFIERS)}."
        )
    return {name: values.get(name) for name in CLASSIFIER_PARAMETER_KEYS[key]}


def validate_classifier_parameters(
    identifier: str, parameters: dict[str, Any]
) -> dict[str, Any]:
    """Return ``parameters`` unchanged, after rejecting empty values.

    A cleared Gradio control submits ``None`` (or an empty string), which the
    Provider would otherwise reject with a raw traceback. Reporting the offending
    names here keeps the message actionable, as the protocol requires.
    """
    empty = sorted(
        name
        for name, value in parameters.items()
        if value is None or (isinstance(value, str) and not value.strip())
    )
    if empty:
        raise ValueError(
            f"Every {identifier} parameter needs a value before running. "
            f"Missing or empty: {', '.join(empty)}."
        )
    return parameters


def whole_number(value: Any, label: str, *, minimum: int) -> int:
    """Coerce one UI value to a whole number, reporting empty input clearly."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{label} is empty. Enter a whole number.")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a whole number.") from None
    if number < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return number


def support_is_complete(
    state: dict[str, Any],
    *,
    minimum_patches: int = TRADITIONAL_MINIMUM_PATCHES_PER_CLASS,
    maximum_patches: int = TRADITIONAL_MAXIMUM_PATCHES_PER_CLASS,
) -> bool:
    """Return whether every class has an allowed number of support patches."""
    groups = state.get("points", [])
    if state.get("image") is None or len(groups) < 2:
        return False
    return all(minimum_patches <= len(group) <= maximum_patches for group in groups)


def features_are_current(
    state: dict[str, Any],
    feature_state: dict[str, Any],
    config: HarnessConfig,
    capabilities: dict[str, Any],
    *,
    symmetry_patch_size: Any,
    device: str,
) -> bool:
    """Return whether the cached eight-channel artifact matches the page state.

    ``symmetry_patch_size`` is accepted as the raw control value: a cleared
    number box must disable the Run button, not raise.
    """
    if not feature_state or not feature_state.get("features_path"):
        return False
    if state.get("image") is None:
        return False
    try:
        _, _, _, expected_key = _feature_request(
            config,
            capabilities,
            state,
            model_identifier=config.model.identifier,
            symmetry_patch_size=symmetry_patch_size,
            device=str(device),
        )
    except (KeyError, TypeError, ValueError):
        return False
    if feature_state.get("cache_key") != expected_key:
        return False
    return bool(
        Path(str(feature_state.get("features_path", ""))).is_file()
        and Path(str(feature_state.get("record_path", ""))).is_file()
    )


def run_is_enabled(
    state: dict[str, Any],
    feature_state: dict[str, Any],
    config: HarnessConfig,
    capabilities: dict[str, Any],
    *,
    symmetry_patch_size: Any,
    device: str,
) -> bool:
    """Return whether the primary Run button may be enabled."""
    if state.get("image") is None:
        return False
    if not support_is_complete(state):
        return False
    return features_are_current(
        state,
        feature_state,
        config,
        capabilities,
        symmetry_patch_size=symmetry_patch_size,
        device=device,
    )


def _configured_capabilities(config: HarnessConfig) -> dict[str, Any]:
    """Build an offline fallback catalog for UI-construction tests."""
    patch_size = int(config.model.classifier_patch_size or
                     TRADITIONAL_DEFAULT_CLASSIFIER_PATCH_SIZE)
    return {
        "contract_version": "symmetry-learn-provider-v1",
        "provider": config.provider.name,
        "provider_version": "0.0.0",
        "operations": [
            "compute_features",
            "traditional_ml_analyze",
        ],
        "models": [
            {
                "identifier": config.model.identifier,
                "display_name": config.model.identifier,
                "available": True,
                "input_channels": config.model.input_channels,
                "pretrained_classes": config.model.pretrained_classes,
                "classifier_patch_size": patch_size,
                "feature_channels": [
                    "image",
                    "reflection_strength",
                    "reflection_sin_2theta",
                    "reflection_cos_2theta",
                    "rotation_2_fold",
                    "rotation_3_fold",
                    "rotation_4_fold",
                    "rotation_6_fold",
                ],
                "feature_pipeline": "eight_channel_v1",
                "fine_tuning_strategy": "Provider-managed fine-tuning",
                "minimum_shots_per_class": config.fine_tuning.minimum_shots_per_class,
                "recommended_shots_per_class": (
                    config.fine_tuning.recommended_shots_per_class
                ),
                "maximum_shots_per_class": config.fine_tuning.maximum_shots_per_class,
                "defaults": {
                    "n_max": config.features.n_max,
                    "symmetry_patch_size": config.features.symmetry_patch_size,
                    "rotation_folds": list(config.features.rotation_folds),
                    "reflection_p": config.features.reflection_p,
                    "normalize_rotation_maps": config.features.normalize_rotation_maps,
                },
                "weights": [],
                "default_weight": None,
            }
        ],
        "traditional_ml": {
            "schema_version": "symmetry-traditional-ml-capability-v1",
            "feature_modes": list(TRADITIONAL_FEATURE_MODES),
            "classifiers": [
                {
                    "identifier": LOGISTIC_REGRESSION,
                    "supports_predict_proba": True,
                    "defaults": {
                        "C": 1.0,
                        "class_weight": "balanced",
                        "max_iter": 1000,
                        "solver": "lbfgs",
                    },
                },
                {
                    "identifier": RANDOM_FOREST,
                    "supports_predict_proba": True,
                    "defaults": {
                        "n_estimators": 300,
                        "max_depth": 12,
                        "min_samples_leaf": 2,
                        "max_features": "sqrt",
                        "class_weight": "balanced",
                        "criterion": "gini",
                        "bootstrap": True,
                        "n_jobs": 1,
                    },
                },
            ],
        },
    }


def _classifier_defaults(
    capabilities: dict[str, Any], identifier: str
) -> dict[str, Any]:
    """Return Provider-declared defaults, falling back to the offline catalog."""
    try:
        return traditional_classifier_defaults(capabilities, identifier)
    except (KeyError, TypeError, ValueError):
        return dict(_FALLBACK_CLASSIFIER_DEFAULTS[str(identifier)])


_FALLBACK_CLASSIFIER_DEFAULTS: dict[str, dict[str, Any]] = {
    LOGISTIC_REGRESSION: {
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 1000,
        "solver": "lbfgs",
    },
    RANDOM_FOREST: {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "criterion": "gini",
        "bootstrap": True,
        "n_jobs": 1,
    },
}


def _downloadable_archive(archive_path: str | Path, run_id: str) -> str:
    """Copy a run archive into the system temporary directory for downloads.

    Gradio only serves files inside the working directory, the system temporary
    directory, or an explicitly allowed path. Run directories live under the
    Harness output root and match none of those, so a browser download needs a
    temporary copy. The run directory stays the durable record.
    """
    destination_directory = Path(tempfile.gettempdir()) / "symmetry-harness-downloads"
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / f"{run_id}-traditional-results.zip"
    shutil.copyfile(archive_path, destination)
    return str(destination)


def build_traditional_app(
    config: HarnessConfig,
    *,
    prepared_input: tuple[np.ndarray, dict[str, Any]] | None = None,
    capabilities: dict[str, Any] | None = None,
):
    """Construct the standalone traditional validation application."""
    import gradio as gr

    catalog = capabilities or _configured_capabilities(config)
    lr_defaults = _classifier_defaults(catalog, LOGISTIC_REGRESSION)
    rf_defaults = _classifier_defaults(catalog, RANDOM_FOREST)

    initial_state: dict[str, Any] = {}
    initial_annotation = None
    initial_status = "Load a single-channel image to begin."
    initial_file = None
    initial_input_record = None
    if prepared_input is not None:
        initial_image, initial_record = prepared_input
        initial_state, load_status = annotation_state(
            initial_image,
            initial_record,
            classifier_patch_size=int(config.model.classifier_patch_size),
        )
        initial_annotation = render_annotations(
            initial_state, int(config.model.classifier_patch_size)
        )
        initial_status = load_status
        initial_file = initial_record["path"]
        initial_input_record = initial_record

    feature_cache_directory = tempfile.TemporaryDirectory(
        prefix="symmetry-harness-traditional-features-"
    )
    feature_cache_root = Path(feature_cache_directory.name)

    with gr.Blocks(title="Symmetry Harness - Traditional ML validation") as app:
        state = gr.State(initial_state)
        feature_state = gr.State({})

        gr.Markdown(
            "# Traditional ML validation\n"
            "Train a conventional scikit-learn classifier on user-selected image "
            "patches and densely predict the full image. This page is an explicit "
            "validation tool: it never loads or requires a pretrained checkpoint. "
            "Conclusions drawn from it are exploratory, not a formal proof that a "
            "pretrained model is necessary."
        )
        status = gr.Markdown(initial_status)

        # --- 1. Input image -----------------------------------------------------
        gr.Markdown(f"## {SECTION_TITLES[0]}")
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
                image_shape_text(initial_input_record), min_width=220, scale=1
            )

        # --- 2. Symmetry maps ---------------------------------------------------
        gr.Markdown(f"## {SECTION_TITLES[1]}")
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                with gr.Row():
                    symmetry_patch_size = gr.Number(
                        label="Symmetry patch size",
                        value=config.features.symmetry_patch_size,
                        precision=0,
                        info=(
                            "Odd local-neighborhood size used to calculate the "
                            "symmetry maps."
                        ),
                    )
                    device = gr.Dropdown(
                        label="Map-computation device",
                        choices=["auto", "cuda", "cpu"],
                        value=config.model.device,
                        info="Used only for symmetry-map computation.",
                    )
            with gr.Column(scale=1, min_width=320):
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
                    elem_id="traditional-compute-status",
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

        # --- 3. Feature mode ----------------------------------------------------
        gr.Markdown(f"## {SECTION_TITLES[2]}")
        feature_mode = gr.Radio(
            label="Traditional ML input features",
            choices=FEATURE_MODE_CHOICES,
            value="image_plus_symmetry_maps",
            info=(
                "Raw image mode uses channel 0 of the same normalized eight-channel "
                "representation. Symmetry-map mode uses all eight channels."
            ),
        )

        # --- 4. Classifier and parameters --------------------------------------
        gr.Markdown(f"## {SECTION_TITLES[3]}")
        classifier_selector = gr.Radio(
            label="Classifier",
            choices=CLASSIFIER_CHOICES,
            value=LOGISTIC_REGRESSION,
        )
        with gr.Column(visible=True) as lr_panel:
            gr.Markdown(
                "**Logistic Regression** — always wrapped in a training-only "
                "`StandardScaler` pipeline."
            )
            with gr.Row():
                lr_c = gr.Number(label="C (regularization strength)", value=lr_defaults["C"])
                lr_class_weight = gr.Radio(
                    label="class_weight",
                    choices=["balanced", "none"],
                    value=lr_defaults["class_weight"],
                )
                lr_max_iter = gr.Number(
                    label="max_iter (advanced)",
                    value=lr_defaults["max_iter"],
                    precision=0,
                )
            gr.Markdown(
                f"Fixed parameters: `solver={CLASSIFIER_FIXED_PARAMETERS[LOGISTIC_REGRESSION]['solver']}`"
            )
        with gr.Column(visible=False) as rf_panel:
            gr.Markdown("**Random Forest** — one job for strict reproducibility.")
            with gr.Row():
                rf_n_estimators = gr.Number(
                    label="n_estimators", value=rf_defaults["n_estimators"], precision=0
                )
                rf_max_depth = gr.Textbox(
                    label="max_depth",
                    value=str(rf_defaults["max_depth"]),
                    info="A positive integer, or 'none' for unlimited depth.",
                )
                rf_min_samples_leaf = gr.Number(
                    label="min_samples_leaf",
                    value=rf_defaults["min_samples_leaf"],
                    precision=0,
                )
            with gr.Row():
                rf_max_features = gr.Radio(
                    label="max_features",
                    choices=["sqrt", "log2", "all"],
                    value=rf_defaults["max_features"],
                )
                rf_class_weight = gr.Radio(
                    label="class_weight",
                    choices=["balanced", "none"],
                    value=rf_defaults["class_weight"],
                )
            gr.Markdown(
                "Fixed parameters: "
                f"`criterion={CLASSIFIER_FIXED_PARAMETERS[RANDOM_FOREST]['criterion']}`, "
                f"`bootstrap={CLASSIFIER_FIXED_PARAMETERS[RANDOM_FOREST]['bootstrap']}`, "
                f"`n_jobs={CLASSIFIER_FIXED_PARAMETERS[RANDOM_FOREST]['n_jobs']}`"
            )
        with gr.Row():
            seed = gr.Number(
                label="Global seed",
                value=TRADITIONAL_DEFAULT_SEED,
                precision=0,
                info="Recorded for every run and controls every stochastic operation.",
            )
            classifier_patch_size = gr.Number(
                label="Classifier patch size",
                value=int(config.model.classifier_patch_size),
                precision=0,
                info=(
                    "Belongs to this validation experiment. Changing it clears all "
                    "support patches."
                ),
            )

        # --- 5. Define classes --------------------------------------------------
        gr.Markdown(f"## {SECTION_TITLES[4]}")
        with gr.Row():
            class_names = gr.Textbox(
                label="Local class names",
                value="Class A, Class B",
                info="Enter comma-separated names in the intended local-label order.",
                scale=2,
            )
            configure_button = gr.Button(
                "Configure classes and patch size", variant="secondary", scale=1
            )

        # --- 6. Select training patches ----------------------------------------
        gr.Markdown(f"## {SECTION_TITLES[5]}")
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                active_class = gr.Radio(
                    label="Active class", choices=[], interactive=True
                )
                annotation_image = gr.Image(
                    label="Click support patches",
                    type="numpy",
                    format="png",
                    interactive=True,
                    buttons=["fullscreen"],
                    value=initial_annotation,
                    height=720,
                    elem_id="traditional-annotation-image",
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
                    undo_button = gr.Button("Undo last patch")
                    clear_button = gr.Button("Clear active class")

        # --- 7. Dense prediction settings --------------------------------------
        gr.Markdown(f"## {SECTION_TITLES[6]}")
        with gr.Row():
            stride = gr.Number(
                label="Prediction stride", value=config.prediction.stride, precision=0
            )
            batch_size = gr.Number(
                label="Prediction batch size",
                value=config.prediction.batch_size,
                precision=0,
            )

        # --- 8. Train and predict ----------------------------------------------
        gr.Markdown(f"## {SECTION_TITLES[7]}")
        run_button = gr.Button(
            "Train traditional ML and predict", variant="primary", interactive=False
        )
        training_progress = gr.HTML(
            progress_bar_html("Training", 0, 1, "Waiting to start.")
        )
        prediction_progress = gr.HTML(
            progress_bar_html("Dense prediction", 0, 1, "Waiting for training.")
        )

        # --- 9. Results ---------------------------------------------------------
        gr.Markdown(f"## {SECTION_TITLES[8]}")
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
        result_json = gr.JSON(label="Run summary")
        summary_table = gr.Dataframe(
            headers=["Class", "Predicted cells", "Fraction"],
            datatype=["str", "number", "number"],
            interactive=False,
            label="Per-class prediction statistics",
        )
        with gr.Row():
            archive_download = gr.DownloadButton(
                "Download results archive", value=None, visible=False
            )
            run_directory_text = gr.Markdown("**Run directory:** —")

        result_outputs = [
            prediction_overlay,
            confidence_image,
            entropy_image,
            result_json,
            summary_table,
            archive_download,
            run_directory_text,
        ]
        # The progress bars belong to the same group as the results: whenever a
        # completed result is invalidated the progress display must return to its
        # idle state, otherwise the page shows a finished bar next to empty
        # results. Protocol section 7.1 also requires a new image to clear run
        # status and progress.
        progress_outputs = [training_progress, prediction_progress]
        invalidation_outputs = [*result_outputs, *progress_outputs]

        def _cleared_results():
            """Return one cleared value per entry in ``invalidation_outputs``."""
            return (
                None,
                None,
                None,
                None,
                [],
                gr.update(value=None, visible=False),
                "**Run directory:** \u2014",
                progress_bar_html("Training", 0, 1, "Waiting to start."),
                progress_bar_html(
                    "Dense prediction", 0, 1, "Waiting for training."
                ),
            )

        def _kept_results():
            """Return one no-op update per entry in ``invalidation_outputs``.

            Used when an event leaves the experiment untouched, for example when
            a click lands outside the valid centre region and no support point is
            actually added.
            """
            return tuple(gr.update() for _ in invalidation_outputs)

        # --- input image --------------------------------------------------------
        def on_image_change(path, current):
            try:
                if not path:
                    raise ValueError("Choose an input image first.")
                image, record = inspect_input(
                    path, config.features.input_normalization
                )
                updated, message = annotation_state(
                    image,
                    record,
                    classifier_patch_size=int(config.model.classifier_patch_size),
                    previous_state=current,
                )
                annotated = render_annotations(
                    updated, int(config.model.classifier_patch_size)
                )
            except Exception as error:
                return (
                    None,
                    {},
                    gr.update(choices=[], value=None),
                    [],
                    None,
                    image_shape_text(None),
                    f"Image validation failed: {error}",
                    {},
                    gr.update(value=None, visible=False),
                    "Load a valid image before computing symmetry maps.",
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    *(
                        _cleared_results()
                    ),
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
                image_shape_text(record),
                message
                + " Selecting another image clears support patches, maps, and results.",
                {},
                gr.update(value=None, visible=False),
                "Input changed; compute the eight-channel symmetry maps.",
                gr.update(interactive=True),
                gr.update(interactive=False),
                *_cleared_results(),
            )

        input_file.change(
            on_image_change,
            inputs=[input_file, state],
            outputs=[
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
                compute_features_button,
                run_button,
                *invalidation_outputs,
            ],
        )

        # --- symmetry-map settings ---------------------------------------------
        def on_feature_setting_change(current):
            message = (
                "Symmetry-map settings changed; compute the maps again. Support "
                "patches are preserved."
                if current.get("image") is not None
                else "Load an image, then compute its eight-channel representation."
            )
            return (
                {},
                gr.update(value=None, visible=False),
                message,
                gr.update(interactive=False),
                *_cleared_results(),
            )

        for trigger in (symmetry_patch_size, device):
            trigger.change(
                on_feature_setting_change,
                inputs=[state],
                outputs=[
                    feature_state,
                    feature_maps,
                    feature_status,
                    run_button,
                    *invalidation_outputs,
                ],
            )

        # --- compute features ---------------------------------------------------
        def on_compute_features(current, sym_size, run_device):
            _, options, fingerprint, cache_key = _feature_request(
                config,
                catalog,
                current,
                model_identifier=config.model.identifier,
                symmetry_patch_size=sym_size,
                device=str(run_device),
            )
            features_path, record_path = _feature_cache_paths(
                feature_cache_root, cache_key
            )
            expected_shape = (
                int(config.model.input_channels),
                int(current["image_shape"][0]),
                int(current["image_shape"][1]),
            )
            cache_hit = False
            try:
                features, channel_names, feature_record = _load_cached_features(
                    features_path, record_path, expected_shape=expected_shape
                )
                if feature_record.get("harness_cache_fingerprint") != fingerprint:
                    raise RuntimeError("Cached feature fingerprint mismatch.")
                cache_hit = True
            except (FileNotFoundError, KeyError, OSError, ValueError, RuntimeError):
                result = run_provider_features(
                    config,
                    np.asarray(current["image"], dtype=np.float32),
                    options=options,
                )
                features = result.features
                channel_names = result.channel_names
                if features.shape != expected_shape:
                    raise RuntimeError(
                        "Provider features do not match the eight-channel contract."
                    )
                feature_record = {
                    **result.record,
                    "harness_cache_fingerprint": fingerprint,
                }
                features_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    features_path, features=features, channel_names=channel_names
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
                f"{source} eight-channel symmetry maps with shape {tuple(features.shape)}."
            )
            feature_message = (
                "Symmetry maps are ready (reused cached result)."
                if cache_hit
                else "Symmetry maps have been computed."
            )
            enabled = run_is_enabled(
                current,
                current_features,
                config,
                catalog,
                symmetry_patch_size=sym_size,
                device=str(run_device),
            )
            return (
                gr.update(value=feature_gallery(features, channel_names), visible=True),
                current_features,
                message,
                feature_message,
                gr.update(interactive=enabled),
            )

        compute_started = compute_features_button.click(
            lambda: "Computing symmetry maps...",
            inputs=[],
            outputs=[feature_status],
            queue=False,
            show_progress="hidden",
        )
        compute_started.then(
            on_compute_features,
            inputs=[state, symmetry_patch_size, device],
            outputs=[feature_maps, feature_state, status, feature_status, run_button],
        )

        # --- feature mode -------------------------------------------------------
        def on_feature_mode_change(mode_label, current, current_features, sym_size, run_device):
            mode = feature_mode_value(mode_label)
            message = (
                f"Feature mode set to `{mode}`. Existing support patches are "
                "preserved; run again to produce new results."
            )
            enabled = run_is_enabled(
                current,
                current_features,
                config,
                catalog,
                symmetry_patch_size=sym_size,
                device=str(run_device),
            )
            return (message, gr.update(interactive=enabled), *_cleared_results())

        feature_mode.change(
            on_feature_mode_change,
            inputs=[feature_mode, state, feature_state, symmetry_patch_size, device],
            outputs=[status, run_button, *invalidation_outputs],
        )

        # --- classifier ---------------------------------------------------------
        def on_classifier_change(identifier, current, current_features, sym_size, run_device):
            lr_visible, rf_visible = classifier_panel_visibility(identifier)
            message = (
                f"Classifier set to `{identifier}`. Support patches and maps are "
                "preserved; run again to produce new results."
            )
            enabled = run_is_enabled(
                current,
                current_features,
                config,
                catalog,
                symmetry_patch_size=sym_size,
                device=str(run_device),
            )
            return (
                gr.update(visible=lr_visible),
                gr.update(visible=rf_visible),
                message,
                gr.update(interactive=enabled),
                *_cleared_results(),
            )

        classifier_selector.change(
            on_classifier_change,
            inputs=[classifier_selector, state, feature_state, symmetry_patch_size, device],
            outputs=[lr_panel, rf_panel, status, run_button, *invalidation_outputs],
        )

        # --- shared settings ----------------------------------------------------
        def on_patch_size_change(selected_patch_size, current):
            """Apply a new classifier patch size, or invalidate it cleanly.

            ``classifier_patch_size`` is a ``gr.Number``, so the user can clear
            it.  A cleared box leaves no patch geometry: drop the support
            patches and every dependent result, fall back to the configured
            patch size for rendering, and keep Run disabled.  Raising instead
            would leave stale annotations and results on screen.
            """
            try:
                patch_size = whole_number(
                    selected_patch_size, "Classifier patch size", minimum=1
                )
            except ValueError as error:
                cleared = dict(current)
                cleared["points"] = [[] for _ in current.get("class_names", [])]
                retained = current.get("classifier_patch_size")
                if not isinstance(retained, int) or retained <= 0:
                    retained = int(config.model.classifier_patch_size)
                cleared["classifier_patch_size"] = retained
                cleared["show_valid_region"] = False
                return (
                    cleared,
                    render_annotations(cleared, retained),
                    annotation_rows(cleared),
                    None,
                    str(error),
                    gr.update(interactive=False),
                    *_cleared_results(),
                )
            updated = dict(current)
            updated["points"] = [[] for _ in current.get("class_names", [])]
            updated["classifier_patch_size"] = patch_size
            updated["show_valid_region"] = False
            message = (
                f"Classifier patch size set to {patch_size}. All support patches were "
                "cleared because valid centers and patch contents changed."
            )
            return (
                updated,
                render_annotations(updated, patch_size),
                annotation_rows(updated),
                None,
                message,
                gr.update(interactive=False),
                *_cleared_results(),
            )

        classifier_patch_size.change(
            on_patch_size_change,
            inputs=[classifier_patch_size, state],
            outputs=[
                state,
                annotation_image,
                support_table,
                patch_preview,
                status,
                run_button,
                *invalidation_outputs,
            ],
        )

        def on_seed_or_prediction_change(current, current_features, sym_size, run_device):
            message = "Settings changed; run again to produce updated results."
            enabled = run_is_enabled(
                current,
                current_features,
                config,
                catalog,
                symmetry_patch_size=sym_size,
                device=str(run_device),
            )
            return (message, gr.update(interactive=enabled), *_cleared_results())

        for trigger in (seed, stride, batch_size, lr_c, lr_class_weight, lr_max_iter,
                        rf_n_estimators, rf_max_depth, rf_min_samples_leaf,
                        rf_max_features, rf_class_weight):
            trigger.change(
                on_seed_or_prediction_change,
                inputs=[state, feature_state, symmetry_patch_size, device],
                outputs=[status, run_button, *invalidation_outputs],
            )

        # --- classes ------------------------------------------------------------
        def on_configure_classes(value, selected_patch_size, current):
            updated, patch_size = configure_class_state(
                value,
                current,
                classifier_patch_size=whole_number(
                    selected_patch_size, "Classifier patch size", minimum=1
                ),
            )
            names = list(updated["class_names"])
            status_message = (
                f"Configured {len(names)} classes with a {patch_size}-pixel classifier "
                f"patch. Select an active class and click at least "
                f"{TRADITIONAL_MINIMUM_PATCHES_PER_CLASS} patches per class."
            )
            return (
                updated,
                gr.update(choices=names, value=names[0] if names else None),
                render_annotations(updated, patch_size),
                annotation_rows(updated),
                None,
                status_message,
                gr.update(interactive=False),
                *_cleared_results(),
            )

        configure_button.click(
            on_configure_classes,
            inputs=[class_names, classifier_patch_size, state],
            outputs=[
                state,
                active_class,
                annotation_image,
                support_table,
                patch_preview,
                status,
                run_button,
                *invalidation_outputs,
            ],
        )

        def on_select(
            current,
            active,
            current_features,
            sym_size,
            run_device,
            event: SelectData,  # type: ignore[valid-type]
        ):
            index = event.index
            if not isinstance(index, (list, tuple)) or len(index) != 2:
                raise ValueError("The image click did not provide a valid pixel coordinate.")
            source_point = display_to_source_point(
                (int(index[0]), int(index[1])), tuple(current["image_shape"])
            )
            updated, patch_size, point_status, invalid = add_point_state(
                current,
                active,
                source_point,
                fallback_patch_size=int(current["classifier_patch_size"]),
                maximum_points=TRADITIONAL_MAXIMUM_PATCHES_PER_CLASS,
            )
            enabled = run_is_enabled(
                updated,
                current_features,
                config,
                catalog,
                symmetry_patch_size=sym_size,
                device=str(run_device),
            )
            if invalid:
                gr.Info(INVALID_SUPPORT_POINT_MESSAGE, duration=1.5, title="Valid region")
                return (
                    updated,
                    render_annotations(updated, patch_size),
                    gr.update(),
                    annotation_rows(updated),
                    gr.update(),
                    gr.update(interactive=enabled),
                    *_kept_results(),
                )
            return (
                updated,
                render_annotations(updated, patch_size),
                source_patch_preview(updated, source_point, patch_size),
                annotation_rows(updated),
                point_status,
                gr.update(interactive=enabled),
                *_cleared_results(),
            )

        annotation_image.select(
            on_select,
            inputs=[state, active_class, feature_state, symmetry_patch_size, device],
            outputs=[
                state,
                annotation_image,
                patch_preview,
                support_table,
                status,
                run_button,
                *invalidation_outputs,
            ],
        )

        def on_undo(current, active, current_features, sym_size, run_device):
            updated, patch_size, _removed, message = undo_point_state(
                current,
                active,
                fallback_patch_size=int(current["classifier_patch_size"]),
            )
            enabled = run_is_enabled(
                updated,
                current_features,
                config,
                catalog,
                symmetry_patch_size=sym_size,
                device=str(run_device),
            )
            return (
                updated,
                render_annotations(updated, patch_size),
                annotation_rows(updated),
                None,
                message,
                gr.update(interactive=enabled),
                *_cleared_results(),
            )

        undo_button.click(
            on_undo,
            inputs=[state, active_class, feature_state, symmetry_patch_size, device],
            outputs=[
                state,
                annotation_image,
                support_table,
                patch_preview,
                status,
                run_button,
                *invalidation_outputs,
            ],
        )

        def on_clear(current, active):
            updated, patch_size, message = clear_class_state(
                current,
                active,
                fallback_patch_size=int(current["classifier_patch_size"]),
            )
            return (
                updated,
                render_annotations(updated, patch_size),
                annotation_rows(updated),
                None,
                message,
                gr.update(interactive=False),
                *_cleared_results(),
            )

        clear_button.click(
            on_clear,
            inputs=[state, active_class],
            outputs=[
                state,
                annotation_image,
                support_table,
                patch_preview,
                status,
                run_button,
                *invalidation_outputs,
            ],
        )

        # --- run ---------------------------------------------------------------
        def on_run(
            current,
            current_features,
            mode_label,
            identifier,
            sym_size,
            run_device,
            seed_value,
            patch_size_value,
            pred_stride,
            pred_batch,
            lr_c_value,
            lr_class_weight_value,
            lr_max_iter_value,
            rf_n_estimators_value,
            rf_max_depth_value,
            rf_min_samples_leaf_value,
            rf_max_features_value,
            rf_class_weight_value,
        ):
            if current.get("image") is None or not current.get("class_names"):
                raise ValueError("Load an image and configure classes before running.")
            if not support_is_complete(current):
                raise ValueError(
                    "Every class needs at least "
                    f"{TRADITIONAL_MINIMUM_PATCHES_PER_CLASS} valid support patches."
                )
            resolved_symmetry_patch = whole_number(
                sym_size, "Symmetry patch size", minimum=1
            )
            resolved_patch_size = whole_number(
                patch_size_value, "Classifier patch size", minimum=1
            )
            resolved_seed = whole_number(seed_value, "Global seed", minimum=0)
            resolved_stride = whole_number(
                pred_stride, "Prediction stride", minimum=1
            )
            resolved_batch = whole_number(
                pred_batch, "Prediction batch size", minimum=1
            )
            if not features_are_current(
                current,
                current_features,
                config,
                catalog,
                symmetry_patch_size=resolved_symmetry_patch,
                device=str(run_device),
            ):
                raise ValueError(
                    "The symmetry maps are missing or stale. Compute them before running."
                )
            selected_patch_size = state_patch_size(current, resolved_patch_size)
            settings = TraditionalMLSettings(
                classifier_patch_size=selected_patch_size,
                seed=resolved_seed,
                stride=resolved_stride,
                batch_size=resolved_batch,
            )
            mode = feature_mode_value(mode_label)
            parameters = validate_classifier_parameters(
                identifier,
                classifier_parameters(
                    identifier,
                    {
                        "C": lr_c_value,
                        "class_weight": (
                            lr_class_weight_value
                            if identifier == LOGISTIC_REGRESSION
                            else rf_class_weight_value
                        ),
                        "max_iter": lr_max_iter_value,
                        "n_estimators": rf_n_estimators_value,
                        "max_depth": rf_max_depth_value,
                        "min_samples_leaf": rf_min_samples_leaf_value,
                        "max_features": rf_max_features_value,
                    },
                ),
            )
            session = create_annotation_session(
                image_path=current["image_path"],
                image_sha256=current["image_sha256"],
                image_shape=tuple(current["image_shape"]),
                classifier_patch_size=selected_patch_size,
                class_names=current["class_names"],
                points_by_class=current["points"],
                colors=current["colors"],
            )
            progress_events: Queue[tuple[str, Any]] = Queue()

            def report_progress(phase: str, completed: int, total: int) -> None:
                progress_events.put(("progress", (phase, completed, total)))

            def execute() -> None:
                try:
                    result = run_traditional_validation(
                        config,
                        image_path=current["image_path"],
                        annotation_session=session,
                        classifier=identifier,
                        feature_mode=mode,
                        settings=settings,
                        parameters=parameters,
                        features_path=current_features["features_path"],
                        features_record_path=current_features["record_path"],
                        progress_callback=report_progress,
                    )
                except Exception as error:  # noqa: BLE001 - forwarded to the UI
                    progress_events.put(("error", error))
                else:
                    progress_events.put(("result", result))

            training_html = progress_bar_html("Training", 0, 1, "Starting training...")
            prediction_html = progress_bar_html(
                "Dense prediction", 0, 1, "Waiting for training."
            )
            yield (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "Traditional ML run started.",
                training_html,
                prediction_html,
            )

            worker = Thread(target=execute, daemon=True)
            worker.start()
            result = None
            while result is None:
                try:
                    event, payload = progress_events.get(timeout=0.25)
                except Empty:
                    continue
                if event == "progress":
                    phase, completed, total = payload
                    if phase == "training":
                        phase_status = (
                            "Training complete."
                            if completed >= total
                            else f"Fitting the classifier ({completed}/{total})."
                        )
                        training_html = progress_bar_html(
                            PROGRESS_PHASE_LABELS["training"], completed, total, phase_status
                        )
                        if completed >= total:
                            prediction_html = progress_bar_html(
                                "Dense prediction", 0, 1, "Starting dense prediction..."
                            )
                    elif phase == "prediction":
                        training_html = progress_bar_html(
                            PROGRESS_PHASE_LABELS["training"], 1, 1, "Training complete."
                        )
                        phase_status = (
                            "Dense prediction complete."
                            if completed >= total
                            else f"Predicting batch {completed} of {total}."
                        )
                        prediction_html = progress_bar_html(
                            "Dense prediction", completed, total, phase_status
                        )
                    yield (
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        training_html,
                        prediction_html,
                    )
                elif event == "error":
                    training_html = progress_bar_html("Training", 0, 1, "Run failed.")
                    prediction_html = progress_bar_html(
                        "Dense prediction", 0, 1, "Run failed."
                    )
                    yield (
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        training_html,
                        prediction_html,
                    )
                    raise payload
                elif event == "result":
                    result = payload

            overlay = np.asarray(Image.open(result["prediction_overlay"]).convert("RGB"))
            confidence = np.asarray(Image.open(result["confidence"]).convert("RGB"))
            entropy = np.asarray(Image.open(result["entropy"]).convert("RGB"))
            statistics = result["class_statistics"]
            table_rows = [
                [
                    result["classes"][entry["index"]]["name"],
                    int(entry["count"]),
                    round(float(entry["fraction"]), 6),
                ]
                for entry in statistics["classes"]
            ]
            archive_value = _downloadable_archive(
                result["results_archive"], result["run_id"]
            )
            message = (
                f"Completed run `{result['run_id']}` with "
                f"`{result['classifier']}`. Artifacts were saved to "
                f"`{result['run_directory']}`."
            )
            yield (
                overlay,
                confidence,
                entropy,
                result,
                table_rows,
                gr.update(value=archive_value, visible=True),
                f"**Run directory:** `{result['run_directory']}`",
                message,
                progress_bar_html("Training", 1, 1, "Training complete."),
                progress_bar_html("Dense prediction", 1, 1, "Dense prediction complete."),
            )

        run_button.click(
            on_run,
            inputs=[
                state,
                feature_state,
                feature_mode,
                classifier_selector,
                symmetry_patch_size,
                device,
                seed,
                classifier_patch_size,
                stride,
                batch_size,
                lr_c,
                lr_class_weight,
                lr_max_iter,
                rf_n_estimators,
                rf_max_depth,
                rf_min_samples_leaf,
                rf_max_features,
                rf_class_weight,
            ],
            outputs=[
                prediction_overlay,
                confidence_image,
                entropy_image,
                result_json,
                summary_table,
                archive_download,
                run_directory_text,
                status,
                training_progress,
                prediction_progress,
            ],
            show_progress="hidden",
        )
    # The Gradio application holds the only reference to the feature cache for
    # the lifetime of the page, so the temporary directory is not collected
    # while the page is open.
    app._traditional_feature_cache = feature_cache_directory  # type: ignore[attr-defined]
    return app.queue(default_concurrency_limit=1)


def launch_traditional_ui(
    config: HarnessConfig,
    *,
    server_name: str,
    server_port: int | None,
    inbrowser: bool,
    prepared_input: tuple[np.ndarray, dict[str, Any]] | None = None,
    capabilities: dict[str, Any] | None = None,
    on_ready: Callable[[str], None] | None = None,
) -> None:
    """Launch the local-only traditional validation page."""
    from .ui import _protect_localhost_from_proxies, _resolve_server_port

    if server_name not in {"127.0.0.1", "localhost"}:
        raise ValueError("The traditional validation page only binds to localhost.")
    if capabilities is None:
        # Resolve the real Provider catalogue instead of silently using the
        # offline construction catalogue: the cached feature fingerprint records
        # the Provider version, and a placeholder version would stop the cache
        # from invalidating after a Provider upgrade.
        try:
            from .provider import provider_capabilities

            capabilities = provider_capabilities(config)
        except Exception:  # noqa: BLE001 - the page still opens with fallbacks
            capabilities = None
    _protect_localhost_from_proxies()
    resolved_port = _resolve_server_port(server_name, server_port)
    app = build_traditional_app(
        config, prepared_input=prepared_input, capabilities=capabilities
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


__all__ = [
    "CLASSIFIER_CHOICES",
    "CLASSIFIER_FIXED_PARAMETERS",
    "CLASSIFIER_LABELS",
    "CLASSIFIER_PARAMETER_KEYS",
    "FEATURE_MODE_CHOICES",
    "FEATURE_MODE_LABELS",
    "LOGISTIC_REGRESSION",
    "PROGRESS_PHASE_LABELS",
    "RANDOM_FOREST",
    "SECTION_TITLES",
    "build_traditional_app",
    "classifier_panel_visibility",
    "classifier_parameters",
    "feature_mode_value",
    "features_are_current",
    "launch_traditional_ui",
    "run_is_enabled",
    "support_is_complete",
    "validate_classifier_parameters",
    "whole_number",
]
