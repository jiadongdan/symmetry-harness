from __future__ import annotations

import copy
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from symmetry_harness.config import load_harness_config
from symmetry_harness.prediction_workflow import PredictionError
from symmetry_harness.ui_readiness import evaluate_predict_readiness, has_blocker
from symmetry_harness.ui_predict import (
    _item_choices,
    _load_package,
    _overrides_are_valid,
    _package_status,
    _provider_supports_prediction,
    _readiness_snapshot,
    _saved_prediction_defaults,
    _stride_for_item,
    _validate_inputs,
    new_presentation_state,
    presentation_colors_text,
    render_selected_item,
    require_symmodel_path,
    split_colors_text,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

CAPABILITIES = {
    "contract_version": "symmetry-learn-provider-v1",
    "provider": "symmetry-learn",
    "provider_version": "0.1.0",
    "operations": ["few_shot_analyze", "predict_with_fine_tuned_model"],
    "models": [
        {
            "identifier": "cnn_8ch_pg17",
            "display_name": "Eight-channel CNN (PG17)",
            "available": True,
            "input_channels": 8,
            "pretrained_classes": 17,
            "classifier_patch_size": 64,
            "feature_pipeline": "eight_channel_v1",
            "feature_channels": [f"channel_{index}" for index in range(8)],
            "fine_tuning_strategy": "adapters",
            "minimum_shots_per_class": 3,
            "recommended_shots_per_class": 5,
            "maximum_shots_per_class": 50,
            "default_weight": {
                "identifier": "pg17-symmetry-v1",
                "version": "1.0.0",
                "sha256": "a" * 64,
                "distribution": "symmetry-learn-default-model",
                "status": "installed",
                "bundled": True,
                "default": True,
            },
            "weights": [
                {
                    "identifier": "pg17-symmetry-v1",
                    "version": "1.0.0",
                    "sha256": "a" * 64,
                    "distribution": "symmetry-learn-default-model",
                    "status": "installed",
                    "bundled": True,
                    "default": True,
                }
            ],
            "defaults": {
                "n_max": 20,
                "symmetry_patch_size": 51,
                "rotation_folds": [2, 3, 4, 6],
                "reflection_p": 2.0,
                "normalize_rotation_maps": False,
                "epochs": 150,
                "learning_rate": 0.0005,
                "weight_decay": 0.0,
                "seed": 42,
                "stride": 4,
                "batch_size": 512,
            },
        }
    ],
}


@pytest.fixture()
def config():
    return load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")


def _labels(app) -> list[str]:
    return [
        component.get("props", {}).get("label")
        for component in app.get_config_file()["components"]
    ]


def _by_key(items) -> dict:
    return {item.key: item for item in items}


def test_interface_builds_both_workspaces(config) -> None:
    gradio = pytest.importorskip("gradio")
    from symmetry_harness.ui import build_app

    app = build_app(config, capabilities=CAPABILITIES)

    labels = _labels(app)
    assert "Fine-tuned model package" in labels
    assert "Prediction images" in labels
    assert "Model input patch size" in labels
    tabs = [
        component
        for component in app.get_config_file()["components"]
        if component.get("type") == "tabitem"
    ]
    assert len(tabs) == 2


def _selected_tab(app) -> object:
    for component in app.get_config_file()["components"]:
        if component.get("type") == "tabs":
            return component.get("props", {}).get("selected")
    raise AssertionError("No tabs component was rendered.")


def test_launch_mode_selects_the_requested_tab_by_id(config) -> None:
    pytest.importorskip("gradio")
    from symmetry_harness.ui import (
        FINE_TUNE_TAB_ID,
        PREDICTION_MODE,
        PREDICTION_TAB_ID,
        build_app,
    )

    fine_tune = build_app(config, capabilities=CAPABILITIES, mode="fine-tune")
    prediction = build_app(config, capabilities=CAPABILITIES, mode=PREDICTION_MODE)

    # Gradio selects tabs by TabItem id. Passing an index silently falls back
    # to the first tab, so these must be ids and must differ per mode.
    assert _selected_tab(fine_tune) == FINE_TUNE_TAB_ID
    assert _selected_tab(prediction) == PREDICTION_TAB_ID
    assert _selected_tab(fine_tune) != _selected_tab(prediction)

    with pytest.raises(ValueError, match="Unknown launch mode"):
        build_app(config, capabilities=CAPABILITIES, mode="train")


def test_tab_ids_match_the_rendered_tab_items(config) -> None:
    pytest.importorskip("gradio")
    from symmetry_harness.ui import build_app

    app = build_app(config, capabilities=CAPABILITIES, mode="predict")
    tab_ids = {
        component.get("props", {}).get("id")
        for component in app.get_config_file()["components"]
        if component.get("type") == "tabitem"
    }
    assert _selected_tab(app) in tab_ids


def test_prediction_state_is_independent_from_fine_tuning(config) -> None:
    pytest.importorskip("gradio")
    from symmetry_harness.ui import build_app

    app = build_app(config, capabilities=CAPABILITIES)

    states = [
        component
        for component in app.get_config_file()["components"]
        if component.get("type") == "state"
    ]
    # The fine-tuning workspace keeps its own state objects; the prediction
    # workspace adds package, status, result and presentation state.
    assert len(states) >= 5


def test_prediction_controls_and_result_images_are_aligned(config) -> None:
    gradio = pytest.importorskip("gradio")
    from symmetry_harness.ui_predict import build_prediction_workspace

    with gradio.Blocks() as app:
        build_prediction_workspace(config, capabilities=CAPABILITIES)
    components = app.get_config_file()["components"]
    by_label = {
        component.get("props", {}).get("label"): component.get("props", {})
        for component in components
    }
    assert by_label["Device"].get("info")
    assert by_label["Prediction stride (saved default)"].get("info")
    assert by_label["Prediction batch size (saved default)"].get("info")
    for label in (
        "Source image",
        "Prediction overlay",
        "Confidence (fixed 0-1 colorbar)",
        "Predictive entropy (fixed 0-ln(N) colorbar)",
    ):
        assert by_label[label].get("height") == 480


def test_color_and_alpha_callbacks_only_refresh_overlay_image(config) -> None:
    gradio = pytest.importorskip("gradio")
    from symmetry_harness.ui_predict import build_prediction_workspace

    with gradio.Blocks() as app:
        build_prediction_workspace(config, capabilities=CAPABILITIES)
    handlers = [
        block_fn
        for block_fn in app.fns.values()
        if getattr(block_fn.fn, "__name__", "") == "on_presentation_render"
    ]
    assert len(handlers) == 2
    for handler in handlers:
        labels = [getattr(component, "label", None) for component in handler.outputs]
        assert "Prediction overlay" in labels
        assert "Source image" not in labels
        assert "Confidence (fixed 0-1 colorbar)" not in labels
        assert "Predictive entropy (fixed 0-ln(N) colorbar)" not in labels


def test_package_loading_reports_validation_and_metadata(
    model_package_factory,
) -> None:
    package, compatibility, message = _load_package(str(model_package_factory()))

    assert package is not None
    assert compatibility == "Ready"
    assert package.task_classes == 2

    missing, blocked, blocked_message = _load_package(None)
    assert missing is None
    assert blocked == "Waiting"

    corrupt = model_package_factory(
        "corrupt.symmodel", model_state=b"tampered", break_checksum=True
    )
    package, blocked, blocked_message = _load_package(str(corrupt))
    assert package is None
    assert blocked == "Blocked"
    assert "checksum" in blocked_message


def test_package_loading_blocks_provider_incompatibility(
    config, model_package_factory
) -> None:
    incompatible = dict(CAPABILITIES)
    incompatible["models"] = [dict(CAPABILITIES["models"][0])]
    incompatible["models"][0]["feature_pipeline"] = "other_pipeline"

    package, compatibility, message = _load_package(
        str(model_package_factory()),
        config=config,
        capabilities=incompatible,
    )

    assert package is not None
    assert compatibility == "Blocked"
    assert "feature pipeline" in message


def test_input_validation_uses_the_saved_contract(
    model_package_factory, unit_image, tmp_path
) -> None:
    package, _, _ = _load_package(str(model_package_factory()))

    valid = _validate_inputs(package, [unit_image(tmp_path / "ok.npy")])
    assert valid[0]["status"] == "valid"

    small = unit_image(tmp_path / "small.npy", size=32)
    rejected = _validate_inputs(package, [small])
    assert rejected[0]["status"] == "invalid"

    missing = _validate_inputs(package, [tmp_path / "absent.npy"])
    assert missing[0]["status"] == "invalid"
    assert "does not exist" in missing[0]["message"]

    without_package = _validate_inputs(None, [unit_image(tmp_path / "ok2.npy")])
    assert without_package[0]["message"] == "Import a valid model package first."


def test_runtime_overrides_are_validated() -> None:
    assert _overrides_are_valid(4, 512)
    assert not _overrides_are_valid(0, 512)
    assert not _overrides_are_valid(4, 0)
    assert not _overrides_are_valid("many", 512)
    assert not _overrides_are_valid(3.8, 512)


def test_prediction_support_checks_the_provider_capabilities() -> None:
    assert _provider_supports_prediction(CAPABILITIES)
    assert not _provider_supports_prediction(
        {"operations": ["few_shot_analyze"]}
    )
    assert not _provider_supports_prediction(None)


# ---------------------------------------------------------------------------
# Saved prediction defaults (§9.2 / §13.2)
# ---------------------------------------------------------------------------
def test_saved_defaults_are_read_from_the_package(model_package_factory) -> None:
    package, _, _ = _load_package(
        str(
            model_package_factory(
                "s8.symmodel",
                manifest_overrides={"prediction_defaults": {"stride": 8, "batch_size": 128}},
            )
        )
    )
    assert _saved_prediction_defaults(package) == {"stride": 8, "batch_size": 128}

    # Another package must not inherit the previous one's values.
    other, _, _ = _load_package(
        str(
            model_package_factory(
                "s3.symmodel",
                manifest_overrides={"prediction_defaults": {"stride": 3, "batch_size": 64}},
            )
        )
    )
    assert _saved_prediction_defaults(other) == {"stride": 3, "batch_size": 64}

    # No package falls back to the application defaults.
    assert _saved_prediction_defaults(None) == {"stride": 4, "batch_size": 512}


def test_require_symmodel_path_rejects_other_suffixes(tmp_path) -> None:
    good = tmp_path / "model.symmodel"
    good.write_bytes(b"not-a-real-package")
    assert require_symmodel_path(str(good)).endswith("model.symmodel")

    upper = tmp_path / "Model.SYMMODEL"
    upper.write_bytes(b"x")
    assert require_symmodel_path(str(upper))

    bad = tmp_path / "model.zip"
    bad.write_bytes(b"x")
    with pytest.raises(PredictionError):
        require_symmodel_path(str(bad))

    with pytest.raises(PredictionError):
        require_symmodel_path(None)


def test_renamed_zip_is_rejected_by_content_validation(tmp_path) -> None:
    renamed = tmp_path / "renamed.symmodel"
    with zipfile.ZipFile(renamed, "w") as archive:
        archive.writestr("notes.json", b"{}")
    package, compatibility, message = _load_package(str(renamed))
    assert package is None
    assert compatibility == "Blocked"
    assert "validation failed" in message


def test_full_run_zip_is_not_a_model_package(tmp_path) -> None:
    run_zip = tmp_path / "run.symmodel"
    with zipfile.ZipFile(run_zip, "w") as archive:
        archive.writestr("input.npy", b"npy")
        archive.writestr("run_record.json", b"{}")
        archive.writestr("report.md", b"# run")
    package, compatibility, _ = _load_package(str(run_zip))
    assert package is None
    assert compatibility == "Blocked"


# ---------------------------------------------------------------------------
# Callback-level behavior (§9.2 / §13.2)
# ---------------------------------------------------------------------------
def _capture_prediction_callbacks(config, capabilities):
    gradio = pytest.importorskip("gradio")
    from symmetry_harness.ui_predict import build_prediction_workspace

    captured: dict[str, object] = {"file_changes": [], "click": None}
    original_file_change = gradio.File.change
    original_click = gradio.Button.click

    def spy_file_change(self, function, **kwargs):
        captured["file_changes"].append(function)
        return original_file_change(self, function, **kwargs)

    def spy_click(self, function, **kwargs):
        captured["click"] = function
        return original_click(self, function, **kwargs)

    gradio.File.change = spy_file_change
    gradio.Button.click = spy_click
    try:
        with gradio.Blocks():
            build_prediction_workspace(config, capabilities=capabilities)
    finally:
        gradio.File.change = original_file_change
        gradio.Button.click = original_click

    # Registration order: package file first, then prediction images.
    captured["package_change"] = captured["file_changes"][0]
    captured["inputs_change"] = captured["file_changes"][1]
    return captured


def test_package_load_hydrates_visible_saved_defaults(
    config, model_package_factory
) -> None:
    captured = _capture_prediction_callbacks(config, CAPABILITIES)
    package_path = model_package_factory(
        "s8.symmodel",
        manifest_overrides={"prediction_defaults": {"stride": 8, "batch_size": 128}},
    )
    outputs = captured["package_change"](str(package_path))

    # Stride and batch controls show the saved default, not 4/512.
    assert outputs[6]["value"] == 8
    assert outputs[7]["value"] == 128
    # The saved class colors are pre-filled for presentation.
    assert outputs[15]["value"] == "#e41a1c\n#377eb8"
    # A new package starts a new input contract, so old image selections clear.
    assert outputs[18]["value"] is None


def test_package_reload_resets_previous_defaults(config, model_package_factory) -> None:
    captured = _capture_prediction_callbacks(config, CAPABILITIES)
    first = model_package_factory(
        "s8.symmodel",
        manifest_overrides={"prediction_defaults": {"stride": 8, "batch_size": 128}},
    )
    second = model_package_factory(
        "s3.symmodel",
        manifest_overrides={"prediction_defaults": {"stride": 3, "batch_size": 64}},
    )

    first_outputs = captured["package_change"](str(first))
    assert (first_outputs[6]["value"], first_outputs[7]["value"]) == (8, 128)

    second_outputs = captured["package_change"](str(second))
    assert (second_outputs[6]["value"], second_outputs[7]["value"]) == (3, 64)


def test_invalid_package_clears_stale_values_and_results(
    config, model_package_factory
) -> None:
    captured = _capture_prediction_callbacks(config, CAPABILITIES)
    good = model_package_factory(
        "s8.symmodel",
        manifest_overrides={"prediction_defaults": {"stride": 8, "batch_size": 128}},
    )
    captured["package_change"](str(good))

    rejected = model_package_factory(
        "corrupt.symmodel", model_state=b"tampered", break_checksum=True
    )
    outputs = captured["package_change"](str(rejected))

    # Controls (re)set to the application defaults; prediction disabled.
    assert outputs[6]["value"] == 4
    assert outputs[7]["value"] == 512
    assert outputs[8]["interactive"] is False
    assert outputs[11] is None  # result_state
    # No stale presentation state or colors survive.
    assert outputs[14] is None  # presentation_state
    assert outputs[15]["value"] == ""  # presentation_colors


def test_non_symmodel_selection_is_blocked(config, tmp_path) -> None:
    captured = _capture_prediction_callbacks(config, CAPABILITIES)
    other = tmp_path / "model.zip"
    other.write_bytes(b"x")
    outputs = captured["package_change"](str(other))

    assert outputs[0] is None  # package_state
    assert outputs[8]["interactive"] is False  # predict button
    assert outputs[14] is None  # presentation state


def test_predict_runs_with_hydrated_saved_defaults(
    config, model_package_factory, monkeypatch, tmp_path
) -> None:
    import symmetry_harness.ui_predict as ui_predict

    captured = _capture_prediction_callbacks(config, CAPABILITIES)
    package_path = model_package_factory(
        "s8.symmodel",
        manifest_overrides={"prediction_defaults": {"stride": 8, "batch_size": 128}},
    )
    package, _, _ = _load_package(str(package_path))

    image = tmp_path / "image.npy"
    np.save(image, np.zeros((96, 96), dtype=np.float32))

    calls: dict[str, object] = {}

    def fake_batch(
        config_,
        *,
        model_package,
        image_paths,
        device=None,
        stride=None,
        batch_size=None,
        progress_callback=None,
        **kwargs,
    ):
        calls["stride"] = stride
        calls["batch_size"] = batch_size
        calls["device"] = device
        return {
            "status": "failed",
            "run_id": "prediction-test",
            "counts": {"requested": 1, "completed": 0, "failed": 1},
            "items": [
                {
                    "item_id": "image-0001",
                    "status": "failed",
                    "source_path": str(image),
                    "error": "synthetic failure",
                }
            ],
            "class_names": ["Phase A", "Phase B"],
            "class_colors": ["#e41a1c", "#377eb8"],
            "download_archive": str(tmp_path / "results.zip"),
        }

    monkeypatch.setattr(ui_predict, "run_saved_model_prediction_batch", fake_batch)

    # The visible controls hold the hydrated values, so what runs is 8/128.
    list(
        captured["click"](
            [str(image)], package, True, "auto", 8, 128
        )
    )

    assert calls["stride"] == 8
    assert calls["batch_size"] == 128


def test_inputs_change_clears_stale_downloads(
    config, model_package_factory
) -> None:
    captured = _capture_prediction_callbacks(config, CAPABILITIES)
    package, _, _ = _load_package(
        str(
            model_package_factory(
                "s8.symmodel",
                manifest_overrides={"prediction_defaults": {"stride": 8, "batch_size": 128}},
            )
        )
    )
    status = _package_status(
        selected_path=str(package.source_path),
        extension_ok=True,
        package=package,
        compatibility="Ready",
        message="Package validated.",
        supported=True,
    )

    outputs = captured["inputs_change"](None, package, True, status, "auto", 8, 128)

    # result_state, selector, batch download and result section all cleared.
    assert outputs[4] is None
    assert outputs[5]["value"] is None
    assert outputs[6]["visible"] is False
    assert outputs[7]["visible"] is False


# ---------------------------------------------------------------------------
# Readiness (§3.11 / §13.4)
# ---------------------------------------------------------------------------
def _snapshot(status, rows, *, device="auto", stride=4, batch_size=512):
    return _readiness_snapshot(
        status,
        rows,
        device=device,
        stride=stride,
        batch_size=batch_size,
    )


def test_predict_readiness_distinguishes_failure_modes(model_package_factory) -> None:
    package, _, _ = _load_package(str(model_package_factory()))
    good_status = _package_status(
        selected_path=str(package.source_path),
        extension_ok=True,
        package=package,
        compatibility="Ready",
        message="",
        supported=True,
    )
    invalid_status = _package_status(
        selected_path="/x/bad.symmodel",
        extension_ok=True,
        package=None,
        compatibility="Blocked",
        message="Package validation failed: bad",
        supported=True,
    )
    extension_status = _package_status(
        selected_path="/x/model.zip",
        extension_ok=False,
        package=None,
        compatibility="Blocked",
        message="Only .symmodel model packages are accepted.",
        supported=True,
    )
    incompatible_status = _package_status(
        selected_path=str(package.source_path),
        extension_ok=True,
        package=package,
        compatibility="Blocked",
        message="Package is incompatible: feature pipeline mismatch",
        supported=True,
    )

    valid_rows = [{"file": "a.npy", "status": "valid", "message": ""}]

    # Wrong extension blocks on the extension item.
    ext_items = _by_key(
        evaluate_predict_readiness(_snapshot(extension_status, []))
    )
    assert ext_items["extension"].status == "blocker"
    assert ext_items["extension"].hint

    # Invalid package blocks on content validation, not on compatibility.
    invalid_items = _by_key(
        evaluate_predict_readiness(_snapshot(invalid_status, valid_rows))
    )
    assert invalid_items["package_valid"].status == "blocker"
    assert invalid_items["extension"].status == "ready"

    # Incompatible package passes content validation but blocks compatibility.
    incompatible_items = _by_key(
        evaluate_predict_readiness(_snapshot(incompatible_status, valid_rows))
    )
    assert incompatible_items["package_valid"].status == "ready"
    assert incompatible_items["compatible"].status == "blocker"

    # No valid images blocks on the images item.
    no_image_items = _by_key(
        evaluate_predict_readiness(_snapshot(good_status, []))
    )
    assert no_image_items["images"].status == "blocker"
    assert has_blocker(evaluate_predict_readiness(_snapshot(good_status, [])))

    # Invalid override blocks on runtime params.
    bad_override_items = _by_key(
        evaluate_predict_readiness(_snapshot(good_status, valid_rows, stride=0))
    )
    assert bad_override_items["runtime_params"].status == "blocker"

    # Fully ready: no blockers, can predict.
    ready_items = _by_key(
        evaluate_predict_readiness(_snapshot(good_status, valid_rows))
    )
    assert ready_items["can_predict"].status == "ready"
    assert not has_blocker(
        evaluate_predict_readiness(_snapshot(good_status, valid_rows))
    )


def test_unsupported_provider_blocks_prediction() -> None:
    status = _package_status(
        selected_path="/x/model.symmodel",
        extension_ok=True,
        package=None,
        compatibility="Waiting",
        message="",
        supported=False,
    )
    items = _by_key(
        evaluate_predict_readiness(
            _snapshot(status, [{"file": "a.npy", "status": "valid", "message": ""}])
        )
    )
    assert items["provider_support"].status == "blocker"
    assert has_blocker(
        evaluate_predict_readiness(
            _snapshot(status, [{"file": "a.npy", "status": "valid", "message": ""}])
        )
    )


# ---------------------------------------------------------------------------
# Item labels (§3.12 / §13.6)
# ---------------------------------------------------------------------------
def test_item_choices_show_filename_status_and_disambiguate() -> None:
    result = {
        "items": [
            {"item_id": "image-0001", "status": "completed", "source_path": "/a/sample.tif"},
            {"item_id": "image-0002", "status": "failed", "source_path": "/b/sample.tif", "error": "boom"},
            {"item_id": "image-0003", "status": "completed", "source_path": "/c/other.tif"},
        ]
    }
    choices = _item_choices(result)
    assert choices[0] == ("sample.tif \u2014 completed", "image-0001")
    assert choices[1] == ("sample.tif \u2014 failed (2)", "image-0002")
    assert choices[2] == ("other.tif \u2014 completed", "image-0003")


# ---------------------------------------------------------------------------
# Presentation rendering (§3.4 / §3.5 / §3.7)
# ---------------------------------------------------------------------------
def _write_completed_prediction(tmp_path, *, stride: int = 4, size: int = 16) -> dict:
    directory = tmp_path / "items" / "image-0001"
    directory.mkdir(parents=True)
    axis = np.arange(0, size, stride, dtype=np.int32)
    grid_shape = (len(axis), len(axis))
    coordinates = np.array(
        [[x, y] for y in axis for x in axis], dtype=np.int32
    )
    sample_count = len(coordinates)
    np.savez_compressed(
        directory / "prediction.npz",
        coordinates_xy=coordinates,
        x_coordinates=axis,
        y_coordinates=axis,
        logits=np.zeros((sample_count, 2), dtype=np.float32),
        probabilities=np.full((sample_count, 2), 0.5, dtype=np.float32),
        predictions=np.zeros(sample_count, dtype=np.int16),
        confidence=np.full(sample_count, 0.5, dtype=np.float32),
        entropy=np.full(sample_count, 0.3, dtype=np.float32),
        prediction_grid=np.zeros(grid_shape, dtype=np.int16),
        confidence_grid=np.full(grid_shape, 0.5, dtype=np.float32),
        entropy_grid=np.full(grid_shape, 0.3, dtype=np.float32),
    )
    y, x = np.mgrid[:size, :size]
    image = ((np.sin(x / 3.0) + np.cos(y / 3.0)) / 2.0).astype(np.float32)
    np.save(directory / "input.npy", image)
    record = {
        "item_id": "image-0001",
        "status": "completed",
        "run_id": "prediction-test",
        "model": {"class_colors": ["#e41a1c", "#377eb8"]},
        "output": {"class_statistics": {"grid_cell_count": 16, "classes": [
            {"index": 0, "count": 16, "fraction": 1.0},
            {"index": 1, "count": 0, "fraction": 0.0},
        ]}},
    }
    (directory / "prediction_record.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    entry = {
        "item_id": "image-0001",
        "status": "completed",
        "source_path": "/a/sample.tif",
        "class_statistics": record["output"]["class_statistics"],
        "artifact_prediction": str(directory / "prediction.npz"),
        "artifact_input": str(directory / "input.npy"),
        "prediction_record": str(directory / "prediction_record.json"),
    }
    return {
        "status": "completed",
        "run_id": "prediction-test",
        "stride": stride,
        "class_names": ["Phase A", "Phase B"],
        "class_colors": ["#e41a1c", "#377eb8"],
        "items": [entry],
    }


def test_stride_for_item_rejects_invalid_result_metadata() -> None:
    with pytest.raises(
        ValueError, match="Recorded prediction stride must be at least 1"
    ):
        _stride_for_item({"stride": 0}, {})


def test_stride_for_item_rejects_corrupt_record(tmp_path: Path) -> None:
    record_path = tmp_path / "prediction_record.json"
    record_path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="could not be read"):
        _stride_for_item({}, {"prediction_record": str(record_path)})


def test_stride_for_item_rejects_invalid_record_metadata(tmp_path: Path) -> None:
    record_path = tmp_path / "prediction_record.json"
    record_path.write_text(
        json.dumps({"prediction_options": {"stride": -2}}), encoding="utf-8"
    )
    with pytest.raises(
        ValueError, match="Recorded prediction stride must be at least 1"
    ):
        _stride_for_item({}, {"prediction_record": str(record_path)})


def test_stride_for_item_uses_default_only_for_absent_legacy_metadata(
    tmp_path: Path, caplog
) -> None:
    record_path = tmp_path / "prediction_record.json"
    record_path.write_text("{}", encoding="utf-8")
    with caplog.at_level("WARNING"):
        stride = _stride_for_item({}, {"prediction_record": str(record_path)})
    assert stride == 4
    assert "legacy default" in caplog.text


def test_render_selected_item_produces_all_variants(tmp_path) -> None:
    result = _write_completed_prediction(tmp_path)
    render = render_selected_item(
        "image-0001",
        result,
        colors_text="#e41a1c\n#377eb8",
        alpha=0.48,
        class_names=["Phase A", "Phase B"],
    )
    assert not render["blocked"]
    assert render["source"] is not None
    assert render["overlay"] is not None
    assert render["confidence"] is not None
    assert render["entropy"] is not None
    assert set(render["downloads"]) == {
        "overlay",
        "overlay_legend",
        "mask",
        "confidence",
        "confidence_colorbar",
        "entropy",
        "entropy_colorbar",
    }
    for path in render["downloads"].values():
        assert Path(path).is_file()
    assert "Predicted grid points" in render["statistics"]
    assert "Cells" not in render["statistics"]
    # The complete prediction record is exposed for the Technical details panel.
    assert render["metadata"]["item_id"] == "image-0001"


def test_render_selected_item_blocks_only_on_invalid_colors(tmp_path) -> None:
    result = _write_completed_prediction(tmp_path)
    render = render_selected_item(
        "image-0001",
        result,
        colors_text="#zzzzzz\n#377eb8",
        alpha=0.48,
        class_names=["Phase A", "Phase B"],
    )
    assert render["blocked"] is True
    assert render["downloads"] == {}
    assert render["overlay"] is None
    assert "invalid" in render["message"].lower()


def test_render_selected_item_does_not_rerun_prediction(
    tmp_path, monkeypatch
) -> None:
    import symmetry_harness.ui_predict as ui_predict

    result = _write_completed_prediction(tmp_path)

    def _boom(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("Presentation rerender must not run prediction.")

    monkeypatch.setattr(ui_predict, "run_saved_model_prediction_batch", _boom)

    render = render_selected_item(
        "image-0001",
        result,
        colors_text="#123456\n#abcdef",
        alpha=0.1,
        class_names=["Phase A", "Phase B"],
    )
    assert not render["blocked"]
    assert render["downloads"]


def test_custom_presentation_colors_do_not_leak_to_record(tmp_path) -> None:
    result = _write_completed_prediction(tmp_path)
    entry = result["items"][0]
    before = copy.deepcopy(entry)
    record_path = Path(entry["prediction_record"])
    record_before = record_path.read_text(encoding="utf-8")

    render_selected_item(
        "image-0001",
        result,
        colors_text="#123456\n#abcdef",
        alpha=0.05,
        class_names=["Phase A", "Phase B"],
    )

    # The record entry and the persisted JSON are untouched.
    assert entry == before
    record_after = record_path.read_text(encoding="utf-8")
    assert record_after == record_before
    assert "#123456" not in record_after
    assert "#abcdef" not in record_after


def test_render_selected_item_exposes_failed_record() -> None:
    result = {
        "run_id": "prediction-test",
        "class_names": ["Phase A", "Phase B"],
        "class_colors": ["#e41a1c", "#377eb8"],
        "items": [
            {
                "item_id": "image-0001",
                "status": "failed",
                "source_path": "/a/bad.tif",
                "error": "Image is smaller than the saved patch.",
            }
        ],
    }
    render = render_selected_item(
        "image-0001",
        result,
        colors_text="#e41a1c\n#377eb8",
        alpha=0.48,
        class_names=["Phase A", "Phase B"],
    )
    assert render["overlay"] is None
    assert render["downloads"] == {}
    assert render["metadata"]["error"].startswith("Image is smaller")
    assert "failed" in render["message"]


def test_presentation_colors_helpers_round_trip() -> None:
    state = new_presentation_state(["Phase A", "Phase B"], ["#e41a1c", "#377eb8"])
    assert presentation_colors_text(state) == "#e41a1c\n#377eb8"
    assert split_colors_text("#e41a1c, #377eb8") == ["#e41a1c", "#377eb8"]
    assert split_colors_text("  \n#fff\n\n") == ["#fff"]
