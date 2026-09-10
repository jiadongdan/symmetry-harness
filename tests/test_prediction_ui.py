from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from symmetry_harness.config import load_harness_config
from symmetry_harness.ui_predict import (
    _load_package,
    _overrides_are_valid,
    _provider_supports_prediction,
    _validate_inputs,
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
    # workspace adds its package and result state.
    assert len(states) >= 5


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


def test_prediction_support_checks_the_provider_capabilities() -> None:
    assert _provider_supports_prediction(CAPABILITIES)
    assert not _provider_supports_prediction(
        {"operations": ["few_shot_analyze"]}
    )
    assert not _provider_supports_prediction(None)
