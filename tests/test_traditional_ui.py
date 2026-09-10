"""Unit tests for the standalone traditional ML validation page."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

from symmetry_harness import cli
from symmetry_harness.annotations import create_annotation_session
from symmetry_harness.config import load_harness_config
from symmetry_harness.contracts import TraditionalMLSettings
from symmetry_harness.ui import build_app
import symmetry_harness.ui as ui_module
import symmetry_harness.ui_annotation as ui_annotation
import symmetry_harness.ui_fine_tune as ui_fine_tune
import symmetry_harness.ui_traditional as traditional_module
from symmetry_harness.ui_traditional import (
    CLASSIFIER_CHOICES,
    CLASSIFIER_FIXED_PARAMETERS,
    FEATURE_MODE_CHOICES,
    FEATURE_MODE_LABELS,
    LOGISTIC_REGRESSION,
    RANDOM_FOREST,
    SECTION_TITLES,
    build_traditional_app,
    classifier_panel_visibility,
    classifier_parameters,
    feature_mode_value,
    features_are_current,
    run_is_enabled,
    support_is_complete,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "config.example.json"


def _config():
    return load_harness_config(CONFIG_PATH)


def _capabilities() -> dict:
    """Return a Provider capability document with traditional ML support."""
    return {
        "contract_version": "symmetry-learn-provider-v1",
        "provider": "symmetry-learn",
        "provider_version": "0.1.1",
        "operations": ["compute_features", "traditional_ml_analyze"],
        "models": [
            {
                "identifier": "cnn_8ch_pg17",
                "available": True,
                "input_channels": 8,
                "pretrained_classes": 17,
                "classifier_patch_size": 64,
                "feature_pipeline": "eight_channel_v1",
                "feature_channels": [f"channel_{index}" for index in range(8)],
            }
        ],
        "traditional_ml": {
            "schema_version": "symmetry-traditional-ml-capability-v1",
            "feature_modes": ["raw_image", "image_plus_symmetry_maps"],
            "classifiers": [
                {
                    "identifier": "logistic_regression",
                    "supports_predict_proba": True,
                    "defaults": {
                        "C": 1.0,
                        "class_weight": "balanced",
                        "max_iter": 1000,
                        "solver": "lbfgs",
                    },
                },
                {
                    "identifier": "random_forest",
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


def _state(**overrides) -> dict:
    state = {
        "image": np.linspace(0.0, 1.0, 128 * 128, dtype=np.float32).reshape(128, 128),
        "image_path": str(REPOSITORY_ROOT / "example.npy"),
        "image_sha256": "a" * 64,
        "normalized_image_sha256": "b" * 64,
        "image_shape": [128, 128],
        "class_names": ["Alpha", "Beta"],
        "colors": ["#e41a1c", "#377eb8"],
        "points": [[(40, 40), (48, 44)], [(80, 80), (88, 84)]],
        "classifier_patch_size": 64,
        "show_valid_region": False,
    }
    state.update(overrides)
    return state


def _label(components: list[dict], label: str) -> dict:
    matches = [
        component
        for component in components
        if component.get("props", {}).get("label") == label
    ]
    assert len(matches) == 1, f"expected exactly one {label!r} control"
    return matches[0]


# --- fixed page order and selectors -----------------------------------------


def test_section_titles_follow_the_protocol_order() -> None:
    assert SECTION_TITLES == (
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


def test_classifier_choices_are_exactly_two_identifiers() -> None:
    assert [value for _label_text, value in CLASSIFIER_CHOICES] == [
        "logistic_regression",
        "random_forest",
    ]
    assert classifier_panel_visibility(LOGISTIC_REGRESSION) == (True, False)
    assert classifier_panel_visibility(RANDOM_FOREST) == (False, True)
    with pytest.raises(ValueError, match="Unsupported classifier"):
        classifier_panel_visibility("support_vector_machine")


def test_feature_mode_choices_and_labels() -> None:
    assert [value for _label_text, value in FEATURE_MODE_CHOICES] == [
        "raw_image",
        "image_plus_symmetry_maps",
    ]
    assert feature_mode_value("Raw image only") == "raw_image"
    assert feature_mode_value("raw_image") == "raw_image"
    assert (
        feature_mode_value(FEATURE_MODE_LABELS["image_plus_symmetry_maps"])
        == "image_plus_symmetry_maps"
    )
    with pytest.raises(ValueError, match="Unsupported traditional ML feature mode"):
        feature_mode_value("principal components")


def test_classifier_parameters_only_include_the_selected_model() -> None:
    values = {
        "C": 2.5,
        "class_weight": "balanced",
        "max_iter": 1500,
        "n_estimators": 120,
        "max_depth": "none",
        "min_samples_leaf": 3,
        "max_features": "all",
    }
    assert classifier_parameters(LOGISTIC_REGRESSION, values) == {
        "C": 2.5,
        "class_weight": "balanced",
        "max_iter": 1500,
    }
    assert classifier_parameters(RANDOM_FOREST, values) == {
        "n_estimators": 120,
        "max_depth": "none",
        "min_samples_leaf": 3,
        "max_features": "all",
        "class_weight": "balanced",
    }
    with pytest.raises(ValueError, match="Unsupported classifier"):
        classifier_parameters("lda", values)


def test_fixed_parameters_are_never_sent_as_overrides() -> None:
    assert CLASSIFIER_FIXED_PARAMETERS[LOGISTIC_REGRESSION] == {"solver": "lbfgs"}
    assert CLASSIFIER_FIXED_PARAMETERS[RANDOM_FOREST] == {
        "criterion": "gini",
        "bootstrap": True,
        "n_jobs": 1,
    }
    for identifier, fixed in CLASSIFIER_FIXED_PARAMETERS.items():
        assert not set(fixed).intersection(
            classifier_parameters(identifier, {}).keys()
        )


# --- state gating -----------------------------------------------------------


def test_support_completeness_requires_two_classes_and_two_patches() -> None:
    assert support_is_complete(_state()) is True
    assert support_is_complete(_state(points=[[(40, 40)], [(80, 80)]])) is False
    assert support_is_complete(_state(points=[[(40, 40)]])) is False
    assert support_is_complete(_state(image=None)) is False


def test_features_are_current_requires_an_existing_matching_cache(tmp_path: Path) -> None:
    config = _config()
    capabilities = _capabilities()
    state = _state()
    feature_state = {
        "cache_key": "not-the-expected-key",
        "features_path": str(tmp_path / "features.npz"),
        "record_path": str(tmp_path / "feature_record.json"),
    }
    assert (
        features_are_current(
            state,
            feature_state,
            config,
            capabilities,
            symmetry_patch_size=51,
            device="cpu",
        )
        is False
    )
    assert (
        run_is_enabled(
            state,
            feature_state,
            config,
            capabilities,
            symmetry_patch_size=51,
            device="cpu",
        )
        is False
    )
    assert (
        run_is_enabled(
            _state(image=None),
            {"cache_key": "x"},
            config,
            capabilities,
            symmetry_patch_size=51,
            device="cpu",
        )
        is False
    )


# --- application structure --------------------------------------------------


def test_standalone_page_builds_without_argument_warnings() -> None:
    pytest.importorskip("gradio")
    config = _config()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        app = build_traditional_app(config)
    assert type(app).__name__ == "Blocks"
    argument_warnings = [
        str(item.message)
        for item in captured
        if "Expected" in str(item.message) and "arguments" in str(item.message)
    ]
    assert argument_warnings == []


def test_standalone_page_has_no_normal_workspace_tabs() -> None:
    pytest.importorskip("gradio")
    app = build_traditional_app(_config())
    components = app.get_config_file()["components"]
    assert not [
        component
        for component in components
        if component.get("type") in {"tab", "tabitem"}
    ]
    labels = {
        str(component.get("props", {}).get("label"))
        for component in components
        if component.get("props", {}).get("label")
    }
    assert "Fine-tune a model" not in labels
    assert "Predict with a saved model" not in labels


def test_normal_app_still_has_only_the_two_workspaces() -> None:
    pytest.importorskip("gradio")
    app = build_app(_config())
    components = app.get_config_file()["components"]
    tab_labels = [
        str(component.get("props", {}).get("label"))
        for component in components
        if component.get("type") in {"tab", "tabitem"}
    ]
    assert tab_labels == ["Fine-tune a model", "Predict with a saved model"]


def test_gradio_controls_expose_the_expected_defaults() -> None:
    pytest.importorskip("gradio")
    config = _config()
    app = build_traditional_app(config, capabilities=_capabilities())
    components = app.get_config_file()["components"]

    classifier = _label(components, "Classifier")
    assert classifier["props"]["choices"] == CLASSIFIER_CHOICES
    assert classifier["props"]["value"] == LOGISTIC_REGRESSION

    feature_mode = _label(components, "Traditional ML input features")
    assert feature_mode["props"]["choices"] == FEATURE_MODE_CHOICES
    assert feature_mode["props"]["value"] == "image_plus_symmetry_maps"

    patch_size = _label(components, "Classifier patch size")
    assert patch_size["props"]["value"] == config.model.classifier_patch_size
    assert patch_size["props"].get("interactive", True) is not False

    seed = _label(components, "Global seed")
    assert seed["props"]["value"] == 42

    assert _label(components, "Prediction stride")["props"]["value"] == (
        config.prediction.stride
    )
    assert _label(components, "Prediction batch size")["props"]["value"] == (
        config.prediction.batch_size
    )

    buttons = {
        str(component.get("props", {}).get("value"))
        for component in components
        if component.get("type") == "button"
    }
    assert "Train traditional ML and predict" in buttons


def test_classifier_panels_start_with_only_logistic_regression_visible() -> None:
    pytest.importorskip("gradio")
    app = build_traditional_app(_config())
    config_file = app.get_config_file()
    components = config_file["components"]
    lr = _label(components, "C (regularization strength)")
    rf = _label(components, "n_estimators")
    assert _enclosing_column_visibility(config_file, lr["id"]) is True
    assert _enclosing_column_visibility(config_file, rf["id"]) is False


def _enclosing_column_visibility(
    config_file: dict, component_id: int
) -> bool | None:
    """Return the visibility of the closest enclosing gradio column.

    Gradio 6 keeps the layout tree under ``config_file["layout"]`` while each
    component's props live under ``config_file["components"]``, so the column a
    control belongs to has to be resolved by walking both.
    """
    by_id = {component["id"]: component for component in config_file["components"]}
    found: list[bool | None] = []

    def walk(node: dict, visible: bool | None) -> None:
        component = by_id.get(node.get("id"))
        if component is not None and component.get("type") == "column":
            visible = bool(component.get("props", {}).get("visible", True))
        if node.get("id") == component_id:
            found.append(visible)
            return
        for child in node.get("children") or []:
            walk(child, visible)

    walk(config_file["layout"], None)
    return found[0] if found else None


# --- annotation refactor regressions ---------------------------------------


def test_fine_tune_annotation_behavior_is_unchanged_after_the_refactor() -> None:
    # _configure_classes imports gradio lazily, so this needs the ui extra.
    pytest.importorskip("gradio")
    config = _config()
    assert ui_fine_tune.render_annotations is ui_annotation.render_annotations
    assert ui_fine_tune._annotation_display_shape is ui_annotation.annotation_display_shape
    assert ui_fine_tune._display_to_source_point is ui_annotation.display_to_source_point
    assert ui_module.render_annotations is ui_annotation.render_annotations

    state = {
        "image": np.zeros((430, 430), dtype=np.float32),
        "image_shape": [430, 430],
        "class_names": [],
        "colors": [],
        "points": [],
    }
    updated, active, annotated, rows, message = ui_fine_tune._configure_classes(
        "Phase A, Phase B", state, config, classifier_patch_size=96
    )
    assert updated["class_names"] == ["Phase A", "Phase B"]
    assert updated["classifier_patch_size"] == 96
    assert updated["show_valid_region"] is True
    assert annotated.shape == (720, 720, 3)
    assert rows == [["Phase A", 0, ""], ["Phase B", 0, ""]]
    assert "at least 3 support points per class" in message
    assert active["value"] == "Phase A"


def test_fine_tune_point_helpers_preserve_their_contract() -> None:
    config = _config()
    state = {
        "image": np.zeros((430, 430), dtype=np.float32),
        "image_shape": [430, 430],
        "class_names": ["Phase A", "Phase B"],
        "colors": ["#e41a1c", "#377eb8"],
        "points": [[], []],
        "classifier_patch_size": 64,
    }
    added = ui_fine_tune._add_point(state, "Phase A", (200, 200), config)
    assert added[0]["points"][0] == [(200, 200)]
    assert added[2] is not None

    invalid = ui_fine_tune._add_point(state, "Phase A", (2, 2), config)
    assert invalid[0]["points"] == [[], []]
    assert invalid[4] == "Please select a support point inside the valid region."

    undone = ui_fine_tune._undo_point(added[0], "Phase A", config)
    assert undone[0]["points"][0] == []

    cleared = ui_fine_tune._clear_class(added[0], "Phase A", config)
    assert cleared[0]["points"][0] == []

    with pytest.raises(ValueError, match="already been assigned"):
        ui_fine_tune._add_point(added[0], "Phase B", (200, 200), config)


def test_shared_annotation_helpers_accept_explicit_shot_policy() -> None:
    state = {
        "image": np.zeros((128, 128), dtype=np.float32),
        "image_shape": [128, 128],
        "class_names": ["A", "B"],
        "colors": ["#e41a1c", "#377eb8"],
        "points": [[], []],
        "classifier_patch_size": 64,
    }
    updated, patch_size = ui_annotation.configure_class_state(
        "A, B", state, classifier_patch_size=64
    )
    assert patch_size == 64
    updated, patch_size, status, invalid = ui_annotation.add_point_state(
        updated, "A", (64, 64), fallback_patch_size=64, maximum_points=2
    )
    assert invalid is False
    assert updated["points"][0] == [(64, 64)]
    assert patch_size == 64
    updated, patch_size, status, invalid = ui_annotation.add_point_state(
        updated, "A", (60, 60), fallback_patch_size=64, maximum_points=2
    )
    assert invalid is False
    assert updated["points"][0] == [(64, 64), (60, 60)]
    with pytest.raises(ValueError, match="support-point limit"):
        ui_annotation.add_point_state(
            updated, "A", (68, 68), fallback_patch_size=64, maximum_points=2
        )


def test_configure_classes_rejects_an_unsized_image() -> None:
    with pytest.raises(ValueError, match="before configuring classes"):
        ui_annotation.configure_class_state(
            "A, B", {}, classifier_patch_size=64
        )


# --- settings ---------------------------------------------------------------


def test_traditional_settings_validate_and_serialize() -> None:
    settings = TraditionalMLSettings()
    assert settings.classifier_patch_size == 64
    assert settings.seed == 42
    assert settings.to_provider_options() == {
        "classifier_patch_size": 64,
        "seed": 42,
        "stride": 4,
        "batch_size": 512,
        "minimum_patches_per_class": 2,
        "maximum_patches_per_class": 5000,
    }
    with pytest.raises(ValueError, match="must be positive"):
        TraditionalMLSettings(classifier_patch_size=0)
    with pytest.raises(ValueError, match="must be positive"):
        TraditionalMLSettings(stride=-1)
    with pytest.raises(ValueError, match="seed must be an integer"):
        TraditionalMLSettings(seed=2**32)


# --- CLI --------------------------------------------------------------------


def test_cli_exposes_the_validate_traditional_command() -> None:
    parser = cli._parser()
    arguments = parser.parse_args(
        [
            "validate-traditional",
            "--config",
            str(CONFIG_PATH),
            "--server-port",
            "0",
            "--no-inbrowser",
        ]
    )
    assert arguments.command == "validate-traditional"
    assert arguments.server_name == "127.0.0.1"
    assert arguments.server_port == 0
    assert arguments.inbrowser is False
    assert arguments.input is None


def test_validate_traditional_emits_ready_and_keeps_the_server_alive(
    monkeypatch, capsys
) -> None:
    readiness = {
        "status": "ready",
        "environment": {"capabilities": _capabilities()},
        "traditional_ml": _capabilities()["traditional_ml"],
        "device": "cpu",
        "issues": [],
        "recommendations": [],
        "timings_seconds": {"total": 0.01},
    }
    calls = {"readiness": 0, "launch": 0}

    def fake_readiness(config):
        calls["readiness"] += 1
        return readiness

    def fake_launch(config, **kwargs):
        calls["launch"] += 1
        assert kwargs["prepared_input"] is None
        assert kwargs["capabilities"] == _capabilities()
        kwargs["on_ready"]("http://127.0.0.1:54321")

    monkeypatch.setattr(cli, "traditional_readiness", fake_readiness)
    monkeypatch.setattr(
        traditional_module, "launch_traditional_ui", fake_launch, raising=True
    )
    arguments = cli._parser().parse_args(
        ["validate-traditional", "--config", str(CONFIG_PATH), "--no-inbrowser"]
    )

    assert cli._execute(arguments) is None
    payload = json.loads(capsys.readouterr().out)
    assert calls == {"readiness": 1, "launch": 1}
    assert payload["status"] == "ready"
    assert payload["mode"] == "validate-traditional"
    assert payload["url"] == "http://127.0.0.1:54321"
    assert payload["pretrained_weight_required"] is False
    assert payload["traditional_ml"]["schema_version"] == (
        "symmetry-traditional-ml-capability-v1"
    )


def test_validate_traditional_blocks_a_non_localhost_bind(monkeypatch, capsys) -> None:
    called = {"launch": 0}

    def fake_launch(config, **kwargs):  # pragma: no cover - must not be reached
        called["launch"] += 1

    monkeypatch.setattr(
        traditional_module, "launch_traditional_ui", fake_launch, raising=True
    )
    arguments = cli._parser().parse_args(
        [
            "validate-traditional",
            "--config",
            str(CONFIG_PATH),
            "--server-name",
            "0.0.0.0",
        ]
    )
    report = cli._execute(arguments)
    assert report is not None
    assert report["status"] == "blocked"
    assert report["phase"] == "server_name"
    assert called["launch"] == 0


def test_main_exits_with_two_when_traditional_validation_is_blocked(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        cli, "_execute", lambda arguments: {"status": "blocked", "phase": "server_name"}
    )
    monkeypatch.setattr(sys, "argv", ["symmetry", "validate-traditional"])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_validate_traditional_reports_blocked_readiness(monkeypatch, capsys) -> None:
    readiness = {
        "status": "blocked",
        "environment": {"status": "blocked"},
        "traditional_ml": None,
        "device": "auto",
        "issues": ["provider does not expose traditional ML analysis"],
        "recommendations": ["Install a compatible symmetry Provider"],
        "timings_seconds": {},
    }
    monkeypatch.setattr(cli, "traditional_readiness", lambda config: readiness)
    arguments = cli._parser().parse_args(
        ["validate-traditional", "--config", str(CONFIG_PATH), "--no-inbrowser"]
    )
    report = cli._execute(arguments)
    assert report is not None
    assert report["status"] == "blocked"
    assert report["phase"] == "traditional_readiness"
    assert report["issues"] == readiness["issues"]


def test_validate_traditional_without_config_returns_actionable_json(capsys) -> None:
    arguments = cli._parser().parse_args(
        ["validate-traditional", "--config", "definitely-missing.json"]
    )
    result = cli._execute(arguments)
    assert result is not None
    assert result["status"] == "blocked"
    assert result["phase"] == "config"
    assert any("symmetry init" in item for item in result["recommendations"])


def test_validate_traditional_launcher_scripts_are_machine_independent() -> None:
    for relative in ("scripts/launch-traditional.ps1", "scripts/launch-traditional.sh"):
        content = (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
        assert "validate-traditional" in content
        assert "python-path.txt" in content
        assert "config-path.txt" in content
        assert "D:\\" not in content
        assert "C:\\" not in content
        assert "/Users/" not in content
        assert "anaconda3" not in content


def test_normal_launch_scripts_do_not_start_the_validation_page() -> None:
    for relative in ("scripts/launch.ps1", "scripts/launch.sh"):
        content = (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
        assert "validate-traditional" not in content


# --- annotation session requirement ----------------------------------------


def test_validation_requires_a_matching_session_patch_size() -> None:
    session = create_annotation_session(
        image_path=str(REPOSITORY_ROOT / "example.npy"),
        image_sha256="a" * 64,
        image_shape=(128, 128),
        classifier_patch_size=64,
        class_names=["Alpha", "Beta"],
        points_by_class=[_state()["points"][0], _state()["points"][1]],
    )
    assert session.classifier_patch_size == 64
    with pytest.raises(ValueError):
        create_annotation_session(
            image_path=str(REPOSITORY_ROOT / "example.npy"),
            image_sha256="a" * 64,
            image_shape=(128, 128),
            classifier_patch_size=64,
            class_names=["Alpha"],
            points_by_class=[_state()["points"][0]],
        )


# --- headless readiness -----------------------------------------------------


def test_readiness_gates_the_gradio_dependency_only_for_the_page(
    monkeypatch,
) -> None:
    """The headless run workflow must not require the optional UI extra."""
    import symmetry_harness.provider as provider_module

    monkeypatch.setattr(provider_module, "find_spec", lambda name: None)
    message = "Gradio is required for the traditional validation page."

    page = provider_module.traditional_readiness(_config())
    headless = provider_module.traditional_readiness(_config(), require_ui=False)

    assert message in page["issues"]
    assert message not in headless["issues"]
    assert "ui_dependency" in page["timings_seconds"]
    assert "ui_dependency" in headless["timings_seconds"]


# --- state invalidation table (protocol section 12) -------------------------

RESULT_LABELS = (
    "Prediction overlay",
    "Confidence — maximum class probability",
    "Predictive entropy — higher means more ambiguous",
    "Run summary",
    "Per-class prediction statistics",
)

ANNOTATION_LABELS = (
    "Click support patches",
    "Selected source patch",
    "Support summary",
)


def _invalidation_index(config_file: dict) -> dict:
    """Return component ids, trigger outputs, and the two state component ids."""
    by_id = {component["id"]: component for component in config_file["components"]}
    outputs_by_trigger: dict[int, set[int]] = {}
    inputs_by_api: dict[str, list[int]] = {}
    for dependency in config_file["dependencies"]:
        outputs = {int(value) for value in dependency.get("outputs") or []}
        for target in dependency.get("targets") or []:
            if not target or target[0] is None:
                continue
            outputs_by_trigger.setdefault(int(target[0]), set()).update(outputs)
        api_name = dependency.get("api_name")
        if api_name:
            inputs_by_api[str(api_name)] = [
                int(value) for value in dependency.get("inputs") or []
            ]

    def state_ids(api_name: str) -> list[int]:
        return [
            value
            for value in inputs_by_api[api_name]
            if by_id.get(value, {}).get("type") == "state"
        ]

    annotation_state = state_ids("on_image_change")
    assert len(annotation_state) == 1
    feature_states = state_ids("on_feature_mode_change")
    feature_state = [
        value for value in feature_states if value != annotation_state[0]
    ]
    assert len(feature_state) == 1
    return {
        "by_id": by_id,
        "outputs_by_trigger": outputs_by_trigger,
        "annotation_state": annotation_state[0],
        "feature_state": feature_state[0],
    }


def _trigger_id(by_id: dict, label: str) -> int:
    return next(
        component["id"]
        for component in by_id.values()
        if component.get("props", {}).get("label") == label
    )


def _trigger_id_by_value(by_id: dict, value: str) -> int:
    return next(
        component["id"]
        for component in by_id.values()
        if component.get("type") == "button"
        and component.get("props", {}).get("value") == value
    )


def _result_ids(by_id: dict) -> set[int]:
    return {_trigger_id(by_id, label) for label in RESULT_LABELS}


def _annotation_ids(by_id: dict) -> set[int]:
    return {_trigger_id(by_id, label) for label in ANNOTATION_LABELS}


def test_state_invalidation_table_is_wired_exactly_once_per_row() -> None:
    pytest.importorskip("gradio")
    config_file = build_traditional_app(
        _config(), capabilities=_capabilities()
    ).get_config_file()
    index = _invalidation_index(config_file)
    by_id = index["by_id"]
    outputs_by_trigger = index["outputs_by_trigger"]

    def outputs(trigger_id: int) -> set[int]:
        return outputs_by_trigger[trigger_id]

    results = _result_ids(by_id)
    annotations = _annotation_ids(by_id)
    annotation_state = index["annotation_state"]
    feature_state = index["feature_state"]

    def assert_results_cleared(trigger_id: int) -> None:
        assert results <= outputs(trigger_id)

    # new image -> invalidate maps, annotations, results
    new_image = _trigger_id(by_id, "Input image")
    assert_results_cleared(new_image)
    assert annotations <= outputs(new_image)
    assert annotation_state in outputs(new_image)
    assert feature_state in outputs(new_image)

    # map-producing setting -> preserve annotations, invalidate maps, results
    for label in ("Symmetry patch size", "Map-computation device"):
        trigger = _trigger_id(by_id, label)
        assert_results_cleared(trigger)
        assert feature_state in outputs(trigger)
        assert annotation_state not in outputs(trigger)
        assert not (annotations & outputs(trigger))

    # feature mode -> preserve maps and annotations, invalidate results
    feature_mode = _trigger_id(by_id, "Traditional ML input features")
    assert_results_cleared(feature_mode)
    assert feature_state not in outputs(feature_mode)
    assert annotation_state not in outputs(feature_mode)

    # classifier choice -> preserve maps and annotations, invalidate results
    classifier = _trigger_id(by_id, "Classifier")
    assert_results_cleared(classifier)
    assert feature_state not in outputs(classifier)
    assert annotation_state not in outputs(classifier)

    # classifier parameter and seed -> preserve maps and annotations
    for label in ("C (regularization strength)", "Global seed", "n_estimators"):
        trigger = _trigger_id(by_id, label)
        assert_results_cleared(trigger)
        assert feature_state not in outputs(trigger)
        assert annotation_state not in outputs(trigger)

    # stride and batch size -> preserve maps and annotations
    for label in ("Prediction stride", "Prediction batch size"):
        trigger = _trigger_id(by_id, label)
        assert_results_cleared(trigger)
        assert feature_state not in outputs(trigger)
        assert annotation_state not in outputs(trigger)

    # patch size -> preserve maps, invalidate annotations and results
    patch_size = _trigger_id(by_id, "Classifier patch size")
    assert_results_cleared(patch_size)
    assert feature_state not in outputs(patch_size)
    assert annotation_state in outputs(patch_size)
    assert annotations <= outputs(patch_size)

    # class definition -> preserve maps, invalidate annotations and results
    configure = _trigger_id_by_value(by_id, "Configure classes and patch size")
    assert_results_cleared(configure)
    assert feature_state not in outputs(configure)
    assert annotation_state in outputs(configure)
    assert annotations <= outputs(configure)

    # every result component is reachable from at least one invalidation path
    cleared_by_something = set().union(*outputs_by_trigger.values())
    assert results <= cleared_by_something


# --- event handler return-count contract ------------------------------------
#
# Gradio only checks that an event handler returns exactly one value per
# declared output when the event actually fires.  ``build_traditional_app``
# therefore succeeds even when a handler returns the wrong number of values,
# and the mismatch only surfaces as a live ``ValueError`` in the browser.
# These tests invoke every handler directly so a dropped output is a red test
# instead of a runtime crash.


class _Event:
    """Minimal stand-in for the gradio ``SelectData`` event object."""

    def __init__(self, index: list[int]) -> None:
        self.index = index


def _handler_invocations() -> list[tuple[str, tuple]]:
    """Return ``(handler_name, positional_args)`` for every safe handler."""
    state = _state()
    features = {
        "cache_key": "cache",
        "features_path": "features.npz",
        "record_path": "features_record.json",
        "fingerprint": "fingerprint",
        "feature_shape": [8, 128, 128],
    }
    return [
        ("on_image_change", (None, state)),
        ("on_feature_setting_change", (state,)),
        ("on_feature_mode_change", ("raw_image", state, features, 51, "auto")),
        ("on_feature_mode_change", ("image_plus_symmetry_maps", state, None, None, "auto")),
        ("on_classifier_change", ("random_forest", state, features, 51, "auto")),
        ("on_classifier_change", ("logistic_regression", state, None, None, "auto")),
        ("on_patch_size_change", (96, state)),
        ("on_seed_or_prediction_change", (state, features, 51, "auto")),
        ("on_seed_or_prediction_change", (state, None, None, "auto")),
        ("on_configure_classes", ("Phase A, Phase B", 64, state)),
        ("on_select", (state, "Alpha", features, 51, "auto", _Event([64, 64]))),
        ("on_select", (state, "Alpha", features, 51, "auto", _Event([2, 2]))),
        ("on_undo", (state, "Alpha", features, 51, "auto")),
        ("on_clear", (state, "Alpha")),
    ]


def test_every_handler_returns_exactly_one_value_per_output() -> None:
    pytest.importorskip("gradio")
    app = build_traditional_app(_config(), capabilities=_capabilities())
    handlers = {}
    for block_fn in app.fns.values():
        handlers.setdefault(block_fn.fn.__name__, block_fn)

    for name, args in _handler_invocations():
        block_fn = handlers.get(name)
        assert block_fn is not None, f"{name} is not registered as an event handler"
        expected = len(block_fn.outputs)
        result = block_fn.fn(*args)
        actual = len(result) if isinstance(result, tuple) else 1
        assert actual == expected, (
            f"{name} returned {actual} values for {expected} declared outputs"
        )


def test_provider_backed_handlers_are_registered_with_outputs() -> None:
    """The two Provider-backed handlers exist and declare their outputs."""
    pytest.importorskip("gradio")
    app = build_traditional_app(_config(), capabilities=_capabilities())
    names = {block_fn.fn.__name__ for block_fn in app.fns.values()}
    for name in ("on_compute_features", "on_run"):
        assert name in names, f"{name} is not registered"


def test_cleared_number_controls_do_not_raise_from_handlers() -> None:
    """A cleared number box is ``None``; handlers must degrade, not crash."""
    pytest.importorskip("gradio")
    config = _config()
    capabilities = _capabilities()
    app = build_traditional_app(config, capabilities=capabilities)
    handlers = {block_fn.fn.__name__: block_fn for block_fn in app.fns.values()}
    state = _state()
    features = {
        "cache_key": "cache",
        "features_path": "features.npz",
        "record_path": "features_record.json",
        "fingerprint": "fingerprint",
        "feature_shape": [8, 128, 128],
    }

    # a cleared patch-size control disables Run instead of raising
    assert (
        features_are_current(
            state,
            features,
            config,
            capabilities,
            symmetry_patch_size=None,
            device="auto",
        )
        is False
    )
    assert (
        run_is_enabled(
            state,
            features,
            config,
            capabilities,
            symmetry_patch_size=None,
            device="auto",
        )
        is False
    )

    for name, args in (
        ("on_seed_or_prediction_change", (state, features, None, "auto")),
        ("on_patch_size_change", (None, state)),
    ):
        block_fn = handlers[name]
        result = block_fn.fn(*args)
        actual = len(result) if isinstance(result, tuple) else 1
        assert actual == len(block_fn.outputs), name
        # the Run button must end up disabled, never left enabled
        run_slot = next(
            index
            for index, component in enumerate(block_fn.outputs)
            if getattr(component, "value", None) == "Train traditional ML and predict"
        )
        assert result[run_slot].get("interactive") is False, name
