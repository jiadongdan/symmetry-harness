"""Focused tests for the Fine-tune workspace integration (protocol sections 13.3 /
13.5 / 13.6, plus the fine-tune parts of 3.3, 3.4, 3.5, 3.10, 3.11).

Everything here is exercised through module-level helpers and the built Gradio
component graph; no server is started.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import symmetry_harness.ui_fine_tune as ft
from symmetry_harness import ui_shared
from symmetry_harness.config import load_harness_config
from symmetry_harness.ui_readiness import readiness_panel_html
from symmetry_harness.ui_fine_tune import (
    PresentationState,
    _current_stage,
    _fine_tune_class_statistics,
    _stage_visibility,
    build_fine_tune_snapshot,
    build_fine_tune_workspace,
    load_fine_tune_result,
    parse_presentation_colors,
    prepare_model_download,
    prepare_presentation_pngs,
    render_fine_tune_variants,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLASS_COLORS = ["#e41a1c", "#377eb8"]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture()
def config():
    return load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")


def _capabilities(*, installed: bool = True) -> dict:
    status = "installed" if installed else "missing"
    weight = {
        "identifier": "pg17-symmetry-v1",
        "version": "1.0.0",
        "sha256": "c" * 64,
        "distribution": "symmetry-learn-default-model",
        "status": status,
        "bundled": True,
        "default": True,
    }
    return {
        "contract_version": "symmetry-learn-provider-v1",
        "provider": "symmetry-learn",
        "provider_version": "0.1.0",
        "operations": ["few_shot_analyze", "compute_features"],
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
                "defaults": {
                    "n_max": 20,
                    "symmetry_patch_size": 51,
                    "rotation_folds": [2, 3, 4, 6],
                    "reflection_p": 2.0,
                    "normalize_rotation_maps": False,
                    "epochs": 150,
                    "learning_rate": 0.0005,
                    "stride": 4,
                    "batch_size": 512,
                    "device": "auto",
                },
                "default_weight": weight,
                "weights": [weight],
            }
        ],
    }


def _build_ft(config, capabilities):
    import gradio as gr

    with gr.Blocks() as app:
        build_fine_tune_workspace(config, capabilities=capabilities)
    return app


def _components(app) -> list[dict]:
    return app.get_config_file()["components"]


def _by_type(app, type_name: str) -> list[dict]:
    return [c for c in _components(app) if c.get("type") == type_name]


def _handler(app, name: str):
    for block_fn in app.fns.values():
        if getattr(block_fn.fn, "__name__", "") == name:
            return block_fn
    raise AssertionError(f"handler {name!r} is not registered")


def _output_index_by_label(block_fn, label: str) -> int:
    for index, component in enumerate(block_fn.outputs):
        if getattr(component, "label", None) == label:
            return index
    raise AssertionError(f"no output labelled {label!r}")


def _make_run(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Create a fake completed run directory + presentation reference."""
    run_dir = tmp_path / "symmetry-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    image = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16)
    np.save(run_dir / "input.npy", image)
    x = np.array([8, 12], dtype=np.int32)
    y = np.array([8, 12], dtype=np.int32)
    coords = np.array([[8, 8], [12, 8], [8, 12], [12, 12]], dtype=np.int32)
    np.savez(
        run_dir / "prediction.npz",
        coordinates_xy=coords,
        x_coordinates=x,
        y_coordinates=y,
        logits=np.zeros((4, 2), dtype=np.float32),
        probabilities=np.full((4, 2), 0.5, dtype=np.float32),
        predictions=np.array([0, 1, 1, 0], dtype=np.int16),
        confidence=np.full(4, 0.5, dtype=np.float32),
        entropy=np.full(4, np.log(2.0), dtype=np.float32),
        prediction_grid=np.array([[0, 1], [1, 0]], dtype=np.int16),
        confidence_grid=np.full((2, 2), 0.5, dtype=np.float32),
        entropy_grid=np.full((2, 2), np.log(2.0), dtype=np.float32),
    )
    (run_dir / "fine_tuned_model.symmodel").write_bytes(b"SYMMODEL-DURABLE-BYTES")
    (run_dir / "symmetry-run-full-run.zip").write_bytes(b"FULL-RUN-ZIP-BYTES")
    (run_dir / "run_record.json").write_text(
        json.dumps(
            {
                "run_id": "symmetry-run",
                "status": "completed",
                "classes": [
                    {"index": 0, "name": "Class A", "color": CLASS_COLORS[0]},
                    {"index": 1, "name": "Class B", "color": CLASS_COLORS[1]},
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    reference = {
        "run_id": "symmetry-run",
        "run_directory": str(run_dir),
        "prediction": str(run_dir / "prediction.npz"),
        "input_array": str(run_dir / "input.npy"),
        "input_preview": str(run_dir / "input_preview.png"),
        "fine_tuned_model": str(run_dir / "fine_tuned_model.symmodel"),
        "full_run_zip": str(run_dir / "symmetry-run-full-run.zip"),
        "stride": "4",
    }
    return run_dir, reference


# ---------------------------------------------------------------------------
# Progressive disclosure / result visibility (protocol 3.10, 13.6)
# ---------------------------------------------------------------------------
def test_four_stage_headers_are_present(config) -> None:
    pytest.importorskip("gradio")
    app = _build_ft(config, _capabilities())
    labels = [c["props"].get("label") for c in _by_type(app, "accordion")]
    for header in (
        "1. Model and image",
        "2. Review symmetry maps",
        "3. Select support points",
        "4. Fine-tune and review",
    ):
        assert header in labels


def test_all_stage_blocks_are_present_before_prerequisites(config) -> None:
    pytest.importorskip("gradio")
    app = _build_ft(config, _capabilities())
    stages = {c["props"].get("label"): c["props"] for c in _by_type(app, "accordion")}
    assert all(stages[label]["visible"] is True for label in STAGE_LABELS)
    buttons = {
        c["props"].get("value"): c["props"] for c in _by_type(app, "button")
    }
    assert buttons["Configure classes and patch size"]["interactive"] is False
    assert buttons["Prepare selected export"]["interactive"] is False


def test_result_controls_and_downloads_are_hidden_before_completion(config) -> None:
    pytest.importorskip("gradio")
    app = _build_ft(config, _capabilities())
    downloads = {
        c["props"].get("label"): c["props"] for c in _by_type(app, "downloadbutton")
    }
    assert downloads["Download model (.symmodel)"]["visible"] is False
    assert downloads["Download complete run (.zip)"]["visible"] is False
    # The result images start empty.
    overlays = [c for c in _by_type(app, "image") if c["props"].get("label", "").startswith("Prediction overlay")]
    assert overlays and overlays[0]["props"].get("value") in (None, "")


def test_fine_tune_result_images_use_one_fixed_height(config) -> None:
    pytest.importorskip("gradio")
    app = _build_ft(config, _capabilities())
    labels = {
        "Prediction overlay (class legend below)",
        "Confidence — maximum class probability (0 to 1)",
        "Predictive entropy — higher means more ambiguous (0 to ln N)",
    }
    images = [
        component["props"]
        for component in _by_type(app, "image")
        if component["props"].get("label") in labels
    ]
    assert len(images) == 3
    assert {image.get("height") for image in images} == {480}


def test_advanced_sections_default_closed(config) -> None:
    pytest.importorskip("gradio")
    app = _build_ft(config, _capabilities())
    accordions = {
        c["props"].get("label"): c["props"].get("open")
        for c in _by_type(app, "accordion")
    }
    for label in (
        "Advanced: custom checkpoint",
        "Advanced: training and prediction settings",
        "Technical details",
    ):
        assert accordions.get(label) is False


def test_symmetry_map_settings_stay_expanded_and_above_the_compute_action(
    config,
) -> None:
    """The patch size is per-image tunable, so it must never be hidden.

    STEM resolution varies from image to image, so the symmetry patch size is
    the one setting users routinely revisit. It is therefore a first-class,
    expanded block at the top of stage 2 -- not a collapsed "Advanced" section
    tucked below the gallery.
    """
    pytest.importorskip("gradio")
    app = _build_ft(config, _capabilities())
    accordions = {c["props"].get("label"): c["props"] for c in _by_type(app, "accordion")}

    settings = accordions["Symmetry map settings"]
    assert settings["open"] is True
    assert "Advanced: symmetry map settings" not in accordions

    components = _components(app)
    labels = [c["props"].get("label") or c["props"].get("value") for c in components]
    settings_index = labels.index("Symmetry map settings")
    compute_index = next(
        index
        for index, value in enumerate(labels)
        if isinstance(value, str) and "Compute / update symmetry maps" in value
    )
    assert settings_index < compute_index


def test_raw_json_is_under_a_default_closed_technical_details_accordion(config) -> None:
    pytest.importorskip("gradio")
    app = _build_ft(config, _capabilities())
    accordions = [c["props"] for c in _by_type(app, "accordion")]
    technical = [a for a in accordions if a.get("label") == "Technical details"]
    assert technical and technical[0].get("open") is False
    json_components = _by_type(app, "json")
    assert len(json_components) == 1
    assert json_components[0]["props"].get("label") == "Run result"


def test_stage_blocks_remain_visible_for_every_readiness_state() -> None:
    expected = {"symmetry": True, "support": True, "run": True}
    assert _stage_visibility({}) == expected
    assert _stage_visibility(
        {"image_valid": True, "features_current": False, "class_names": ["A", "B"]}
    ) == expected
    assert _stage_visibility(
        {"image_valid": True, "features_current": True, "class_names": ["A", "B"]}
    ) == expected


STAGE_LABELS = [
    "1. Model and image",
    "2. Review symmetry maps",
    "3. Select support points",
    "4. Fine-tune and review",
]


def _ready_snapshot(**overrides) -> dict:
    """A snapshot with no blocking readiness item."""
    snapshot = {
        "model_supported": True,
        "model_name": "CNN",
        "checkpoint_ready": True,
        "checkpoint_detail": "Installed.",
        "image_valid": True,
        "image_name": "source.npy",
        "features_current": True,
        "features_detail": "current",
        "class_names": ["Class A", "Class B"],
        "class_counts": [5, 5],
        "minimum_shots": 3,
        "recommended_shots": 5,
        "maximum_shots": 50,
        "symmetry_patch_size": 51,
        "epochs": 150,
        "learning_rate": 0.0005,
        "stride": 4,
        "batch_size": 512,
        "device": "cpu",
        "stale_state": False,
        "stale_detail": "",
        "has_result": False,
    }
    snapshot.update(overrides)
    return snapshot


def test_current_stage_is_the_first_revealed_incomplete_stage() -> None:
    # Stage 1: no valid image yet (this also covers the empty snapshot).
    assert _current_stage({}) == "model_and_image"
    assert _current_stage(_ready_snapshot(image_valid=False)) == "model_and_image"
    # Stage 2: image is valid but the symmetry maps are not current.
    assert _current_stage(_ready_snapshot(features_current=False)) == "symmetry"
    # Stage 3: maps current but a blocker remains (a second class is missing here).
    assert _current_stage(_ready_snapshot(class_names=["Class A"])) == "support"
    # Stage 4: everything is ready, so fine-tuning is the current stage.
    assert _current_stage(_ready_snapshot()) == "run"


def test_current_stage_advances_monotonically_with_readiness() -> None:
    stages = [
        _current_stage(_ready_snapshot(image_valid=False)),
        _current_stage(_ready_snapshot(features_current=False)),
        _current_stage(_ready_snapshot(class_names=["Class A"])),
        _current_stage(_ready_snapshot()),
    ]
    assert stages == ["model_and_image", "symmetry", "support", "run"]


def test_stage_accordions_default_open_only_for_stage_one(config) -> None:
    pytest.importorskip("gradio")
    app = _build_ft(config, _capabilities())
    stages = {c["props"].get("label"): c["props"] for c in _by_type(app, "accordion")}
    opens = [stages[label]["open"] for label in STAGE_LABELS]
    assert opens == [True, False, False, False]


def test_refresh_is_additive_and_never_collapses_a_revealed_stage(config) -> None:
    pytest.importorskip("gradio")
    app = _build_ft(config, _capabilities())
    block_fn = _handler(app, "on_refresh")

    def _refresh(announced):
        return block_fn.fn(
            {},  # state (no image yet)
            {},  # feature_state
            "cnn_8ch_pg17",  # model id
            None,  # checkpoint path
            _capabilities(),  # catalog
            51,  # symmetry patch size
            150,  # epochs
            0.0005,  # learning rate
            4,  # stride
            512,  # batch size
            "cpu",  # device
            PresentationState(),  # presentation state
            announced,  # last announced stage
        )

    # First refresh announces the stage and opens it.
    changed = _refresh(None)
    assert len(changed) == len(block_fn.outputs)
    assert changed[6] == "model_and_image"
    stage_updates = changed[2:6]
    assert [update["visible"] for update in stage_updates] == [True, True, True, True]
    assert stage_updates[0]["open"] is True
    # Every block remains present. A refresh must not collapse a stage the user
    # is working in.
    assert all("open" not in update for update in stage_updates[1:])

    # A transition to a new current stage opens that stage and leaves the rest.
    reannounced = _refresh("run")
    assert reannounced[6] == "model_and_image"
    assert reannounced[2]["open"] is True
    assert all("open" not in update for update in reannounced[3:6])

    # The same stage leaves open untouched, so a manual expansion survives:
    # no "open" key is pushed for any stage.
    unchanged = _refresh("model_and_image")
    assert unchanged[6] == "model_and_image"
    assert all("open" not in update for update in unchanged[2:6])


def test_stage_three_is_revealed_once_the_symmetry_maps_are_current(
    config, tmp_path
) -> None:
    """Completing stage 2 must reveal stage 3, or the user is stuck.

    Regression guard: the refresh used to push ``open=False`` for every stage
    except the current one, so finishing the symmetry maps collapsed stage 2 and
    left no obvious way forward.
    """
    pytest.importorskip("gradio")
    app = _build_ft(config, _capabilities())
    block_fn = _handler(app, "on_refresh")

    state = {
        "image": np.zeros((96, 96), dtype=np.float32),
        "image_path": str(tmp_path / "source.png"),
        "image_sha256": "a" * 64,
        "normalized_image_sha256": "b" * 64,
        "image_shape": [96, 96],
        "class_names": [],
        "points": [],
    }
    _, _, _, cache_key = ft._feature_request(
        config,
        _capabilities(),
        state,
        model_identifier="cnn_8ch_pg17",
        symmetry_patch_size=51,
        device="cpu",
    )
    features_path = tmp_path / "features.npz"
    record_path = tmp_path / "record.json"
    np.savez_compressed(features_path, features=np.zeros((1, 1, 1, 1), np.float32))
    record_path.write_text("{}", encoding="utf-8")
    feature_state = {
        "cache_key": cache_key,
        "features_path": str(features_path),
        "record_path": str(record_path),
        "fingerprint": {},
        "feature_shape": [1, 1, 1, 1],
    }

    updates = block_fn.fn(
        state,
        feature_state,
        "cnn_8ch_pg17",
        None,
        _capabilities(),
        51,
        150,
        0.0005,
        4,
        512,
        "cpu",
        PresentationState(),
        "symmetry",
    )
    stages = updates[2:6]
    # Stages 1 and 2 stay revealed, and stage 2 is not collapsed by the refresh
    # that ran inside it.
    assert [update["visible"] for update in stages] == [True, True, True, True]
    assert "open" not in stages[1]
    # Stage 3 is where the workflow now waits, so it is revealed and opened.
    assert stages[2]["open"] is True
    assert updates[6] == "support"
    configure_index = next(
        index
        for index, component in enumerate(block_fn.outputs)
        if getattr(component, "value", None) == "Configure classes and patch size"
    )
    export_index = next(
        index
        for index, component in enumerate(block_fn.outputs)
        if getattr(component, "value", None) == "Prepare selected export"
    )
    assert updates[configure_index]["interactive"] is True
    assert updates[export_index]["interactive"] is True


def test_presentation_rerender_works_when_the_run_recorded_its_result(config, tmp_path) -> None:
    """Regression guard: the rerender used to error all three images.

    A completed run must store its persisted artifact paths in
    ``numerical_result_reference``. Without them the rerender guard fired and
    Gradio turned every result image into an error box.
    """
    pytest.importorskip("gradio")
    app = _build_ft(config, _capabilities())
    block_fn = _handler(app, "on_update_presentation")

    image_path = tmp_path / "input.npy"
    np.save(image_path, np.zeros((8, 8), dtype=np.float32))
    prediction_path = tmp_path / "prediction.npz"
    grid = np.zeros((2, 2), dtype=np.int64)
    coordinates = np.array(
        [[0, 0], [4, 0], [0, 4], [4, 4]], dtype=np.int64
    )
    logits = np.zeros((4, 2), dtype=np.float32)
    np.savez_compressed(
        prediction_path,
        coordinates_xy=coordinates,
        x_coordinates=np.array([0, 4], dtype=np.int64),
        y_coordinates=np.array([0, 4], dtype=np.int64),
        logits=logits,
        probabilities=np.full((4, 2), 0.5, dtype=np.float32),
        predictions=np.zeros(4, dtype=np.int64),
        confidence=np.full(4, 0.5, dtype=np.float32),
        entropy=np.full(4, 0.1, dtype=np.float32),
        prediction_grid=grid,
        confidence_grid=np.full((2, 2), 0.5, dtype=np.float32),
        entropy_grid=np.full((2, 2), 0.1, dtype=np.float32),
    )
    reference = {
        "input_array": str(image_path),
        "prediction": str(prediction_path),
        "stride": "4",
    }
    presentation = PresentationState(
        class_names=["A", "B"],
        display_colors=["#e41a1c", "#377eb8"],
        alpha=0.48,
        numerical_result_reference=reference,
    )

    updates = block_fn.fn(presentation, "#111111, #222222", 0.30)
    rendered = updates[:3]
    assert all(isinstance(value, np.ndarray) for value in rendered), updates
    assert all(value.ndim == 3 and value.shape[2] == 3 for value in rendered)
    # The rerendered palette is what the caller asked for, and alpha changed.
    assert updates[4].display_colors == ["#111111", "#222222"]
    assert abs(updates[4].alpha - 0.30) < 1e-9
    # The numerical reference survives, so repeated rerenders stay possible.
    assert updates[4].numerical_result_reference == reference


def test_fine_tune_reuses_the_shared_readiness_renderer(config) -> None:
    pytest.importorskip("gradio")
    # Fix 2 convergence: exactly one HTML renderer (ui_readiness); ui_fine_tune
    # must not keep a private copy.
    assert not hasattr(ft, "_readiness_items_html")
    assert ft.readiness_panel_html is readiness_panel_html

    app = _build_ft(config, _capabilities())
    values = [
        c["props"].get("value", "")
        for c in _by_type(app, "html")
        if isinstance(c["props"].get("value"), str)
        and "readiness-panel" in c["props"].get("value", "")
    ]
    assert values, "the fine-tune workspace must render the readiness panel"
    # The sticky column owns the id, so the shared HTML keeps the class only.
    assert all('class="readiness-panel"' in value for value in values)
    assert all('id="readiness-panel"' not in value for value in values)


# ---------------------------------------------------------------------------
# Model selection (protocol 3.2, 3.3, 8.1, 13.3)
# ---------------------------------------------------------------------------
def test_one_model_selector_and_no_official_weight_dropdown(config) -> None:
    pytest.importorskip("gradio")
    app = _build_ft(config, _capabilities())
    dropdowns = _by_type(app, "dropdown")
    assert sum(1 for c in dropdowns if c["props"].get("label") == "Model") == 1
    labels = [c["props"].get("label", "") for c in _components(app)]
    assert not any("Registered weight" in str(label) for label in labels)
    assert not any("Weight source" in str(label) for label in labels)


def test_installed_checkpoint_hides_install_action(config) -> None:
    pytest.importorskip("gradio")
    app = _build_ft(config, _capabilities(installed=True))
    buttons = [
        c["props"]
        for c in _by_type(app, "button")
        if c["props"].get("value") == "Install selected weight"
    ]
    assert len(buttons) == 1
    assert buttons[0].get("visible") is False


def test_missing_checkpoint_requires_explicit_install(config) -> None:
    pytest.importorskip("gradio")
    app = _build_ft(config, _capabilities(installed=False))
    buttons = [
        c["props"]
        for c in _by_type(app, "button")
        if c["props"].get("value") == "Install selected weight"
    ]
    assert len(buttons) == 1
    assert buttons[0].get("visible") is True
    snapshot = build_fine_tune_snapshot(
        config,
        _capabilities(installed=False),
        state={},
        feature_state={},
        model_identifier="cnn_8ch_pg17",
        checkpoint_path=None,
        symmetry_patch_size=51,
        epochs=150,
        learning_rate=0.0005,
        stride=4,
        batch_size=512,
        device="cpu",
    )
    assert snapshot["model_supported"] is True
    assert snapshot["checkpoint_ready"] is False
    assert ft.has_blocker(ft.evaluate_fine_tune_readiness(snapshot))


def test_custom_checkpoint_lives_under_advanced_and_targets_the_model(config) -> None:
    pytest.importorskip("gradio")
    app = _build_ft(config, _capabilities())
    accordions = {c["props"].get("label"): c["props"] for c in _by_type(app, "accordion")}
    assert accordions["Advanced: custom checkpoint"]["open"] is False
    files = [c["props"] for c in _by_type(app, "file")]
    assert sum(1 for f in files if f.get("label") == "Custom checkpoint") == 1


def test_custom_checkpoint_change_auto_computes_sha256(config, tmp_path) -> None:
    pytest.importorskip("gradio")
    app = _build_ft(config, _capabilities())
    handler = _handler(app, "on_custom_checkpoint_change")
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"custom-checkpoint-bytes")
    from symmetry_harness.image_io import file_sha256

    digest_text, status_text, install_update = handler.fn(
        str(checkpoint), "cnn_8ch_pg17", _capabilities()
    )
    assert file_sha256(checkpoint) in digest_text
    assert "ready" in status_text
    assert install_update.get("visible") is False


def test_model_change_clears_model_state_and_stale_downloads(config, tmp_path) -> None:
    pytest.importorskip("gradio")
    capabilities = _capabilities()
    app = _build_ft(config, capabilities)
    block_fn = _handler(app, "on_model_change")
    source = tmp_path / "source.npy"
    np.save(source, np.linspace(0.0, 1.0, 96 * 96, dtype=np.float32).reshape(96, 96))
    previous_state = {
        "image": np.zeros((96, 96), dtype=np.float32),
        "image_path": str(source),
        "image_sha256": "a" * 64,
        "normalized_image_sha256": "b" * 64,
        "image_shape": [96, 96],
        "class_names": ["Class A", "Class B"],
        "colors": CLASS_COLORS,
        "points": [[(40, 40)], [(50, 50)]],
        "classifier_patch_size": 64,
        "show_valid_region": True,
    }
    result = block_fn.fn("cnn_8ch_pg17", capabilities, previous_state, None)
    assert isinstance(result, tuple)
    assert len(result) == len(block_fn.outputs)

    # Only the source path survives, and it is re-read under the new contract.
    states = [value for value in result if isinstance(value, dict) and "class_names" in value]
    assert len(states) == 1
    assert states[0]["class_names"] == []
    assert states[0]["points"] == []
    assert states[0]["image_path"] == str(source)

    # Every result-derived download is cleared and hidden.
    model_index = _output_index_by_label(block_fn, "Download model (.symmodel)")
    zip_index = _output_index_by_label(block_fn, "Download complete run (.zip)")
    assert result[model_index].get("value") is None
    assert result[model_index].get("visible") is False
    assert result[zip_index].get("value") is None
    assert result[zip_index].get("visible") is False

    presentations = [value for value in result if isinstance(value, PresentationState)]
    assert len(presentations) == 1
    assert presentations[0].numerical_result_reference == {}
    assert presentations[0].display_colors == []

    # New model defaults replace the previous settings.
    patch_index = _output_index_by_label(block_fn, "Symmetry patch size")
    epochs_index = _output_index_by_label(block_fn, "Fine-tuning epochs")
    stride_index = _output_index_by_label(block_fn, "Prediction stride")
    assert result[patch_index] == 51
    assert result[epochs_index] == 150
    assert result[stride_index] == 4


def test_model_change_refreshes_locked_patch_size_and_class_text(config, tmp_path) -> None:
    pytest.importorskip("gradio")
    capabilities = _capabilities()
    alternate = copy.deepcopy(capabilities["models"][0])
    alternate["identifier"] = "cnn_8ch_small_patch"
    alternate["display_name"] = "Eight-channel CNN (small patch)"
    alternate["classifier_patch_size"] = 32
    capabilities["models"].append(alternate)
    app = _build_ft(config, capabilities)
    block_fn = _handler(app, "on_model_change")
    source = tmp_path / "source.npy"
    np.save(source, np.zeros((96, 96), dtype=np.float32))
    previous_state = {
        "image": np.zeros((96, 96), dtype=np.float32),
        "image_path": str(source),
        "image_sha256": "a" * 64,
        "normalized_image_sha256": "b" * 64,
        "image_shape": [96, 96],
        "class_names": ["Old A", "Old B"],
        "colors": CLASS_COLORS,
        "points": [[], []],
        "classifier_patch_size": 64,
        "show_valid_region": False,
    }

    result = block_fn.fn(
        "cnn_8ch_small_patch", capabilities, previous_state, None
    )
    patch_index = _output_index_by_label(block_fn, "Model input patch size")
    names_index = _output_index_by_label(block_fn, "Local class names")
    assert result[patch_index] == 32
    assert result[names_index]["value"] == "Class A, Class B"


def test_run_setting_change_clears_results_and_refreshes_readiness(config) -> None:
    pytest.importorskip("gradio")
    capabilities = _capabilities()
    app = _build_ft(config, capabilities)
    block_fn = _handler(app, "on_run_setting_change")

    result = block_fn.fn(
        {},
        {},
        "cnn_8ch_pg17",
        None,
        capabilities,
        51,
        0,
        0.0005,
        4,
        512,
        "cpu",
        None,
    )
    assert len(result) == len(block_fn.outputs)
    model_index = _output_index_by_label(block_fn, "Download model (.symmodel)")
    assert result[model_index]["value"] is None
    assert result[model_index]["visible"] is False
    button_index = next(
        index
        for index, component in enumerate(block_fn.outputs)
        if getattr(component, "value", None) == "Fine-tune and predict"
    )
    assert result[button_index]["interactive"] is False


def test_image_change_clears_stale_downloads(config, tmp_path) -> None:
    pytest.importorskip("gradio")
    capabilities = _capabilities()
    app = _build_ft(config, capabilities)
    block_fn = _handler(app, "on_image_change")
    source = tmp_path / "source.npy"
    np.save(source, np.linspace(0.0, 1.0, 96 * 96, dtype=np.float32).reshape(96, 96))
    previous_state = {
        "image": np.zeros((96, 96), dtype=np.float32),
        "image_path": str(source),
        "image_sha256": "a" * 64,
        "normalized_image_sha256": "b" * 64,
        "image_shape": [96, 96],
        "class_names": ["Class A", "Class B"],
        "colors": CLASS_COLORS,
        "points": [[], []],
        "classifier_patch_size": 64,
        "show_valid_region": True,
    }
    result = block_fn.fn(str(source), previous_state, "cnn_8ch_pg17", capabilities)
    assert len(result) == len(block_fn.outputs)
    model_index = _output_index_by_label(block_fn, "Download model (.symmodel)")
    zip_index = _output_index_by_label(block_fn, "Download complete run (.zip)")
    assert result[model_index].get("value") is None
    assert result[zip_index].get("value") is None
    assert result[model_index].get("visible") is False
    assert result[zip_index].get("visible") is False


# ---------------------------------------------------------------------------
# Readiness checklist (protocol 3.11, 13.4)
# ---------------------------------------------------------------------------
def test_readiness_html_uses_icons_and_states_next_action() -> None:
    snapshot = {
        "model_supported": True,
        "model_name": "CNN",
        "checkpoint_ready": False,
        "checkpoint_detail": "Checkpoint is missing.",
        "image_valid": False,
        "image_name": "source.npy",
        "features_current": False,
        "class_names": ["Class A", "Class B"],
        "class_counts": [3, 3],
        "minimum_shots": 3,
        "recommended_shots": 5,
        "maximum_shots": 50,
        "symmetry_patch_size": 51,
        "epochs": 150,
        "learning_rate": 0.0005,
        "stride": 4,
        "batch_size": 512,
        "device": "cpu",
    }
    items = ft.evaluate_fine_tune_readiness(snapshot)
    html = readiness_panel_html(items)
    assert "\u2713" in html  # ready
    assert "\u00d7" in html  # blocker
    assert "Next:" in html


def test_three_points_per_class_is_runnable(config) -> None:
    snapshot = build_fine_tune_snapshot(
        config,
        _capabilities(),
        state={
            "image": np.zeros((96, 96), dtype=np.float32),
            "image_path": "source.npy",
            "image_sha256": "a" * 64,
            "image_shape": [96, 96],
            "class_names": ["Class A", "Class B"],
            "points": [[(1, 1), (2, 2), (3, 3)], [(4, 4), (5, 5), (6, 6)]],
        },
        feature_state={
            "cache_key": "does-not-match",
            "features_path": "missing.npz",
            "record_path": "missing.json",
        },
        model_identifier="cnn_8ch_pg17",
        checkpoint_path=None,
        symmetry_patch_size=51,
        epochs=150,
        learning_rate=0.0005,
        stride=4,
        batch_size=512,
        device="cpu",
    )
    # Missing/stale features still block here; the point is that three shots do
    # not block the min-support criterion.
    items = ft.evaluate_fine_tune_readiness(snapshot)
    min_item = next(item for item in items if item.key == "min_support")
    assert min_item.status == "ready"


# ---------------------------------------------------------------------------
# Result terminology (protocol 3.12, 13.6)
# ---------------------------------------------------------------------------
def test_no_cells_terminology_in_user_facing_components(config) -> None:
    pytest.importorskip("gradio")
    app = _build_ft(config, _capabilities())
    for component in _components(app):
        assert "Cells" not in str(component.get("props", {}))


def test_class_statistics_display_merges_names_colors_counts_fractions() -> None:
    statistics = _fine_tune_class_statistics(
        np.array([[0, 1], [1, 0]], dtype=np.int16), 2
    )
    html = ui_shared.class_statistics_html(
        statistics, class_names=["Class A", "Class B"], class_colors=CLASS_COLORS
    )
    assert "Cells" not in html
    assert "Predicted grid points" in html
    assert "Fraction of predicted grid" in html
    assert "Class A" in html and CLASS_COLORS[0] in html
    assert ">2<" in html
    assert "50.0%" in html
    assert ui_shared.CLASS_STATISTICS_NOTE.split(".")[0] in html


# ---------------------------------------------------------------------------
# Presentation rerender never triggers numerical work (protocol 3.4, 3.5, 13.5)
# ---------------------------------------------------------------------------
def test_color_or_alpha_rerender_never_runs_numerics(tmp_path, monkeypatch) -> None:
    run_dir, reference = _make_run(tmp_path)

    def _boom(*_args, **_kwargs):
        raise AssertionError(
            "presentation rerender must not trigger a numerical or Provider call"
        )

    monkeypatch.setattr(ft, "run_analysis", _boom)
    monkeypatch.setattr(ft, "run_provider_features", _boom)
    monkeypatch.setattr(ft, "install_registered_weight", _boom)

    before_bytes = (run_dir / "prediction.npz").read_bytes()
    before_grid = load_fine_tune_result(reference)[1].prediction_grid.copy()

    base = render_fine_tune_variants(reference, ["Class A", "Class B"], CLASS_COLORS, 0.48)
    recolored = render_fine_tune_variants(
        reference, ["Class A", "Class B"], ["#00ff00", "#0000ff"], 0.48
    )
    transparent = render_fine_tune_variants(
        reference, ["Class A", "Class B"], CLASS_COLORS, 0.9
    )

    # Class colors change the overlay and the categorical mask.
    assert not np.array_equal(base["overlay"], recolored["overlay"])
    assert not np.array_equal(base["mask"], recolored["mask"])
    # Alpha changes only the overlay.
    assert not np.array_equal(base["overlay"], transparent["overlay"])
    assert np.array_equal(base["mask"], transparent["mask"])
    # Scalar maps never depend on colors or alpha.
    for key in ("confidence", "entropy", "confidence_colorbar", "entropy_colorbar"):
        assert np.array_equal(base[key], recolored[key])
        assert np.array_equal(base[key], transparent[key])

    # The numerical result is untouched.
    assert (run_dir / "prediction.npz").read_bytes() == before_bytes
    assert np.array_equal(
        load_fine_tune_result(reference)[1].prediction_grid, before_grid
    )


def test_presentation_edits_do_not_leak_into_persistent_artifacts(tmp_path) -> None:
    run_dir, reference = _make_run(tmp_path)
    before_files = {p.name for p in run_dir.iterdir()}
    model_before = (run_dir / "fine_tuned_model.symmodel").read_bytes()
    record_before = (run_dir / "run_record.json").read_text(encoding="utf-8")
    zip_before = (run_dir / "symmetry-run-full-run.zip").read_bytes()

    paths = prepare_presentation_pngs(
        reference,
        ["Class A", "Class B"],
        ["#102030", "#405060"],
        0.7,
        list(ft.RESULT_EXPORT_NAMES),
    )
    assert len(paths) == len(ft.RESULT_EXPORT_NAMES)
    for path in paths:
        with Image.open(path) as image:
            image.load()

    # No new files in the run directory, and no transient colors in metadata.
    assert {p.name for p in run_dir.iterdir()} == before_files
    assert (run_dir / "fine_tuned_model.symmodel").read_bytes() == model_before
    assert (run_dir / "run_record.json").read_text(encoding="utf-8") == record_before
    assert (run_dir / "symmetry-run-full-run.zip").read_bytes() == zip_before
    assert "#102030" not in record_before
    assert "0.7" not in record_before


def test_model_download_bytes_match_durable_file(tmp_path) -> None:
    run_dir, reference = _make_run(tmp_path)
    served = prepare_model_download(reference)
    assert Path(served).read_bytes() == (
        run_dir / "fine_tuned_model.symmodel"
    ).read_bytes()


def test_invalid_hex_blocks_rerender_but_not_the_result(tmp_path) -> None:
    run_dir, reference = _make_run(tmp_path)
    before_grid = load_fine_tune_result(reference)[1].prediction_grid.copy()

    with pytest.raises(ft.PresentationError):
        parse_presentation_colors("not-a-color, #377eb8", 2)
    with pytest.raises(ft.PresentationError):
        render_fine_tune_variants(
            reference, ["Class A", "Class B"], ["#12", "#377eb8"], 0.48
        )
    with pytest.raises(ft.PresentationError):
        prepare_presentation_pngs(
            reference, ["Class A", "Class B"], ["#12", "#377eb8"], 0.48, ["overlay"]
        )

    # The completed numerical result remains valid and unchanged.
    assert (run_dir / "prediction.npz").is_file()
    assert np.array_equal(
        load_fine_tune_result(reference)[1].prediction_grid, before_grid
    )


def test_identical_colors_are_allowed() -> None:
    assert parse_presentation_colors("#112233, #112233", 2) == ["#112233", "#112233"]
    with pytest.raises(ft.PresentationError):
        parse_presentation_colors("112233, #112233", 2)


def test_presentation_state_roundtrip() -> None:
    state = PresentationState.defaults(["Class A", "Class B"], CLASS_COLORS)
    restored = PresentationState.from_dict(state.to_dict())
    assert restored.to_dict() == state.to_dict()
    assert restored.alpha == ft.DEFAULT_OVERLAY_ALPHA
