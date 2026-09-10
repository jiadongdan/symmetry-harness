"""Unit tests for the portable symmetry-harness core."""

from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import warnings
import zipfile

import numpy as np
import pytest

from symmetry_harness import cli
from symmetry_harness.catalog import choose_model, weight_capability
from symmetry_harness.analysis import (
    dense_prediction_from_arrays,
    prediction_display_bounds,
    render_prediction_overlay,
    render_scalar_map,
    DensePrediction,
)
from symmetry_harness.annotations import (
    create_annotation_session,
    load_annotation_session,
    save_annotation_session,
    validate_annotation_session,
)
from symmetry_harness.config import load_harness_config
from symmetry_harness.image_io import file_sha256, normalize_image
from symmetry_harness.initialization import initialize_config
from symmetry_harness.provider import (
    PROVIDER_CONTRACT_VERSION,
    default_provider_install_command,
    validate_provider_capabilities,
)
import symmetry_harness.provider as provider_module
import symmetry_harness.ui as ui_module
import symmetry_harness.workflow as workflow_module
from symmetry_harness.ui import (
    _annotation_display_shape,
    _display_to_source_point,
    _feature_preview,
    _feature_request,
    _loaded_image_outputs,
    _protect_localhost_from_proxies,
    _resolve_server_port,
    build_app,
    feature_gallery,
    parse_class_names,
    prepare_feature_exports,
    progress_bar_html,
)
from symmetry_harness.workflow import build_run_options, recorded_run_options


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CJK_PATTERN = re.compile(
    r"[\u2e80-\u2eff\u3000-\u303f\u3040-\u30ff\u3100-\u312f"
    r"\u31a0-\u31bf\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\uff00-\uffef\U00020000-\U0002fa1f]"
)
TEXT_SUFFIXES = {
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {".gitattributes", ".gitignore", "LICENSE"}


def _source_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source = str(REPOSITORY_ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source if not existing else os.pathsep.join((source, existing))
    )
    return environment


def _session(points_per_class: int = 3):
    first = [(40 + index, 40) for index in range(points_per_class)]
    second = [(70 + index, 70) for index in range(points_per_class)]
    return create_annotation_session(
        image_path=str(REPOSITORY_ROOT / "example.npy"),
        image_sha256="a" * 64,
        image_shape=(128, 128),
        classifier_patch_size=64,
        class_names=["Phase A", "Phase B"],
        points_by_class=[first, second],
    )


def _capabilities() -> dict:
    return {
        "contract_version": PROVIDER_CONTRACT_VERSION,
        "provider": "symmetry-learn",
        "provider_version": "0.1.0",
        "operations": [
            "compute_features",
            "few_shot_analyze",
            "few_shot_analyze_precomputed_features",
            "probe_model",
        ],
        "models": [
            {
                "identifier": "cnn_8ch_pg17",
                "display_name": "Eight-channel CNN (PG17)",
                "available": True,
                "checkpoint_required": False,
                "input_channels": 8,
                "pretrained_classes": 17,
                "classifier_patch_size": 64,
                "feature_pipeline": "eight_channel_v1",
                "fine_tuning_strategy": (
                    "frozen pretrained network with residual adapters and a new local head"
                ),
                "minimum_shots_per_class": 3,
                "recommended_shots_per_class": 5,
                "maximum_shots_per_class": 50,
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
                "defaults": {
                    "n_max": 20,
                    "symmetry_patch_size": 51,
                    "rotation_folds": [2, 3, 4, 6],
                    "reflection_p": 2.0,
                    "normalize_rotation_maps": False,
                    "classifier_patch_size": 64,
                    "minimum_shots_per_class": 3,
                    "maximum_shots_per_class": 50,
                    "adapter_bottleneck": 16,
                    "epochs": 150,
                    "learning_rate": 0.0005,
                    "weight_decay": 0.0,
                    "seed": 42,
                    "stride": 4,
                    "batch_size": 512,
                    "device": "auto",
                },
                "default_weight": {
                    "identifier": "pg17-symmetry-v1",
                    "distribution": "symmetry-learn-default-model",
                    "version": "1.0.0",
                    "sha256": "c" * 64,
                    "size_bytes": 1024,
                    "installed_size_bytes": 1024,
                    "installed_path": "/models/pg17-symmetry-v1.pth",
                    "status": "installed",
                    "default": True,
                    "bundled": True,
                },
                "weights": [
                    {
                        "identifier": "pg17-symmetry-v1",
                        "distribution": "symmetry-learn-default-model",
                        "version": "1.0.0",
                        "sha256": "c" * 64,
                        "size_bytes": 1024,
                        "installed_size_bytes": 1024,
                        "installed_path": "/models/pg17-symmetry-v1.pth",
                        "status": "installed",
                        "default": True,
                        "bundled": True,
                    }
                ],
            }
        ],
    }


def test_repository_contains_no_cjk_text() -> None:
    violations = []
    for path in REPOSITORY_ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in TEXT_FILENAMES:
            continue
        if CJK_PATTERN.search(path.read_text(encoding="utf-8")):
            violations.append(str(path.relative_to(REPOSITORY_ROOT)))
    assert violations == []


def test_repository_does_not_vendor_models_or_private_paths() -> None:
    assert (REPOSITORY_ROOT / "SKILL.md").is_file()
    assert (REPOSITORY_ROOT / "pyproject.toml").is_file()
    tracked_weights = subprocess.run(
        ["git", "ls-files", "*.pt", "*.pth"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    assert tracked_weights == []
    skill = (REPOSITORY_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert "D:\\" not in skill
    assert "C:\\" not in skill


def test_installer_persists_runtime_and_copies_skill_resources(tmp_path) -> None:
    script_path = REPOSITORY_ROOT / "scripts" / "install.py"
    specification = importlib.util.spec_from_file_location(
        "symmetry_harness_install_script", script_path
    )
    assert specification is not None
    assert specification.loader is not None
    installer = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(installer)

    config_path = tmp_path / "source" / "symmetry-harness.json"
    config_path.parent.mkdir()
    config_path.write_text("{}\n", encoding="utf-8")
    state_directory = tmp_path / "state"
    runtime_path = installer.write_runtime_state(
        state_directory,
        harness_python=Path(sys.executable),
        config_path=config_path,
    )
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert runtime["schema_version"] == "symmetry-harness-runtime-v1"
    assert Path(runtime["harness_python"]) == Path(sys.executable).resolve()
    assert Path(runtime["config_path"]) == config_path.resolve()

    skill_path = installer.install_codex_skill(
        REPOSITORY_ROOT, tmp_path / "codex"
    )
    assert (skill_path / "SKILL.md").is_file()
    assert (skill_path / "agents" / "openai.yaml").is_file()
    assert (skill_path / "scripts" / "launch.ps1").is_file()
    assert (skill_path / "scripts" / "launch.sh").is_file()


def test_example_configuration_is_portable_and_valid() -> None:
    config = load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")
    assert config.model.identifier == "cnn_8ch_pg17"
    assert config.model.weight_identifier == "pg17-symmetry-v1"
    assert config.model.checkpoint_path is None
    assert config.model.input_channels == 8
    assert config.model.pretrained_classes == 17
    assert config.model.classifier_patch_size == 64
    assert config.provider.module == "symmlearn.provider.worker"
    assert Path(config.model.python_executable).is_file()
    assert config.features.rotation_folds == (2, 3, 4, 6)
    assert config.fine_tuning.minimum_shots_per_class == 3
    assert config.fine_tuning.recommended_shots_per_class == 5


def test_recorded_options_restore_all_execution_parameters() -> None:
    config = load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")
    payload = build_run_options(config).to_dict()
    payload["n_max"] = 31
    payload["adapter_bottleneck"] = 7
    restored = recorded_run_options(payload)
    assert restored.n_max == 31
    assert restored.adapter_bottleneck == 7
    assert restored.minimum_shots_per_class == 3


def test_classifier_patch_size_is_an_allowed_integer_run_override() -> None:
    config = load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")

    current_model = build_run_options(
        config, {"classifier_patch_size": 64}
    )
    future_model = build_run_options(
        config, {"classifier_patch_size": "96"}
    )

    assert current_model.classifier_patch_size == 64
    assert future_model.classifier_patch_size == 96
    with pytest.raises(ValueError, match="classifier_patch_size must be positive"):
        build_run_options(config, {"classifier_patch_size": 0})


def test_reproduction_matches_a_registered_weight_by_checksum(monkeypatch) -> None:
    config = load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        image_path = directory / "input.npy"
        np.save(image_path, np.linspace(0.0, 1.0, 128 * 128).reshape(128, 128))
        session = create_annotation_session(
            image_path=str(image_path),
            image_sha256=file_sha256(image_path),
            image_shape=(128, 128),
            classifier_patch_size=64,
            class_names=["Phase A", "Phase B"],
            points_by_class=[
                [(40, 40), (41, 40), (42, 40)],
                [(70, 70), (71, 70), (72, 70)],
            ],
        )
        annotation_path = directory / "annotation_session.json"
        save_annotation_session(annotation_path, session)
        record_path = directory / "run_record.json"
        record_path.write_text(
            json.dumps(
                {
                    "contract_version": "symmetry-harness-run-v1",
                    "status": "completed",
                    "model": {
                        "identifier": "cnn_8ch_pg17",
                        "checkpoint": {"sha256": "c" * 64},
                    },
                    "annotations": {"path": str(annotation_path)},
                    "artifacts": {"input_array": str(image_path)},
                    "input": {"path": str(image_path)},
                    "options": build_run_options(config).to_dict(),
                    "configuration": {
                        "fine_tuning": {
                            "minimum_shots_per_class": 3,
                            "maximum_shots_per_class": 50,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        monkeypatch.setattr(
            workflow_module,
            "provider_capabilities",
            lambda current: _capabilities(),
        )
        monkeypatch.setattr(
            workflow_module,
            "probe_model",
            lambda current: {"details": {"checkpoint": {"sha256": "c" * 64}}},
        )
        monkeypatch.setattr(
            workflow_module,
            "run_analysis",
            lambda current, **kwargs: {
                "status": "completed",
                "resolved_options": kwargs["resolved_options"].to_dict(),
            },
        )
        result = workflow_module.reproduce_analysis(config, record_path)

    assert result["status"] == "completed"
    assert result["resolved_options"]["classifier_patch_size"] == 64


def test_initialization_hashes_checkpoint_without_copying_it() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        checkpoint = directory / "trusted.pth"
        checkpoint.write_bytes(b"trusted checkpoint")
        destination = directory / "symmetry-harness.json"
        result = initialize_config(
            destination,
            checkpoint=checkpoint,
            provider_python=sys.executable,
            force=False,
            capabilities=_capabilities(),
        )
        payload = json.loads(destination.read_text(encoding="utf-8"))
        assert result["checkpoint_configured"] is True
        assert len(payload["model"]["checkpoint_sha256"]) == 64
        assert payload["model"]["checkpoint_path"] == str(checkpoint.resolve())
        assert checkpoint.read_bytes() == b"trusted checkpoint"
        config = load_harness_config(destination)
        method = provider_module._method_payload(config)
        assert method["checkpoint_path"] == str(checkpoint.resolve())
        assert method["checkpoint_sha256"] == payload["model"]["checkpoint_sha256"]
        assert "weight_identifier" not in method


def test_initialization_selects_the_installed_default_weight() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary) / "symmetry-harness.json"
        result = initialize_config(
            destination,
            checkpoint=None,
            provider_python=sys.executable,
            force=False,
            capabilities=_capabilities(),
        )
        config = load_harness_config(destination)

    assert result["checkpoint_configured"] is False
    assert result["weight_identifier"] == "pg17-symmetry-v1"
    assert result["weight_status"] == "installed"
    assert result["guidance"] == []
    assert config.model.weight_identifier == "pg17-symmetry-v1"
    assert config.model.checkpoint_path is None


def test_model_catalog_and_provider_payload_support_registered_weights() -> None:
    capabilities = _capabilities()
    model = choose_model(capabilities, None)
    weight = weight_capability(model, None)
    config = load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")
    payload = provider_module._method_payload(config)

    assert model["identifier"] == "cnn_8ch_pg17"
    assert weight["identifier"] == "pg17-symmetry-v1"
    assert payload == {
        "identifier": "cnn_8ch_pg17",
        "weight_identifier": "pg17-symmetry-v1",
    }

    multiple = json.loads(json.dumps(capabilities))
    second = dict(multiple["models"][0])
    second["identifier"] = "future_model"
    multiple["models"].append(second)
    with pytest.raises(ValueError, match="--model"):
        choose_model(multiple, None)
    assert choose_model(multiple, "future_model")["identifier"] == "future_model"


def test_normalization_contracts() -> None:
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    unit, record = normalize_image(image, "minmax_0_1")
    assert record["original_range"] == [0.0, 63.0]
    assert unit.dtype == np.float32
    assert float(unit.min()) == 0.0
    assert float(unit.max()) == 1.0
    with pytest.raises(ValueError, match="constant"):
        normalize_image(np.ones((8, 8)), "minmax_0_1")


def test_annotation_session_round_trip_and_bounds() -> None:
    session = _session()
    report = validate_annotation_session(
        session, minimum_shots=3, maximum_shots=5
    )
    assert report is None
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "annotations.json"
        save_annotation_session(path, session)
        loaded = load_annotation_session(path)
    assert loaded.to_dict() == session.to_dict()


def test_annotation_rejects_too_few_points_and_cross_class_duplicates() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        validate_annotation_session(_session(2), minimum_shots=3, maximum_shots=5)
    with pytest.raises(ValueError, match="multiple classes"):
        create_annotation_session(
            image_path=str(REPOSITORY_ROOT / "example.npy"),
            image_sha256="b" * 64,
            image_shape=(128, 128),
            classifier_patch_size=64,
            class_names=["Phase A", "Phase B"],
            points_by_class=[[(40, 40)], [(40, 40)]],
        )


def test_provider_capability_contract_and_install_guidance() -> None:
    config = load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")
    assert validate_provider_capabilities(config, _capabilities()) == []
    incompatible = _capabilities()
    incompatible["contract_version"] = "unsupported"
    assert "provider contract version mismatch" in validate_provider_capabilities(
        config, incompatible
    )
    command = default_provider_install_command("python")
    assert "symmetry-learn[provider]" in command


def test_provider_progress_file_is_forwarded_without_duplication(
    monkeypatch, tmp_path: Path
) -> None:
    config = load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")
    progress_path = tmp_path / "progress.json"

    class FakeProcess:
        returncode = 0

        def __init__(self) -> None:
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                progress_path.write_text(
                    json.dumps(
                        {"phase": "fine_tuning", "current": 4, "total": 10}
                    ),
                    encoding="utf-8",
                )
                raise subprocess.TimeoutExpired("provider", timeout)
            return json.dumps({"status": "completed"}), ""

        def kill(self) -> None:
            self.returncode = -1

    monkeypatch.setattr(
        provider_module.subprocess, "Popen", lambda *args, **kwargs: FakeProcess()
    )
    updates = []
    result = provider_module._invoke_provider_json(
        config,
        "--job",
        tmp_path / "job.json",
        timeout_seconds=5,
        progress_path=progress_path,
        progress_callback=lambda *values: updates.append(values),
    )

    assert result == {"status": "completed"}
    assert updates == [("fine_tuning", 4, 10)]


def test_doctor_reports_stage_timings(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        checkpoint = directory / "trusted.pth"
        checkpoint.write_bytes(b"trusted checkpoint")
        destination = directory / "symmetry-harness.json"
        initialize_config(
            destination,
            checkpoint=checkpoint,
            provider_python=sys.executable,
            force=False,
            capabilities=_capabilities(),
        )
        config = load_harness_config(destination)
        monkeypatch.setattr(
            provider_module, "_invoke_provider_json", lambda *args, **kwargs: _capabilities()
        )
        monkeypatch.setattr(
            provider_module,
            "probe_model",
            lambda current: {"identifier": current.model.identifier},
        )
        monkeypatch.setattr(provider_module, "find_spec", lambda name: object())
        report = provider_module.doctor(config, require_ui=True)

    assert report["status"] == "ready"
    assert set(report["timings_seconds"]) == {
        "provider_capabilities",
        "checkpoint_sha256",
        "model_probe",
        "ui_dependency",
        "total",
    }
    assert all(value >= 0 for value in report["timings_seconds"].values())


def test_doctor_accepts_an_installed_registered_default_weight(monkeypatch) -> None:
    config = load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")

    def fake_invoke(current, action, path=None, *, timeout_seconds):
        if action == "--capabilities":
            return _capabilities()
        assert action == "--probe"
        return {
            "provider_contract_version": PROVIDER_CONTRACT_VERSION,
            "provider": "symmetry-learn",
            "identifier": "cnn_8ch_pg17",
            "details": {
                "checkpoint": {
                    "path": "/models/pg17-symmetry-v1.pth",
                    "sha256": "c" * 64,
                    "strict_load": True,
                    "weight_identifier": "pg17-symmetry-v1",
                    "source": "bundled",
                    "bundled": True,
                }
            },
        }

    monkeypatch.setattr(provider_module, "_invoke_provider_json", fake_invoke)
    report = provider_module.doctor(config)

    assert report["status"] == "ready"
    assert report["issues"] == []
    assert report["model"]["weight_identifier"] == "pg17-symmetry-v1"
    assert report["checkpoint"]["kind"] == "registered"
    assert report["checkpoint"]["bundled"] is True
    assert report["checkpoint"]["strict_load"] is True


def test_doctor_blocks_a_missing_registered_weight_with_install_guidance(
    monkeypatch,
) -> None:
    config = load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")
    capabilities = _capabilities()
    model = capabilities["models"][0]
    model["default_weight"]["status"] = "missing"
    model["default_weight"]["installed_path"] = None
    model["weights"][0]["status"] = "missing"
    model["weights"][0]["installed_path"] = None
    monkeypatch.setattr(
        provider_module,
        "_invoke_provider_json",
        lambda *args, **kwargs: capabilities,
    )
    monkeypatch.setattr(
        provider_module,
        "find_spec",
        lambda name: object() if name == "gradio" else None,
    )

    report = provider_module.doctor(config)

    assert report["status"] == "blocked"
    assert "not installed" in report["issues"][0]
    assert any(
        "symmetry-learn-default-model==1.0.0" in item
        for item in report["recommendations"]
    )

    ui_report = provider_module.doctor(config, require_ui=True)
    assert ui_report["status"] == "ready"
    assert ui_report["model"]["status"] == "selection_required"
    assert ui_report["model"]["probe"] is None


def test_registered_weight_installation_is_explicit_and_refreshes_capabilities(
    monkeypatch,
) -> None:
    config = load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")
    missing = _capabilities()
    missing_model = missing["models"][0]
    missing_model["default_weight"]["status"] = "missing"
    missing_model["weights"][0]["status"] = "missing"
    installed = _capabilities()
    discovered = iter((missing, installed))
    commands = []
    monkeypatch.setattr(
        provider_module,
        "provider_capabilities",
        lambda current: next(discovered),
    )

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="installed", stderr="")

    monkeypatch.setattr(provider_module.subprocess, "run", fake_run)
    result = provider_module.install_registered_weight(
        config,
        model_identifier="cnn_8ch_pg17",
        weight_identifier="pg17-symmetry-v1",
    )

    assert commands == [
        [
            config.model.python_executable,
            "-m",
            "pip",
            "install",
            "symmetry-learn-default-model==1.0.0",
        ]
    ]
    assert result["status"] == "installed"
    assert result["weight"]["status"] == "installed"


def test_harness_source_has_no_in_process_model_runtime_dependency() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (REPOSITORY_ROOT / "src" / "symmetry_harness").glob("*.py")
    )
    assert "import torch" not in source
    assert "from torch" not in source
    assert "jdml" not in source.lower()
    assert "symmlearn.maps" not in source


def test_prediction_overlay_crops_to_dense_prediction_region() -> None:
    x_values = np.array([32, 36], dtype=np.int32)
    y_values = np.array([32, 36], dtype=np.int32)
    coordinates = np.array([[32, 32], [36, 32], [32, 36], [36, 36]], dtype=np.int32)
    prediction = DensePrediction(
        coordinates_xy=coordinates,
        x_coordinates=x_values,
        y_coordinates=y_values,
        logits=np.zeros((4, 2), dtype=np.float32),
        probabilities=np.full((4, 2), 0.5, dtype=np.float32),
        predictions=np.array([0, 1, 1, 0], dtype=np.int16),
        confidence=np.full(4, 0.5, dtype=np.float32),
        entropy=np.full(4, np.log(2), dtype=np.float32),
        prediction_grid=np.array([[0, 1], [1, 0]], dtype=np.int16),
        confidence_grid=np.full((2, 2), 0.5, dtype=np.float32),
        entropy_grid=np.full((2, 2), np.log(2), dtype=np.float32),
    )
    overlay = render_prediction_overlay(
        np.zeros((96, 96), dtype=np.float32),
        prediction,
        ["#ff0000", "#0000ff"],
        stride=4,
    )
    assert prediction_display_bounds(prediction, (96, 96), stride=4) == (
        30,
        30,
        38,
        38,
    )
    assert overlay.shape == (8, 8, 3)


def test_progress_bar_html_clamps_values_and_escapes_text() -> None:
    rendered = progress_bar_html("Fine <tuning>", 12, 10, "Done & saved")
    assert "Fine &lt;tuning&gt;" in rendered
    assert "Done &amp; saved" in rendered
    assert "100%" in rendered
    assert "width: 100%" in rendered


def test_provider_prediction_arrays_are_validated() -> None:
    arrays = {
        "coordinates_xy": np.array([[32, 32]], dtype=np.int32),
        "x_coordinates": np.array([32], dtype=np.int32),
        "y_coordinates": np.array([32], dtype=np.int32),
        "logits": np.array([[0.0, 1.0]], dtype=np.float32),
        "probabilities": np.array([[0.25, 0.75]], dtype=np.float32),
        "predictions": np.array([1], dtype=np.int16),
        "confidence": np.array([0.75], dtype=np.float32),
        "entropy": np.array([0.5], dtype=np.float32),
        "prediction_grid": np.array([[1]], dtype=np.int16),
        "confidence_grid": np.array([[0.75]], dtype=np.float32),
        "entropy_grid": np.array([[0.5]], dtype=np.float32),
    }
    prediction = dense_prediction_from_arrays(arrays)
    assert prediction.probabilities.shape == (1, 2)
    with pytest.raises(RuntimeError, match="missing arrays"):
        dense_prediction_from_arrays({})


def test_scalar_map_uses_fixed_value_range() -> None:
    rendered = render_scalar_map(
        np.full((2, 2), 0.5, dtype=np.float32),
        (4, 4),
        value_range=(0.0, 1.0),
    )
    assert rendered.shape == (4, 4, 3)
    assert rendered[0, 0].tolist() == [128, 255, 128]


def test_class_name_parser() -> None:
    assert parse_class_names("Alpha, Beta") == ["Alpha", "Beta"]
    with pytest.raises(ValueError, match="at least two"):
        parse_class_names("Alpha")
    with pytest.raises(ValueError, match="unique"):
        parse_class_names("Alpha, Alpha")


def test_feature_cache_key_ignores_support_points_but_tracks_feature_options() -> None:
    config = load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")
    image = np.linspace(0.0, 1.0, 128 * 128, dtype=np.float32).reshape(128, 128)
    record = {
        "path": str(REPOSITORY_ROOT / "example.npy"),
        "sha256": "a" * 64,
        "shape": [128, 128],
        "dtype": "float32",
        "normalization": {"policy": "minmax_0_1"},
    }
    _, state, _ = _loaded_image_outputs(image, record, config)
    _, _, first_fingerprint, first_key = _feature_request(
        config,
        _capabilities(),
        state,
        model_identifier="cnn_8ch_pg17",
        symmetry_patch_size=51,
        device="cpu",
    )
    state["class_names"] = ["A", "B"]
    state["points"] = [[(40, 40)], [(80, 80)]]
    _, _, _, support_changed_key = _feature_request(
        config,
        _capabilities(),
        state,
        model_identifier="cnn_8ch_pg17",
        symmetry_patch_size=51,
        device="cpu",
    )
    _, _, _, option_changed_key = _feature_request(
        config,
        _capabilities(),
        state,
        model_identifier="cnn_8ch_pg17",
        symmetry_patch_size=53,
        device="cpu",
    )

    assert support_changed_key == first_key
    assert option_changed_key != first_key
    assert first_fingerprint == json.loads(json.dumps(first_fingerprint))


def test_feature_previews_use_fixed_color_ranges() -> None:
    unsigned = np.asarray([[-0.5, 0.0, 0.25, 1.0, 1.5]], dtype=np.float32)
    signed = np.asarray([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)

    assert _feature_preview(unsigned, "rotation_4_fold").tolist() == [
        [0, 0, 64, 255, 255]
    ]
    assert _feature_preview(signed, "reflection_sin_2theta").tolist() == [
        [0, 0, 128, 255, 255]
    ]
    gallery = feature_gallery(
        np.stack((unsigned, signed)),
        np.asarray(("rotation_4_fold", "reflection_sin_2theta")),
    )
    assert gallery[0][1].endswith("[color range 0 to 1]")
    assert gallery[1][1].endswith("[color range -1 to 1]")


def test_feature_gallery_and_exports_preserve_all_eight_channels(
    tmp_path: Path,
) -> None:
    names = np.asarray(_capabilities()["models"][0]["feature_channels"])
    features = np.stack(
        [np.full((16, 16), index / 7.0, dtype=np.float32) for index in range(8)]
    )
    feature_path = tmp_path / "features.npz"
    record_path = tmp_path / "feature_record.json"
    np.savez_compressed(feature_path, features=features, channel_names=names)
    record_path.write_text("{}", encoding="utf-8")
    state = {
        "features_path": str(feature_path),
        "record_path": str(record_path),
        "feature_shape": list(features.shape),
    }

    gallery = feature_gallery(features, names)
    exports, _ = prepare_feature_exports(state, ["PNG", "NPY"])

    assert len(gallery) == 8
    assert all(item[0].shape == (16, 16) for item in gallery)
    assert {Path(path).suffix for path in exports} == {".zip", ".npy"}
    assert np.array_equal(np.load(next(Path(path) for path in exports if path.endswith(".npy"))), features)
    archive_path = next(Path(path) for path in exports if path.endswith(".zip"))
    with zipfile.ZipFile(archive_path) as archive:
        names_in_archive = archive.namelist()
    assert len(names_in_archive) == 9
    assert "symmetry_features_montage.png" in names_in_archive


def test_gradio_interface_builds_when_ui_extra_is_installed() -> None:
    pytest.importorskip("gradio")
    config = load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        app = build_app(config)
    argument_warnings = [
        str(item.message)
        for item in captured
        if "Expected" in str(item.message) and "arguments" in str(item.message)
    ]
    assert type(app).__name__ == "Blocks"
    assert argument_warnings == []
    components = app.get_config_file()["components"]
    patch_controls = [
        component
        for component in components
        if component.get("props", {}).get("label") == "Model input patch size"
    ]
    assert len(patch_controls) == 1
    assert patch_controls[0]["props"]["value"] == 64
    assert patch_controls[0]["props"]["interactive"] is False
    configure_buttons = [
        component
        for component in components
        if component.get("props", {}).get("value")
        == "Configure classes and patch size"
    ]
    assert len(configure_buttons) == 1


def test_loading_a_new_image_starts_an_empty_annotation_state() -> None:
    config = load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")
    image = np.linspace(0.0, 1.0, 96 * 96, dtype=np.float32).reshape(96, 96)
    record = {
        "path": str(REPOSITORY_ROOT / "replacement.npy"),
        "sha256": "d" * 64,
        "shape": [96, 96],
        "dtype": "float32",
    }
    previous = {
        "class_names": ["Phase A", "Phase B"],
        "colors": ["#e41a1c", "#377eb8"],
        "points": [[(40, 40)], [(70, 70)]],
    }
    annotated, state, status = _loaded_image_outputs(
        image, record, config, previous
    )

    assert annotated.shape == (720, 720, 3)
    assert state["class_names"] == ["Phase A", "Phase B"]
    assert state["points"] == [[], []]
    assert state["image_sha256"] == "d" * 64
    assert state["classifier_patch_size"] == 64
    assert state["show_valid_region"] is False
    assert "Class definitions were retained" in status


def test_annotation_display_coordinates_map_back_to_source_pixels() -> None:
    assert _annotation_display_shape((430, 430)) == (720, 720)
    assert _display_to_source_point((0, 0), (430, 430)) == (0, 0)
    assert _display_to_source_point((360, 360), (430, 430)) == (215, 215)
    assert _display_to_source_point((719, 719), (430, 430)) == (429, 429)

    assert _annotation_display_shape((400, 430)) == (670, 720)
    assert _display_to_source_point((719, 669), (400, 430)) == (429, 399)
    with pytest.raises(ValueError, match="outside"):
        _display_to_source_point((720, 0), (430, 430))


def test_mapped_display_click_remains_under_rendered_point_marker() -> None:
    display_point = (480, 240)
    source_point = _display_to_source_point(display_point, (430, 430))
    state = {
        "image": np.zeros((430, 430), dtype=np.float32),
        "image_shape": [430, 430],
        "class_names": ["A"],
        "colors": ["#e41a1c"],
        "points": [[source_point]],
    }

    annotated = ui_module.render_annotations(state, patch_size=64)
    display_x, display_y = display_point
    neighborhood = annotated[
        display_y - 2 : display_y + 3,
        display_x - 2 : display_x + 3,
    ]
    red_marker = (
        (neighborhood[..., 0] > 180)
        & (neighborhood[..., 1] < 100)
        & (neighborhood[..., 2] < 100)
    )

    assert annotated.shape == (720, 720, 3)
    assert red_marker.any()


def test_valid_region_and_patch_preview_follow_configured_patch_size() -> None:
    pytest.importorskip("gradio")
    config = load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")
    state = {
        "image": np.zeros((430, 430), dtype=np.float32),
        "image_shape": [430, 430],
        "class_names": [],
        "colors": [],
        "points": [],
    }

    configured = ui_module._configure_classes(
        "Phase A, Phase B",
        state,
        config,
        classifier_patch_size=96,
    )
    configured_state = configured[0]
    annotated = configured[2]

    assert configured_state["classifier_patch_size"] == 96
    assert configured_state["show_valid_region"] is True
    valid_edge = round(48 * 719 / 429)
    assert np.all(annotated[valid_edge, valid_edge] >= 250)

    accepted = ui_module._add_point(
        configured_state, "Phase A", (100, 100), config
    )
    assert accepted[2].shape == (96, 96)

    rejected = ui_module._add_point(
        accepted[0], "Phase A", (47, 100), config
    )
    assert rejected[0]["points"] == accepted[0]["points"]
    assert rejected[4] == ui_module.INVALID_SUPPORT_POINT_MESSAGE


def test_zero_server_port_resolves_to_available_local_port() -> None:
    port = _resolve_server_port("127.0.0.1", 0)
    assert isinstance(port, int)
    assert 0 < port <= 65535


def test_gradio_localhost_bypasses_system_proxies(monkeypatch) -> None:
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setenv("NO_PROXY", "example.test")
    _protect_localhost_from_proxies()

    assert os.environ["NO_PROXY"] == "example.test,127.0.0.1,localhost"
    assert os.environ["no_proxy"] == "example.test,127.0.0.1,localhost"


def test_launch_preflights_once_and_emits_ready_json(monkeypatch, capsys) -> None:
    config_path = REPOSITORY_ROOT / "configs" / "config.example.json"
    input_path = REPOSITORY_ROOT / "example.npy"
    readiness = {
        "status": "ready",
        "environment": {"capabilities": _capabilities()},
        "model": {"identifier": "cnn_8ch_pg17", "probe": {}},
        "checkpoint": {},
        "issues": [],
        "recommendations": [],
        "timings_seconds": {"total": 0.01},
    }
    calls = {"doctor": 0, "inspect": 0, "launch": 0}

    def fake_doctor(config, *, require_ui=False):
        calls["doctor"] += 1
        assert require_ui is True
        return readiness

    image = np.zeros((96, 96), dtype=np.float32)
    input_record = {
        "path": str(input_path),
        "sha256": "b" * 64,
        "shape": [96, 96],
        "dtype": "float32",
        "finite": True,
        "normalization": {"policy": "minmax_0_1"},
    }

    def fake_inspect(path, normalization):
        calls["inspect"] += 1
        assert Path(path) == input_path
        assert normalization == "minmax_0_1"
        return image, input_record

    def fake_launch_ui(config, **kwargs):
        calls["launch"] += 1
        assert kwargs["server_port"] == 0
        prepared_image, prepared_record = kwargs["prepared_input"]
        assert prepared_image is image
        assert prepared_record is input_record
        kwargs["on_ready"]("http://127.0.0.1:54321")

    monkeypatch.setattr(cli, "doctor", fake_doctor)
    monkeypatch.setattr(cli, "inspect_input", fake_inspect)
    monkeypatch.setattr(ui_module, "launch_ui", fake_launch_ui)
    arguments = cli._parser().parse_args(
        [
            "launch",
            "--config",
            str(config_path),
            "--input",
            str(input_path),
            "--no-inbrowser",
        ]
    )

    assert cli._execute(arguments) is None
    payload = json.loads(capsys.readouterr().out)
    assert calls == {"doctor": 1, "inspect": 1, "launch": 1}
    assert payload["status"] == "ready"
    assert payload["url"] == "http://127.0.0.1:54321"
    assert payload["input"]["sha256"] == "b" * 64
    assert payload["timings_seconds"]["total"] >= 0


def test_launch_without_input_waits_for_gradio_selection(monkeypatch, capsys) -> None:
    config_path = REPOSITORY_ROOT / "configs" / "config.example.json"
    readiness = {
        "status": "ready",
        "environment": {"capabilities": _capabilities()},
        "model": {
            "identifier": "cnn_8ch_pg17",
            "weight_identifier": "pg17-symmetry-v1",
            "probe": {},
        },
        "checkpoint": {},
        "issues": [],
        "recommendations": [],
        "timings_seconds": {"total": 0.01},
    }

    monkeypatch.setattr(
        cli, "doctor", lambda config, *, require_ui=False: readiness
    )

    def fake_launch_ui(config, **kwargs):
        assert kwargs["prepared_input"] is None
        assert kwargs["capabilities"] == _capabilities()
        kwargs["on_ready"]("http://127.0.0.1:54322")

    monkeypatch.setattr(ui_module, "launch_ui", fake_launch_ui)
    arguments = cli._parser().parse_args(
        [
            "launch",
            "--config",
            str(config_path),
            "--no-inbrowser",
        ]
    )

    assert cli._execute(arguments) is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ready"
    assert payload["input"] is None
    assert payload["input_status"] == "awaiting_user_selection"


def test_cli_exposes_public_workflow_commands() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "symmetry_harness.cli", "--help"],
        text=True,
        capture_output=True,
        check=False,
        env=_source_environment(),
    )
    assert completed.returncode == 0
    for command in (
        "init",
        "doctor",
        "models",
        "inspect",
        "launch",
        "run",
        "reproduce",
        "ui",
    ):
        assert command in completed.stdout


def test_doctor_without_config_returns_actionable_json() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        missing = Path(temporary) / "missing.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "symmetry_harness.cli",
                "doctor",
                "--config",
                str(missing),
            ],
            text=True,
            capture_output=True,
            check=False,
            env=_source_environment(),
        )
        assert completed.returncode == 0
        payload = json.loads(completed.stdout)
        assert payload["status"] == "blocked"
        assert any("symmetry init" in item for item in payload["recommendations"])
