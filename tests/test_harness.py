"""Unit tests for the portable symmetry-harness core."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import pytest

from symmetry_harness.analysis import (
    dense_prediction_from_arrays,
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
from symmetry_harness.image_io import normalize_image
from symmetry_harness.initialization import initialize_config
from symmetry_harness.provider import (
    PROVIDER_CONTRACT_VERSION,
    default_provider_install_command,
    validate_provider_capabilities,
)
from symmetry_harness.ui import build_app, parse_class_names
from symmetry_harness.workflow import build_run_options, recorded_run_options


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CJK_PATTERN = re.compile(
    r"[\u2e80-\u2eff\u3000-\u303f\u3040-\u30ff\u3100-\u312f"
    r"\u31a0-\u31bf\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\uff00-\uffef\U00020000-\U0002fa1f]"
)
TEXT_SUFFIXES = {".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
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
        "models": [
            {
                "identifier": "cnn_8ch_pg17",
                "available": True,
                "input_channels": 8,
                "pretrained_classes": 17,
                "classifier_patch_size": 64,
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
    assert not list(REPOSITORY_ROOT.rglob("*.pt"))
    assert not list(REPOSITORY_ROOT.rglob("*.pth"))
    skill = (REPOSITORY_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert "D:\\" not in skill
    assert "C:\\" not in skill


def test_example_configuration_is_portable_and_valid() -> None:
    config = load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")
    assert config.model.identifier == "cnn_8ch_pg17"
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


def test_harness_source_has_no_in_process_model_runtime_dependency() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (REPOSITORY_ROOT / "src" / "symmetry_harness").glob("*.py")
    )
    assert "import torch" not in source
    assert "from torch" not in source
    assert "jdml" not in source.lower()
    assert "symmlearn.maps" not in source


def test_prediction_overlay_preserves_source_shape() -> None:
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
    assert overlay.shape == (96, 96, 3)


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


def test_cli_exposes_public_workflow_commands() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "symmetry_harness.cli", "--help"],
        text=True,
        capture_output=True,
        check=False,
        env=_source_environment(),
    )
    assert completed.returncode == 0
    for command in ("init", "doctor", "models", "inspect", "run", "reproduce", "ui"):
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
