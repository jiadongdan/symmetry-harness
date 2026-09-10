"""Harness orchestration tests for the traditional ML validation workflow.

The Provider is mocked here on purpose: the real Provider contract is locked by
``tests/test_traditional_integration.py``, which runs the actual worker. These
tests stay fast and CPU-only so they can cover result validation, run artifacts,
malformed Provider payloads, and the failure record.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

import symmetry_harness.traditional_workflow as workflow_module
from symmetry_harness.analysis import (
    TRADITIONAL_REQUIRED_ARRAYS,
    traditional_prediction_from_arrays,
)
from symmetry_harness.annotations import create_annotation_session
from symmetry_harness.config import load_harness_config
from symmetry_harness.contracts import TraditionalMLSettings
from symmetry_harness.image_io import file_sha256
from symmetry_harness.provider import ProviderTraditionalMLResult
from symmetry_harness.traditional_workflow import artifact_paths, run_traditional_validation


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "config.example.json"

IMAGE_SIZE = 96
PATCH_SIZE = 64
STRIDE = 8

CLASS_NAMES = ("Alpha", "Beta")
SUPPORT_POINTS = (
    ((40, 40), (44, 44)),
    ((52, 52), (56, 56)),
)

CHANNEL_NAMES = [f"channel_{index}" for index in range(8)]


def _config():
    return load_harness_config(CONFIG_PATH)


def _capabilities() -> dict:
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
                "feature_pipeline": "eight_channel_v1",
                "feature_channels": list(CHANNEL_NAMES),
            }
        ],
        "traditional_ml": {
            "schema_version": "symmetry-traditional-ml-capability-v1",
            "feature_modes": ["raw_image", "image_plus_symmetry_maps"],
            "classifiers": [
                {
                    "identifier": "logistic_regression",
                    "supports_predict_proba": True,
                    "defaults": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
                },
                {
                    "identifier": "random_forest",
                    "supports_predict_proba": True,
                    "defaults": {"n_estimators": 300, "n_jobs": 1},
                },
            ],
        },
    }


def _readiness(config=None, *, require_ui: bool = True) -> dict:
    capabilities = _capabilities()
    return {
        "status": "ready",
        "environment": {"status": "ready", "capabilities": capabilities},
        "traditional_ml": capabilities["traditional_ml"],
        "device": "cpu",
        "issues": [],
        "recommendations": [],
        "timings_seconds": {},
    }


def _unit_image(size: int = IMAGE_SIZE) -> tuple[np.ndarray, Path]:
    y, x = np.mgrid[:size, :size]
    values = np.sin(x / 6.0) + np.cos(y / 7.0)
    return ((values - values.min()) / np.ptp(values)).astype(np.float32)


def _setup(tmp_path: Path) -> dict:
    image = _unit_image()
    image_path = tmp_path / "input.npy"
    np.save(image_path, image)

    features = np.random.default_rng(3).random((8, *image.shape), dtype=np.float32)
    features_path = tmp_path / "features.npz"
    np.savez_compressed(
        features_path,
        features=features,
        channel_names=np.asarray(CHANNEL_NAMES),
    )
    features_record_path = tmp_path / "features_record.json"
    features_record_path.write_text(
        json.dumps(
            {
                "features": {
                    "shape": [int(value) for value in features.shape],
                    "channel_names": list(CHANNEL_NAMES),
                }
            }
        ),
        encoding="utf-8",
    )

    session = create_annotation_session(
        image_path=str(image_path),
        image_sha256=file_sha256(image_path),
        image_shape=(image.shape[0], image.shape[1]),
        classifier_patch_size=PATCH_SIZE,
        class_names=CLASS_NAMES,
        points_by_class=SUPPORT_POINTS,
    )
    return {
        "config": _config(),
        "image_path": image_path,
        "features_path": features_path,
        "features_record_path": features_record_path,
        "session": session,
        "settings": TraditionalMLSettings(
            classifier_patch_size=PATCH_SIZE, stride=STRIDE, batch_size=32
        ),
        "tmp_path": tmp_path,
    }


def _arrays(grid_shape: tuple[int, int] = (2, 2), class_count: int = 2) -> dict:
    y_coordinates = np.arange(grid_shape[0], dtype=np.int32) * STRIDE
    x_coordinates = np.arange(grid_shape[1], dtype=np.int32) * STRIDE
    sample_count = int(np.prod(grid_shape))
    probabilities = np.full((sample_count, class_count), 1.0 / class_count, np.float32)
    return {
        "coordinates_xy": np.asarray(
            [(int(x), int(y)) for y in y_coordinates for x in x_coordinates],
            dtype=np.int32,
        ),
        "x_coordinates": x_coordinates,
        "y_coordinates": y_coordinates,
        "probabilities": probabilities,
        "predictions": np.zeros(sample_count, dtype=np.int16),
        "confidence": np.full(sample_count, 1.0 / class_count, dtype=np.float32),
        "entropy": np.full(sample_count, float(np.log(class_count)), dtype=np.float32),
        "prediction_grid": np.zeros(grid_shape, dtype=np.int16),
        "confidence_grid": np.full(grid_shape, 1.0 / class_count, dtype=np.float32),
        "entropy_grid": np.full(
            grid_shape, float(np.log(class_count)), dtype=np.float32
        ),
    }


def _record(**overrides) -> dict:
    record = {
        "provider": "symmetry-learn",
        "provider_version": "0.1.1",
        "provider_contract_version": "symmetry-learn-provider-v1",
        "worker_schema_version": "symmetry-provider-worker-v1",
        "classifier": {
            "identifier": "logistic_regression",
            "parameters": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
        },
        "feature_mode": "image_plus_symmetry_maps",
        "channel_count": 8,
        "channel_names": list(CHANNEL_NAMES),
        "class_names": list(CLASS_NAMES),
        "class_count": 2,
        "support_counts": {"Alpha": 2, "Beta": 2},
        "support": {"sha256": "c" * 64},
        "feature_sha256": "d" * 64,
        "options": {"classifier_patch_size": PATCH_SIZE, "seed": 42},
        "seed": 42,
        "stride": STRIDE,
        "batch_size": 32,
        "classifier_patch_size": PATCH_SIZE,
        "grid_shape": [2, 2],
        "sample_count": 4,
        "training_matrix_shape": [4, 32768],
        "support_training_accuracy": 1.0,
        "timings_seconds": {"fit": 0.01, "prediction": 0.02},
        "runtime": {"device": "cpu"},
        "warnings": [],
    }
    record.update(overrides)
    return record


def _fake_provider(monkeypatch, *, arrays=None, record=None):
    """Install a fake Provider that writes the artifacts the workflow expects."""
    captured: dict = {}

    def fake_run(config, **kwargs):
        captured.update(kwargs)
        resolved_arrays = _arrays() if arrays is None else arrays
        resolved_record = _record() if record is None else record
        if resolved_arrays is not None:
            np.savez_compressed(kwargs["output_path"], **resolved_arrays)
        if resolved_record is not None:
            Path(kwargs["record_path"]).write_text(
                json.dumps(resolved_record, indent=2, sort_keys=True), encoding="utf-8"
            )
        progress_path = kwargs.get("progress_path")
        if progress_path is not None:
            Path(progress_path).write_text(
                json.dumps({"phase": "prediction", "current": 4, "total": 4}),
                encoding="utf-8",
            )
        callback = kwargs.get("progress_callback")
        if callback is not None:
            callback("prediction", 4, 4)
        return ProviderTraditionalMLResult(
            arrays=resolved_arrays if resolved_arrays is not None else {},
            record=resolved_record if resolved_record is not None else {},
        )

    monkeypatch.setattr(workflow_module, "traditional_readiness", _readiness)
    monkeypatch.setattr(workflow_module, "run_provider_traditional_ml", fake_run)
    return captured


def _run(setup, tmp_path, *, feature_mode="image_plus_symmetry_maps", classifier=None):
    return run_traditional_validation(
        setup["config"],
        image_path=setup["image_path"],
        annotation_session=setup["session"],
        classifier=classifier or "logistic_regression",
        feature_mode=feature_mode,
        settings=setup["settings"],
        features_path=setup["features_path"],
        features_record_path=setup["features_record_path"],
        output_root=tmp_path / "runs",
    )


# --- result array validation ------------------------------------------------


def test_array_validation_accepts_a_logit_free_contract() -> None:
    payload = _arrays()
    assert "logits" not in payload
    prediction = traditional_prediction_from_arrays(payload, class_count=2)
    assert prediction.probabilities.shape == (4, 2)
    assert prediction.prediction_grid.shape == (2, 2)


def test_array_validation_reports_every_missing_array() -> None:
    payload = _arrays()
    payload.pop("entropy")
    payload.pop("confidence_grid")
    with pytest.raises(RuntimeError, match="missing arrays"):
        traditional_prediction_from_arrays(payload, class_count=2)


def test_array_validation_rejects_a_single_class_distribution() -> None:
    payload = _arrays(class_count=1)
    with pytest.raises(RuntimeError, match="two classes"):
        traditional_prediction_from_arrays(payload)


def test_array_validation_rejects_nonfinite_negative_and_unnormalized_rows() -> None:
    nonfinite = _arrays()
    nonfinite["probabilities"][0, 0] = np.nan
    with pytest.raises(RuntimeError, match="nonfinite"):
        traditional_prediction_from_arrays(nonfinite)

    negative = _arrays()
    negative["probabilities"][0] = [-0.5, 1.5]
    with pytest.raises(RuntimeError, match="negative"):
        traditional_prediction_from_arrays(negative)

    unnormalized = _arrays()
    unnormalized["probabilities"][0] = [0.2, 0.2]
    with pytest.raises(RuntimeError, match="sum to one"):
        traditional_prediction_from_arrays(unnormalized)


def test_array_validation_rejects_misaligned_shapes() -> None:
    misaligned = _arrays()
    misaligned["confidence"] = misaligned["confidence"][:3]
    with pytest.raises(RuntimeError, match="vector outputs"):
        traditional_prediction_from_arrays(misaligned)

    grid = _arrays()
    grid["prediction_grid"] = np.zeros((3, 3), dtype=np.int16)
    with pytest.raises(RuntimeError, match="grid outputs"):
        traditional_prediction_from_arrays(grid)

    wrong_class_count = _arrays()
    with pytest.raises(RuntimeError, match="requested classes"):
        traditional_prediction_from_arrays(wrong_class_count, class_count=3)


def test_traditional_required_arrays_do_not_include_logits() -> None:
    assert "logits" not in TRADITIONAL_REQUIRED_ARRAYS
    assert "probabilities" in TRADITIONAL_REQUIRED_ARRAYS


# --- completed runs ---------------------------------------------------------


def test_completed_run_writes_every_artifact_and_archive_member(
    tmp_path, monkeypatch
) -> None:
    setup = _setup(tmp_path)
    captured = _fake_provider(monkeypatch)

    result = _run(setup, tmp_path)

    assert result["status"] == "completed"
    assert result["seed"] == 42
    assert result["classifier_patch_size"] == PATCH_SIZE
    run_directory = Path(result["run_directory"])
    assert run_directory.name.startswith("traditional-validation-")

    for name, artifact in artifact_paths(run_directory).items():
        assert Path(artifact).is_file(), name

    archive_path = Path(result["results_archive"])
    with ZipFile(archive_path) as archive:
        members = set(archive.namelist())
    assert {
        "run_record.json",
        "training_summary.json",
        "provider_record.json",
        "annotation_session.json",
        "features.npz",
        "features_record.json",
        "traditional_prediction.npz",
        "progress.json",
        "prediction_overlay.png",
        "confidence.png",
        "entropy.png",
        "report.md",
        "input.npy",
    } <= members
    assert "results.zip" not in members

    record = json.loads(Path(result["run_record"]).read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["pretrained_checkpoint_accessed"] is False
    assert record["traditional_ml"]["class_names"] == list(CLASS_NAMES)
    assert record["traditional_ml"]["seed"] == 42
    assert record["features"]["shape"] == [8, IMAGE_SIZE, IMAGE_SIZE]
    assert record["provider"]["operation"] == "traditional_ml_analyze"
    assert any("exploratory" in warning for warning in record["warnings"])

    summary = json.loads(Path(result["training_summary"]).read_text(encoding="utf-8"))
    assert summary["support_training_accuracy_is_validation"] is False
    assert "not held-out validation accuracy" in summary["support_training_accuracy_statement"]

    # The Provider job must never carry a weight, checkpoint, or model state.
    assert "checkpoint_path" not in captured
    assert "weight_identifier" not in captured
    assert captured["feature_mode"] == "image_plus_symmetry_maps"
    assert captured["classifier"] == "logistic_regression"


def test_progress_callback_receives_provider_events(tmp_path, monkeypatch) -> None:
    setup = _setup(tmp_path)
    _fake_provider(monkeypatch)
    events: list[tuple[str, int, int]] = []

    run_traditional_validation(
        setup["config"],
        image_path=setup["image_path"],
        annotation_session=setup["session"],
        classifier="random_forest",
        feature_mode="raw_image",
        settings=setup["settings"],
        features_path=setup["features_path"],
        features_record_path=setup["features_record_path"],
        output_root=tmp_path / "runs",
        progress_callback=lambda phase, current, total: events.append(
            (phase, current, total)
        ),
    )
    assert ("prediction", 4, 4) in events


def test_run_uses_the_provider_support_counts_and_class_order(
    tmp_path, monkeypatch
) -> None:
    setup = _setup(tmp_path)
    _fake_provider(monkeypatch)
    result = _run(setup, tmp_path)
    assert [entry["name"] for entry in result["classes"]] == list(CLASS_NAMES)
    assert [entry["support_count"] for entry in result["classes"]] == [2, 2]


# --- malformed Provider payloads --------------------------------------------


def test_missing_provider_arrays_fail_and_record_the_failure(
    tmp_path, monkeypatch
) -> None:
    setup = _setup(tmp_path)
    incomplete = _arrays()
    incomplete.pop("probabilities")
    _fake_provider(monkeypatch, arrays=incomplete)

    with pytest.raises(RuntimeError, match="missing arrays"):
        _run(setup, tmp_path)

    run_directories = list((tmp_path / "runs").iterdir())
    assert len(run_directories) == 1
    record = json.loads(
        (run_directories[0] / "run_record.json").read_text(encoding="utf-8")
    )
    assert record["status"] == "failed"
    assert record["phase"] == "traditional_validation"
    assert record["error_type"] == "RuntimeError"


def test_changed_class_order_is_rejected(tmp_path, monkeypatch) -> None:
    setup = _setup(tmp_path)
    _fake_provider(monkeypatch, record=_record(class_names=["Beta", "Alpha"]))

    with pytest.raises(RuntimeError, match="class order"):
        _run(setup, tmp_path)


def test_grid_shape_disagreement_is_rejected(tmp_path, monkeypatch) -> None:
    setup = _setup(tmp_path)
    _fake_provider(monkeypatch, record=_record(grid_shape=[3, 3]))

    with pytest.raises(RuntimeError, match="grid shape"):
        _run(setup, tmp_path)


def test_a_missing_feature_artifact_is_refused_before_the_provider_runs(
    tmp_path, monkeypatch
) -> None:
    setup = _setup(tmp_path)
    captured = _fake_provider(monkeypatch)
    setup["features_path"].unlink()

    with pytest.raises(ValueError, match="feature artifact is missing"):
        _run(setup, tmp_path)
    assert captured == {}


def test_an_unsupported_feature_mode_is_refused(tmp_path, monkeypatch) -> None:
    setup = _setup(tmp_path)
    captured = _fake_provider(monkeypatch)

    with pytest.raises(ValueError, match="Unsupported traditional ML feature mode"):
        _run(setup, tmp_path, feature_mode="everything")
    assert captured == {}


def test_a_blocked_runtime_stops_before_any_run_directory(
    tmp_path, monkeypatch
) -> None:
    setup = _setup(tmp_path)
    captured = _fake_provider(monkeypatch)
    monkeypatch.setattr(
        workflow_module,
        "traditional_readiness",
        lambda config, *, require_ui=True: {
            "status": "blocked",
            "issues": ["provider does not expose traditional ML analysis"],
        },
    )

    with pytest.raises(RuntimeError, match="blocked"):
        _run(setup, tmp_path)
    assert captured == {}
    assert not (tmp_path / "runs").exists()


def test_workflow_readiness_never_requires_the_ui_extra(
    tmp_path, monkeypatch
) -> None:
    """The headless workflow must ask for a Provider-only readiness check."""
    setup = _setup(tmp_path)
    observed: list[bool] = []

    def fake_readiness(config, *, require_ui=True):
        observed.append(require_ui)
        return _readiness()

    monkeypatch.setattr(workflow_module, "traditional_readiness", fake_readiness)
    monkeypatch.setattr(
        workflow_module,
        "run_provider_traditional_ml",
        lambda config, **kwargs: SimpleNamespace(
            arrays=_arrays(), record=_record()
        ),
    )

    run_traditional_validation(
        setup["config"],
        image_path=setup["image_path"],
        annotation_session=setup["session"],
        classifier="logistic_regression",
        feature_mode="image_plus_symmetry_maps",
        settings=setup["settings"],
        features_path=setup["features_path"],
        features_record_path=setup["features_record_path"],
        output_root=tmp_path / "runs",
    )
    assert observed == [False]
