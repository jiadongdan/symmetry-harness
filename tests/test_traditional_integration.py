"""Real cross-repository traditional ML validation through the Provider worker.

These tests exercise the actual ``symmetry-learn`` Provider over its documented
subprocess contract: a real feature computation, a real scikit-learn fit, and a
real dense prediction. Nothing here is mocked.

The module is deliberately written to **fail instead of skip**. The dedicated
cross-repository CI job installs both repositories plus torch and scikit-learn
and runs this file by marker, so a missing Provider must surface as a failure
rather than a green, silently-skipped run. The generic CI test job deselects the
``traditional_integration`` marker.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import pytest

from symmetry_harness.annotations import create_annotation_session
from symmetry_harness.config import load_harness_config
from symmetry_harness.contracts import TraditionalMLSettings
from symmetry_harness.image_io import file_sha256
from symmetry_harness.provider import (
    provider_capabilities,
    run_provider_features,
    traditional_readiness,
    validate_traditional_capabilities,
)
from symmetry_harness.traditional_workflow import (
    artifact_paths,
    run_traditional_validation,
)


pytestmark = pytest.mark.traditional_integration

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

IMAGE_SIZE = 96
PATCH_SIZE = 64
STRIDE = 16
BATCH_SIZE = 64

# Small feature options keep the real map computation fast while still
# exercising the same eight-channel contract the page uses.
FEATURE_OPTIONS = {
    "n_max": 4,
    "symmetry_patch_size": 5,
    "rotation_folds": [2, 3, 4, 6],
    "reflection_p": 2.0,
    "normalize_rotation_maps": False,
}

CLASS_NAMES = ("Phase A", "Phase B")
SUPPORT_POINTS = (
    ((40, 40), (44, 44)),
    ((52, 52), (56, 56)),
)


def _unit_image(size: int = IMAGE_SIZE) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    values = np.sin(x / 6.0) + np.cos(y / 7.0)
    return ((values - values.min()) / np.ptp(values)).astype(np.float32)


def _harness_config(tmp_path: Path, *, orphan_checkpoint: bool = False):
    """Write a portable harness config that uses the running interpreter.

    ``orphan_checkpoint`` points the model section at a checkpoint path that does
    not exist and clears the registered weight, so a successful run proves the
    traditional path resolved no pretrained checkpoint at all.
    """
    payload = json.loads(
        (REPOSITORY_ROOT / "configs" / "config.example.json").read_text(
            encoding="utf-8"
        )
    )
    payload["project_root"] = "."
    payload["output_root"] = "runs"
    payload["model"]["python_executable"] = sys.executable
    if orphan_checkpoint:
        payload["model"]["weight_identifier"] = None
        payload["model"]["checkpoint_path"] = str(tmp_path / "unreachable.pt")
        payload["model"]["checkpoint_sha256"] = None
    path = tmp_path / "harness_config.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return load_harness_config(path)


def _assert_provider_is_ready(config) -> dict:
    """Fail, never skip, when the cross-repository Provider is unavailable."""
    if not Path(config.model.python_executable).is_file():
        pytest.fail(
            "The configured Provider interpreter does not exist: "
            f"{config.model.python_executable}"
        )
    try:
        capabilities = provider_capabilities(config)
    except Exception as error:  # noqa: BLE001 - reported as a hard failure
        pytest.fail(
            "The symmetry-learn Provider could not be reached. This integration "
            f"test must run for real and never skip. Original error: {error!r}"
        )
    issues = validate_traditional_capabilities(config, capabilities)
    if issues:
        pytest.fail(
            "The symmetry-learn Provider does not satisfy the traditional ML "
            "contract: " + "; ".join(issues)
        )
    return capabilities


@pytest.fixture()
def validation_setup(tmp_path):
    """Return a ready-to-run traditional validation harness configuration."""
    config = _harness_config(tmp_path)
    _assert_provider_is_ready(config)

    image = _unit_image()
    image_path = tmp_path / "input.npy"
    np.save(image_path, image)

    features = run_provider_features(config, image, options=dict(FEATURE_OPTIONS))
    features_path = tmp_path / "features.npz"
    np.savez_compressed(
        features_path,
        features=np.asarray(features.features, dtype=np.float32),
        channel_names=np.asarray(features.channel_names),
    )
    features_record_path = tmp_path / "features_record.json"
    features_record_path.write_text(
        json.dumps(features.record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    session = create_annotation_session(
        image_path=str(image_path),
        image_sha256=file_sha256(image_path),
        image_shape=(image.shape[0], image.shape[1]),
        classifier_patch_size=PATCH_SIZE,
        class_names=CLASS_NAMES,
        points_by_class=SUPPORT_POINTS,
    )
    settings = TraditionalMLSettings(
        classifier_patch_size=PATCH_SIZE, stride=STRIDE, batch_size=BATCH_SIZE
    )
    return {
        "config": config,
        "image": image,
        "image_path": image_path,
        "features_path": features_path,
        "features_record_path": features_record_path,
        "session": session,
        "settings": settings,
        "tmp_path": tmp_path,
    }


def _run(setup: dict, classifier: str, output_name: str, *, feature_mode: str):
    return run_traditional_validation(
        setup["config"],
        image_path=setup["image_path"],
        annotation_session=setup["session"],
        classifier=classifier,
        feature_mode=feature_mode,
        settings=setup["settings"],
        features_path=setup["features_path"],
        features_record_path=setup["features_record_path"],
        output_root=setup["tmp_path"] / output_name,
    )


def _prediction_grid(result: dict) -> np.ndarray:
    with np.load(result["traditional_prediction"], allow_pickle=False) as archive:
        return np.asarray(archive["prediction_grid"]).copy()


def _assert_completed_run(result: dict, classifier: str) -> dict:
    assert result["status"] == "completed"
    assert result["classifier"] == classifier
    record_path = Path(result["run_record"])
    assert record_path.is_file()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["traditional_ml"]["classifier"]["identifier"] == classifier
    assert record["traditional_ml"]["class_names"] == list(CLASS_NAMES)
    assert record["pretrained_checkpoint_accessed"] is False

    for artifact in artifact_paths(Path(result["run_directory"])).values():
        assert Path(artifact).is_file(), artifact

    archive_path = Path(result["results_archive"])
    with ZipFile(archive_path) as archive:
        members = set(archive.namelist())
    assert {
        "run_record.json",
        "training_summary.json",
        "provider_record.json",
        "annotation_session.json",
        "features.npz",
        "traditional_prediction.npz",
        "report.md",
    } <= members
    return record


def test_logistic_regression_runs_through_the_real_provider(validation_setup) -> None:
    result = _run(
        validation_setup,
        "logistic_regression",
        "runs-lr",
        feature_mode="image_plus_symmetry_maps",
    )
    record = _assert_completed_run(result, "logistic_regression")
    assert record["traditional_ml"]["feature_mode"] == "image_plus_symmetry_maps"
    assert record["traditional_ml"]["channel_count"] == 8

    with np.load(result["traditional_prediction"], allow_pickle=False) as archive:
        probabilities = np.asarray(archive["probabilities"])
        prediction_grid = np.asarray(archive["prediction_grid"])
    assert probabilities.shape[1] == len(CLASS_NAMES)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4)
    assert prediction_grid.size == int(np.prod(record["traditional_ml"]["grid_shape"]))


def test_random_forest_runs_through_the_real_provider(validation_setup) -> None:
    result = _run(
        validation_setup,
        "random_forest",
        "runs-rf",
        feature_mode="image_plus_symmetry_maps",
    )
    record = _assert_completed_run(result, "random_forest")
    parameters = record["traditional_ml"]["classifier"]["parameters"]
    assert parameters["n_jobs"] == 1
    assert parameters["n_estimators"] == 300
    # The seed is applied by the Provider when it builds the estimator; it is not
    # a user-overridable classifier parameter.
    assert "random_state" not in parameters
    assert record["traditional_ml"]["seed"] == 42


def test_raw_image_mode_delegates_channel_zero(validation_setup) -> None:
    result = _run(
        validation_setup,
        "logistic_regression",
        "runs-raw",
        feature_mode="raw_image",
    )
    record = _assert_completed_run(result, "logistic_regression")
    assert record["traditional_ml"]["feature_mode"] == "raw_image"
    assert record["traditional_ml"]["channel_count"] == 1


def test_repeated_runs_with_one_seed_are_reproducible(validation_setup) -> None:
    first = _run(
        validation_setup,
        "random_forest",
        "runs-rf-a",
        feature_mode="image_plus_symmetry_maps",
    )
    second = _run(
        validation_setup,
        "random_forest",
        "runs-rf-b",
        feature_mode="image_plus_symmetry_maps",
    )
    assert np.array_equal(_prediction_grid(first), _prediction_grid(second))
    with np.load(first["traditional_prediction"], allow_pickle=False) as archive:
        first_probabilities = np.asarray(archive["probabilities"])
    with np.load(second["traditional_prediction"], allow_pickle=False) as archive:
        second_probabilities = np.asarray(archive["probabilities"])
    assert np.array_equal(first_probabilities, second_probabilities)

    logistic_a = _run(
        validation_setup,
        "logistic_regression",
        "runs-lr-a",
        feature_mode="image_plus_symmetry_maps",
    )
    logistic_b = _run(
        validation_setup,
        "logistic_regression",
        "runs-lr-b",
        feature_mode="image_plus_symmetry_maps",
    )
    assert np.array_equal(_prediction_grid(logistic_a), _prediction_grid(logistic_b))


def test_workflow_never_resolves_a_pretrained_checkpoint(tmp_path) -> None:
    config = _harness_config(tmp_path, orphan_checkpoint=True)
    assert config.model.checkpoint_path == (tmp_path / "unreachable.pt").resolve()
    assert not config.model.checkpoint_path.exists()
    _assert_provider_is_ready(config)

    image = _unit_image()
    image_path = tmp_path / "input.npy"
    np.save(image_path, image)
    features = run_provider_features(config, image, options=dict(FEATURE_OPTIONS))
    features_path = tmp_path / "features.npz"
    np.savez_compressed(
        features_path,
        features=np.asarray(features.features, dtype=np.float32),
        channel_names=np.asarray(features.channel_names),
    )
    features_record_path = tmp_path / "features_record.json"
    features_record_path.write_text(
        json.dumps(features.record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    session = create_annotation_session(
        image_path=str(image_path),
        image_sha256=file_sha256(image_path),
        image_shape=(image.shape[0], image.shape[1]),
        classifier_patch_size=PATCH_SIZE,
        class_names=CLASS_NAMES,
        points_by_class=SUPPORT_POINTS,
    )

    result = run_traditional_validation(
        config,
        image_path=image_path,
        annotation_session=session,
        classifier="logistic_regression",
        feature_mode="image_plus_symmetry_maps",
        settings=TraditionalMLSettings(
            classifier_patch_size=PATCH_SIZE, stride=STRIDE, batch_size=BATCH_SIZE
        ),
        features_path=features_path,
        features_record_path=features_record_path,
        output_root=tmp_path / "runs-orphan",
    )
    record = _assert_completed_run(result, "logistic_regression")
    assert record["pretrained_checkpoint_accessed"] is False

    provider_record = json.loads(
        Path(result["provider_record"]).read_text(encoding="utf-8")
    )
    serialized = json.dumps(provider_record).lower()
    for forbidden in ("checkpoint", "weight_identifier", "model_state", "adapter"):
        assert forbidden not in serialized
    checkpoint_like = [
        artifact
        for artifact in Path(result["run_directory"]).rglob("*")
        if artifact.suffix in {".pt", ".pth", ".ckpt", ".symmodel"}
    ]
    assert checkpoint_like == []


def test_headless_readiness_does_not_require_the_ui_extra(validation_setup) -> None:
    readiness = traditional_readiness(validation_setup["config"], require_ui=False)
    assert readiness["status"] == "ready"
    assert readiness["pretrained_weight_required"] is False
    assert readiness["traditional_ml"]["schema_version"] == (
        "symmetry-traditional-ml-capability-v1"
    )
