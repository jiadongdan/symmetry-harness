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


def _harness_config(tmp_path: Path):
    """Write a portable harness config that uses the running interpreter."""
    payload = json.loads(
        (REPOSITORY_ROOT / "configs" / "config.example.json").read_text(
            encoding="utf-8"
        )
    )
    payload["project_root"] = "."
    payload["output_root"] = "runs"
    payload["model"]["python_executable"] = sys.executable
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


@pytest.fixture(scope="module")
def validation_setup(tmp_path_factory):
    """Return a ready-to-run traditional validation harness configuration.

    Module-scoped on purpose: building the configuration pays for a real
    ``compute_features`` Provider call (tens of seconds), and every test in this
    module consumes the exact same prepared inputs read-only. Only the run
    output directory differs per test, and that comes from the test's own
    ``tmp_path`` (see :func:`_run`).
    """
    base = tmp_path_factory.mktemp("traditional-validation")
    config = _harness_config(base)
    _assert_provider_is_ready(config)

    image = _unit_image()
    image_path = base / "input.npy"
    np.save(image_path, image)

    features = run_provider_features(config, image, options=dict(FEATURE_OPTIONS))
    features_path = base / "features.npz"
    np.savez_compressed(
        features_path,
        features=np.asarray(features.features, dtype=np.float32),
        channel_names=np.asarray(features.channel_names),
    )
    features_record_path = base / "features_record.json"
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
        "base": base,
    }


def _run(
    setup: dict,
    classifier: str,
    output_name: str,
    *,
    feature_mode: str,
    output_root: Path,
):
    return run_traditional_validation(
        setup["config"],
        image_path=setup["image_path"],
        annotation_session=setup["session"],
        classifier=classifier,
        feature_mode=feature_mode,
        settings=setup["settings"],
        features_path=setup["features_path"],
        features_record_path=setup["features_record_path"],
        output_root=output_root,
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


def test_logistic_regression_runs_through_the_real_provider(
    validation_setup, tmp_path
) -> None:
    result = _run(
        validation_setup,
        "logistic_regression",
        "runs-lr",
        feature_mode="image_plus_symmetry_maps",
        output_root=tmp_path / "runs-lr",
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


def test_raw_image_mode_delegates_channel_zero(validation_setup, tmp_path) -> None:
    result = _run(
        validation_setup,
        "logistic_regression",
        "runs-raw",
        feature_mode="raw_image",
        output_root=tmp_path / "runs-raw",
    )
    record = _assert_completed_run(result, "logistic_regression")
    assert record["traditional_ml"]["feature_mode"] == "raw_image"
    assert record["traditional_ml"]["channel_count"] == 1


def test_repeated_runs_with_one_seed_are_reproducible(
    validation_setup, tmp_path
) -> None:
    first = _run(
        validation_setup,
        "random_forest",
        "runs-rf-a",
        feature_mode="image_plus_symmetry_maps",
        output_root=tmp_path / "runs-rf-a",
    )
    second = _run(
        validation_setup,
        "random_forest",
        "runs-rf-b",
        feature_mode="image_plus_symmetry_maps",
        output_root=tmp_path / "runs-rf-b",
    )
    # The first run also carries the random-forest parameter contract, so this
    # single pair of runs covers both "the RF classifier runs for real" and
    # "one seed makes it reproducible".
    record = _assert_completed_run(first, "random_forest")
    parameters = record["traditional_ml"]["classifier"]["parameters"]
    assert parameters["n_jobs"] == 1
    assert parameters["n_estimators"] == 300
    # The seed is applied by the Provider when it builds the estimator; it is not
    # a user-overridable classifier parameter.
    assert "random_state" not in parameters
    assert record["traditional_ml"]["seed"] == 42

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
        output_root=tmp_path / "runs-lr-a",
    )
    logistic_b = _run(
        validation_setup,
        "logistic_regression",
        "runs-lr-b",
        feature_mode="image_plus_symmetry_maps",
        output_root=tmp_path / "runs-lr-b",
    )
    assert np.array_equal(_prediction_grid(logistic_a), _prediction_grid(logistic_b))


def test_headless_readiness_does_not_require_the_ui_extra(validation_setup) -> None:
    readiness = traditional_readiness(validation_setup["config"], require_ui=False)
    assert readiness["status"] == "ready"
    assert readiness["pretrained_weight_required"] is False
    assert readiness["traditional_ml"]["schema_version"] == (
        "symmetry-traditional-ml-capability-v1"
    )
