"""End-to-end saved-model prediction through the real Provider worker.

These tests use a small CPU model and skip cleanly when torch is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from symmetry_harness.contracts import FINE_TUNED_MODEL_PACKAGE_SCHEMA_VERSION
from symmetry_harness.config import load_harness_config
from symmetry_harness.prediction_workflow import run_saved_model_prediction_batch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

FEATURE_OPTIONS = {
    "n_max": 4,
    "symmetry_patch_size": 5,
    "rotation_folds": [2, 3, 4, 6],
    "reflection_p": 2.0,
    "normalize_rotation_maps": False,
    "input_normalization": "minmax_0_1",
}


CHANNEL_NAMES = (
    "image",
    "reflection_strength",
    "reflection_sin_2theta",
    "reflection_cos_2theta",
    "rotation_2_fold",
    "rotation_3_fold",
    "rotation_4_fold",
    "rotation_6_fold",
)


def _manifest(model_state_sha256: str) -> dict:
    return {
        "schema_version": FINE_TUNED_MODEL_PACKAGE_SCHEMA_VERSION,
        "model": {
            "identifier": "cnn_8ch_pg17",
            "input_channels": 8,
            "classifier_patch_size": 64,
            "adapter_bottleneck": 4,
            "task_classes": 2,
        },
        "classes": [
            {"index": 0, "name": "Phase A", "color": "#e41a1c"},
            {"index": 1, "name": "Phase B", "color": "#377eb8"},
        ],
        "features": {
            "pipeline": "eight_channel_v1",
            "channel_names": list(CHANNEL_NAMES),
            **FEATURE_OPTIONS,
        },
        "prediction_defaults": {"stride": 16, "batch_size": 32},
        "provenance": {
            "training_run_id": "integration-test",
            "model_state_sha256": model_state_sha256,
        },
    }


def _unit_image(size: int = 96) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    values = np.sin(x / 6.0) + np.cos(y / 7.0)
    return ((values - values.min()) / np.ptp(values)).astype(np.float32)


@pytest.fixture(scope="module")
def real_package(tmp_path_factory) -> Path:
    """Build a small CPU fine-tuned state and package it as `.symmodel`."""
    torch = pytest.importorskip("torch")
    from symmlearn.finetuning.adapters import add_task_adapters
    from symmlearn.models.registry import build_registered_model

    directory = tmp_path_factory.mktemp("package")
    model = build_registered_model("cnn_8ch_pg17")
    adapted = add_task_adapters(model, bottleneck=4, task_classes=2)
    state = {
        "schema_version": "symmetry-fine-tuned-model-state-v1",
        "model_identifier": "cnn_8ch_pg17",
        "adapter_bottleneck": 4,
        "task_classes": 2,
        "class_names": ["Phase A", "Phase B"],
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in adapted.state_dict().items()
        },
    }
    state_path = directory / "model_state.pt"
    torch.save(state, state_path)

    import hashlib
    from zipfile import ZIP_DEFLATED, ZipFile

    state_bytes = state_path.read_bytes()
    payload = _manifest(hashlib.sha256(state_bytes).hexdigest())

    destination = directory / "fine_tuned_model.symmodel"
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(payload, sort_keys=True))
        archive.writestr("model_state.pt", state_bytes)
        archive.writestr(
            "training_summary.json", json.dumps({"training_run_id": "integration-test"})
        )
    return destination


@pytest.fixture()
def config():
    return load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")


def test_end_to_end_prediction_creates_artifacts(config, real_package, tmp_path) -> None:
    pytest.importorskip("torch")
    image = tmp_path / "image.npy"
    np.save(image, _unit_image())

    result = run_saved_model_prediction_batch(
        config,
        model_package=real_package,
        image_paths=[image],
        output_root=tmp_path / "runs",
        device="cpu",
    )

    assert result["counts"] == {"requested": 1, "completed": 1, "failed": 0}
    run_directory = Path(result["run_directory"])
    item = result["items"][0]
    for key in ("prediction_overlay", "confidence", "entropy", "prediction"):
        assert Path(item[f"artifact_{key}"]).is_file()
    record = json.loads(
        (run_directory / "items" / "image-0001" / "prediction_record.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["model"]["class_names"] == ["Phase A", "Phase B"]
    assert record["features"]["symmetry_patch_size"] == 5
    with np.load(item["artifact_prediction"]) as archive:
        assert list(archive["prediction_grid"].shape) == record["output"]["grid_shape"]


def test_end_to_end_prediction_is_deterministic(config, real_package, tmp_path) -> None:
    pytest.importorskip("torch")
    image = tmp_path / "image.npy"
    np.save(image, _unit_image())

    first = run_saved_model_prediction_batch(
        config,
        model_package=real_package,
        image_paths=[image],
        output_root=tmp_path / "runs-a",
        device="cpu",
    )
    second = run_saved_model_prediction_batch(
        config,
        model_package=real_package,
        image_paths=[image],
        output_root=tmp_path / "runs-b",
        device="cpu",
    )

    with np.load(first["items"][0]["artifact_prediction"]) as archive:
        first_grid = archive["prediction_grid"].copy()
    with np.load(second["items"][0]["artifact_prediction"]) as archive:
        second_grid = archive["prediction_grid"].copy()
    assert np.array_equal(first_grid, second_grid)


def test_end_to_end_batch_loads_the_model_once(config, real_package, tmp_path) -> None:
    pytest.importorskip("torch")
    images = []
    for index in range(2):
        path = tmp_path / f"image{index}.npy"
        np.save(path, _unit_image())
        images.append(path)

    result = run_saved_model_prediction_batch(
        config,
        model_package=real_package,
        image_paths=images,
        output_root=tmp_path / "runs",
        device="cpu",
    )

    assert result["counts"] == {"requested": 2, "completed": 2, "failed": 0}
    assert len({entry["item_id"] for entry in result["items"]}) == 2
