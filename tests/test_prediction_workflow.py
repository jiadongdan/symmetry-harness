from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from symmetry_harness.config import load_harness_config
from symmetry_harness.model_package import ModelPackageError
from symmetry_harness.prediction_workflow import (
    PredictionError,
    run_saved_model_prediction_batch,
    validate_prediction_request,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _capabilities(operations=None, **overrides) -> dict:
    payload = {
        "contract_version": "symmetry-learn-provider-v1",
        "provider": "symmetry-learn",
        "provider_version": "0.1.0",
        "operations": operations
        if operations is not None
        else [
            "compute_features",
            "few_shot_analyze",
            "predict_with_fine_tuned_model",
        ],
        "models": [
            {
                "identifier": "cnn_8ch_pg17",
                "available": True,
                "input_channels": 8,
                "pretrained_classes": 17,
                "classifier_patch_size": 64,
                "feature_pipeline": "eight_channel_v1",
                "feature_channels": [f"channel_{index}" for index in range(8)],
                "fine_tuning_strategy": "adapters",
                "minimum_shots_per_class": 3,
                "maximum_shots_per_class": 50,
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
    payload.update(overrides)
    return payload


def _fake_provider(monkeypatch, *, capabilities=None, fail_item: str | None = None):
    """Capture the submitted batch job and write deterministic predictions."""
    captured: dict = {}

    def fake_batch(config, **kwargs):
        # Record the state path while the private temporary directory still
        # exists; the orchestrator deletes it once the batch job returns.
        captured["kwargs"] = kwargs
        captured["model_state_existed"] = Path(kwargs["model_state_path"]).is_file()
        captured["model_state_bytes"] = Path(kwargs["model_state_path"]).read_bytes()
        from symmetry_harness.provider import ProviderPredictionBatch

        items = []
        for item in kwargs["items"]:
            item_id = item["item_id"]
            output_path = Path(item["output_path"])
            record_path = Path(item["record_path"])
            if item_id == fail_item:
                items.append(
                    {
                        "item_id": item_id,
                        "status": "failed",
                        "error": "Prediction input must lie in the shared [0, 1] space.",
                    }
                )
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_path,
                features=np.zeros((8, 96, 96), dtype=np.float32),
                channel_names=np.asarray([f"channel_{index}" for index in range(8)]),
                coordinates_xy=np.array(
                    [[32, 32], [36, 32], [32, 36], [36, 36]], dtype=np.int32
                ),
                x_coordinates=np.array([32, 36], dtype=np.int32),
                y_coordinates=np.array([32, 36], dtype=np.int32),
                logits=np.zeros((4, 2), dtype=np.float32),
                probabilities=np.full((4, 2), 0.5, dtype=np.float32),
                predictions=np.array([0, 1, 1, 0], dtype=np.int16),
                confidence=np.full(4, 0.5, dtype=np.float32),
                entropy=np.full(4, np.log(2), dtype=np.float32),
                prediction_grid=np.array([[0, 1], [1, 0]], dtype=np.int16),
                confidence_grid=np.full((2, 2), 0.5, dtype=np.float32),
                entropy_grid=np.full((2, 2), np.log(2), dtype=np.float32),
            )
            record_path.write_text(
                json.dumps(
                    {
                        "provider": "symmetry-learn",
                        "provider_version": "0.1.0",
                        "provider_contract_version": "symmetry-learn-provider-v1",
                    }
                ),
                encoding="utf-8",
            )
            items.append({"item_id": item_id, "status": "completed"})
        return ProviderPredictionBatch(
            summary={
                "provider": "symmetry-learn",
                "provider_contract_version": "symmetry-learn-provider-v1",
                "item_count": len(items),
                "completed_count": sum(
                    1 for entry in items if entry["status"] == "completed"
                ),
                "failed_count": sum(
                    1 for entry in items if entry["status"] == "failed"
                ),
                "items": items,
            }
        )

    monkeypatch.setattr(
        "symmetry_harness.prediction_workflow.run_provider_prediction_batch",
        fake_batch,
    )
    monkeypatch.setattr(
        "symmetry_harness.prediction_workflow.provider_capabilities",
        lambda config: capabilities or _capabilities(),
    )
    return captured


@pytest.fixture()
def config():
    return load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")


def test_prediction_request_restores_saved_defaults(
    config, tmp_path, monkeypatch, model_package_factory, unit_image
) -> None:
    _fake_provider(monkeypatch)
    package = model_package_factory()
    image = unit_image(tmp_path / "image.npy")

    request = validate_prediction_request(
        config, model_package=package, image_paths=[image]
    )

    assert request.stride == 4
    assert request.batch_size == 512
    assert request.device == "auto"
    assert request.package.model_identifier == "cnn_8ch_pg17"
    assert request.package.input_normalization == "minmax_0_1"


def test_prediction_request_allows_only_runtime_overrides(
    config, tmp_path, monkeypatch, model_package_factory, unit_image
) -> None:
    _fake_provider(monkeypatch)
    package = model_package_factory()
    image = unit_image(tmp_path / "image.npy")

    request = validate_prediction_request(
        config,
        model_package=package,
        image_paths=[image],
        device="cpu",
        stride=8,
        batch_size=64,
    )

    assert (request.device, request.stride, request.batch_size) == ("cpu", 8, 64)

    with pytest.raises(PredictionError, match="device must be"):
        validate_prediction_request(
            config, model_package=package, image_paths=[image], device="tpu"
        )
    with pytest.raises(PredictionError, match="at least 1"):
        validate_prediction_request(
            config, model_package=package, image_paths=[image], stride=0
        )
    with pytest.raises(PredictionError, match="at least one image"):
        validate_prediction_request(config, model_package=package, image_paths=[])


def test_prediction_blocks_an_unsupported_or_mismatched_provider(
    config, tmp_path, monkeypatch, model_package_factory, unit_image
) -> None:
    _fake_provider(
        monkeypatch, capabilities=_capabilities(operations=["few_shot_analyze"])
    )
    package = model_package_factory()
    image = unit_image(tmp_path / "image.npy")

    with pytest.raises(PredictionError, match="does not support saved-model"):
        validate_prediction_request(
            config, model_package=package, image_paths=[image]
        )

    mismatched = _capabilities()
    mismatched["models"][0]["feature_pipeline"] = "other_pipeline"
    _fake_provider(monkeypatch, capabilities=mismatched)
    with pytest.raises(PredictionError, match="feature pipeline"):
        validate_prediction_request(
            config, model_package=package, image_paths=[image]
        )


def test_prediction_applies_semantic_runtime_version_policy(
    config, tmp_path, monkeypatch, model_package_factory, unit_image
) -> None:
    image = unit_image(tmp_path / "image.npy")
    incompatible_provider = model_package_factory(
        "provider.symmodel",
        manifest_overrides={
            "software": {"symmetry_learn_version": "0.2.0"}
        },
    )
    _fake_provider(monkeypatch, capabilities=_capabilities())
    with pytest.raises(PredictionError, match="incompatible symmetry-learn"):
        validate_prediction_request(
            config, model_package=incompatible_provider, image_paths=[image]
        )

    incompatible_torch = model_package_factory(
        "torch.symmodel",
        manifest_overrides={"software": {"torch_version": "3.0.0"}},
    )
    _fake_provider(
        monkeypatch,
        capabilities=_capabilities(runtime={"torch_version": "2.1.0"}),
    )
    with pytest.raises(PredictionError, match="PyTorch major"):
        validate_prediction_request(
            config, model_package=incompatible_torch, image_paths=[image]
        )

    patch_difference = model_package_factory(
        "patch.symmodel",
        manifest_overrides={
            "software": {
                "symmetry_learn_version": "0.1.1",
                "torch_version": "2.1.1",
            }
        },
    )
    request = validate_prediction_request(
        config, model_package=patch_difference, image_paths=[image]
    )
    assert len(request.compatibility_warnings) == 2


def test_prediction_rejects_an_invalid_package_and_missing_input(
    config, tmp_path, monkeypatch, model_package_factory, unit_image
) -> None:
    _fake_provider(monkeypatch)
    image = unit_image(tmp_path / "image.npy")

    with pytest.raises(ModelPackageError):
        validate_prediction_request(
            config, model_package=tmp_path / "absent.symmodel", image_paths=[image]
        )

    package = model_package_factory()
    with pytest.raises(PredictionError, match="does not exist"):
        validate_prediction_request(
            config, model_package=package, image_paths=[tmp_path / "absent.npy"]
        )


def test_prediction_rejects_images_smaller_than_the_saved_patch(
    config, tmp_path, monkeypatch, model_package_factory, unit_image
) -> None:
    _fake_provider(monkeypatch)
    package = model_package_factory()

    # The saved classifier patch is 64 pixels, so any smaller input is rejected
    # before a Provider job is ever submitted.
    small = unit_image(tmp_path / "small.npy", size=32)
    with pytest.raises(PredictionError, match="small.npy"):
        validate_prediction_request(
            config, model_package=package, image_paths=[small]
        )

    adequate = unit_image(tmp_path / "adequate.npy", size=64)
    request = validate_prediction_request(
        config, model_package=package, image_paths=[adequate]
    )
    assert request.image_paths == (adequate,)


def test_batch_prediction_writes_artifacts_and_statistics(
    config, tmp_path, monkeypatch, model_package_factory, unit_image
) -> None:
    captured = _fake_provider(monkeypatch)
    package = model_package_factory()
    images = [unit_image(tmp_path / f"image{index}.npy") for index in range(2)]

    result = run_saved_model_prediction_batch(
        config,
        model_package=package,
        image_paths=images,
        output_root=tmp_path / "runs",
        device="cpu",
    )

    assert result["status"] == "completed"
    assert result["counts"] == {"requested": 2, "completed": 2, "failed": 0}
    run_directory = Path(result["run_directory"])
    assert (run_directory / "batch_record.json").is_file()
    assert (run_directory / "report.md").is_file()
    assert (run_directory / "results.zip").is_file()

    first = result["items"][0]
    assert first["class_statistics"]["grid_cell_count"] == 4
    assert [entry["count"] for entry in first["class_statistics"]["classes"]] == [2, 2]
    for key in ("prediction_overlay", "confidence", "entropy", "prediction"):
        assert Path(first[f"artifact_{key}"]).is_file()
    assert Path(first["artifact_features"]).is_file()
    assert json.loads(json.dumps(result))["status"] == "completed"

    item_record = json.loads(
        (
            run_directory / "items" / "image-0001" / "prediction_record.json"
        ).read_text(encoding="utf-8")
    )
    assert item_record["contract_version"] == "symmetry-harness-prediction-run-v1"
    assert item_record["model"]["class_names"] == ["Phase A", "Phase B"]
    assert item_record["prediction_options"] == {
        "device": "cpu",
        "stride": 4,
        "batch_size": 512,
    }
    assert item_record["package"]["model_state_sha256"]

    job = captured["kwargs"]
    assert job["feature_options"]["symmetry_patch_size"] == 51
    assert job["feature_options"]["rotation_folds"] == [2, 3, 4, 6]
    assert "input_normalization" not in job["feature_options"]
    assert job["expected_model"]["task_classes"] == 2
    assert [entry["item_id"] for entry in job["items"]] == [
        "image-0001",
        "image-0002",
    ]
    assert captured["model_state_existed"]
    with zipfile.ZipFile(package) as archive:
        assert captured["model_state_bytes"] == archive.read("model_state.pt")


def test_batch_prediction_reports_partial_failure(
    config, tmp_path, monkeypatch, model_package_factory, unit_image
) -> None:
    _fake_provider(monkeypatch, fail_item="image-0001")
    package = model_package_factory()
    images = [unit_image(tmp_path / f"image{index}.npy") for index in range(2)]

    result = run_saved_model_prediction_batch(
        config,
        model_package=package,
        image_paths=images,
        output_root=tmp_path / "runs",
    )

    assert result["counts"] == {"requested": 2, "completed": 1, "failed": 1}
    failed = next(entry for entry in result["items"] if entry["status"] == "failed")
    assert failed["item_id"] == "image-0001"
    assert failed["error"]
    assert "failed" in Path(result["report"]).read_text(encoding="utf-8")


def test_batch_prediction_records_invalid_input_and_continues(
    config, tmp_path, monkeypatch, model_package_factory, unit_image
) -> None:
    _fake_provider(monkeypatch)
    package = model_package_factory()
    valid = unit_image(tmp_path / "valid.npy")
    invalid = unit_image(tmp_path / "small.npy", size=32)

    result = run_saved_model_prediction_batch(
        config,
        model_package=package,
        image_paths=[invalid, valid],
        output_root=tmp_path / "runs",
    )

    assert result["counts"] == {"requested": 2, "completed": 1, "failed": 1}
    assert [entry["item_id"] for entry in result["items"]] == [
        "image-0001",
        "image-0002",
    ]
    failed = result["items"][0]
    assert failed["phase"] == "input_validation"
    assert Path(failed["prediction_record"]).is_file()


def test_batch_archive_contains_completed_items_only(
    config, tmp_path, monkeypatch, model_package_factory, unit_image
) -> None:
    _fake_provider(monkeypatch, fail_item="image-0002")
    package = model_package_factory()
    images = [unit_image(tmp_path / f"image{index}.npy") for index in range(2)]

    result = run_saved_model_prediction_batch(
        config,
        model_package=package,
        image_paths=images,
        output_root=tmp_path / "runs",
    )

    with zipfile.ZipFile(result["results_archive"]) as archive:
        names = archive.namelist()
    assert "batch_record.json" in names
    assert "report.md" in names
    assert any("image-0001" in name for name in names)
    assert "items/image-0002/prediction_record.json" in names


def test_source_package_is_unchanged_after_a_batch(
    config, tmp_path, monkeypatch, model_package_factory, unit_image
) -> None:
    _fake_provider(monkeypatch)
    package = model_package_factory()
    before = package.read_bytes()

    run_saved_model_prediction_batch(
        config,
        model_package=package,
        image_paths=[unit_image(tmp_path / "image.npy")],
        output_root=tmp_path / "runs",
    )

    assert package.read_bytes() == before


def test_download_archive_is_servable_and_run_archive_is_kept(
    config, tmp_path, monkeypatch, model_package_factory, unit_image
) -> None:
    """Gradio only serves files in the cwd, temp dir, or allowed paths."""
    import tempfile

    _fake_provider(monkeypatch)
    package = model_package_factory()

    result = run_saved_model_prediction_batch(
        config,
        model_package=package,
        image_paths=[unit_image(tmp_path / "image.npy")],
        output_root=tmp_path / "runs",
    )

    run_archive = Path(result["results_archive"])
    download = Path(result["download_archive"])
    system_temp = Path(tempfile.gettempdir()).resolve()

    # The durable record stays in the run directory.
    assert run_archive.is_file()
    assert run_archive.parent == Path(result["run_directory"])
    # The download copy lives where Gradio is allowed to serve files.
    assert download.is_file()
    assert download.read_bytes() == run_archive.read_bytes()
    assert system_temp in download.resolve().parents


def test_temporary_model_state_is_cleaned_up(
    config, tmp_path, monkeypatch, model_package_factory, unit_image
) -> None:
    captured = _fake_provider(monkeypatch)
    package = model_package_factory()

    run_saved_model_prediction_batch(
        config,
        model_package=package,
        image_paths=[unit_image(tmp_path / "image.npy")],
        output_root=tmp_path / "runs",
    )

    temporary = Path(captured["kwargs"]["model_state_path"]).parent
    assert not temporary.exists()
