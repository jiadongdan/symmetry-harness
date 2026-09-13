"""Focused tests for the complete run archive and the extended run result."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

from symmetry_harness.annotations import create_annotation_session
from symmetry_harness.config import load_harness_config
from symmetry_harness.image_io import file_sha256
from symmetry_harness.provider import ProviderAnalysis
import symmetry_harness.workflow as workflow_module
from symmetry_harness.workflow import create_run_archive, run_analysis


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

_ARTIFACT_NAMES = (
    "annotation_session.json",
    "input.npy",
    "input_preview.png",
    "features.npz",
    "support_patches.npz",
    "adapter_head.pt",
    "fine_tuned_model.symmodel",
    "provider_record.json",
    "training_history.json",
    "prediction.npz",
    "prediction_overlay.png",
    "confidence.png",
    "entropy.png",
    "run_record.json",
    "report.md",
)


def _write_image(path: Path, size: int = 96) -> Path:
    y, x = np.mgrid[:size, :size]
    values = np.sin(x / 6.0) + np.cos(y / 7.0)
    np.save(path, ((values - values.min()) / np.ptp(values)).astype(np.float32))
    return path


def _session(image_path: Path):
    return create_annotation_session(
        image_path=str(image_path),
        image_sha256=file_sha256(image_path),
        image_shape=(96, 96),
        classifier_patch_size=64,
        class_names=["Class A", "Class B"],
        points_by_class=[
            [(32, 32), (40, 40), (48, 48)],
            [(52, 52), (58, 58), (62, 62)],
        ],
    )


def _provider_record() -> dict:
    return {
        "provider": "symmetry-learn",
        "provider_version": "0.1.0",
        "provider_contract_version": "symmetry-learn-provider-v1",
        "identifier": "cnn_8ch_pg17",
        "model": {
            "identifier": "cnn_8ch_pg17",
            "input_channels": 8,
            "classifier_patch_size": 64,
            "feature_pipeline": "eight_channel_v1",
            "checkpoint": {
                "sha256": "a" * 64,
                "weight_identifier": "pg17-symmetry-v1",
                "source": "registered",
            },
        },
        "features": {"channel_names": [f"channel_{index}" for index in range(8)]},
        "runtime": {"torch_version": "2.1.0"},
        "training": {
            "best_support_loss": 0.1,
            "final_support_accuracy": 1.0,
            "trainable_parameters": 128,
        },
        "prediction": {"sample_count": 4},
        "warnings": [],
    }


def _install_fake_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        workflow_module,
        "doctor",
        lambda config: {"status": "ready", "issues": []},
    )

    def fake_analysis(
        config,
        image,
        *,
        coordinates_xy,
        labels,
        class_names,
        options,
        features_path=None,
        features_record_path=None,
        progress_callback=None,
    ):
        coordinates = np.asarray(coordinates_xy, dtype=np.int32)
        support_labels = np.asarray(labels, dtype=np.int64)
        patch = int(options["classifier_patch_size"])
        arrays = {
            "features": np.zeros((8, image.shape[0], image.shape[1]), dtype=np.float32),
            "channel_names": np.asarray([f"channel_{index}" for index in range(8)]),
            "support_patches": np.zeros(
                (len(coordinates), 8, patch, patch), dtype=np.float32
            ),
            "support_labels": support_labels,
            "support_coordinates_xy": coordinates,
            "coordinates_xy": np.array(
                [[32, 32], [36, 32], [32, 36], [36, 36]], dtype=np.int32
            ),
            "x_coordinates": np.array([32, 36], dtype=np.int32),
            "y_coordinates": np.array([32, 36], dtype=np.int32),
            "logits": np.zeros((4, 2), dtype=np.float32),
            "probabilities": np.full((4, 2), 0.5, dtype=np.float32),
            "predictions": np.array([0, 1, 1, 0], dtype=np.int16),
            "confidence": np.full(4, 0.5, dtype=np.float32),
            "entropy": np.full(4, np.log(2), dtype=np.float32),
            "prediction_grid": np.array([[0, 1], [1, 0]], dtype=np.int16),
            "confidence_grid": np.full((2, 2), 0.5, dtype=np.float32),
            "entropy_grid": np.full((2, 2), np.log(2), dtype=np.float32),
        }
        return ProviderAnalysis(
            arrays=arrays,
            adapter_checkpoint=b"adapter-bytes",
            fine_tuned_model_state=b"fine-tuned-model-state",
            record=_provider_record(),
        )

    monkeypatch.setattr(workflow_module, "run_provider_analysis", fake_analysis)


def _run(tmp_path: Path, monkeypatch) -> dict:
    _install_fake_provider(monkeypatch)
    config = load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")
    image_path = _write_image(tmp_path / "sample.npy")
    session = _session(image_path)
    return run_analysis(
        config,
        image_path=image_path,
        annotation_session=session,
        output_root=tmp_path / "runs",
    )


# ---------------------------------------------------------------------------
# Standalone archive helper
# ---------------------------------------------------------------------------
def test_create_run_archive_is_a_safe_deterministic_sibling(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "symmetry-demo"
    run_dir.mkdir(parents=True)
    for name in _ARTIFACT_NAMES:
        (run_dir / name).write_bytes(f"payload:{name}".encode("utf-8"))
    # An unrelated external file that must never be added to the archive.
    external = tmp_path / "original_upload.npy"
    external.write_bytes(b"external-source")

    first = create_run_archive(run_dir, output_root=tmp_path / "runs")
    second = create_run_archive(run_dir, output_root=tmp_path / "runs")

    assert first == second
    assert first.parent == run_dir.parent  # sibling, never inside the run dir
    assert first.name == "symmetry-demo-full-run.zip"
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
    assert sorted(names) == sorted(_ARTIFACT_NAMES)
    assert all(not name.startswith("/") and ".." not in Path(name).parts for name in names)
    assert "original_upload.npy" not in names


def test_create_run_archive_rejects_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        create_run_archive(tmp_path / "does-not-exist", output_root=tmp_path)


def test_create_run_archive_failure_preserves_previous_archive(
    tmp_path: Path, monkeypatch
) -> None:
    import symmetry_harness.workflow as workflow

    run_dir = tmp_path / "runs" / "symmetry-demo"
    run_dir.mkdir(parents=True)
    (run_dir / "run_record.json").write_text("{}", encoding="utf-8")
    target = run_dir.parent / "symmetry-demo-full-run.zip"
    target.write_bytes(b"previous-complete-archive")

    def fail_zip(*args, **kwargs):
        raise OSError("synthetic archive failure")

    monkeypatch.setattr(workflow, "ZipFile", fail_zip)
    with pytest.raises(OSError, match="synthetic archive failure"):
        create_run_archive(run_dir, output_root=run_dir.parent)

    assert target.read_bytes() == b"previous-complete-archive"
    assert list(run_dir.parent.glob(".symmetry-demo-*.zip.tmp")) == []


# ---------------------------------------------------------------------------
# run_analysis integration
# ---------------------------------------------------------------------------
def test_run_analysis_returns_extended_keys_and_no_numeric_change(
    tmp_path: Path, monkeypatch
) -> None:
    result = _run(tmp_path, monkeypatch)

    for key in (
        "prediction",
        "input_array",
        "input_preview",
        "full_run_zip",
        "stride",
        "class_names",
    ):
        assert key in result, key

    assert result["status"] == "completed"
    assert result["stride"] == 4
    assert result["class_names"] == ["Class A", "Class B"]
    assert Path(result["prediction"]).name == "prediction.npz"
    assert Path(result["input_array"]).name == "input.npy"

    run_dir = Path(result["run_directory"])
    archive = Path(result["full_run_zip"])
    assert archive.parent == run_dir.parent
    with zipfile.ZipFile(archive) as handle:
        members = sorted(handle.namelist())
    assert members == sorted(_ARTIFACT_NAMES)  # every ordinary run artifact
    # The normalized input is archived; the original external upload is not.
    assert "input.npy" in members
    assert "sample.npy" not in members


def test_run_analysis_confidence_and_entropy_use_fixed_colormaps(
    tmp_path: Path, monkeypatch
) -> None:
    from symmetry_harness.colormaps import magma_lut, viridis_lut

    result = _run(tmp_path, monkeypatch)
    confidence = np.asarray(Image.open(result["confidence"]).convert("RGB"))
    entropy = np.asarray(Image.open(result["entropy"]).convert("RGB"))
    # confidence_grid is all 0.5 over bounds (0, 1) -> viridis[128].
    assert np.array_equal(np.unique(confidence.reshape(-1, 3), axis=0), viridis_lut()[128:129])
    # entropy_grid is all ln(2) over bounds (0, ln(2)) -> magma[255].
    assert np.array_equal(np.unique(entropy.reshape(-1, 3), axis=0), magma_lut()[255:256])


def test_archive_failure_preserves_the_completed_numerical_run(
    tmp_path: Path, monkeypatch
) -> None:
    import symmetry_harness.workflow as workflow

    monkeypatch.setattr(
        workflow,
        "create_run_archive",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("synthetic archive failure")
        ),
    )
    result = _run(tmp_path, monkeypatch)

    assert result["status"] == "completed"
    assert result["full_run_zip"] is None
    assert Path(result["fine_tuned_model"]).is_file()
    assert any("complete run ZIP could not be created" in item for item in result["warnings"])
    record = json.loads(Path(result["run_record"]).read_text(encoding="utf-8"))
    assert record["artifact_errors"]["full_run_zip"] == "synthetic archive failure"


def test_create_run_archive_never_mutates_source_files(
    tmp_path: Path, monkeypatch
) -> None:
    result = _run(tmp_path, monkeypatch)
    run_dir = Path(result["run_directory"])
    before = {
        path.name: path.read_bytes()
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
    }
    create_run_archive(run_dir, output_root=run_dir.parent)
    after = {
        path.name: path.read_bytes()
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
    }
    assert before == after
