"""Tests for the fixed-template run report generator."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

from symmetry_harness import cli
from symmetry_harness import run_report as rr


CHANNEL_NAMES = [
    "image",
    "reflection_strength",
    "reflection_sin_2theta",
    "reflection_cos_2theta",
    "rotation_2_fold",
    "rotation_3_fold",
    "rotation_4_fold",
    "rotation_6_fold",
]


def _dense_arrays(size: int = 96, stride: int = 4, classes: int = 2) -> dict:
    step = np.arange(stride // 2, size, stride, dtype=np.int32)
    coordinates = np.array(
        [[int(x), int(y)] for y in step for x in step], dtype=np.int32
    )
    grid = (
        np.add.outer(np.arange(step.size), np.arange(step.size)) % classes
    ).astype(np.int16)
    count = int(coordinates.shape[0])
    return {
        "coordinates_xy": coordinates,
        "x_coordinates": step,
        "y_coordinates": step,
        "logits": np.zeros((count, classes), dtype=np.float32),
        "probabilities": np.full((count, classes), 1.0 / classes, dtype=np.float32),
        "predictions": grid.reshape(-1).astype(np.int16),
        "confidence": np.full(count, 0.75, dtype=np.float32),
        "entropy": np.full(count, 0.3, dtype=np.float32),
        "prediction_grid": grid,
        "confidence_grid": (grid.astype(np.float32) * 0.5 + 0.25),
        "entropy_grid": np.full(grid.shape, 0.3, dtype=np.float32),
    }


def _image(size: int = 96) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    values = np.sin(x / 6.0) + np.cos(y / 7.0)
    return ((values - values.min()) / np.ptp(values)).astype(np.float32)


def _write_fine_tune_run(root: Path, *, name: str = "symmetry-20260101T000000Z-aaaa") -> Path:
    run = root / name
    run.mkdir(parents=True)
    np.save(run / "input.npy", _image())
    features = np.stack([_image() * 0.5 for _ in range(8)]).astype(np.float32)
    np.savez(
        run / "features.npz",
        features=features,
        channel_names=np.asarray(CHANNEL_NAMES),
    )
    np.savez(run / "prediction.npz", **_dense_arrays())
    (run / "annotation_session.json").write_text(
        json.dumps(
            {
                "schema_version": "symmetry-annotation-session-v1",
                "classifier_patch_size": 64,
                "classes": [
                    {
                        "index": 0,
                        "name": "Phase A",
                        "color": "#e41a1c",
                        "points_xy": [[10, 12], [20, 24]],
                    },
                    {
                        "index": 1,
                        "name": "Phase B",
                        "color": "#377eb8",
                        "points_xy": [[40, 44], [50, 54], [60, 66]],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "training_history.json").write_text(
        json.dumps({"loss": [1.0, 0.1, 0.001], "support_accuracy": [0.5, 0.9, 1.0]}),
        encoding="utf-8",
    )
    (run / "run_record.json").write_text(
        json.dumps(
            {
                "contract_version": "symmetry-harness-run-v1",
                "run_id": name,
                "status": "completed",
                "created_utc": "2026-01-01T00:00:00+00:00",
                "options": {
                    "stride": 4,
                    "epochs": 150,
                    "learning_rate": 0.0005,
                    "weight_decay": 0.0,
                    "seed": 42,
                    "batch_size": 512,
                    "adapter_bottleneck": 16,
                    "classifier_patch_size": 64,
                    "symmetry_patch_size": 31,
                    "n_max": 20,
                    "rotation_folds": [2, 3, 4, 6],
                    "reflection_p": 2.0,
                    "input_normalization": "minmax_0_1",
                },
                "model": {
                    "identifier": "cnn_8ch_pg17",
                    "checkpoint": {"path": "weights.pth", "sha256": "deadbeef"},
                },
                "input": {"shape": [96, 96], "normalization": {"policy": "minmax_0_1"}},
                "prediction": {
                    "grid_shape": [24, 24],
                    "sample_count": 576,
                    "confidence_range": [0.5, 1.0],
                    "entropy_range": [0.0, 0.6931],
                },
                "training": {
                    "epochs": 150,
                    "best_support_loss": 0.0001234,
                    "final_support_loss": 0.0001234,
                    "final_support_accuracy": 1.0,
                    "trainable_parameters": 32258,
                    "runtime_seconds": 4.5,
                },
                "provider": {
                    "name": "symmetry-learn",
                    "version": "0.1.1",
                    "contract_version": "symmetry-learn-provider-v1",
                },
                "runtime": {
                    "resolved_device": "cuda",
                    "torch_version": "2.1.0+cu121",
                    "python_version": "3.10.18",
                    "platform": "Windows",
                    "harness_total_seconds": 17.0,
                },
                "warnings": ["Confidence and entropy do not establish physical correctness."],
                "artifacts": {"adapter_head": "adapter_head.pt"},
            }
        ),
        encoding="utf-8",
    )
    return run


def _write_prediction_run(root: Path, *, name: str = "prediction-20260101T000000Z-bbbb") -> Path:
    run = root / name
    item = run / "items" / "image-0001"
    item.mkdir(parents=True)
    np.save(item / "input.npy", _image())
    np.savez(
        item / "features.npz",
        features=np.stack([_image() * 0.5 for _ in range(8)]).astype(np.float32),
        channel_names=np.asarray(CHANNEL_NAMES),
    )
    np.savez(item / "prediction.npz", **_dense_arrays())
    (item / "prediction_record.json").write_text(
        json.dumps(
            {
                "contract_version": "symmetry-harness-prediction-run-v1",
                "run_id": name,
                "item_id": "image-0001",
                "status": "completed",
                "created_utc": "2026-01-01T00:00:00+00:00",
                "model": {"identifier": "cnn_8ch_pg17"},
                "input": {"shape": [96, 96]},
                "output": {
                    "grid_shape": [24, 24],
                    "sample_count": 576,
                    "confidence_range": [0.5, 1.0],
                    "entropy_range": [0.0, 0.6931],
                },
                "provider": {"name": "symmetry-learn", "version": "0.1.1"},
                "warnings": [],
                "artifacts": {"prediction": "prediction.npz"},
            }
        ),
        encoding="utf-8",
    )
    (run / "batch_record.json").write_text(
        json.dumps(
            {
                "contract_version": "symmetry-harness-prediction-run-v1",
                "run_id": name,
                "status": "completed",
                "created_utc": "2026-01-01T00:00:00+00:00",
                "symmetry_harness_version": "0.2.3",
                "runtime_seconds": 3.0,
                "input_normalization": "minmax_0_1",
                "features": {"n_max": 20, "symmetry_patch_size": 31},
                "prediction_options": {"stride": 4, "batch_size": 512},
                "model": {
                    "identifier": "cnn_8ch_pg17",
                    "task_classes": 2,
                    "class_names": ["Phase A", "Phase B"],
                    "class_colors": ["#e41a1c", "#377eb8"],
                },
                "package": {
                    "path": "model.symmodel",
                    "package_sha256": "abc",
                    "model_state_sha256": "def",
                    "training_run_id": "symmetry-20260101T000000Z-aaaa",
                    "schema_version": "symmetry-fine-tuned-model-package-v1",
                },
                "counts": {"requested": 1, "completed": 1, "failed": 0},
                "items": [{"item_id": "image-0001"}],
                "warnings": ["Confidence and entropy do not establish physical correctness."],
                "artifacts": {"batch_record": "batch_record.json"},
            }
        ),
        encoding="utf-8",
    )
    return run


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def test_discover_and_resolve_latest_prefers_newest_matching_prefix(tmp_path: Path) -> None:
    _write_fine_tune_run(tmp_path, name="symmetry-20260101T000000Z-aaaa")
    newest = _write_fine_tune_run(tmp_path, name="symmetry-20260202T000000Z-bbbb")
    _write_prediction_run(tmp_path)

    assert rr.resolve_run("latest", tmp_path, prefix=rr.FINE_TUNE_PREFIX) == newest
    prediction = rr.resolve_run("latest", tmp_path, prefix=rr.PREDICTION_PREFIX)
    assert prediction.name.startswith("prediction-")
    assert rr.resolve_run(newest.name, tmp_path) == newest


def test_resolve_run_rejects_missing_prefix(tmp_path: Path) -> None:
    _write_prediction_run(tmp_path)
    with pytest.raises(rr.ReportError):
        rr.resolve_run("latest", tmp_path, prefix=rr.FINE_TUNE_PREFIX)


def test_detect_kind_and_unknown_directory(tmp_path: Path) -> None:
    fine = _write_fine_tune_run(tmp_path)
    batch = _write_prediction_run(tmp_path)
    assert rr.detect_kind(fine) == rr.KIND_FINE_TUNE
    assert rr.detect_kind(batch) == rr.KIND_PREDICTION
    empty = tmp_path / "not-a-run"
    empty.mkdir()
    with pytest.raises(rr.ReportError):
        rr.detect_kind(empty)


# ---------------------------------------------------------------------------
# Fine-tune report
# ---------------------------------------------------------------------------
def test_fine_tune_report_has_every_section_and_figure(tmp_path: Path) -> None:
    run = _write_fine_tune_run(tmp_path)
    result = rr.build_run_report(run, notes="Session notes body.")

    assert result.kind == rr.KIND_FINE_TUNE
    assert result.report_path == run / rr.REPORT_FILENAME
    text = result.report_path.read_text(encoding="utf-8")

    for heading in (
        "## 1. Summary",
        "## 2. Workflow Overview",
        "## 3. Session Notes",
        "## 4. Input & Annotation",
        "## 5. Symmetry Features",
        "## 6. Fine-tuning",
        "## 7. Dense Prediction",
        "## 8. Configuration & Provenance",
        "## 9. Caveats",
        "## 10. Artifacts",
    ):
        assert heading in text

    assert "Session notes body." in text
    assert "Phase A" in text and "Phase B" in text
    # The workflow must quote the run's actual options, not config defaults.
    assert "patch=31" in text

    figures = {path.name for path in result.figure_paths}
    assert figures == {
        rr.FIG_INPUT,
        rr.FIG_SUPPORT,
        rr.FIG_SYMMETRY,
        rr.FIG_TRAINING,
        rr.FIG_MASK,
        rr.FIG_OVERLAY,
        rr.FIG_CONFIDENCE,
        rr.FIG_ENTROPY,
    }
    for path in result.figure_paths:
        assert path.is_file()
        assert path.parent == run / rr.FIGURES_DIRNAME
        assert f"]({path.name})" in text


def test_fine_tune_report_notes_default_placeholder(tmp_path: Path) -> None:
    run = _write_fine_tune_run(tmp_path)
    result = rr.build_run_report(run, figures=False)
    assert "_Not provided._" in result.report_path.read_text(encoding="utf-8")


def test_no_figures_produces_text_only_report(tmp_path: Path) -> None:
    run = _write_fine_tune_run(tmp_path)
    result = rr.build_run_report(run, figures=False)
    assert result.figure_paths == ()
    assert not (run / rr.FIGURES_DIRNAME).exists()
    assert "![" not in result.report_path.read_text(encoding="utf-8")


def test_missing_optional_artifacts_still_report(tmp_path: Path) -> None:
    run = _write_fine_tune_run(tmp_path)
    (run / "features.npz").unlink()
    (run / "prediction.npz").unlink()
    (run / "training_history.json").unlink()

    result = rr.build_run_report(run)
    text = result.report_path.read_text(encoding="utf-8")
    assert "feature bundle not persisted" in text
    # The unsupported sections degrade to their tables without figures.
    names = {path.name for path in result.figure_paths}
    assert rr.FIG_INPUT in names
    assert rr.FIG_SYMMETRY not in names
    assert rr.FIG_MASK not in names


def test_custom_output_path(tmp_path: Path) -> None:
    run = _write_fine_tune_run(tmp_path)
    target = tmp_path / "elsewhere" / "custom.md"
    result = rr.build_run_report(run, figures=False, output=target)
    assert result.report_path == target
    assert target.is_file()


# ---------------------------------------------------------------------------
# Prediction batch report
# ---------------------------------------------------------------------------
def test_prediction_batch_writes_index_and_item_reports(tmp_path: Path) -> None:
    run = _write_prediction_run(tmp_path)
    result = rr.build_run_report(run, notes="Prediction notes.")

    assert result.kind == rr.KIND_PREDICTION
    assert result.report_path == run / rr.REPORT_FILENAME
    assert len(result.item_reports) == 1
    item_report = result.item_reports[0]
    assert item_report == run / "items" / "image-0001" / rr.REPORT_FILENAME

    index_text = result.report_path.read_text(encoding="utf-8")
    assert "Symmetry Prediction Batch Report" in index_text
    assert "items/image-0001/report_summary.md" in index_text
    assert "Prediction notes." in index_text

    item_text = item_report.read_text(encoding="utf-8")
    assert "Symmetry Prediction Report" in item_text
    assert "Phase A" in item_text

    item_figures = {path.name for path in result.figure_paths}
    assert rr.FIG_SUPPORT not in item_figures
    assert rr.FIG_TRAINING not in item_figures
    assert rr.FIG_SYMMETRY in item_figures
    assert rr.FIG_MASK in item_figures


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def test_feature_tile_uses_fixed_range_per_channel() -> None:
    """Signed channels span [-1, 1]; unsigned channels span [0, 1]."""
    signed = "reflection_sin_2theta"
    unsigned = "reflection_strength"

    assert rr._feature_display_range(signed) == (-1.0, 1.0)
    assert rr._feature_display_range(unsigned) == (0.0, 1.0)

    # 0.0 is mid-grey only on a signed channel; on an unsigned channel it is black.
    assert 126 <= int(rr._feature_tile(np.full((2, 2), 0.0), signed)[0, 0]) <= 129
    assert int(rr._feature_tile(np.full((2, 2), 0.0), unsigned)[0, 0]) == 0
    assert int(rr._feature_tile(np.full((2, 2), -0.5), signed)[0, 0]) == 64
    assert 126 <= int(rr._feature_tile(np.full((2, 2), 0.5), unsigned)[0, 0]) <= 129


def test_symmetry_montage_geometry() -> None:
    features = np.full((8, 16, 16), 0.5, dtype=np.float32)
    montage = rr.build_symmetry_maps_figure(features, CHANNEL_NAMES)
    assert montage.dtype == np.uint8
    assert montage.ndim == 3 and montage.shape[2] == 3
    assert montage.shape[0] == 2 * (rr._MONTAGE_TILE + rr._MONTAGE_LABEL_HEIGHT)
    assert montage.shape[1] == 4 * rr._MONTAGE_TILE


def test_feature_tile_requires_matching_channel_count() -> None:
    with pytest.raises(rr.ReportError):
        rr.build_symmetry_maps_figure(np.zeros((7, 8, 8), dtype=np.float32), CHANNEL_NAMES)


def test_training_figure_renders_for_degenerate_inputs() -> None:
    flat = rr.build_training_figure([1.0, 1.0], [1.0, 1.0])
    assert flat.dtype == np.uint8 and flat.ndim == 3
    empty = rr.build_training_figure([], [])
    assert empty.shape == (
        rr._TRAINING_MARGIN + 2 * (rr._TRAINING_PANEL_HEIGHT + rr._TRAINING_GAP),
        rr._TRAINING_WIDTH,
        3,
    )


def test_support_overlay_marks_every_point() -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    classes = [
        rr._ClassEntry(0, "A", "#e41a1c", [(10, 10)]),
        rr._ClassEntry(1, "B", "#377eb8", [(40, 40)]),
    ]
    figure = rr.build_support_overlay_figure(image, classes)
    assert figure.ndim == 3 and figure.shape[2] == 3
    # The legend is appended below the image, so the figure is taller than 64.
    assert figure.shape[0] > 64


def test_figures_are_valid_png(tmp_path: Path) -> None:
    run = _write_fine_tune_run(tmp_path)
    result = rr.build_run_report(run)
    for path in result.figure_paths:
        with Image.open(path) as handle:
            handle.verify()


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------
def test_cli_report_writes_json_summary(tmp_path, monkeypatch, capsys) -> None:
    run = _write_fine_tune_run(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["symmetry", "report", "--run", str(run), "--notes-text", "cli notes"],
    )
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["kind"] == rr.KIND_FINE_TUNE
    assert Path(payload["report"]) == run / rr.REPORT_FILENAME
    assert "cli notes" in (run / rr.REPORT_FILENAME).read_text(encoding="utf-8")


def test_cli_report_accepts_notes_file(tmp_path, monkeypatch, capsys) -> None:
    run = _write_fine_tune_run(tmp_path)
    notes = tmp_path / "notes.md"
    notes.write_text("notes from a file", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["symmetry", "report", "--run", str(run), "--notes", str(notes), "--no-figures"],
    )
    cli.main()
    capsys.readouterr()
    assert "notes from a file" in (run / rr.REPORT_FILENAME).read_text(encoding="utf-8")


def test_cli_report_latest_with_output_root(tmp_path, monkeypatch, capsys) -> None:
    run = _write_fine_tune_run(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "symmetry",
            "report",
            "--run",
            "latest",
            "--output-root",
            str(tmp_path),
            "--no-figures",
        ],
    )
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert Path(payload["report"]) == run / rr.REPORT_FILENAME


def test_cli_config_path_uses_installed_pointer(tmp_path, monkeypatch) -> None:
    """The recorded config must resolve from any working directory."""
    from symmetry_harness.runtime_state import CONFIG_PATH_NAME

    state = tmp_path / "state"
    state.mkdir()
    config = tmp_path / "installed.json"
    config.write_text("{}", encoding="utf-8")
    # Older installers wrote a UTF-8 BOM and CRLF; both must still resolve.
    (state / CONFIG_PATH_NAME).write_bytes(
        b"\xef\xbb\xbf" + str(config).encode("utf-8") + b"\r\n"
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    monkeypatch.setenv("SYMMETRY_HARNESS_HOME", str(state))
    monkeypatch.delenv("SYMMETRY_HARNESS_CONFIG", raising=False)
    monkeypatch.chdir(elsewhere)
    assert cli._config_path(None) == config


def test_cli_config_path_prefers_environment_over_installed(
    tmp_path, monkeypatch
) -> None:
    from symmetry_harness.runtime_state import CONFIG_PATH_NAME

    state = tmp_path / "state"
    state.mkdir()
    installed = tmp_path / "installed.json"
    installed.write_text("{}", encoding="utf-8")
    (state / CONFIG_PATH_NAME).write_text(str(installed), encoding="utf-8")

    explicit = tmp_path / "explicit.json"
    explicit.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("SYMMETRY_HARNESS_HOME", str(state))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SYMMETRY_HARNESS_CONFIG", str(explicit))
    assert cli._config_path(None) == explicit
