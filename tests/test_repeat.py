"""Tests for the repeated fine-tuning stability loop."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from symmetry_harness import repeat as repeat_module
from symmetry_harness.analysis import DensePrediction
from symmetry_harness.config import load_harness_config
from symmetry_harness.repeat import (
    Accumulated,
    RepeatError,
    RepeatOptions,
    agreement_with,
    class_map_from_labels,
    evaluate_round,
    history_statistics,
    interior_points,
    load_label_map,
    run_repeat,
    run_repeat_batch,
    score_prediction,
    select_points,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def config(tmp_path: Path):
    config = load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")
    from dataclasses import replace as dataclass_replace

    return dataclass_replace(config, output_root=tmp_path / "runs")


def two_phase_image(height: int = 128, width: int = 128, split: int = 64) -> np.ndarray:
    image = np.zeros((height, width), dtype=np.float32)
    image[:, split:] = 1.0
    return image


def two_phase_labels(height: int = 128, width: int = 128, split: int = 64) -> np.ndarray:
    labels = np.full((height, width), 14, dtype=np.int32)
    labels[:, split:] = 17
    return labels


def _grid_axes() -> tuple[np.ndarray, np.ndarray]:
    values = np.arange(32, 97, 4, dtype=np.int64)
    return values, values


def fake_prediction(*, inverted: bool = False, confident: float = 0.99) -> DensePrediction:
    """Build one internally consistent dense prediction on the 17x17 test grid."""
    xs, ys = _grid_axes()
    grid_x = np.broadcast_to(xs[None, :], (len(ys), len(xs)))
    grid = ((grid_x >= 64) != inverted).astype(np.int16)

    confidence = np.full(grid.shape, confident, dtype=np.float32)
    confidence[:, :2] = 0.55  # a few ambiguous cells on purpose

    coordinates = np.asarray(
        [(int(x), int(y)) for y in ys for x in xs], dtype=np.int32
    )
    flat_grid = grid.reshape(-1)
    flat_confidence = confidence.reshape(-1)
    probabilities = np.stack(
        [
            np.where(flat_grid == 0, flat_confidence, 1.0 - flat_confidence),
            np.where(flat_grid == 1, flat_confidence, 1.0 - flat_confidence),
        ],
        axis=1,
    ).astype(np.float32)
    logits = np.log(np.clip(probabilities, 1e-9, 1.0)).astype(np.float32)
    entropy = (-(probabilities * np.log(np.clip(probabilities, 1e-9, 1.0))).sum(axis=1)).astype(
        np.float32
    )
    return DensePrediction(
        coordinates_xy=coordinates,
        x_coordinates=xs,
        y_coordinates=ys,
        logits=logits,
        probabilities=probabilities,
        predictions=flat_grid.astype(np.int16),
        confidence=flat_confidence,
        entropy=entropy,
        prediction_grid=grid,
        confidence_grid=confidence,
        entropy_grid=entropy.reshape(grid.shape),
    )


def prediction_arrays(prediction: DensePrediction) -> dict[str, np.ndarray]:
    return {
        "coordinates_xy": prediction.coordinates_xy,
        "x_coordinates": prediction.x_coordinates,
        "y_coordinates": prediction.y_coordinates,
        "logits": prediction.logits,
        "probabilities": prediction.probabilities,
        "predictions": prediction.predictions,
        "confidence": prediction.confidence,
        "entropy": prediction.entropy,
        "prediction_grid": prediction.prediction_grid,
        "confidence_grid": prediction.confidence_grid,
        "entropy_grid": prediction.entropy_grid,
    }


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def test_class_map_maps_codes_positionally() -> None:
    labels = two_phase_labels()
    class_map = class_map_from_labels(labels, (14, 17))
    assert class_map.dtype == np.int8
    assert set(np.unique(class_map)) == {0, 1}
    assert class_map[0, 0] == 0 and class_map[0, -1] == 1

    flipped = class_map_from_labels(labels, (17, 14))
    assert flipped[0, 0] == 1 and flipped[0, -1] == 0


def test_class_map_rejects_unknown_codes() -> None:
    labels = two_phase_labels()
    labels[0, 0] = 9
    with pytest.raises(RepeatError, match="not part of label_codes"):
        class_map_from_labels(labels, (14, 17))


def test_load_label_map_requires_npy_integer(tmp_path: Path) -> None:
    float_path = tmp_path / "labels.npy"
    np.save(float_path, np.zeros((8, 8), dtype=np.float32))
    with pytest.raises(RepeatError, match="integer raster"):
        load_label_map(float_path)

    missing = tmp_path / "nope.npy"
    with pytest.raises(RepeatError, match="does not exist"):
        load_label_map(missing)

    wrong_suffix = tmp_path / "labels.png"
    wrong_suffix.write_bytes(b"not a npy")
    with pytest.raises(RepeatError, match="must be a .npy"):
        load_label_map(wrong_suffix)


# ---------------------------------------------------------------------------
# Seed points
# ---------------------------------------------------------------------------


def test_interior_points_respect_class_and_bounds() -> None:
    labels = two_phase_labels()
    class_map = class_map_from_labels(labels, (14, 17))
    group_a, group_b = interior_points(
        class_map, patch_size=64, points_per_class=5, margin=16, seed=42
    )
    assert len(group_a) == len(group_b) == 5
    for x, y in group_a:
        assert class_map[y, x] == 0
        assert 32 <= x <= 96 and 32 <= y <= 96
    for x, y in group_b:
        assert class_map[y, x] == 1
    assert len(set(group_a) & set(group_b)) == 0


def test_interior_points_are_reproducible() -> None:
    labels = two_phase_labels()
    class_map = class_map_from_labels(labels, (14, 17))
    first = interior_points(class_map, patch_size=64, points_per_class=5, margin=16, seed=7)
    second = interior_points(class_map, patch_size=64, points_per_class=5, margin=16, seed=7)
    third = interior_points(class_map, patch_size=64, points_per_class=5, margin=16, seed=8)
    assert first == second
    assert first != third


def test_interior_points_shrink_the_margin_for_a_narrow_phase() -> None:
    """A phase narrower than twice the margin must not stop the loop."""
    labels = np.full((128, 128), 14, dtype=np.int32)
    labels[58:70] = 17  # a 12 px band: eroding by 32 wipes it out
    class_map = class_map_from_labels(labels, (14, 17))
    group_a, group_b = interior_points(
        class_map, patch_size=8, points_per_class=2, margin=32, seed=1
    )
    assert len(group_a) == 2 and len(group_b) == 2
    for x, y in group_b:
        assert class_map[y, x] == 1


def test_interior_points_reject_a_missing_phase() -> None:
    labels = np.full((128, 128), 14, dtype=np.int32)
    class_map = class_map_from_labels(labels, (14, 17))
    with pytest.raises(RepeatError, match="interior pixels"):
        interior_points(class_map, patch_size=8, points_per_class=2, margin=8, seed=1)


# ---------------------------------------------------------------------------
# Accumulation and sampling
# ---------------------------------------------------------------------------


def test_accumulated_averages_confidence_and_votes() -> None:
    first = fake_prediction()
    accumulated = Accumulated.empty(first.confidence_grid.shape)
    accumulated.update(first)
    assert accumulated.rounds == 1
    assert np.allclose(accumulated.confidence, first.confidence_grid)

    inverted = fake_prediction(inverted=True)
    accumulated.update(inverted)
    assert accumulated.rounds == 2
    expected = (first.confidence_grid + inverted.confidence_grid) / 2.0
    assert np.allclose(accumulated.confidence, expected)
    # One vote each per class: the argmax breaks the tie towards class 0.
    assert set(np.unique(accumulated.labels)) == {0}


def _cell(prediction: DensePrediction, x: int, y: int) -> tuple[int, int]:
    """Translate an image coordinate into the dense-grid index pair."""
    column = int((x - int(prediction.x_coordinates[0])) // 4)
    row = int((y - int(prediction.y_coordinates[0])) // 4)
    return row, column


def test_select_points_respects_tau_and_class() -> None:
    prediction = fake_prediction()
    accumulated = Accumulated.empty(prediction.confidence_grid.shape)
    accumulated.update(prediction)
    xs, ys = prediction.x_coordinates, prediction.y_coordinates
    sampling = select_points(
        accumulated, xs, ys, points_per_class=4, tau=0.95, seed=11
    )
    points_a, points_b = sampling.points_by_class
    assert len(points_a) == len(points_b) == 4
    assert len(set(points_a) & set(points_b)) == 0
    for x, y in points_a:
        row, column = _cell(prediction, x, y)
        assert prediction.prediction_grid[row, column] == 0
        assert prediction.confidence_grid[row, column] >= 0.95
    for x, y in points_b:
        row, column = _cell(prediction, x, y)
        assert prediction.prediction_grid[row, column] == 1
        assert prediction.confidence_grid[row, column] >= 0.95
    assert sampling.pool_sizes[0] > 0 and sampling.pool_sizes[1] > 0


def test_select_points_is_seed_deterministic() -> None:
    prediction = fake_prediction()
    accumulated = Accumulated.empty(prediction.confidence_grid.shape)
    accumulated.update(prediction)
    xs, ys = prediction.x_coordinates, prediction.y_coordinates
    first = select_points(accumulated, xs, ys, points_per_class=5, tau=0.95, seed=3)
    second = select_points(accumulated, xs, ys, points_per_class=5, tau=0.95, seed=3)
    third = select_points(accumulated, xs, ys, points_per_class=5, tau=0.95, seed=4)
    assert first.points_by_class == second.points_by_class
    assert first.points_by_class != third.points_by_class


def test_select_points_rejects_small_pool() -> None:
    prediction = fake_prediction()
    accumulated = Accumulated.empty(prediction.confidence_grid.shape)
    accumulated.update(prediction)
    xs, ys = prediction.x_coordinates, prediction.y_coordinates
    with pytest.raises(RepeatError, match="confident cells"):
        select_points(accumulated, xs, ys, points_per_class=500, tau=0.95, seed=1)
    with pytest.raises(RepeatError, match="confident cells"):
        select_points(accumulated, xs, ys, points_per_class=1, tau=0.999, seed=1)


# ---------------------------------------------------------------------------
# Metrics and stop rules
# ---------------------------------------------------------------------------


def test_score_prediction_matches_a_perfect_grid() -> None:
    labels = two_phase_labels()
    class_map = class_map_from_labels(labels, (14, 17))
    prediction = fake_prediction()
    metrics = score_prediction(prediction, class_map)
    assert metrics["cells"] == 289
    assert metrics["acc"] == pytest.approx(1.0)
    assert metrics["iou_class_a"] == pytest.approx(1.0)
    assert metrics["iou_class_b"] == pytest.approx(1.0)
    # The grid spans x = 32..96 in steps of 4: eight cells left of the split at
    # 64 and nine cells on the right, so class B covers 9/17 of the grid.
    assert metrics["area_frac_gt"] == pytest.approx(9 / 17)
    assert metrics["area_frac_pred"] == pytest.approx(9 / 17)


def test_score_prediction_detects_an_inverted_map() -> None:
    labels = two_phase_labels()
    class_map = class_map_from_labels(labels, (14, 17))
    metrics = score_prediction(fake_prediction(inverted=True), class_map)
    assert metrics["acc"] == pytest.approx(0.0)
    assert metrics["iou_class_a"] == pytest.approx(0.0)


def test_agreement_with_handles_missing_previous() -> None:
    grid = np.zeros((4, 4), dtype=np.int16)
    assert np.isnan(agreement_with(None, grid))
    assert agreement_with(grid, grid) == pytest.approx(1.0)
    assert agreement_with(grid, np.ones((4, 4), dtype=np.int16)) == pytest.approx(0.0)
    assert np.isnan(agreement_with(np.zeros((2, 2), dtype=np.int16), grid))


def test_history_statistics_ignores_nan() -> None:
    statistics = history_statistics([0.9, float("nan"), 0.95, 1.0])
    assert statistics["count"] == 3
    assert statistics["mean"] == pytest.approx(0.95)
    assert statistics["range"] == pytest.approx(0.1)
    empty = history_statistics([])
    assert empty["count"] == 0 and np.isnan(empty["mean"])


def test_stop_rules() -> None:
    options = RepeatOptions()
    healthy = evaluate_round({"acc": 0.97, "iou_class_a": 0.9, "iou_class_b": 0.9}, [0.96, 0.97], options)
    assert not healthy.stop and healthy.status == "ok"

    below_floor = evaluate_round({"acc": 0.5, "iou_class_a": 0.4, "iou_class_b": 0.4}, [0.97], options)
    assert below_floor.stop and below_floor.status == "alarm"
    assert "floor" in below_floor.reason

    drift = evaluate_round(
        {"acc": 0.90, "iou_class_a": 0.8, "iou_class_b": 0.8},
        [0.99, 0.98, 1.0],
        RepeatOptions(alarm_floor=0.5),
    )
    assert drift.stop and "sigma" in drift.reason

    collapsed = evaluate_round(
        {"acc": 0.95, "iou_class_a": 0.9, "iou_class_b": 0.05},
        [0.96, 0.97],
        options,
    )
    assert collapsed.stop and "class collapse" in collapsed.reason

    observing = evaluate_round(
        {"acc": 0.5, "iou_class_a": 0.1, "iou_class_b": 0.1},
        [0.97],
        RepeatOptions(stop_on_alarm=False),
    )
    assert not observing.stop and observing.status == "alarm"


def test_option_validation() -> None:
    RepeatOptions().validate()
    for bad in (
        RepeatOptions(rounds=-1),
        RepeatOptions(points_per_class=0),
        RepeatOptions(tau=0.2),
        RepeatOptions(alarm_drop=0),
        RepeatOptions(label_codes=(14, 14)),
        RepeatOptions(class_names=("A", "A")),
    ):
        with pytest.raises(RepeatError):
            bad.validate()


# ---------------------------------------------------------------------------
# Loop engine
# ---------------------------------------------------------------------------


def _install_fake_provider(monkeypatch: pytest.MonkeyPatch, *, bad_round: int | None = None):
    """Replace the numerical path with a deterministic fake prediction."""
    calls: list[dict] = []

    def fake_doctor(config):
        return {"status": "ready", "issues": [], "recommendations": []}

    def fake_features(config, image, *, options):
        raise RuntimeError("feature reuse is unavailable in this test")

    def fake_run_analysis(config, *, image_path, annotation_session, output_root, resolved_options=None, overrides=None, features_path=None, features_record_path=None):
        index = len(calls)
        session = annotation_session
        run_dir = Path(output_root) / f"symmetry-test-{index:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        prediction = fake_prediction(inverted=(bad_round is not None and index == bad_round))
        np.savez_compressed(run_dir / "prediction.npz", **prediction_arrays(prediction))
        (run_dir / "training_history.json").write_text(
            json.dumps({"final_support_accuracy": 1.0, "final_support_loss": 1e-6}),
            encoding="utf-8",
        )
        for name in ("run_record.json", "confidence.png", "prediction_overlay.png", "fine_tuned_model.symmodel"):
            (run_dir / name).write_bytes(b"stub")
        (run_dir.parent / f"{run_dir.name}-full-run.zip").write_bytes(b"zip")
        calls.append(
            {
                "seed": None if resolved_options is None else resolved_options.seed,
                "minimum_shots": (
                    None if resolved_options is None else resolved_options.minimum_shots_per_class
                ),
                "points": [[tuple(point) for point in entry.points] for entry in session.classes],
                "counts": [len(entry.points) for entry in session.classes],
                "features": features_path,
            }
        )
        return {
            "status": "completed",
            "run_id": run_dir.name,
            "run_directory": str(run_dir),
        }

    monkeypatch.setattr(repeat_module, "doctor", fake_doctor)
    monkeypatch.setattr(repeat_module, "run_provider_features", fake_features)
    monkeypatch.setattr(repeat_module, "run_analysis", fake_run_analysis)
    return calls


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "image.npy"
    label_path = tmp_path / "labels.npy"
    np.save(image_path, two_phase_image())
    np.save(label_path, two_phase_labels())
    return image_path, label_path


def test_run_repeat_keeps_rounds_independent(config, tmp_path, monkeypatch) -> None:
    calls = _install_fake_provider(monkeypatch)
    image_path, label_path = _write_inputs(tmp_path)
    options = RepeatOptions(rounds=2, points_per_class=3, tau=0.95, base_seed=42, interior_margin=8)

    result = run_repeat(
        config, image_path=image_path, label_path=label_path, options=options
    )

    assert result["status"] == "completed"
    assert result["rounds_completed"] == 3
    assert len(calls) == 3
    # The support set never accumulates: every round carries exactly k points.
    assert all(call["counts"] == [3, 3] for call in calls)
    # Seeds advance with the round index.
    assert [call["seed"] for call in calls] == [42, 43, 44]

    loop_dir = Path(result["loop_directory"])
    assert (loop_dir / "loop_manifest.json").is_file()
    assert (loop_dir / "loop_rounds.csv").is_file()
    assert (loop_dir / "loop_summary.md").is_file()
    csv_lines = (loop_dir / "loop_rounds.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) == 4

    for index in range(3):
        round_dir = loop_dir / "rounds" / f"round-{index:03d}"
        assert (round_dir / f"session_round{index:03d}.json").is_file()
        assert (round_dir / "metrics.json").is_file()
        assert (round_dir / "stop_decision.json").is_file()
        kept = round_dir / "kept"
        assert (kept / "prediction.npz").is_file()
        assert (kept / "fine_tuned_model.symmodel").is_file()
        assert (kept / "annotation_session.json").is_file()
    # The run directories stay, but their two ~46 MB duplicates are removed.
    leftovers = [
        path
        for path in (loop_dir / "rounds").iterdir()
        if path.is_dir() and path.name.startswith("symmetry-")
    ]
    assert len(leftovers) == 3
    for leftover in leftovers:
        assert not (leftover / "fine_tuned_model.symmodel").is_file()
    assert list((loop_dir / "rounds").glob("*.zip")) == []
    assert (loop_dir / "figures" / "acc_vs_round.png").is_file()


def test_run_repeat_stops_on_alarm(config, tmp_path, monkeypatch) -> None:
    _install_fake_provider(monkeypatch, bad_round=2)
    image_path, label_path = _write_inputs(tmp_path)
    options = RepeatOptions(rounds=4, points_per_class=3, tau=0.95, interior_margin=8)

    result = run_repeat(
        config, image_path=image_path, label_path=label_path, options=options
    )

    assert result["status"] == "alarm"
    assert result["rounds_completed"] == 3
    assert result["alarm"] and "floor" in result["alarm"]
    summary = Path(result["summary"]).read_text(encoding="utf-8")
    assert "## Alarm" in summary


def test_run_repeat_can_continue_past_an_alarm(config, tmp_path, monkeypatch) -> None:
    _install_fake_provider(monkeypatch, bad_round=2)
    image_path, label_path = _write_inputs(tmp_path)
    options = RepeatOptions(rounds=3, points_per_class=3, stop_on_alarm=False, interior_margin=8)

    result = run_repeat(
        config, image_path=image_path, label_path=label_path, options=options
    )

    assert result["status"] == "completed"
    assert result["rounds_completed"] == 4


def test_run_repeat_resumes_from_completed_rounds(config, tmp_path, monkeypatch) -> None:
    calls = _install_fake_provider(monkeypatch)
    image_path, label_path = _write_inputs(tmp_path)
    first = run_repeat(
        config,
        image_path=image_path,
        label_path=label_path,
        options=RepeatOptions(rounds=1, points_per_class=3, interior_margin=8),
    )
    assert first["rounds_completed"] == 2

    calls.clear()
    second = run_repeat(
        config,
        image_path=image_path,
        label_path=label_path,
        options=RepeatOptions(rounds=3, points_per_class=3, interior_margin=8),
        resume=first["loop_directory"],
    )
    assert second["loop_directory"] == first["loop_directory"]
    assert second["rounds_completed"] == 4
    assert len(calls) == 2
    assert [call["seed"] for call in calls] == [44, 45]


def test_run_repeat_refuses_a_different_image(config, tmp_path, monkeypatch) -> None:
    _install_fake_provider(monkeypatch)
    image_path, label_path = _write_inputs(tmp_path)
    first = run_repeat(
        config,
        image_path=image_path,
        label_path=label_path,
        options=RepeatOptions(rounds=0, points_per_class=3, interior_margin=8),
    )

    other_image = tmp_path / "other.npy"
    np.save(other_image, np.flip(two_phase_image(), axis=1))
    with pytest.raises(RepeatError, match="different image"):
        run_repeat(
            config,
            image_path=other_image,
            label_path=label_path,
            options=RepeatOptions(rounds=1, points_per_class=3, interior_margin=8),
            resume=first["loop_directory"],
        )


def test_run_repeat_rejects_a_mismatched_label_shape(config, tmp_path, monkeypatch) -> None:
    _install_fake_provider(monkeypatch)
    image_path, label_path = _write_inputs(tmp_path)
    np.save(label_path, two_phase_labels(64, 64))
    with pytest.raises(RepeatError, match="shapes differ"):
        run_repeat(
            config,
            image_path=image_path,
            label_path=label_path,
            options=RepeatOptions(rounds=1, points_per_class=3, interior_margin=8),
        )


def test_run_repeat_batch_reports_every_image(config, tmp_path, monkeypatch) -> None:
    _install_fake_provider(monkeypatch)
    image_path, label_path = _write_inputs(tmp_path)
    other_image = tmp_path / "second.npy"
    other_labels = tmp_path / "second_labels.npy"
    np.save(other_image, two_phase_image())
    np.save(other_labels, two_phase_labels())

    payload = run_repeat_batch(
        config,
        [
            {"image": image_path, "gt": label_path},
            {"image": other_image, "gt": other_labels},
        ],
        options=RepeatOptions(rounds=0, points_per_class=3, interior_margin=8),
    )

    assert payload["status"] == "completed"
    assert len(payload["loops"]) == 2
    assert all(item["status"] == "completed" for item in payload["loops"])
    assert (Path(payload["batch_directory"]) / "repeat_batch.json").is_file()


def test_run_repeat_relaxes_the_shot_floor_for_one_shot_runs(config, tmp_path, monkeypatch) -> None:
    calls = _install_fake_provider(monkeypatch)
    image_path, label_path = _write_inputs(tmp_path)
    # The example config demands three points per class; a one-shot replicate
    # must still be allowed to run.
    run_repeat(
        config,
        image_path=image_path,
        label_path=label_path,
        options=RepeatOptions(rounds=0, points_per_class=1, interior_margin=8),
    )
    assert [call["counts"] for call in calls] == [[1, 1]]
    assert calls[0]["minimum_shots"] == 1


def test_run_repeat_reports_reuse_failure(config, tmp_path, monkeypatch) -> None:
    _install_fake_provider(monkeypatch)
    image_path, label_path = _write_inputs(tmp_path)
    messages: list[str] = []
    result = run_repeat(
        config,
        image_path=image_path,
        label_path=label_path,
        options=RepeatOptions(rounds=0, points_per_class=3, interior_margin=8),
        progress=messages.append,
    )
    assert result["status"] == "completed"
    assert any("feature reuse unavailable" in message for message in messages)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def test_point_groups_reads_both_row_shapes() -> None:
    from symmetry_harness.repeat_report import _point_groups

    nested = {"points_by_class": [[(1, 2), (3, 4)], [(5, 6)]]}
    assert _point_groups(nested) == [[(1, 2), (3, 4)], [(5, 6)]]

    # loop_rounds.csv keeps the same payload under points_xy, as a JSON string.
    from_csv = {"points_xy": json.dumps([[[1, 2]], [[3, 4], [5, 6]]])}
    assert _point_groups(from_csv) == [[[1, 2]], [[3, 4], [5, 6]]]
    assert _point_groups({}) == []


def test_evenly_keeps_the_first_and_the_last() -> None:
    from symmetry_harness.repeat_report import _evenly

    assert _evenly(list(range(5)), 12) == [0, 1, 2, 3, 4]
    picked = _evenly(list(range(100)), 12)
    assert len(picked) == 12
    assert picked[0] == 0 and picked[-1] == 99


def test_round_sheet_draws_one_row_per_round(tmp_path: Path) -> None:
    from symmetry_harness.repeat_report import build_round_sheet

    prediction = fake_prediction()
    rounds_dir = tmp_path / "rounds"
    kept = rounds_dir / "round-000" / "kept"
    kept.mkdir(parents=True)
    np.savez_compressed(kept / "prediction.npz", **prediction_arrays(prediction))

    path = build_round_sheet(
        tmp_path / "sheet.png",
        two_phase_image(),
        class_map_from_labels(two_phase_labels(), (14, 17)),
        [
            {
                "round": 0,
                "acc": "0.97",
                "points_by_class": [[(40, 40)], [(80, 80)]],
            }
        ],
        rounds_dir=rounds_dir,
        class_colors=("#e41a1c", "#377eb8"),
        patch_size=16,
        stride=4,
    )
    assert path is not None and path.is_file()
    assert path.stat().st_size > 0


def test_round_sheet_skips_rounds_without_a_prediction(tmp_path: Path) -> None:
    from symmetry_harness.repeat_report import build_round_sheet

    path = build_round_sheet(
        tmp_path / "sheet.png",
        two_phase_image(),
        None,
        [{"round": 0, "acc": "0.97", "points_by_class": [[(40, 40)], [(80, 80)]]}],
        rounds_dir=tmp_path / "rounds",
        class_colors=("#e41a1c", "#377eb8"),
        patch_size=16,
        stride=4,
    )
    assert path is None
    assert not (tmp_path / "sheet.png").exists()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def test_point_groups_reads_both_row_shapes() -> None:
    from symmetry_harness.repeat_report import _point_groups

    nested = {"points_by_class": [[(1, 2), (3, 4)], [(5, 6)]]}
    assert _point_groups(nested) == [[(1, 2), (3, 4)], [(5, 6)]]

    # loop_rounds.csv keeps the same payload under points_xy, as a JSON string.
    from_csv = {"points_xy": json.dumps([[[1, 2]], [[3, 4], [5, 6]]])}
    assert _point_groups(from_csv) == [[[1, 2]], [[3, 4], [5, 6]]]
    assert _point_groups({}) == []


def test_evenly_keeps_the_first_and_the_last() -> None:
    from symmetry_harness.repeat_report import _evenly

    assert _evenly(list(range(5)), 12) == [0, 1, 2, 3, 4]
    picked = _evenly(list(range(100)), 12)
    assert len(picked) == 12
    assert picked[0] == 0 and picked[-1] == 99


def test_round_sheet_draws_one_row_per_round(tmp_path: Path) -> None:
    from symmetry_harness.repeat_report import build_round_sheet

    prediction = fake_prediction()
    rounds_dir = tmp_path / "rounds"
    kept = rounds_dir / "round-000" / "kept"
    kept.mkdir(parents=True)
    np.savez_compressed(kept / "prediction.npz", **prediction_arrays(prediction))

    path = build_round_sheet(
        tmp_path / "sheet.png",
        two_phase_image(),
        class_map_from_labels(two_phase_labels(), (14, 17)),
        [
            {
                "round": 0,
                "acc": "0.97",
                "points_by_class": [[(40, 40)], [(80, 80)]],
            }
        ],
        rounds_dir=rounds_dir,
        class_colors=("#e41a1c", "#377eb8"),
        patch_size=16,
        stride=4,
    )
    assert path is not None and path.is_file()
    assert path.stat().st_size > 0


def test_round_sheet_skips_rounds_without_a_prediction(tmp_path: Path) -> None:
    from symmetry_harness.repeat_report import build_round_sheet

    path = build_round_sheet(
        tmp_path / "sheet.png",
        two_phase_image(),
        None,
        [{"round": 0, "acc": "0.97", "points_by_class": [[(40, 40)], [(80, 80)]]}],
        rounds_dir=tmp_path / "rounds",
        class_colors=("#e41a1c", "#377eb8"),
        patch_size=16,
        stride=4,
    )
    assert path is None
    assert not (tmp_path / "sheet.png").exists()
