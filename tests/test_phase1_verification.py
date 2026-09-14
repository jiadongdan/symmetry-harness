"""Independent Phase 1 verification set (QA stage A: core layer + Predict).

These tests are written by the QA engineer to *independently* prove that the
frozen core layer and the Predict workspace honour the Phase 1 protocol. They
deliberately target behaviour the engineers' own suites do not cover, and they
push the boundaries and error paths instead of repeating happy paths.

Covered protocol clauses:
    section 3.1 / 3.6 / 3.7 / 3.9 / 3.10 (predict) / 5 / 10 / 11 / 13.1 / 13.2 /
    13.5 / 13.7 / 14.3

Only public / documented behaviour is asserted; production ``src/`` files are
never imported for mutation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import warnings
import zipfile

import numpy as np
from PIL import Image
import pytest

from symmetry_harness import result_exports as rx
from symmetry_harness import colormaps
from symmetry_harness.analysis import (
    DensePrediction,
    prediction_display_bounds,
    render_scalar_map,
)
from symmetry_harness.model_package import ModelPackageError, inspect_model_package
from symmetry_harness.ui_readiness import evaluate_predict_readiness
from symmetry_harness.ui_predict import (
    _load_package,
    _saved_prediction_defaults,
    new_presentation_state,
    render_selected_item,
    require_symmodel_path,
)
from symmetry_harness.workflow import _artifact_paths, create_run_archive


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "symmetry_harness"

CAPABILITIES = {
    "contract_version": "symmetry-learn-provider-v1",
    "provider": "symmetry-learn",
    "provider_version": "0.1.0",
    "operations": ["few_shot_analyze", "predict_with_fine_tuned_model"],
    "models": [
        {
            "identifier": "cnn_8ch_pg17",
            "display_name": "Eight-channel CNN (PG17)",
            "available": True,
            "input_channels": 8,
            "pretrained_classes": 17,
            "classifier_patch_size": 64,
            "feature_pipeline": "eight_channel_v1",
            "feature_channels": [f"channel_{index}" for index in range(8)],
            "fine_tuning_strategy": "adapters",
            "minimum_shots_per_class": 3,
            "recommended_shots_per_class": 5,
            "maximum_shots_per_class": 50,
            "default_weight": {
                "identifier": "pg17-symmetry-v1",
                "version": "1.0.0",
                "sha256": "a" * 64,
                "distribution": "symmetry-learn-default-model",
                "status": "installed",
                "bundled": True,
                "default": True,
            },
            "weights": [
                {
                    "identifier": "pg17-symmetry-v1",
                    "version": "1.0.0",
                    "sha256": "a" * 64,
                    "distribution": "symmetry-learn-default-model",
                    "status": "installed",
                    "bundled": True,
                    "default": True,
                }
            ],
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


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------
def _prediction(
    *,
    grid: np.ndarray | None = None,
    confidence_grid: np.ndarray | None = None,
    entropy_grid: np.ndarray | None = None,
) -> DensePrediction:
    x_values = np.array([32, 36], dtype=np.int32)
    y_values = np.array([32, 36], dtype=np.int32)
    coordinates = np.array([[32, 32], [36, 32], [32, 36], [36, 36]], dtype=np.int32)
    return DensePrediction(
        coordinates_xy=coordinates,
        x_coordinates=x_values,
        y_coordinates=y_values,
        logits=np.zeros((4, 2), dtype=np.float32),
        probabilities=np.full((4, 2), 0.5, dtype=np.float32),
        predictions=np.array([0, 1, 1, 0], dtype=np.int16),
        confidence=np.full(4, 0.5, dtype=np.float32),
        entropy=np.full(4, np.log(2), dtype=np.float32),
        prediction_grid=(
            np.array([[0, 1], [1, 0]], dtype=np.int16) if grid is None else grid
        ),
        confidence_grid=(
            np.full((2, 2), 0.5, dtype=np.float32)
            if confidence_grid is None
            else confidence_grid
        ),
        entropy_grid=(
            np.full((2, 2), np.log(2), dtype=np.float32)
            if entropy_grid is None
            else entropy_grid
        ),
    )


def _image(size: int = 96, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((size, size), dtype=np.float32)


def _region(prediction: DensePrediction, shape: tuple[int, int], stride: int):
    left, top, right, bottom = prediction_display_bounds(
        prediction, shape, stride=stride
    )
    return (bottom - top, right - left)


def _config():
    from symmetry_harness.config import load_harness_config

    return load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")


def _raw_zip(path: Path, members: dict[str, bytes], *, duplicates: bool = False) -> Path:
    """Build a ZIP by hand so duplicate member names can be crafted."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
        if duplicates:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr("manifest.json", b"{}")
    return path


def _capture_predict_callbacks(config, capabilities):
    """Return (package_change, image_change, click) callbacks of the Predict UI."""
    gradio = pytest.importorskip("gradio")
    from symmetry_harness.ui_predict import build_prediction_workspace

    captured: dict[str, object] = {"file_changes": [], "click": None}
    original_file_change = gradio.File.change
    original_click = gradio.Button.click

    def spy_file_change(self, function, **kwargs):
        captured["file_changes"].append(function)
        return original_file_change(self, function, **kwargs)

    def spy_click(self, function, **kwargs):
        captured["click"] = function
        return original_click(self, function, **kwargs)

    gradio.File.change = spy_file_change
    gradio.Button.click = spy_click
    try:
        with gradio.Blocks():
            build_prediction_workspace(config, capabilities=capabilities)
    finally:
        gradio.File.change = original_file_change
        gradio.Button.click = original_click

    file_changes = captured["file_changes"]
    assert len(file_changes) >= 2, "expected the package and image file inputs"
    return file_changes[0], file_changes[1], captured["click"]


def _completed_prediction(tmp_path: Path, *, stride: int = 4, size: int = 24) -> dict:
    """Write one completed per-item prediction NPZ + input and return the batch."""
    directory = tmp_path / "items" / "image-0001"
    directory.mkdir(parents=True)
    axis = np.arange(0, size, stride, dtype=np.int32)
    grid_shape = (len(axis), len(axis))
    coordinates = np.array([[x, y] for y in axis for x in axis], dtype=np.int32)
    sample_count = len(coordinates)
    np.savez_compressed(
        directory / "prediction.npz",
        coordinates_xy=coordinates,
        x_coordinates=axis,
        y_coordinates=axis,
        logits=np.zeros((sample_count, 2), dtype=np.float32),
        probabilities=np.full((sample_count, 2), 0.5, dtype=np.float32),
        predictions=np.zeros(sample_count, dtype=np.int16),
        confidence=np.full(sample_count, 0.5, dtype=np.float32),
        entropy=np.full(sample_count, 0.3, dtype=np.float32),
        prediction_grid=np.zeros(grid_shape, dtype=np.int16),
        confidence_grid=np.full(grid_shape, 0.5, dtype=np.float32),
        entropy_grid=np.full(grid_shape, 0.3, dtype=np.float32),
    )
    np.save(directory / "input.npy", _image(size).astype(np.float32))
    record = {
        "item_id": "image-0001",
        "status": "completed",
        "run_id": "prediction-verify",
        "output": {
            "class_statistics": {
                "grid_cell_count": int(np.prod(grid_shape)),
                "classes": [
                    {"index": 0, "count": int(np.prod(grid_shape)), "fraction": 1.0},
                    {"index": 1, "count": 0, "fraction": 0.0},
                ],
            }
        },
    }
    (directory / "prediction_record.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    entry = {
        "item_id": "image-0001",
        "status": "completed",
        "source_path": "/a/sample.tif",
        "class_statistics": record["output"]["class_statistics"],
        "artifact_prediction": str(directory / "prediction.npz"),
        "artifact_input": str(directory / "input.npy"),
        "prediction_record": str(directory / "prediction_record.json"),
        "item_directory": str(directory),
    }
    return {
        "status": "completed",
        "run_id": "prediction-verify",
        "stride": stride,
        "class_names": ["Phase A", "Phase B"],
        "class_colors": ["#e41a1c", "#377eb8"],
        "items": [entry],
    }


# ===========================================================================
# 13.1 Package and download
# ===========================================================================
def test_valid_symmodel_still_passes_complete_validation(model_package_factory):
    package = inspect_model_package(model_package_factory("valid.symmodel"))
    assert package.task_classes == 2
    assert package.prediction_defaults == {"stride": 4, "batch_size": 512}


def test_arbitrary_renamed_zip_is_rejected_by_content_not_suffix(tmp_path):
    renamed = _raw_zip(tmp_path / "renamed.symmodel", {"notes.json": b"{}"})
    package, compatibility, message = _load_package(str(renamed))
    assert package is None
    assert compatibility == "Blocked"
    assert "unexpected member" in message
    # The suffix alone is accepted by the user-facing policy ...
    assert require_symmodel_path(str(renamed)).endswith("renamed.symmodel")
    # ... but content validation is what actually blocks it.


def test_full_run_zip_is_never_a_model_package(tmp_path):
    run_zip = _raw_zip(
        tmp_path / "run.symmodel",
        {
            "input.npy": b"npy",
            "run_record.json": b"{}",
            "report.md": b"# report",
            "manifest.json": b"{}",
        },
    )
    with pytest.raises(ModelPackageError):
        inspect_model_package(run_zip)


def test_predict_user_flow_rejects_non_symmodel_path(tmp_path):
    plain = tmp_path / "model.zip"
    plain.write_bytes(b"x")
    with pytest.raises(Exception) as info:
        require_symmodel_path(str(plain))
    assert ".symmodel" in str(info.value)

    good = tmp_path / "ok.symmodel"
    good.write_bytes(b"x")
    assert require_symmodel_path(str(good)).endswith("ok.symmodel")


@pytest.mark.parametrize(
    "member",
    ["../escape.json", "/etc/absolute.json", "nested/../../escape.json"],
)
def test_model_package_rejects_unsafe_member_paths(model_package_factory, member):
    path = model_package_factory(f"unsafe-{abs(hash(member))}.symmodel")
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr(member, b"{}")
    with pytest.raises(ModelPackageError, match="unsafe member"):
        inspect_model_package(path)


def test_model_package_rejects_duplicate_members(tmp_path):
    path = tmp_path / "dup.symmodel"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", b"{}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            archive.writestr("manifest.json", b"{}")
        archive.writestr("model_state.pt", b"state")
        archive.writestr("training_summary.json", b"{}")
    with pytest.raises(ModelPackageError, match="duplicate members"):
        inspect_model_package(path)


def test_model_package_rejects_missing_manifest(model_package_factory):
    path = model_package_factory(
        "nomanifest.symmodel", omit=("manifest.json",)
    )
    with pytest.raises(ModelPackageError, match="missing"):
        inspect_model_package(path)


def test_model_package_rejects_model_state_checksum_mismatch(model_package_factory):
    path = model_package_factory(
        "tamper.symmodel", model_state=b"tampered", break_checksum=True
    )
    with pytest.raises(ModelPackageError, match="checksum"):
        inspect_model_package(path)


def test_download_copy_matches_bytes_and_never_mutates_the_durable_artifact(
    model_package_factory,
):
    package = model_package_factory("download.symmodel")
    before = package.read_bytes()
    digest_before = hashlib.sha256(before).hexdigest()

    copy = rx.prepare_download_copy(
        package, run_id="symmetry-download", filename="fine_tuned_model.symmodel"
    )

    assert copy.is_file()
    assert copy.read_bytes() == before
    assert hashlib.sha256(copy.read_bytes()).hexdigest() == digest_before
    # The durable artifact is byte-identical and lives elsewhere (temp copy).
    assert package.read_bytes() == before
    assert copy.resolve() != package.resolve()


def test_run_archive_contains_every_ordinary_artifact_and_no_self(tmp_path):
    run_dir = tmp_path / "runs" / "symmetry-demo"
    run_dir.mkdir(parents=True)
    artifact_names = {Path(path).name for path in _artifact_paths(run_dir).values()}
    assert len(artifact_names) == 15
    for name in artifact_names:
        (run_dir / name).write_bytes(f"payload:{name}".encode("utf-8"))
    # A nested file must keep a safe relative member name.
    nested = run_dir / "nested"
    nested.mkdir()
    (nested / "extra.bin").write_bytes(b"nested")
    # An unrelated external file must never be swept in.
    (tmp_path / "external_source.npy").write_bytes(b"external")

    # Force the destination *inside* the source tree to prove self-exclusion.
    archive = create_run_archive(run_dir, output_root=run_dir)
    with zipfile.ZipFile(archive) as handle:
        members = handle.namelist()
    assert "nested/extra.bin" in members
    assert archive.name not in members  # never contains itself
    assert "external_source.npy" not in members
    for name in members:
        assert not name.startswith(("/", "\\")) and ".." not in Path(name).parts
    assert artifact_names <= set(members)


# ===========================================================================
# 13.2 Prediction defaults (the correctness fix)
# ===========================================================================
def _s8_package(model_package_factory, name="s8.symmodel", stride=8, batch=128):
    return model_package_factory(
        name,
        manifest_overrides={"prediction_defaults": {"stride": stride, "batch_size": batch}},
    )


def test_package_load_hydrates_visible_controls_to_saved_defaults(
    model_package_factory,
):
    package, _, _ = _load_package(str(_s8_package(model_package_factory)))
    assert _saved_prediction_defaults(package) == {"stride": 8, "batch_size": 128}


def test_run_without_user_edits_uses_hydrated_saved_defaults(
    model_package_factory, monkeypatch, tmp_path
):
    import symmetry_harness.ui_predict as ui_predict

    config = _config()
    package_change, _, click = _capture_predict_callbacks(config, CAPABILITIES)

    path = _s8_package(model_package_factory)
    outputs = package_change(str(path))
    visible_stride = outputs[6]["value"]
    visible_batch = outputs[7]["value"]
    assert (visible_stride, visible_batch) == (8, 128)

    image = tmp_path / "image.npy"
    np.save(image, np.zeros((96, 96), dtype=np.float32))

    seen: dict[str, object] = {}

    def fake_batch(config_, *, model_package, image_paths, device=None,
                   stride=None, batch_size=None, progress_callback=None, **kwargs):
        seen["stride"] = stride
        seen["batch_size"] = batch_size
        return {
            "status": "failed",
            "run_id": "prediction-hydrate",
            "counts": {"requested": 1, "completed": 0, "failed": 1},
            "items": [
                {
                    "item_id": "image-0001",
                    "status": "failed",
                    "source_path": str(image),
                    "error": "synthetic",
                }
            ],
            "class_names": ["Phase A", "Phase B"],
            "class_colors": ["#e41a1c", "#377eb8"],
            "download_archive": str(tmp_path / "results.zip"),
        }

    monkeypatch.setattr(ui_predict, "run_saved_model_prediction_batch", fake_batch)

    # Feed the *hydrated* visible values back in, exactly as Gradio would.
    list(
        click(
            [str(image)],
            outputs[0],
            outputs[1],
            "auto",
            visible_stride,
            visible_batch,
        )
    )
    assert seen["stride"] == 8
    assert seen["batch_size"] == 128


def test_user_edited_values_are_passed_as_overrides(
    model_package_factory, monkeypatch, tmp_path
):
    import symmetry_harness.ui_predict as ui_predict

    config = _config()
    _, _, click = _capture_predict_callbacks(config, CAPABILITIES)
    package, _, _ = _load_package(str(_s8_package(model_package_factory)))

    image = tmp_path / "image.npy"
    np.save(image, np.zeros((96, 96), dtype=np.float32))
    seen: dict[str, object] = {}

    def fake_batch(config_, *, model_package, image_paths, device=None,
                   stride=None, batch_size=None, progress_callback=None, **kwargs):
        seen["stride"] = stride
        seen["batch_size"] = batch_size
        return {
            "status": "failed",
            "run_id": "prediction-override",
            "counts": {"requested": 1, "completed": 0, "failed": 1},
            "items": [{"item_id": "image-0001", "status": "failed",
                       "source_path": str(image), "error": "synthetic"}],
            "class_names": ["Phase A", "Phase B"],
            "class_colors": ["#e41a1c", "#377eb8"],
            "download_archive": str(tmp_path / "results.zip"),
        }

    monkeypatch.setattr(ui_predict, "run_saved_model_prediction_batch", fake_batch)
    list(click([str(image)], package, True, "auto", 16, 64))
    assert seen["stride"] == 16
    assert seen["batch_size"] == 64


def test_loading_another_package_resets_the_previous_defaults(model_package_factory):
    config = _config()
    package_change, _, _ = _capture_predict_callbacks(config, CAPABILITIES)

    first = package_change(str(_s8_package(model_package_factory, "s8.symmodel", 8, 128)))
    assert (first[6]["value"], first[7]["value"]) == (8, 128)

    second = package_change(
        str(_s8_package(model_package_factory, "s3.symmodel", 3, 64))
    )
    assert (second[6]["value"], second[7]["value"]) == (3, 64)


def test_invalid_package_selection_disables_and_clears_stale_state(
    model_package_factory,
):
    config = _config()
    package_change, _, _ = _capture_predict_callbacks(config, CAPABILITIES)

    package_change(str(_s8_package(model_package_factory, "s8.symmodel", 8, 128)))

    corrupt = model_package_factory(
        "corrupt.symmodel", model_state=b"tampered", break_checksum=True
    )
    outputs = package_change(str(corrupt))

    assert outputs[0] is None  # package_state cleared
    assert outputs[8]["interactive"] is False  # predict disabled
    assert outputs[11] is None  # result_state cleared
    assert outputs[14] is None  # presentation_state cleared
    assert outputs[15]["value"] == ""  # presentation colors cleared
    assert outputs[6]["value"] == 4 and outputs[7]["value"] == 512  # app fallback


def test_no_hardcoded_default_is_passed_as_override_in_on_predict():
    """Reverse check: ``on_predict`` must not smuggle 4/512 as an override."""
    source = (SOURCE_ROOT / "ui_predict.py").read_text(encoding="utf-8")
    start = source.index("def on_predict(")
    end = source.index("predict_button.click(")
    body = source[start:end]
    assert "DEFAULT_STRIDE" not in body
    assert "DEFAULT_BATCH_SIZE" not in body
    assert "stride=resolved_stride" in body
    assert "batch_size=resolved_batch_size" in body
    assert "require_whole_number(" in body
    # No numeric literal is used where the runtime overrides are forwarded.
    assert "stride=4" not in body and "batch_size=512" not in body


# ===========================================================================
# 13.5 Visualization boundaries
# ===========================================================================
@pytest.mark.parametrize(
    "value, ok",
    [
        ("#A1B2C3", True),
        ("#abcdef", True),
        ("#AbCdEf", True),
        ("   #123456   ", True),
        ("#000000", True),
        ("#ffffff", True),
        ("#12345", False),
        ("#GGGGGG", False),
        ("#gggggg", False),
        ("123456", False),
        ("#1234567", False),
        ("", False),
        ("   ", False),
        ("rgb(1,2,3)", False),
        ("#123456 ", True),
        ("#123 456", False),
        ("#", False),
    ],
)
def test_hex_color_boundary_matrix(value, ok):
    if ok:
        normalized = rx.validate_hex_color(value)
        assert normalized == value.strip().lower()
    else:
        with pytest.raises(rx.PresentationError):
            rx.validate_hex_color(value)


def test_duplicate_class_colors_are_allowed_without_warning():
    prediction = _prediction()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        variants = rx.render_result_variants(
            _image(),
            prediction,
            class_names=["A", "B"],
            colors=["#ff0000", "#ff0000"],  # identical colours: explicitly allowed
            alpha=0.48,
            stride=4,
        )
    assert variants["mask"].shape[-1] == 3
    assert rx.normalize_palette(["#123456", "#123456"], class_count=2) == (
        "#123456",
        "#123456",
    )


def test_alpha_boundaries_and_non_numeric_are_enforced():
    assert rx.validate_alpha(0.0) == 0.0
    assert rx.validate_alpha(1.0) == 1.0
    for bad in (-1e-9, 1.0 + 1e-9, "many", None, float("nan"), float("inf")):
        with pytest.raises(rx.PresentationError):
            rx.validate_alpha(bad)


def test_color_and_alpha_changes_never_invoke_prediction(
    tmp_path, monkeypatch
):
    import symmetry_harness.ui_predict as ui_predict

    result = _completed_prediction(tmp_path)
    array_before = np.load(result["items"][0]["artifact_prediction"])[
        "prediction_grid"
    ].copy()

    def _boom(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("presentation rerender must not run prediction")

    monkeypatch.setattr(ui_predict, "run_saved_model_prediction_batch", _boom)

    base = render_selected_item(
        "image-0001", result, colors_text="#e41a1c\n#377eb8", alpha=0.2,
        class_names=["Phase A", "Phase B"],
    )
    recolored = render_selected_item(
        "image-0001", result, colors_text="#111111\n#eeeeee", alpha=0.2,
        class_names=["Phase A", "Phase B"],
    )
    higher_alpha = render_selected_item(
        "image-0001", result, colors_text="#e41a1c\n#377eb8", alpha=0.9,
        class_names=["Phase A", "Phase B"],
    )

    # Compare the in-memory rendered arrays: repeated renders with the same
    # run_id/item_id intentionally reuse the same temporary download filename,
    # so reading the PNGs back would only ever show the most recent render.
    base_overlay = np.asarray(base["overlay"])
    recolored_overlay = np.asarray(recolored["overlay"])
    alpha_overlay = np.asarray(higher_alpha["overlay"])
    # Colors and alpha change overlay pixels ...
    assert not np.array_equal(base_overlay, recolored_overlay)
    assert not np.array_equal(base_overlay, alpha_overlay)
    # ... while the pure (title-free) overlay download decodes as RGB.
    with Image.open(higher_alpha["downloads"]["overlay"]) as overlay_png:
        assert overlay_png.mode == "RGB"
    # ... while the persisted numerical array is untouched.
    array_after = np.load(result["items"][0]["artifact_prediction"])[
        "prediction_grid"
    ]
    assert np.array_equal(array_before, array_after)


def test_presentation_change_does_not_touch_the_symmodel_bytes(
    tmp_path, model_package_factory
):
    """Section 14.3 persistence audit on the portable model package."""
    package = model_package_factory("audit.symmodel")
    digest_before = hashlib.sha256(package.read_bytes()).hexdigest()

    result = _completed_prediction(tmp_path)
    render_selected_item(
        "image-0001", result, colors_text="#123456\n#abcdef", alpha=0.03,
        class_names=["Phase A", "Phase B"],
    )

    after = package.read_bytes()
    assert hashlib.sha256(after).hexdigest() == digest_before
    assert b"#123456" not in after and b"#abcdef" not in after


def test_categorical_mask_is_nearest_neighbour_independent_of_source_and_alpha():
    prediction = _prediction()
    colors = ["#e41a1c", "#377eb8"]
    mask_low = rx.render_result_variants(
        _image(seed=1), prediction, class_names=["A", "B"], colors=colors,
        alpha=0.0, stride=4,
    )["mask"]
    mask_high = rx.render_result_variants(
        _image(seed=9), prediction, class_names=["A", "B"], colors=colors,
        alpha=1.0, stride=4,
    )["mask"]
    # Source image and alpha have no effect on the categorical mask.
    assert np.array_equal(mask_low, mask_high)
    unique = {tuple(int(c) for c in pixel) for pixel in mask_low.reshape(-1, 3)}
    allowed = {(228, 26, 28), (55, 126, 184)}
    assert unique <= allowed and len(unique) <= 2


def test_confidence_uses_fixed_bounds_even_when_array_is_constant_high():
    # A confidence array that is uniformly 0.9 must map through the fixed 0..1
    # range, i.e. to a single mid-to-high viridis index, never stretched to 255.
    prediction = _prediction(
        confidence_grid=np.full((2, 2), 0.9, dtype=np.float32)
    )
    confidence = rx.render_result_variants(
        _image(), prediction, class_names=["A", "B"],
        colors=["#e41a1c", "#377eb8"], alpha=0.48, stride=4,
    )["confidence"]
    expected_index = int(
        colormaps.lut_index(np.array([0.9], dtype=np.float32))[0]
    )
    expected = colormaps.viridis_lut()[expected_index]
    unique = np.unique(confidence.reshape(-1, 3), axis=0)
    assert np.array_equal(unique, expected.reshape(1, 3))
    assert expected_index not in (0, 255)


def test_entropy_uses_fixed_ln_n_bound_regardless_of_extrema():
    upper = float(np.log(3))
    prediction = _prediction(
        entropy_grid=np.full((2, 2), upper, dtype=np.float32)
    )
    entropy = rx.render_result_variants(
        _image(), prediction, class_names=["A", "B", "C"],
        colors=["#e41a1c", "#377eb8", "#4daf4a"], alpha=0.48, stride=4,
        class_count=3,
    )["entropy"]
    unique = np.unique(entropy.reshape(-1, 3), axis=0)
    # entropy == ln(N) with upper bound ln(N) -> magma top entry.
    assert np.array_equal(unique, colormaps.magma_lut()[255].reshape(1, 3))


def test_pure_exports_have_no_margin_and_decorated_append_only_margin():
    image = _image()
    prediction = _prediction()
    variants = rx.render_result_variants(
        image, prediction, class_names=["A", "B"],
        colors=["#e41a1c", "#377eb8"], alpha=0.48, stride=4,
    )
    region = _region(prediction, image.shape, 4)
    # Pure variants exactly equal the predicted region: no title band, no margin.
    for name in ("overlay", "mask", "confidence", "entropy"):
        assert variants[name].shape == region + (3,), name
    # The legend is stacked below the overlay rather than beside it, so the
    # annotated map keeps the source width and renders at the same size as the
    # confidence and entropy maps.
    assert variants["overlay_legend"].shape[1] >= region[1]
    assert variants["overlay_legend"].shape[0] > region[0]
    assert variants["overlay_legend"].shape[0] == (
        region[0] + rx.LEGEND_PADDING * 2 + rx.LEGEND_ROW_HEIGHT
    )
    assert variants["overlay_legend"].shape[1] >= max(
        region[1], rx.LEGEND_MARGIN_WIDTH
    )
    assert variants["confidence_colorbar"].shape == (
        region[0],
        region[1] + rx.COLORBAR_MARGIN_WIDTH,
        3,
    )
    assert variants["entropy_colorbar"].shape == (
        region[0],
        region[1] + rx.COLORBAR_MARGIN_WIDTH,
        3,
    )
    # The decorated top-left scientific region is copied byte-for-byte.
    assert np.array_equal(
        variants["confidence_colorbar"][:, : region[1]], variants["confidence"]
    )


def test_luts_are_deterministic_with_frozen_anchors():
    assert np.array_equal(colormaps.viridis_lut(), colormaps.viridis_lut())
    assert np.array_equal(colormaps.magma_lut(), colormaps.magma_lut())
    assert colormaps.viridis_lut()[128].tolist() == [33, 145, 140]
    assert colormaps.magma_lut()[128].tolist() == [183, 55, 121]
    for name in (colormaps.LEGACY_RAINBOW_NAME, "LEGACY_RAINBOW"):
        with pytest.raises(ValueError, match="legacy_rainbow"):
            colormaps.lut(name)


def test_every_variant_writes_a_decodable_rgb_png(tmp_path):
    variants = rx.render_result_variants(
        _image(), _prediction(), class_names=["A", "B"],
        colors=["#e41a1c", "#377eb8"], alpha=0.48, stride=4,
    )
    for name, array in variants.items():
        path = rx.write_png(array, tmp_path / f"{name}.png")
        with Image.open(path) as loaded:
            assert loaded.mode == "RGB"
            assert loaded.size == (array.shape[1], array.shape[0])


# ===========================================================================
# 3.6 / 4 Legacy rainbow regression (independently reconstructed)
# ===========================================================================
def test_legacy_rainbow_is_byte_identical_to_the_historical_formula():
    values = np.random.default_rng(11).random((37, 53)).astype(np.float32)
    rendered = render_scalar_map(
        values, (53, 37), value_range=(0.0, 1.0), colormap="legacy_rainbow"
    )
    n = values.astype(np.float32)
    reference = np.rint(
        np.stack((n, 1.0 - np.abs(2.0 * n - 1.0), 1.0 - n), axis=-1) * 255.0
    ).astype(np.uint8)
    assert np.array_equal(rendered, reference)

    # A value chosen so that 8-bit index quantization would visibly disagree:
    # the historical ramp is computed arithmetically, not via a 256-entry table.
    probe = np.array([[0.3]], dtype=np.float32)
    probe_render = render_scalar_map(
        probe, (1, 1), value_range=(0.0, 1.0), colormap="legacy_rainbow"
    )
    assert probe_render[0, 0].tolist() == [
        int(round(0.3 * 255)),
        int(round((1.0 - abs(2.0 * 0.3 - 1.0)) * 255)),
        int(round((1.0 - 0.3) * 255)),
    ]


def test_traditional_workflow_pins_the_legacy_rainbow_contract():
    source = (SOURCE_ROOT / "traditional_workflow.py").read_text(encoding="utf-8")
    assert source.count('colormap="legacy_rainbow"') == 2


# ===========================================================================
# 5 / 13.7 Purity and import boundaries
# ===========================================================================
def test_no_harness_module_imports_matplotlib_torch_or_symmlearn():
    offenders: list[str] = []
    for path in sorted(SOURCE_ROOT.glob("*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            for token in ("matplotlib", "torch", "symmlearn"):
                if token in stripped:
                    offenders.append(f"{path.name}: {stripped!r}")
    assert offenders == [], offenders


def test_pure_modules_respect_their_dependency_floor():
    colormaps_src = (SOURCE_ROOT / "colormaps.py").read_text(encoding="utf-8")
    assert "PIL" not in colormaps_src and "import gradio" not in colormaps_src
    for name in ("result_exports.py", "ui_readiness.py"):
        src = (SOURCE_ROOT / name).read_text(encoding="utf-8")
        assert "import gradio" not in src and "from gradio" not in src


def test_predict_readiness_never_raises_on_a_plain_empty_snapshot():
    items = evaluate_predict_readiness({})
    assert items  # structured items, never an exception
    assert all(item.status in {"ready", "recommendation", "blocker"} for item in items)


def test_predict_module_is_the_only_source_of_saved_defaults():
    """The un-hydrated fallback is only used before a package validates."""
    assert _saved_prediction_defaults(None) == {"stride": 4, "batch_size": 512}
    state = new_presentation_state(["A", "B"], ["#e41a1c", "#377eb8"])
    assert state["alpha"] == rx.DEFAULT_OVERLAY_ALPHA == 0.48


# ===========================================================================
# STAGE B — Fine-tune integration + the two T04 fixes + real-run audit
# ===========================================================================
STAGE_LABELS = (
    "1. Model and image",
    "2. Review symmetry maps",
    "3. Select support points",
    "4. Fine-tune and review",
)


def _ft_capabilities(*, installed: bool = True) -> dict:
    status = "installed" if installed else "missing"
    weight = {
        "identifier": "pg17-symmetry-v1",
        "version": "1.0.0",
        "sha256": "c" * 64,
        "distribution": "symmetry-learn-default-model",
        "status": status,
        "bundled": True,
        "default": True,
    }
    return {
        "contract_version": "symmetry-learn-provider-v1",
        "provider": "symmetry-learn",
        "provider_version": "0.1.0",
        "operations": ["few_shot_analyze", "compute_features"],
        "models": [
            {
                "identifier": "cnn_8ch_pg17",
                "display_name": "Eight-channel CNN (PG17)",
                "available": True,
                "input_channels": 8,
                "pretrained_classes": 17,
                "classifier_patch_size": 64,
                "feature_pipeline": "eight_channel_v1",
                "feature_channels": [f"channel_{index}" for index in range(8)],
                "fine_tuning_strategy": "adapters",
                "minimum_shots_per_class": 3,
                "recommended_shots_per_class": 5,
                "maximum_shots_per_class": 50,
                "defaults": {
                    "n_max": 20,
                    "symmetry_patch_size": 51,
                    "rotation_folds": [2, 3, 4, 6],
                    "reflection_p": 2.0,
                    "normalize_rotation_maps": False,
                    "epochs": 150,
                    "learning_rate": 0.0005,
                    "stride": 4,
                    "batch_size": 512,
                    "device": "auto",
                },
                "default_weight": weight,
                "weights": [weight],
            }
        ],
    }


def _build_full_app():
    gradio = pytest.importorskip("gradio")
    from symmetry_harness.ui import build_app

    return gradio, build_app(_config(), capabilities=_ft_capabilities())


def _build_ft(config, capabilities):
    gradio = pytest.importorskip("gradio")
    import symmetry_harness.ui_fine_tune as ft

    with gradio.Blocks() as app:
        ft.build_fine_tune_workspace(config, capabilities=capabilities)
    return app


def _components(app) -> list[dict]:
    return app.get_config_file()["components"]


def _by_type(app, type_name: str) -> list[dict]:
    return [c for c in _components(app) if c.get("type") == type_name]


def _ft_handler(app, name: str):
    for block_fn in app.fns.values():
        if getattr(block_fn.fn, "__name__", "") == name:
            return block_fn
    raise AssertionError(f"handler {name!r} is not registered")


def _output_index_by_label(block_fn, label: str) -> int:
    for index, component in enumerate(block_fn.outputs):
        if getattr(component, "label", None) == label:
            return index
    raise AssertionError(f"no output labelled {label!r}")


def _fake_fine_tune_run(tmp_path: Path) -> tuple[Path, dict]:
    run_dir = tmp_path / "symmetry-fake-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
    np.save(run_dir / "input.npy", image)
    axis = np.array([16, 24], dtype=np.int32)
    coords = np.array([[16, 16], [24, 16], [16, 24], [24, 24]], dtype=np.int32)
    np.savez(
        run_dir / "prediction.npz",
        coordinates_xy=coords,
        x_coordinates=axis,
        y_coordinates=axis,
        logits=np.zeros((4, 2), dtype=np.float32),
        probabilities=np.full((4, 2), 0.5, dtype=np.float32),
        predictions=np.array([0, 1, 1, 0], dtype=np.int16),
        confidence=np.full(4, 0.5, dtype=np.float32),
        entropy=np.full(4, np.log(2.0), dtype=np.float32),
        prediction_grid=np.array([[0, 1], [1, 0]], dtype=np.int16),
        confidence_grid=np.full((2, 2), 0.5, dtype=np.float32),
        entropy_grid=np.full((2, 2), np.log(2.0), dtype=np.float32),
    )
    model_bytes = b"QA-DURABLE-SYMMODEL-BYTES-0123456789"
    (run_dir / "fine_tuned_model.symmodel").write_bytes(model_bytes)
    with zipfile.ZipFile(run_dir / "symmetry-fake-run-full-run.zip", "w") as archive:
        archive.writestr("input.npy", b"npy")
        archive.writestr("run_record.json", b"{}")
    reference = {
        "run_id": "symmetry-fake-run",
        "run_directory": str(run_dir),
        "prediction": str(run_dir / "prediction.npz"),
        "input_array": str(run_dir / "input.npy"),
        "input_preview": str(run_dir / "input_preview.png"),
        "fine_tuned_model": str(run_dir / "fine_tuned_model.symmodel"),
        "full_run_zip": str(run_dir / "symmetry-fake-run-full-run.zip"),
        "stride": "4",
    }
    return run_dir, reference


# ---------------------------------------------------------------------------
# B1 / B3 / B10 — shell heading, tabs, import purity
# ---------------------------------------------------------------------------
def test_b1_duplicate_fine_tune_heading_is_gone():
    fine_tune = (SOURCE_ROOT / "ui_fine_tune.py").read_text(encoding="utf-8")
    shell = (SOURCE_ROOT / "ui.py").read_text(encoding="utf-8")
    assert "Symmetry Harness" not in fine_tune
    assert shell.count("Symmetry Harness") == 2  # Blocks(title=) + Markdown heading


def test_b3_len_tabs_is_two():
    gradio, app = _build_full_app()
    tabs = [c for c in _components(app) if c.get("type") == "tabitem"]
    assert len(tabs) == 2
    # Both workspaces also contribute independent state objects.
    states = [c for c in _components(app) if c.get("type") == "state"]
    assert len(states) >= 7


def test_b10_fine_tune_and_shell_do_not_import_banned_modules():
    for name in ("ui_fine_tune.py", "ui.py"):
        source = (SOURCE_ROOT / name).read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            for token in ("matplotlib", "torch", "symmlearn"):
                assert token not in stripped, f"{name}: {stripped!r}"


def test_b2_select_handler_inputs_are_the_stable_seven_tuple():
    gradio, app = _build_full_app()
    # The user-facing select registration must keep exactly seven inputs while
    # the injected SelectData event stays a positional parameter.
    source = (SOURCE_ROOT / "ui_fine_tune.py").read_text(encoding="utf-8")
    start = source.index("annotation_image.select(")
    end = source.index(").then(on_refresh", start)
    body = source[start:end]
    assert "feature_state" in body and "symmetry_patch_size" in body and "device" in body
    assert body.count(",") >= 7
    assert "event: SelectData" in source
    # The handler must remain importable/registered (the dedicated test_harness
    # regression exercises the injected-event call path).
    pytest.importorskip("gradio")


# ---------------------------------------------------------------------------
# B4 — stable annotation wrapper return shapes
# ---------------------------------------------------------------------------
def test_b4_annotation_wrapper_return_shapes(tmp_path):
    pytest.importorskip("gradio")
    import symmetry_harness.ui_fine_tune as ft
    from symmetry_harness.image_io import inspect_input

    config = _config()
    image_path = tmp_path / "source.npy"
    np.save(image_path, np.linspace(0.0, 1.0, 96 * 96, dtype=np.float32).reshape(96, 96))
    image, record = inspect_input(str(image_path), config.features.input_normalization)

    loaded = ft._loaded_image_outputs(image, record, config)
    assert isinstance(loaded, tuple) and len(loaded) == 3

    configured = ft._configure_classes(
        "Class A, Class B", loaded[1], config, classifier_patch_size=64
    )
    assert isinstance(configured, tuple) and len(configured) == 5

    added = ft._add_point(configured[0], "Class A", (40, 40), config)
    assert isinstance(added, tuple) and len(added) == 5


# ---------------------------------------------------------------------------
# B5 / B8 — hidden-until-ready and stale-download clearing
# ---------------------------------------------------------------------------
def test_b5_results_hidden_and_downloads_absent_before_completion():
    gradio, app = _build_full_app()
    downloads = {
        c["props"].get("label"): c["props"] for c in _by_type(app, "downloadbutton")
    }
    assert downloads["Download model (.symmodel)"]["visible"] is False
    assert downloads["Download complete run (.zip)"]["visible"] is False
    assert downloads["Download model (.symmodel)"].get("value") in (None, "")
    assert downloads["Download complete run (.zip)"].get("value") in (None, "")
    overlays = [
        c
        for c in _by_type(app, "image")
        if str(c["props"].get("label", "")).startswith("Prediction overlay")
    ]
    assert overlays and overlays[0]["props"].get("value") in (None, "")


def test_b8_upstream_change_clears_every_result_download(tmp_path):
    pytest.importorskip("gradio")
    config = _config()
    capabilities = _ft_capabilities()
    app = _build_ft(config, capabilities)
    source = tmp_path / "source.npy"
    np.save(source, np.linspace(0.0, 1.0, 96 * 96, dtype=np.float32).reshape(96, 96))
    previous_state = {
        "image": np.zeros((96, 96), dtype=np.float32),
        "image_path": str(source),
        "image_sha256": "a" * 64,
        "normalized_image_sha256": "b" * 64,
        "image_shape": [96, 96],
        "class_names": ["Class A", "Class B"],
        "colors": ["#e41a1c", "#377eb8"],
        "points": [[], []],
        "classifier_patch_size": 64,
        "show_valid_region": True,
    }

    for handler_name, args in (
        ("on_image_change", (str(source), previous_state, "cnn_8ch_pg17", capabilities)),
        ("on_model_change", ("cnn_8ch_pg17", capabilities, previous_state, None)),
    ):
        block_fn = _ft_handler(app, handler_name)
        result = block_fn.fn(*args)
        assert len(result) == len(block_fn.outputs)
        for label in ("Download model (.symmodel)", "Download complete run (.zip)"):
            index = _output_index_by_label(block_fn, label)
            update = result[index]
            assert update.get("value") is None, (handler_name, label)
            assert update.get("visible") is False, (handler_name, label)


# ---------------------------------------------------------------------------
# B6 / B7 / B9 — presentation rerender, download bytes, state isolation
# ---------------------------------------------------------------------------
def test_b6_rerender_never_runs_numerics(tmp_path, monkeypatch):
    import symmetry_harness.ui_fine_tune as ft
    import symmetry_harness.workflow as workflow
    import symmetry_harness.provider as provider

    run_dir, reference = _fake_fine_tune_run(tmp_path)
    counts = {"run_analysis": 0, "provider_analysis": 0, "provider_features": 0}

    def _count_run_analysis(*_a, **_k):  # pragma: no cover - must not run
        counts["run_analysis"] += 1
        raise AssertionError("rerender must not run analysis")

    def _count_provider(*_a, **_k):  # pragma: no cover - must not run
        counts["provider_analysis"] += 1
        raise AssertionError("rerender must not reach the Provider")

    monkeypatch.setattr(ft, "run_analysis", _count_run_analysis)
    monkeypatch.setattr(workflow, "run_provider_analysis", _count_provider)
    monkeypatch.setattr(provider, "run_provider_analysis", _count_provider)

    grid_before = ft.load_fine_tune_result(reference)[1].prediction_grid.copy()
    npz_before = (run_dir / "prediction.npz").read_bytes()

    base = ft.render_fine_tune_variants(reference, ["Class A", "Class B"], ["#e41a1c", "#377eb8"], 0.48)
    recolored = ft.render_fine_tune_variants(reference, ["Class A", "Class B"], ["#102030", "#405060"], 0.48)
    alpha = ft.render_fine_tune_variants(reference, ["Class A", "Class B"], ["#e41a1c", "#377eb8"], 0.9)

    assert not np.array_equal(base["overlay"], recolored["overlay"])
    assert not np.array_equal(base["overlay"], alpha["overlay"])
    assert np.array_equal(base["confidence"], recolored["confidence"])
    assert np.array_equal(base["entropy"], alpha["entropy"])
    assert counts == {"run_analysis": 0, "provider_analysis": 0, "provider_features": 0}
    assert (run_dir / "prediction.npz").read_bytes() == npz_before
    assert np.array_equal(ft.load_fine_tune_result(reference)[1].prediction_grid, grid_before)


def test_b7_model_and_run_zip_downloads_are_safe_and_faithful(tmp_path):
    import symmetry_harness.ui_fine_tune as ft

    run_dir, reference = _fake_fine_tune_run(tmp_path)
    durable = (run_dir / "fine_tuned_model.symmodel").read_bytes()

    served_model = Path(ft.prepare_model_download(reference))
    assert served_model.read_bytes() == durable  # byte-for-byte
    assert served_model.resolve() != (run_dir / "fine_tuned_model.symmodel").resolve()

    served_zip = Path(ft.prepare_run_zip_download(reference))
    assert zipfile.is_zipfile(served_zip)
    with zipfile.ZipFile(served_zip) as archive:
        members = archive.namelist()
    for name in members:
        assert not name.startswith(("/", "\\")) and ".." not in Path(name).parts
    # The durable artifacts are untouched by the served copies.
    assert (run_dir / "fine_tuned_model.symmodel").read_bytes() == durable


def test_b9_fine_tune_and_predict_presentation_states_are_independent():
    import symmetry_harness.ui_fine_tune as ft
    import symmetry_harness.ui_predict as ui_predict

    fine_tune_state = ft.PresentationState.defaults(["A", "B"], ["#e41a1c", "#377eb8"])
    predict_state = ui_predict.new_presentation_state(["A", "B"], ["#e41a1c", "#377eb8"])

    # Mutating one must never touch the other.
    fine_tune_state.display_colors = ["#000000", "#ffffff"]
    fine_tune_state.alpha = 0.1
    assert predict_state["display_colors"] == ["#e41a1c", "#377eb8"]
    assert predict_state["alpha"] == 0.48

    predict_state["display_colors"] = ["#111111", "#222222"]
    assert fine_tune_state.display_colors == ["#000000", "#ffffff"]


# ---------------------------------------------------------------------------
# C11 / C12 / C13 / C14 — the two T04 fixes
# ---------------------------------------------------------------------------
def test_c11_static_stage_open_flags_are_one_then_closed():
    gradio, app = _build_full_app()
    stages = {c["props"].get("label"): c["props"] for c in _by_type(app, "accordion")}
    opens = [stages[label]["open"] for label in STAGE_LABELS]
    assert opens == [True, False, False, False]


def test_c12_current_stage_semantics():
    import symmetry_harness.ui_fine_tune as ft

    def snap(**overrides):
        base = {
            "model_supported": True,
            "model_name": "CNN",
            "checkpoint_ready": True,
            "checkpoint_detail": "Installed.",
            "image_valid": True,
            "image_name": "s.npy",
            "features_current": True,
            "class_names": ["Class A", "Class B"],
            "class_counts": [5, 5],
            "minimum_shots": 3,
            "recommended_shots": 5,
            "maximum_shots": 50,
            "symmetry_patch_size": 51,
            "epochs": 150,
            "learning_rate": 0.0005,
            "stride": 4,
            "batch_size": 512,
            "device": "cpu",
            "stale_state": False,
            "has_result": False,
        }
        base.update(overrides)
        return base

    assert ft._current_stage({}) == "model_and_image"
    assert ft._current_stage(snap(image_valid=False)) == "model_and_image"
    assert ft._current_stage(snap(features_current=False)) == "symmetry"
    assert ft._current_stage(snap(class_names=["Class A"])) == "support"
    assert ft._current_stage(snap()) == "run"
    # With an existing result the current stage stays "run".
    assert ft._current_stage(snap(has_result=True)) == "run"


def test_c13_refresh_is_additive_and_never_collapses_a_revealed_stage():
    gradio, app = _build_full_app()
    block_fn = _ft_handler(app, "on_refresh")

    def refresh(announced):
        return block_fn.fn(
            {},
            {},
            "cnn_8ch_pg17",
            None,
            _ft_capabilities(),
            51,
            150,
            0.0005,
            4,
            512,
            "cpu",
            ft_module().PresentationState(),
            announced,
        )

    changed = refresh(None)
    assert len(changed) == len(block_fn.outputs)
    assert [u["visible"] for u in changed[2:6]] == [True, True, True, True]
    assert changed[2]["open"] is True
    # Disclosure is additive: a revealed stage is never collapsed by a refresh,
    # so running a command inside a stage leaves that block open. Stages that are
    # not current keep whatever open state the user left them in.
    assert all("open" not in u for u in changed[3:6])

    reannounced = refresh("run")
    assert reannounced[2]["open"] is True
    assert all("open" not in u for u in reannounced[3:6])

    unchanged = refresh("model_and_image")
    # Same current stage: no "open" key is pushed for any stage, else a manual
    # collapse or expansion would be overwritten.
    assert all("open" not in update for update in unchanged[2:6])


def ft_module():
    import symmetry_harness.ui_fine_tune as ft

    return ft


def test_c14_single_readiness_renderer_and_byte_identical_default():
    import symmetry_harness.ui_fine_tune as ft
    import symmetry_harness.ui_readiness as ui_readiness
    from symmetry_harness.ui_readiness import ReadinessItem

    # Exactly one renderer implementation exists.
    assert not hasattr(ft, "_readiness_items_html")
    assert ft.readiness_panel_html is ui_readiness.readiness_panel_html

    items = [
        ReadinessItem("a", "Hello", "blocker", detail="detail", hint="hint"),
        ReadinessItem("b", "Ready", "ready", detail="ok"),
    ]
    default_html = ui_readiness.readiness_panel_html(items)
    none_html = ui_readiness.readiness_panel_html(items, elem_id=None)

    # elem_id=None drops the id but keeps the sticky class.
    assert none_html.count('id="readiness-panel"') == 0
    assert 'class="readiness-panel"' in none_html
    # Default output is byte-identical to the Stage-A rendering: it is the
    # elem_id=None output with exactly the id attribute inserted once.
    assert default_html == none_html.replace(
        '<div class="readiness-panel">',
        '<div id="readiness-panel" class="readiness-panel">',
        1,
    )
    # Frozen literal for a single blocker item (Stage-A format).
    frozen = (
        '<div id="readiness-panel" class="readiness-panel">'
        '<div class="readiness-title">Readiness</div>'
        '<div class="readiness-item readiness-blocker">'
        '<span class="readiness-icon">\u00d7</span>'
        '<span class="readiness-body">'
        '<span class="readiness-label">Hello</span>'
        '<span class="readiness-detail">detail</span>'
        '<span class="readiness-hint">Next: hint</span>'
        "</span></div></div>"
    )
    assert ui_readiness.readiness_panel_html([items[0]]) == frozen


# ---------------------------------------------------------------------------
# B16 — Saved locally vs browser download copy
# ---------------------------------------------------------------------------
def test_b16_ui_distinguishes_durable_path_from_download_copy(tmp_path):
    import symmetry_harness.ui_fine_tune as ft

    source = (SOURCE_ROOT / "ui_fine_tune.py").read_text(encoding="utf-8")
    assert "Saved locally" in source  # durable path is labelled as such
    assert "Download complete run (.zip)" in source
    assert "Download model (.symmodel)" in source

    run_dir, reference = _fake_fine_tune_run(tmp_path)
    served = Path(ft.prepare_model_download(reference))
    import tempfile

    temp_root = Path(tempfile.gettempdir()).resolve()
    assert str(served.resolve()).startswith(str(temp_root))
    assert run_dir not in served.resolve().parents


# ---------------------------------------------------------------------------
# D15 — REAL run_analysis persistence audit (section 14.3)
# ---------------------------------------------------------------------------
def test_d15_real_run_persistence_audit(tmp_path):
    """Run a real Provider fine-tune, then prove presentation edits are inert."""
    import time

    from symmetry_harness.annotations import create_annotation_session
    from symmetry_harness.image_io import file_sha256
    import symmetry_harness.ui_fine_tune as ft
    from symmetry_harness.workflow import run_analysis

    config = _config()
    try:
        from symmetry_harness.provider import doctor

        if doctor(config)["status"] != "ready":
            pytest.skip("The numerical Provider is not ready in this environment.")
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"Provider doctor unavailable: {error!r}")

    size = 96
    y, x = np.mgrid[:size, :size]
    values = np.sin(x / 7.0) + np.cos(y / 9.0)
    image_path = tmp_path / "audit_source.npy"
    np.save(image_path, ((values - values.min()) / np.ptp(values)).astype(np.float32))

    session = create_annotation_session(
        image_path=str(image_path),
        image_sha256=file_sha256(image_path),
        image_shape=(size, size),
        classifier_patch_size=64,
        class_names=["Class A", "Class B"],
        points_by_class=[
            [(32, 32), (40, 40), (48, 48)],
            [(52, 52), (58, 58), (62, 62)],
        ],
    )

    started = time.perf_counter()
    result = run_analysis(
        config,
        image_path=image_path,
        annotation_session=session,
        overrides={"epochs": 2, "stride": 8, "batch_size": 64},
        output_root=tmp_path / "runs",
    )
    elapsed = time.perf_counter() - started
    assert result["status"] == "completed"

    run_dir = Path(result["run_directory"])
    zip_path = Path(result["full_run_zip"])
    durable_paths = {
        "symmodel": run_dir / "fine_tuned_model.symmodel",
        "run_record": run_dir / "run_record.json",
        "report": run_dir / "report.md",
        "provider_record": run_dir / "provider_record.json",
        "zip": zip_path,
        "prediction": run_dir / "prediction.npz",
        "input": run_dir / "input.npy",
    }
    for name, path in durable_paths.items():
        assert path.is_file(), f"{name} missing: {path}"

    digests_before = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in durable_paths.items()
    }
    with zipfile.ZipFile(zip_path) as archive:
        members_before = sorted(archive.namelist())
    files_before = sorted(p.name for p in run_dir.rglob("*") if p.is_file())

    # --- presentation edit + rerender (must be inert) --------------------
    reference = {
        "run_id": str(result["run_id"]),
        "run_directory": str(run_dir),
        "prediction": str(result["prediction"]),
        "input_array": str(result["input_array"]),
        "input_preview": str(result["input_preview"]),
        "fine_tuned_model": str(result["fine_tuned_model"]),
        "full_run_zip": str(zip_path),
        "stride": str(result["stride"]),
    }
    names = [str(entry["name"]) for entry in result["classes"]]
    original_colors = [str(entry["color"]) for entry in result["classes"]]
    presentation = ft.PresentationState.defaults(names, original_colors)
    assert presentation.alpha == ft.DEFAULT_OVERLAY_ALPHA

    base_variants = ft.render_fine_tune_variants(
        reference, names, presentation.display_colors, presentation.alpha
    )
    changed_variants = ft.render_fine_tune_variants(
        reference, names, ["#123456", "#abcdef"], 0.9
    )
    assert not np.array_equal(base_variants["overlay"], changed_variants["overlay"])
    assert not np.array_equal(base_variants["mask"], changed_variants["mask"])
    assert np.array_equal(base_variants["confidence"], changed_variants["confidence"])

    exported = ft.prepare_presentation_pngs(
        reference, names, ["#123456", "#abcdef"], 0.9, list(ft.RESULT_EXPORT_NAMES)
    )
    assert len(exported) == len(ft.RESULT_EXPORT_NAMES)
    for path in exported:
        with Image.open(path) as image:
            image.load()

    # --- every durable artifact byte-identical ---------------------------
    digests_after = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in durable_paths.items()
    }
    assert digests_before == digests_after
    with zipfile.ZipFile(zip_path) as archive:
        members_after = sorted(archive.namelist())
    assert members_before == members_after
    assert files_before == sorted(p.name for p in run_dir.rglob("*") if p.is_file())

    # The transient colors never reached any durable text artifact.
    for name in ("run_record", "report", "provider_record"):
        payload = durable_paths[name].read_bytes()
        assert b"#123456" not in payload and b"#abcdef" not in payload

    # Make the observed timing visible in the report.
    print(f"[QA] real run_analysis persistence audit elapsed: {elapsed:.1f}s")
    print(f"[QA] symmodel sha256 before: {digests_before['symmodel']}")
    print(f"[QA] symmodel sha256 after : {digests_after['symmodel']}")
    print(f"[QA] run_record sha256 before/after: {digests_before['run_record']}")
    print(f"[QA] zip members: {members_before}")

    # Sanity: the real archive really contains the ordinary run artifacts.
    assert "input.npy" in members_before
    assert "fine_tuned_model.symmodel" in members_before


# ---------------------------------------------------------------------------
# Section 15 red lines
# ---------------------------------------------------------------------------
def test_redline_scalar_maps_use_fixed_bounds_not_extrema():
    prediction = _prediction(
        confidence_grid=np.full((2, 2), 0.75, dtype=np.float32),
        entropy_grid=np.full((2, 2), 0.25, dtype=np.float32),
    )
    variants = rx.render_result_variants(
        _image(), prediction, class_names=["A", "B"],
        colors=["#e41a1c", "#377eb8"], alpha=0.48, stride=4,
    )
    conf_index = int(colormaps.lut_index(np.array([0.75], np.float32))[0])
    ent_index = int(
        colormaps.lut_index(np.array([0.25 / np.log(2)], np.float32))[0]
    )
    assert np.array_equal(
        np.unique(variants["confidence"].reshape(-1, 3), axis=0),
        colormaps.viridis_lut()[conf_index].reshape(1, 3),
    )
    assert np.array_equal(
        np.unique(variants["entropy"].reshape(-1, 3), axis=0),
        colormaps.magma_lut()[ent_index].reshape(1, 3),
    )
    assert conf_index not in (0, 255)


def test_redline_symmodel_validation_is_not_weakened(model_package_factory):
    # A .symmodel must still be rejected when an ordinary run archive is passed.
    run_like = model_package_factory(
        "archive.symmodel", omit=("model_state.pt", "training_summary.json")
    )
    with pytest.raises(ModelPackageError):
        inspect_model_package(run_like)
