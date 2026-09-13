"""Focused tests for the pure presentation/export layer (protocol section 13.5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from symmetry_harness import result_exports as rx
from symmetry_harness.analysis import (
    DensePrediction,
    prediction_display_bounds,
    render_scalar_map,
)
from symmetry_harness.colormaps import (
    lut,
    lut_index,
    magma_lut,
    map_scalar_to_rgb,
    viridis_lut,
)


def _prediction(
    *,
    grid: np.ndarray | None = None,
    confidence_grid: np.ndarray | None = None,
    entropy_grid: np.ndarray | None = None,
) -> DensePrediction:
    x_values = np.array([32, 36], dtype=np.int32)
    y_values = np.array([32, 36], dtype=np.int32)
    coordinates = np.array([[32, 32], [36, 32], [32, 36], [36, 36]], dtype=np.int32)
    if grid is None:
        grid = np.array([[0, 1], [1, 0]], dtype=np.int16)
    if confidence_grid is None:
        confidence_grid = np.full((2, 2), 0.5, dtype=np.float32)
    if entropy_grid is None:
        entropy_grid = np.full((2, 2), np.log(2), dtype=np.float32)
    return DensePrediction(
        coordinates_xy=coordinates,
        x_coordinates=x_values,
        y_coordinates=y_values,
        logits=np.zeros((4, 2), dtype=np.float32),
        probabilities=np.full((4, 2), 0.5, dtype=np.float32),
        predictions=np.array([0, 1, 1, 0], dtype=np.int16),
        confidence=np.full(4, 0.5, dtype=np.float32),
        entropy=np.full(4, np.log(2), dtype=np.float32),
        prediction_grid=grid,
        confidence_grid=confidence_grid,
        entropy_grid=entropy_grid,
    )


def _image() -> np.ndarray:
    return np.linspace(0.0, 1.0, 96 * 96, dtype=np.float32).reshape(96, 96)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_validate_hex_color_accepts_only_strict_rrggbb() -> None:
    assert rx.validate_hex_color("#A1B2C3") == "#a1b2c3"
    assert rx.validate_hex_color("  #abcdef  ") == "#abcdef"
    for bad in ("abc", "#abc", "red", "#12345", "#1234567", "#gggggg", "123456", ""):
        with pytest.raises(rx.PresentationError):
            rx.validate_hex_color(bad)


def test_normalize_palette_requires_class_alignment_and_allows_duplicates() -> None:
    assert rx.normalize_palette(["#FF0000", "#00FF00"], class_count=2) == (
        "#ff0000",
        "#00ff00",
    )
    # Duplicate / identical colors are explicitly allowed.
    assert rx.normalize_palette(["#ff0000", "#ff0000"], class_count=2) == (
        "#ff0000",
        "#ff0000",
    )
    with pytest.raises(rx.PresentationError):
        rx.normalize_palette(["#ff0000"], class_count=2)


def test_validate_alpha_enforces_bounds() -> None:
    assert rx.validate_alpha(0.0) == 0.0
    assert rx.validate_alpha(1.0) == 1.0
    assert rx.validate_alpha(0.48) == 0.48
    for bad in (-0.0001, 1.0001, "many", None):
        with pytest.raises(rx.PresentationError):
            rx.validate_alpha(bad)


# ---------------------------------------------------------------------------
# LUT determinism
# ---------------------------------------------------------------------------
def test_luts_are_deterministic_with_frozen_anchors() -> None:
    viridis = viridis_lut()
    magma = magma_lut()
    assert viridis.shape == (256, 3) and viridis.dtype == np.uint8
    assert magma.shape == (256, 3) and magma.dtype == np.uint8
    assert np.array_equal(viridis, viridis_lut())
    assert np.array_equal(magma, magma_lut())
    assert viridis[0].tolist() == [68, 1, 84]
    assert viridis[128].tolist() == [33, 145, 140]
    assert viridis[255].tolist() == [253, 231, 37]
    assert magma[0].tolist() == [0, 0, 4]
    assert magma[128].tolist() == [183, 55, 121]
    assert magma[255].tolist() == [252, 253, 191]


def test_lut_index_contract_and_named_lookup() -> None:
    assert lut_index(np.array([0.0, 0.5, 1.0]))[1] == 128
    assert lut_index(np.array([0.0, 0.5, 1.0])).tolist() == [0, 128, 255]
    assert np.array_equal(lut("viridis"), viridis_lut())
    assert np.array_equal(lut("MAGMA"), magma_lut())
    with pytest.raises(ValueError):
        lut("not_a_colormap")


def test_lut_rejects_legacy_rainbow_loudly() -> None:
    # ``legacy_rainbow`` has no table: an 8-bit index quantization would shift
    # pixels away from the historical output. ``lut`` must fail loudly so nobody
    # silently routes the traditional page through a subtly wrong table.
    with pytest.raises(ValueError, match="legacy_rainbow"):
        lut("legacy_rainbow")
    with pytest.raises(ValueError, match="legacy_rainbow"):
        lut("LEGACY_RAINBOW")


def test_map_scalar_to_rgb_uses_declared_bounds() -> None:
    mapped = map_scalar_to_rgb(
        np.array([0.0, 0.5, 1.0], dtype=np.float32),
        value_range=(0.0, 1.0),
        colormap="viridis",
    )
    assert np.array_equal(mapped[0], viridis_lut()[0])
    assert np.array_equal(mapped[1], viridis_lut()[128])
    assert np.array_equal(mapped[2], viridis_lut()[255])


def test_legacy_rainbow_render_is_byte_identical_to_history() -> None:
    values = np.random.default_rng(7).random((5, 7)).astype(np.float32)
    rendered = render_scalar_map(
        values, (7, 5), value_range=(0.0, 1.0), colormap="legacy_rainbow"
    )
    legacy = np.stack(
        (values, 1.0 - np.abs(2.0 * values - 1.0), 1.0 - values), axis=2
    )
    expected = np.rint(legacy * 255.0).astype(np.uint8)
    assert np.array_equal(rendered, expected)


def test_render_scalar_map_requires_named_colormap() -> None:
    with pytest.raises(TypeError):
        render_scalar_map(np.zeros((2, 2), np.float32), (2, 2), value_range=(0, 1))


def test_missing_matplotlib_optional_full_table_comparison() -> None:
    """Optionally compare against matplotlib when available; skip otherwise."""
    matplotlib = pytest.importorskip("matplotlib")
    from matplotlib import colormaps as mpl_colormaps

    for name, table in (("viridis", viridis_lut()), ("magma", magma_lut())):
        expected = np.rint(
            mpl_colormaps[name](np.linspace(0.0, 1.0, 256))[:, :3] * 255
        ).astype(np.uint8)
        assert np.array_equal(table, expected)


# ---------------------------------------------------------------------------
# Overlay / mask / variant structure
# ---------------------------------------------------------------------------
def test_overlay_and_mask_share_the_prediction_region() -> None:
    prediction = _prediction()
    image = _image()
    variants = rx.render_result_variants(
        image,
        prediction,
        class_names=["A", "B"],
        colors=["#e41a1c", "#377eb8"],
        alpha=0.48,
        stride=4,
    )
    left, top, right, bottom = prediction_display_bounds(prediction, image.shape, stride=4)
    region = (bottom - top, right - left)
    for name in ("overlay", "mask", "confidence", "entropy"):
        assert variants[name].shape == region + (3,)
        assert variants[name].dtype == np.uint8


def test_alpha_change_affects_overlay_only_and_colors_change_mask_too() -> None:
    prediction = _prediction()
    image = _image()
    before_grid = prediction.prediction_grid.copy()
    before_conf = prediction.confidence_grid.copy()

    low = rx.render_result_variants(
        image,
        prediction,
        class_names=["A", "B"],
        colors=["#e41a1c", "#377eb8"],
        alpha=0.2,
        stride=4,
    )
    high = rx.render_result_variants(
        image,
        prediction,
        class_names=["A", "B"],
        colors=["#e41a1c", "#377eb8"],
        alpha=0.9,
        stride=4,
    )

    # Alpha affects the overlay but never the mask / confidence / entropy.
    assert not np.array_equal(low["overlay"], high["overlay"])
    assert np.array_equal(low["mask"], high["mask"])
    assert np.array_equal(low["confidence"], high["confidence"])
    assert np.array_equal(low["entropy"], high["entropy"])

    recolored = rx.render_result_variants(
        image,
        prediction,
        class_names=["A", "B"],
        colors=["#111111", "#eeeeee"],
        alpha=0.2,
        stride=4,
    )
    # Class colors affect the overlay and the categorical mask ...
    assert not np.array_equal(low["overlay"], recolored["overlay"])
    assert not np.array_equal(low["mask"], recolored["mask"])
    # ... but never the scalar confidence / entropy maps.
    assert np.array_equal(low["confidence"], recolored["confidence"])
    assert np.array_equal(low["entropy"], recolored["entropy"])

    # The numerical arrays are untouched.
    assert np.array_equal(prediction.prediction_grid, before_grid)
    assert np.array_equal(prediction.confidence_grid, before_conf)


def test_mask_uses_nearest_neighbour_colors_without_alpha() -> None:
    prediction = _prediction()
    colors = ["#e41a1c", "#377eb8"]
    mask = rx.render_categorical_mask(
        prediction, (96, 96), colors=colors, stride=4
    )
    unique = {tuple(int(c) for c in pixel) for pixel in mask.reshape(-1, 3)}
    allowed = {(228, 26, 28), (55, 126, 184)}
    assert unique <= allowed
    # Nearest-neighbour only: no blended intermediate colors appear.
    assert len(unique) <= 2


def test_confidence_uses_fixed_zero_to_one_bounds_regardless_of_extrema() -> None:
    prediction = _prediction(
        confidence_grid=np.array([[5.0, -3.0], [0.0, 1.0]], dtype=np.float32)
    )
    variants = rx.render_result_variants(
        _image(),
        prediction,
        class_names=["A", "B"],
        colors=["#e41a1c", "#377eb8"],
        alpha=0.48,
        stride=4,
    )
    confidence = variants["confidence"]
    # A 2x2 grid resized to 8x8 with NEAREST: columns 0-3 are grid column 0.
    assert confidence[0, 0].tolist() == viridis_lut()[255].tolist()
    assert confidence[0, 4].tolist() == viridis_lut()[0].tolist()


def test_entropy_uses_fixed_zero_to_ln_n_bounds_regardless_of_extrema() -> None:
    upper = float(np.log(2))
    prediction = _prediction(
        entropy_grid=np.array([[99.0, -5.0], [0.0, upper]], dtype=np.float32)
    )
    variants = rx.render_result_variants(
        _image(),
        prediction,
        class_names=["A", "B"],
        colors=["#e41a1c", "#377eb8"],
        alpha=0.48,
        stride=4,
    )
    entropy = variants["entropy"]
    assert entropy[0, 0].tolist() == magma_lut()[255].tolist()
    assert entropy[0, 4].tolist() == magma_lut()[0].tolist()


def test_pure_variants_have_no_margin_and_decorated_variants_append_it() -> None:
    variants = rx.render_result_variants(
        _image(),
        _prediction(),
        class_names=["A", "B"],
        colors=["#e41a1c", "#377eb8"],
        alpha=0.48,
        stride=4,
    )
    overlay = variants["overlay"]
    # The legend is stacked *below* the overlay: a right-hand margin made the
    # annotated map wider than the confidence and entropy maps, so it rendered
    # smaller than them when the three were shown side by side.
    assert variants["overlay_legend"].shape[1] >= overlay.shape[1]
    assert variants["overlay_legend"].shape[0] > overlay.shape[0]
    confidence = variants["confidence"]
    assert variants["confidence_colorbar"].shape[1] == confidence.shape[1] + rx.COLORBAR_MARGIN_WIDTH
    entropy = variants["entropy"]
    assert variants["entropy_colorbar"].shape[1] == entropy.shape[1] + rx.COLORBAR_MARGIN_WIDTH
    # The scientific subimage is copied byte-for-byte, never rescaled.
    assert np.array_equal(
        variants["overlay_legend"][: overlay.shape[0], : overlay.shape[1]],
        overlay,
    )
    assert np.array_equal(
        variants["confidence_colorbar"][:, : confidence.shape[1]], confidence
    )
    assert np.array_equal(variants["entropy_colorbar"][:, : entropy.shape[1]], entropy)


def test_legend_sits_below_without_rescaling_or_clipping_labels() -> None:
    variants = rx.render_result_variants(
        _image(),
        _prediction(),
        class_names=["A", "B"],
        colors=["#e41a1c", "#377eb8"],
        alpha=0.48,
        stride=4,
    )
    overlay_legend = variants["overlay_legend"]
    confidence_colorbar = variants["confidence_colorbar"]
    entropy_colorbar = variants["entropy_colorbar"]
    assert overlay_legend.shape[1] >= variants["overlay"].shape[1]
    assert np.array_equal(
        overlay_legend[
            : variants["overlay"].shape[0], : variants["overlay"].shape[1]
        ],
        variants["overlay"],
    )
    # The scientific region of every map keeps its native resolution.
    assert np.array_equal(
        confidence_colorbar[:, : variants["confidence"].shape[1]],
        variants["confidence"],
    )
    assert np.array_equal(
        entropy_colorbar[:, : variants["entropy"].shape[1]], variants["entropy"]
    )


def test_long_legend_labels_expand_instead_of_being_truncated() -> None:
    short = rx.build_categorical_legend(
        [{"index": 0, "name": "A", "color": "#e41a1c"}]
    )
    long = rx.build_categorical_legend(
        [
            {
                "index": 0,
                "name": "A deliberately long scientific class label",
                "color": "#e41a1c",
            }
        ]
    )
    assert long.shape[1] > short.shape[1]


def test_categorical_legend_wraps_after_three_classes() -> None:
    three = rx.build_categorical_legend(
        [
            {"index": index, "name": f"Class {index}", "color": "#e41a1c"}
            for index in range(3)
        ]
    )
    four = rx.build_categorical_legend(
        [
            {"index": index, "name": f"Class {index}", "color": "#e41a1c"}
            for index in range(4)
        ]
    )
    assert three.shape[0] == rx.LEGEND_PADDING * 2 + rx.LEGEND_ROW_HEIGHT
    assert four.shape[0] == rx.LEGEND_PADDING * 2 + 2 * rx.LEGEND_ROW_HEIGHT


def test_result_export_names_cover_all_expected_variants() -> None:
    assert rx.RESULT_EXPORT_NAMES == (
        "overlay",
        "overlay_legend",
        "mask",
        "confidence",
        "confidence_colorbar",
        "entropy",
        "entropy_colorbar",
    )
    assert rx.DEFAULT_OVERLAY_ALPHA == 0.48


# ---------------------------------------------------------------------------
# PNG writing / decode / download copies
# ---------------------------------------------------------------------------
def test_every_variant_writes_a_decodable_rgb_png(tmp_path: Path) -> None:
    variants = rx.render_result_variants(
        _image(),
        _prediction(),
        class_names=["A", "B"],
        colors=["#e41a1c", "#377eb8"],
        alpha=0.48,
        stride=4,
    )
    for name, array in variants.items():
        path = rx.write_png(array, tmp_path / f"{name}.png")
        assert path.is_file()
        with Image.open(path) as loaded:
            assert loaded.mode == "RGB"
            assert loaded.size == (array.shape[1], array.shape[0])


def test_prepare_download_copy_never_touches_the_source(tmp_path: Path) -> None:
    source = rx.write_png(
        np.zeros((4, 4, 3), dtype=np.uint8), tmp_path / "prediction_overlay.png"
    )
    original = source.read_bytes()
    copy = rx.prepare_download_copy(
        source, run_id="symmetry-demo", filename="prediction_overlay.png"
    )
    assert copy.is_file()
    assert copy.read_bytes() == original
    assert source.read_bytes() == original
    assert copy.name == "symmetry-demo-prediction_overlay.png"


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------
def test_presentation_modules_do_not_import_gradio_torch_or_symmlearn() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "symmetry_harness"
    banned = ("gradio", "torch", "symmlearn", "matplotlib")
    for name in ("colormaps.py", "result_exports.py", "ui_readiness.py"):
        for line in (root / name).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            for token in banned:
                assert token not in stripped, f"{name} imports {token}: {stripped!r}"
