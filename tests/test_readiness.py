"""Focused tests for pure readiness evaluation and presentation terminology."""

from __future__ import annotations

from symmetry_harness.ui_readiness import (
    STATUS_BLOCKER,
    STATUS_READY,
    STATUS_RECOMMENDATION,
    ReadinessItem,
    evaluate_fine_tune_readiness,
    evaluate_predict_readiness,
    has_blocker,
    readiness_panel_html,
)
from symmetry_harness import ui_shared


def _by_key(items, key):
    return next(item for item in items if item.key == key)


def _ready_fine_tune(**overrides) -> dict:
    snapshot = {
        "model_supported": True,
        "model_name": "CNN 8ch",
        "checkpoint_ready": True,
        "checkpoint_detail": "Installed.",
        "image_valid": True,
        "image_name": "sample.npy",
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
    }
    snapshot.update(overrides)
    return snapshot


# ---------------------------------------------------------------------------
# Fine-tune readiness
# ---------------------------------------------------------------------------
def test_fine_tune_full_readiness_has_no_blocker() -> None:
    items = evaluate_fine_tune_readiness(_ready_fine_tune())
    assert len(items) == 10
    assert not has_blocker(items)
    assert all(item.status == STATUS_READY for item in items)


def test_three_points_per_class_is_ready_when_minimum_is_three() -> None:
    items = evaluate_fine_tune_readiness(_ready_fine_tune(class_counts=[3, 3]))
    assert _by_key(items, "min_support").status == STATUS_READY
    assert not has_blocker(items)


def test_fewer_than_five_points_is_advisory_only() -> None:
    items = evaluate_fine_tune_readiness(_ready_fine_tune(class_counts=[4, 3]))
    assert _by_key(items, "min_support").status == STATUS_READY
    advice = _by_key(items, "support_advice")
    assert advice.status == STATUS_RECOMMENDATION
    assert advice.hint  # explains that five is recommended
    assert not has_blocker(items)


def test_fewer_than_three_points_blocks() -> None:
    items = evaluate_fine_tune_readiness(_ready_fine_tune(class_counts=[3, 2]))
    blocker = _by_key(items, "min_support")
    assert blocker.status == STATUS_BLOCKER
    assert "Class B" in blocker.hint
    assert has_blocker(items)


def test_stale_features_block_with_an_actionable_next_step() -> None:
    items = evaluate_fine_tune_readiness(_ready_fine_tune(features_current=False))
    blocker = _by_key(items, "features_current")
    assert blocker.status == STATUS_BLOCKER
    assert "symmetry maps" in blocker.hint.lower()
    assert has_blocker(items)


def test_stale_downstream_state_blocks_with_next_step() -> None:
    items = evaluate_fine_tune_readiness(
        _ready_fine_tune(stale_state=True, stale_detail="Result predates current classes.")
    )
    blocker = _by_key(items, "no_stale_state")
    assert blocker.status == STATUS_BLOCKER
    assert blocker.hint
    assert has_blocker(items)


def test_invalid_runtime_parameters_block() -> None:
    items = evaluate_fine_tune_readiness(
        _ready_fine_tune(epochs=0, stride=-1, batch_size=0, device="quantum")
    )
    blocker = _by_key(items, "runtime_params")
    assert blocker.status == STATUS_BLOCKER
    for token in ("epochs", "stride", "batch size", "device"):
        assert token in blocker.detail
    assert has_blocker(items)


def test_even_symmetry_patch_size_blocks() -> None:
    items = evaluate_fine_tune_readiness(_ready_fine_tune(symmetry_patch_size=52))
    assert _by_key(items, "symmetry_patch").status == STATUS_BLOCKER


# ---------------------------------------------------------------------------
# Predict readiness
# ---------------------------------------------------------------------------
def _ready_predict(**overrides) -> dict:
    snapshot = {
        "provider_supported": True,
        "package_path": "model.symmodel",
        "package_valid": True,
        "package_compatible": True,
        "compatibility_warnings": [],
        "package_detail": "Package validated.",
        "valid_images": 2,
        "invalid_images": 0,
        "device": "auto",
        "stride": 8,
        "batch_size": 128,
    }
    snapshot.update(overrides)
    return snapshot


def test_predict_full_readiness_has_no_blocker() -> None:
    items = evaluate_predict_readiness(_ready_predict())
    assert len(items) == 7
    assert not has_blocker(items)


def test_predict_extension_policy_blocks_non_symmodel() -> None:
    items = evaluate_predict_readiness(_ready_predict(package_path="model.zip"))
    blocker = _by_key(items, "extension")
    assert blocker.status == STATUS_BLOCKER
    assert ".symmodel" in blocker.hint
    assert has_blocker(items)


def test_predict_distinguishes_invalid_and_incompatible_packages() -> None:
    invalid = evaluate_predict_readiness(
        _ready_predict(package_valid=False, package_detail="Checksum mismatch.")
    )
    assert _by_key(invalid, "package_valid").status == STATUS_BLOCKER
    assert _by_key(invalid, "compatible").status == STATUS_BLOCKER
    assert _by_key(invalid, "can_predict").status == STATUS_BLOCKER

    incompatible = evaluate_predict_readiness(
        _ready_predict(
            package_compatible=False,
            package_detail="The saved model is not installed in this Provider.",
        )
    )
    assert _by_key(incompatible, "package_valid").status == STATUS_READY
    compat_blocker = _by_key(incompatible, "compatible")
    assert compat_blocker.status == STATUS_BLOCKER
    assert "not installed" in compat_blocker.detail


def test_predict_compatible_with_warnings_is_advisory() -> None:
    items = evaluate_predict_readiness(
        _ready_predict(compatibility_warnings=["Patch versions differ."])
    )
    assert _by_key(items, "compatible").status == STATUS_RECOMMENDATION
    assert not has_blocker(items)


def test_predict_invalid_images_block() -> None:
    items = evaluate_predict_readiness(_ready_predict(valid_images=0, invalid_images=3))
    blocker = _by_key(items, "images")
    assert blocker.status == STATUS_BLOCKER
    assert "0 valid / 3 invalid" in blocker.detail
    assert _by_key(items, "can_predict").status == STATUS_BLOCKER


def test_predict_invalid_overrides_block() -> None:
    items = evaluate_predict_readiness(_ready_predict(stride=0, batch_size=-1))
    blocker = _by_key(items, "runtime_params")
    assert blocker.status == STATUS_BLOCKER
    assert "stride" in blocker.detail and "batch size" in blocker.detail


def test_predict_provider_without_support_blocks() -> None:
    items = evaluate_predict_readiness(_ready_predict(provider_supported=False))
    assert _by_key(items, "provider_support").status == STATUS_BLOCKER
    assert has_blocker(items)


# ---------------------------------------------------------------------------
# HTML rendering / consistency
# ---------------------------------------------------------------------------
def test_button_interactivity_agrees_with_readiness() -> None:
    ready = evaluate_fine_tune_readiness(_ready_fine_tune())
    blocked = evaluate_fine_tune_readiness(_ready_fine_tune(model_supported=False))
    # The UI disables the primary button exactly when a blocker is present.
    assert not has_blocker(ready)
    assert has_blocker(blocked)


def test_readiness_panel_html_reports_next_action_without_button_color() -> None:
    items = evaluate_predict_readiness(_ready_predict(valid_images=0))
    html = readiness_panel_html(items)
    assert 'id="readiness-panel"' in html
    # The immediate next action is stated in text, not conveyed by button color.
    assert "Next:" in html
    assert "Select at least one valid prediction image." in html


def test_readiness_panel_html_uses_status_icons() -> None:
    items = [
        ReadinessItem("a", "Alpha", STATUS_READY),
        ReadinessItem("b", "Beta", STATUS_RECOMMENDATION, detail="advisory"),
        ReadinessItem("c", "Gamma", STATUS_BLOCKER, hint="do the thing"),
    ]
    html = readiness_panel_html(items)
    assert "\u2713" in html
    assert "!" in html
    assert "\u00d7" in html
    assert "Next: do the thing" in html


def test_readiness_panel_html_shows_recommendation_text() -> None:
    items = evaluate_fine_tune_readiness(
        _ready_fine_tune(class_counts=[3, 4], recommended_shots=5)
    )
    html = readiness_panel_html(items)
    assert "Recommended:" in html
    assert "5 support points per class" in html
    assert "Minimum 3; recommended 5; maximum 50." in html


def test_readiness_panel_html_default_elem_id_is_the_predict_panel() -> None:
    items = evaluate_fine_tune_readiness(_ready_fine_tune())
    html = readiness_panel_html(items)
    assert 'id="readiness-panel"' in html
    assert 'class="readiness-panel"' in html


def test_readiness_panel_html_elem_id_none_omits_the_id_but_keeps_the_class() -> None:
    items = evaluate_fine_tune_readiness(_ready_fine_tune())
    html = readiness_panel_html(items, elem_id=None)
    assert 'class="readiness-panel"' in html
    assert 'id="readiness-panel"' not in html
    # No wrapper id leaks in when the caller disables it.
    assert "id=\"" not in html.split("readiness-title")[0]


def test_readiness_panel_html_accepts_a_custom_elem_id() -> None:
    items = evaluate_fine_tune_readiness(_ready_fine_tune())
    html = readiness_panel_html(items, elem_id="custom-readiness")
    assert 'id="custom-readiness"' in html
    assert 'class="readiness-panel"' in html
    assert 'id="readiness-panel"' not in html


# ---------------------------------------------------------------------------
# Shared terminology helpers
# ---------------------------------------------------------------------------
def test_class_statistics_uses_correct_terminology_and_merges_colors() -> None:
    statistics = {
        "grid_cell_count": 4,
        "classes": [
            {"index": 0, "count": 2, "fraction": 0.5},
            {"index": 1, "count": 2, "fraction": 0.5},
        ],
    }
    html = ui_shared.class_statistics_html(
        statistics, class_names=["Phase A", "Phase B"], class_colors=["#e41a1c", "#377eb8"]
    )
    assert "Cells" not in html
    assert "Predicted grid points" in html
    assert "Fraction of predicted grid" in html
    assert "Phase A" in html and "Phase B" in html
    assert "#e41a1c" in html and "#377eb8" in html
    assert ">2<" in html
    assert "50.0%" in html
    assert ui_shared.CLASS_STATISTICS_NOTE.split(".")[0] in html
    assert html.count("<th>") == 3
    assert html.count("<td") == 6


def test_class_statistics_remains_callable_without_names() -> None:
    statistics = {"classes": [{"index": 0, "count": 1, "fraction": 1.0}]}
    html = ui_shared.class_statistics_html(statistics)
    assert "Cells" not in html
    assert "Class 1" in html


def test_class_legend_uses_a_three_column_wrapping_grid() -> None:
    html = ui_shared.class_legend_html(
        [
            {"index": index, "name": f"Class {index}", "color": "#e41a1c"}
            for index in range(5)
        ]
    )
    assert html.count('class="class-legend-item"') == 5
    assert "<table" not in html
    assert "repeat(3, minmax(0, 1fr))" in ui_shared.APP_CSS


def test_result_item_label_disambiguates_duplicate_basenames() -> None:
    seen: dict[str, int] = {}
    first = ui_shared.result_item_label(
        "/a/sample.tif", "completed", item_id="image-0001", seen=seen
    )
    second = ui_shared.result_item_label(
        "/b/sample.tif", "failed", item_id="image-0002", seen=seen
    )
    third = ui_shared.result_item_label(
        "/c/other.tif", "completed", item_id="image-0003", seen=seen
    )
    assert first == "sample.tif \u2014 completed"
    assert second == "sample.tif \u2014 failed (2)"
    assert third == "other.tif \u2014 completed"


def test_app_css_keeps_readiness_panel_sticky_on_narrow_screens() -> None:
    assert "#readiness-panel" in ui_shared.APP_CSS
    assert "@media (max-width: 1100px)" in ui_shared.APP_CSS
    narrow_rule = ui_shared.APP_CSS.split("@media (max-width: 1100px)", 1)[1]
    assert "position: sticky" in narrow_rule
    assert "order: -1" in narrow_rule
    assert "position: static" not in narrow_rule
