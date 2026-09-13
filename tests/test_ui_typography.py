"""Typography contract for the shared application stylesheet.

The Phase 2 typography protocol fixes font size, line height and font weight
for every readable element of the Fine-tune and Predict workspaces. These tests
parse :data:`symmetry_harness.ui_shared.APP_CSS` into selector/declaration pairs
and assert the frozen numbers, so a later edit cannot quietly reintroduce
sub-13-px helper text or the nested ``em`` scaling in the readiness panel.

They deliberately assert semantics and values only: never the full CSS text,
its whitespace, its declaration order or its rule count.
"""

from __future__ import annotations

import re

import pytest

from symmetry_harness import ui_shared


CSS = ui_shared.APP_CSS


# ---------------------------------------------------------------------------
# Small CSS parsing helpers (no full stylesheet engine)
# ---------------------------------------------------------------------------
def parse_declarations(body: str) -> dict[str, str]:
    """Return ``{property: value}`` for one declaration block."""
    declarations: dict[str, str] = {}
    for chunk in body.split(";"):
        if ":" not in chunk:
            continue
        name, value = chunk.split(":", 1)
        declarations[name.strip()] = value.strip()
    return declarations


def iter_rules(css: str):
    """Yield ``(selector, declarations, at_rule)`` for every rule in ``css``.

    ``at_rule`` is the enclosing media query prelude, or ``None`` for a
    top-level rule. Values in this stylesheet never contain ``{`` or ``}``.
    """
    stack: list[str] = []
    prelude_start = 0
    index = 0
    while index < len(css):
        char = css[index]
        if char == "{":
            prelude = css[prelude_start:index].strip()
            if prelude.startswith("@"):
                stack.append(prelude)
                prelude_start = index + 1
            else:
                end = css.index("}", index)
                yield prelude, parse_declarations(css[index + 1 : end]), (
                    stack[-1] if stack else None
                )
                index = end
                prelude_start = end + 1
        elif char == "}":
            if stack:
                stack.pop()
            prelude_start = index + 1
        index += 1


RULES = list(iter_rules(CSS))


def declarations(selector: str, at_rule: str | None = None) -> dict[str, str]:
    """Return the declarations of the single top-level rule for ``selector``."""
    for rule_selector, body, media in RULES:
        if rule_selector == selector and media == at_rule:
            return body
    raise AssertionError(f"no rule for {selector!r} (at_rule={at_rule!r})")


def find_rule(*selector_tokens: str, at_rule: str | None = None) -> dict[str, str]:
    """Return the declarations of the first rule whose selector contains tokens."""
    for rule_selector, body, media in RULES:
        if media != at_rule:
            continue
        if all(token in rule_selector for token in selector_tokens):
            return body
    raise AssertionError(f"no rule containing {selector_tokens!r}")


# ---------------------------------------------------------------------------
# 1. Frozen theme tokens
# ---------------------------------------------------------------------------
FROZEN_TOKENS = {
    "--text-sm": "13px",
    "--text-md": "15px",
    "--text-lg": "18px",
    "--text-xl": "24px",
    "--text-xxl": "30px",
    "--body-text-size": "15px",
    "--prose-text-size": "15px",
    "--input-text-size": "15px",
    "--block-label-text-size": "14px",
    "--block-info-text-size": "13px",
    "--block-title-text-size": "14px",
    "--section-header-text-size": "15px",
    "--checkbox-label-text-size": "14px",
    "--button-large-text-size": "15px",
    "--button-medium-text-size": "15px",
    "--button-small-text-size": "13px",
}


def test_typography_tokens_are_declared_with_the_frozen_values() -> None:
    root = declarations(":root")
    for name, value in FROZEN_TOKENS.items():
        assert root.get(name) == value, f"{name} is {root.get(name)!r}, expected {value!r}"


def test_only_typography_tokens_are_overridden() -> None:
    allowed = re.compile(
        r"^--(text-(sm|md|lg|xl|xxl)"
        r"|body-text-size|prose-text-size|input-text-size"
        r"|block-(label|info|title)-text-(size|weight)"
        r"|section-header-text-(size|weight)"
        r"|checkbox-label-text-size"
        r"|button-(large|medium|small)-text-size)$"
    )
    for name in declarations(":root"):
        assert allowed.match(name), f"{name} is not a typography token"


def test_field_label_and_block_title_weights() -> None:
    root = declarations(":root")
    # Gradio renders the field label and the block title through its own
    # ``--block-label-*`` / ``--block-title-*`` tokens, so the 600 weight of a
    # field title has to be declared there rather than on a hashed class.
    assert root["--block-label-text-weight"] == "600"
    assert root["--block-title-text-weight"] == "600"
    assert root["--section-header-text-weight"] == "600"
    # Helper text stays regular: it is explanatory, not a title.
    assert root["--block-label-text-size"] == "14px"
    assert root["--block-info-text-size"] == "13px"


def test_no_importants_and_no_versioned_selectors() -> None:
    assert "!important" not in CSS
    assert ".svelte-" not in CSS
    assert not re.search(r"\.gradio-container-\d", CSS)


def test_the_font_family_is_left_to_the_theme() -> None:
    # No web font is requested and no family is redefined: only sizes change.
    assert "font-family" not in CSS
    assert "@font-face" not in CSS


# ---------------------------------------------------------------------------
# 2. Headings, prose and inline code
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("selector", "size", "line_height"),
    [
        (".prose h1", "30px", "38px"),
        (".prose h2", "24px", "32px"),
        (".prose h3", "18px", "26px"),
    ],
)
def test_prose_heading_contract(selector, size, line_height) -> None:
    body = declarations(selector)
    assert body["font-size"] == size
    assert body["line-height"] == line_height
    assert body["font-weight"] == "600"


def test_prose_body_and_inline_code_contract() -> None:
    paragraph = declarations(".prose p")
    assert paragraph["font-size"] == "15px"
    assert paragraph["line-height"] == "23px"
    code = declarations(".prose code")
    assert code["font-size"] == "13px"
    assert code["line-height"] == "20px"


def test_mobile_step_sizes_down_only_h1() -> None:
    mobile = "@media (max-width: 700px)"
    h1 = declarations(".prose h1", at_rule=mobile)
    assert h1["font-size"] == "28px"
    assert h1["line-height"] == "36px"
    # H2, H3 and body text keep their desktop size on narrow screens.
    for selector in (".prose h2", ".prose h3", ".prose p"):
        with pytest.raises(AssertionError):
            declarations(selector, at_rule=mobile)


# ---------------------------------------------------------------------------
# 3. Navigation: tabs and accordions
# ---------------------------------------------------------------------------
def test_tab_and_accordion_contract() -> None:
    tab = declarations('[role="tab"]')
    assert tab["font-size"] == "15px"
    assert tab["line-height"] == "22px"
    assert tab["font-weight"] == "600"

    accordion = declarations(".label-wrap")
    assert accordion["font-size"] == "15px"
    assert accordion["line-height"] == "22px"
    assert accordion["font-weight"] == "600"


def test_tab_rule_does_not_replace_the_accent_treatment() -> None:
    # Only typography is declared for tabs: colours and the underline stay
    # with the theme.
    for name in declarations('[role="tab"]'):
        assert name in {"font-size", "line-height", "font-weight"}


# ---------------------------------------------------------------------------
# 4. Controls, helper text and tables
# ---------------------------------------------------------------------------
def test_text_controls_reach_fifteen_px_over_twenty_two_px() -> None:
    controls = find_rule("textarea", "select", at_rule=None)
    assert controls["font-size"] == "15px"
    assert controls["line-height"] == "22px"
    selector = next(
        rule_selector
        for rule_selector, _, media in RULES
        if media is None and "textarea" in rule_selector and "select" in rule_selector
    )
    # Checkboxes and radios keep their own label size: they are excluded.
    for excluded in ('[type="checkbox"]', '[type="radio"]', '[type="range"]'):
        assert excluded in selector


def test_helper_text_contract() -> None:
    helper = declarations(".info-text")
    assert helper["font-size"] == "13px"
    assert helper["line-height"] == "19px"


def test_table_contract() -> None:
    cells = declarations("table th, table td")
    assert cells["font-size"] == "14px"
    assert cells["line-height"] == "20px"
    assert declarations("table th")["font-weight"] == "600"
    dataframe = declarations(".header-table, .virtual-table-viewport")
    assert dataframe["font-size"] == "14px"
    assert dataframe["line-height"] == "20px"


def test_button_line_heights_follow_the_button_size() -> None:
    assert declarations("button.sm")["line-height"] == "18px"
    assert declarations("button.md, button.lg")["line-height"] == "22px"


# ---------------------------------------------------------------------------
# 5. Readiness, progress and the scientific note
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("selector", "size", "line_height"),
    [
        (".readiness-title", "16px", "24px"),
        (".readiness-item", "14px", "20px"),
        (".readiness-detail", "13px", "18px"),
        (".readiness-hint", "13px", "18px"),
    ],
)
def test_readiness_levels_use_explicit_pixel_sizes(selector, size, line_height) -> None:
    body = declarations(selector)
    assert body["font-size"] == size
    assert body["line-height"] == line_height
    assert "em" not in body["font-size"]


def test_readiness_stops_scaling_with_nested_em() -> None:
    assert "0.9em" not in CSS
    assert "0.92em" not in CSS
    assert "0.85em" not in CSS


def test_readiness_keeps_its_layout_and_colours() -> None:
    item = declarations(".readiness-item")
    assert item["display"] == "flex"
    assert item["gap"] == "8px"
    assert item["padding"] == "3px 0"
    detail = declarations(".readiness-detail")
    assert detail["color"] == "var(--body-text-color-subdued)"
    hint = declarations(".readiness-hint")
    assert hint["font-weight"] == "600"
    icon = declarations(".readiness-icon")
    assert icon["font-weight"] == "700"
    assert icon["width"] == "1em"


def test_progress_and_scientific_note_contract() -> None:
    label = declarations(".phase-progress-label")
    assert label["font-size"] == "14px"
    assert label["line-height"] == "20px"
    assert label["font-weight"] == "600"
    status = declarations(".phase-progress-status")
    assert status["font-size"] == "13px"
    assert status["line-height"] == "18px"
    note = declarations(".class-statistics-note")
    assert note["font-size"] == "13px"
    assert note["line-height"] == "19px"


def test_progress_bar_geometry_is_untouched() -> None:
    track = declarations(".phase-progress-track")
    assert track["height"] == "14px"
    assert track["border-radius"] == "7px"
    fill = declarations(".phase-progress-fill")
    assert fill["transition"] == "width 0.15s ease"


# ---------------------------------------------------------------------------
# 6. Non-regression of the frozen presentation rules
# ---------------------------------------------------------------------------
def test_class_legend_grid_layout_is_unchanged() -> None:
    grid = declarations(".class-legend-grid")
    assert grid["grid-template-columns"] == "repeat(3, minmax(0, 1fr))"
    assert grid["gap"] == "8px 16px"
    assert grid["padding"] == "8px 4px"


def test_readiness_panel_sticky_rule_is_unchanged() -> None:
    panel = declarations("#readiness-panel, .readiness-sticky")
    assert panel["position"] == "sticky"
    assert panel["max-height"] == "calc(100vh - 24px)"
    narrow = declarations("#readiness-panel, .readiness-sticky", at_rule="@media (max-width: 1100px)")
    assert narrow["position"] == "sticky"
    assert narrow["order"] == "-1"


def test_no_core_workspace_rule_declares_a_size_below_thirteen_px() -> None:
    offenders: list[tuple[str, str]] = []
    for selector, body, _ in RULES:
        size = body.get("font-size", "")
        if size.endswith("px") and float(size[:-2]) < 13:
            offenders.append((selector, size))
    assert offenders == []


def test_readiness_and_progress_html_still_render_the_font_styled_classes() -> None:
    # The stylesheet only works if the renderers keep emitting these hooks.
    items = [
        ui_shared.ReadinessItem("a", "Alpha", "blocker", detail="d", hint="h"),
    ]
    html = ui_shared.readiness_panel_html(items)
    for hook in (
        "readiness-title",
        "readiness-item",
        "readiness-label",
        "readiness-detail",
        "readiness-hint",
    ):
        assert hook in html
    progress = ui_shared.progress_bar_html("Fine-tuning", 1, 4, "Epoch 1 of 2")
    for hook in ("phase-progress-label", "phase-progress-status"):
        assert hook in progress
