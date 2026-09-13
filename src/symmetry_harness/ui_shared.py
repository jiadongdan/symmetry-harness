"""Rendering and presentation helpers shared by both interface workspaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path

from .annotations import DEFAULT_CLASS_COLORS
from .ui_readiness import ReadinessItem, readiness_panel_html  # noqa: F401  (re-exported)


REGISTERED_WEIGHT_SOURCE = "Registered weight"
CUSTOM_CHECKPOINT_SOURCE = "Custom checkpoint"
ANNOTATION_DISPLAY_MAX_EDGE = 720
INVALID_SUPPORT_POINT_MESSAGE = (
    "Please select a support point inside the valid region."
)

CLASS_STATISTICS_NOTE = (
    "Counts and fractions are calculated over the predicted sampling grid. "
    "They are not ground-truth accuracy or direct physical area measurements."
)

APP_CSS = (
    "#annotation-image {max-width: 760px; margin: 0 auto;} "
    "#symmetry-compute-status {min-height: 72px; display: flex; "
    "align-items: center; padding: 12px 16px; border: 1px solid "
    "var(--border-color-primary); border-left: 4px solid var(--color-accent); "
    "border-radius: var(--radius-lg); background: var(--background-fill-secondary);} "
    "#symmetry-compute-status p {margin: 0; font-weight: 600;} "
    ".phase-progress {padding: 8px 2px 4px;} "
    ".phase-progress-label {display: flex; justify-content: space-between; "
    "font-weight: 600; margin-bottom: 6px;} "
    ".phase-progress-track {height: 14px; overflow: hidden; border-radius: 7px; "
    "background: var(--background-fill-secondary); border: 1px solid "
    "var(--border-color-primary);} "
    ".phase-progress-fill {height: 100%; background: var(--color-accent); "
    "transition: width 0.15s ease;} "
    ".phase-progress-status {font-size: 0.85em; color: var(--body-text-color-subdued); "
    "margin-top: 4px;} "
    # Sticky has to *not* be stretched: a full-height grid/flex item has no room
    # left to travel, so ``position: sticky`` silently degenerates to static and
    # the panel scrolls out of view. ``align-self: start`` works for both flex and
    # grid, and ``height: fit-content`` guarantees the item keeps its own size.
    "#readiness-panel, .readiness-sticky {position: sticky; top: 12px; "
    "align-self: start; height: fit-content; "
    "max-height: calc(100vh - 24px); overflow-y: auto; z-index: 5;} "
    ".readiness-panel {border: 1px solid var(--border-color-primary); "
    "border-radius: var(--radius-lg); padding: 10px 12px; "
    "background: var(--background-fill-secondary);} "
    ".readiness-title {font-weight: 600; margin-bottom: 8px;} "
    ".readiness-item {display: flex; gap: 8px; padding: 3px 0; "
    "font-size: 0.9em; line-height: 1.3;} "
    ".readiness-icon {font-weight: 700; width: 1em; text-align: center;} "
    ".readiness-ready .readiness-icon {color: #2e7d32;} "
    ".readiness-recommendation .readiness-icon {color: #b26a00;} "
    ".readiness-blocker .readiness-icon {color: #c62828;} "
    ".readiness-body {display: flex; flex-direction: column;} "
    ".readiness-detail {color: var(--body-text-color-subdued); font-size: 0.92em;} "
    ".readiness-hint {font-weight: 600; font-size: 0.92em;} "
    ".class-statistics-note {color: var(--body-text-color-subdued); "
    "font-size: 0.85em; margin-top: 4px;} "
    ".class-legend-grid {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); "
    "gap: 8px 16px; padding: 8px 4px;} "
    ".class-legend-item {display: flex; align-items: center; gap: 8px; min-width: 0;} "
    ".class-legend-label {overflow-wrap: anywhere;} "
    "@media (max-width: 1100px) {"
    "  #readiness-panel, .readiness-sticky "
    "{position: sticky; top: 0; order: -1; align-self: start; width: 100%; height: auto; "
    "max-height: 35vh; overflow-y: auto; z-index: 20;}"
    "}"
    "@media (max-width: 700px) {"
    "  .class-legend-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}"
    "}"
    "@media (max-width: 460px) {"
    "  .class-legend-grid {grid-template-columns: minmax(0, 1fr);}"
    "}"
)


def progress_bar_html(label: str, current: int, total: int, status: str) -> str:
    """Render one deterministic progress bar for a long-running phase."""
    safe_total = max(1, int(total))
    safe_current = min(max(0, int(current)), safe_total)
    percent = int(round(100 * safe_current / safe_total))
    return (
        '<div class="phase-progress">'
        '<div class="phase-progress-label">'
        f"<span>{escape(label)}</span><span>{percent}%</span></div>"
        '<div class="phase-progress-track">'
        f'<div class="phase-progress-fill" style="width: {percent}%"></div></div>'
        f'<div class="phase-progress-status">{escape(status)}</div></div>'
    )


def parse_class_names(value: str) -> list[str]:
    """Parse unique comma-separated local class names."""
    names = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(names) < 2:
        raise ValueError("Enter at least two comma-separated class names.")
    if len(set(names)) != len(names):
        raise ValueError("Class names must be unique.")
    if len(names) > len(DEFAULT_CLASS_COLORS):
        raise ValueError(f"The interface supports at most {len(DEFAULT_CLASS_COLORS)} classes.")
    return names


def _swatch_html(color: str) -> str:
    return (
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'background:{escape(str(color))};'
        f'border:1px solid var(--border-color-primary)"></span>'
    )


def class_legend_html(classes: list[dict]) -> str:
    """Render an ordered class legend in rows of at most three classes."""
    items = "".join(
        (
            '<div class="class-legend-item">'
            f'{_swatch_html(entry.get("color", "#ffffff"))}'
            f'<span class="class-legend-label">{escape(str(entry.get("name", "")))}</span>'
            "</div>"
        )
        for entry in classes
    )
    return f'<div class="class-legend-grid">{items}</div>'


def class_statistics_html(
    statistics: Mapping[str, object],
    *,
    class_names: Sequence[str] = (),
    class_colors: Sequence[str] = (),
) -> str:
    """Render per-class counts and fractions for one prediction.

    Each row combines the class color swatch, the class name, the predicted
    grid-point count and the fraction of the predicted grid. ``index`` is merged
    with ``class_names`` / ``class_colors`` by position (the numerical layer only
    reports indices). The ``Cells`` terminology is intentionally gone.
    """
    entries = statistics.get("classes", []) if isinstance(statistics, Mapping) else []
    rows = []
    for entry in entries:
        index = int(entry.get("index", 0))
        name = (
            str(class_names[index])
            if index < len(class_names)
            else f"Class {index + 1}"
        )
        color = class_colors[index] if index < len(class_colors) else "#ffffff"
        fraction = float(entry.get("fraction", 0.0))
        rows.append(
            "<tr>"
            f'<td style="padding-right:12px">{_swatch_html(color)} '
            f'{escape(name)}</td>'
            f"<td style=\"padding-right:12px\">{int(entry.get('count', 0))}</td>"
            f"<td>{fraction * 100:.1f}%</td>"
            "</tr>"
        )
    header = (
        "<tr><th>Class</th><th>Predicted grid points</th>"
        "<th>Fraction of predicted grid</th></tr>"
    )
    table = f'<table style="border-collapse:collapse">{header}{"".join(rows)}</table>'
    return f'{table}<p class="class-statistics-note">{escape(CLASS_STATISTICS_NOTE)}</p>'


def result_item_label(
    source_name: str,
    status: str,
    *,
    item_id: str,
    seen: dict[str, int],
) -> str:
    """Return a user-facing ``filename — status`` label for one batch item.

    The stable internal ``item_id`` is never changed. Duplicate basenames are
    disambiguated deterministically (``(2)``, ``(3)`` ...) in encounter order via
    the shared ``seen`` counter dictionary.
    """
    base = Path(str(source_name)).name
    count = seen.get(base, 0) + 1
    seen[base] = count
    label = f"{base} \u2014 {status}"
    if count > 1:
        label = f"{label} ({count})"
    return label
