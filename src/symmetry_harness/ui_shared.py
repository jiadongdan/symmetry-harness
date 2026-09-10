"""Rendering and presentation helpers shared by both interface workspaces."""

from __future__ import annotations

from html import escape

from .annotations import DEFAULT_CLASS_COLORS


REGISTERED_WEIGHT_SOURCE = "Registered weight"
CUSTOM_CHECKPOINT_SOURCE = "Custom checkpoint"
ANNOTATION_DISPLAY_MAX_EDGE = 720
INVALID_SUPPORT_POINT_MESSAGE = (
    "Please select a support point inside the valid region."
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
    "margin-top: 4px;}"
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


def class_legend_html(classes: list[dict]) -> str:
    """Render an ordered class legend with saved display colors."""
    rows = "".join(
        (
            "<tr>"
            f'<td style="padding-right:8px">'
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{escape(str(entry.get("color", "#ffffff")))};'
            f'border:1px solid var(--border-color-primary)"></span></td>'
            f"<td>{escape(str(entry.get('name', '')))}</td>"
            "</tr>"
        )
        for entry in classes
    )
    return f'<table style="border-collapse:collapse">{rows}</table>'


def class_statistics_html(statistics: dict) -> str:
    """Render per-class counts and fractions for one prediction."""
    entries = statistics.get("classes", [])
    rows = "".join(
        (
            "<tr>"
            f'<td style="padding-right:12px">{escape(str(entry.get("index")))}</td>'
            f"<td>{int(entry.get('count', 0))}</td>"
            f"<td>{float(entry.get('fraction', 0.0)) * 100:.1f}%</td>"
            "</tr>"
        )
        for entry in entries
    )
    header = "<tr><th>Class</th><th>Cells</th><th>Fraction</th></tr>"
    return f'<table style="border-collapse:collapse">{header}{rows}</table>'
