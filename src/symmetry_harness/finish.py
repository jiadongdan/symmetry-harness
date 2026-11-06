"""Close a finished fine-tune out and hand it to the conversation.

The interface and the conversation are separate processes with no channel
between them: a button cannot wake an assistant up. What a button *can* do is
make the handover unambiguous. It writes the fixed-template report, a
self-contained HTML copy that survives outside the run directory, and a small
record naming the run, so the next conversation does not have to guess which
fine-tune is meant.

Nothing here trains or re-scores anything. The report is a pure view over
artifacts the run already produced, which is why it takes seconds.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import html as html_module
import json
import os
from pathlib import Path
import re
from typing import Any, Sequence

FINISH_SCHEMA_VERSION = "symmetry-run-handoff-v1"
HANDOFF_NAME = "ui_handoff.json"
REPORT_HTML_NAME = "report_view.html"

#: Defaults the conversation layer passes to ``symmetry repeat``. Twenty rounds
#: at roughly twelve seconds each is four minutes: enough for a distribution,
#: short enough that a person waits for it.
REPEAT_DEFAULT_ROUNDS = 20
REPEAT_DEFAULT_POINTS = 5

NEXT_STEP = (
    "Report written. To check reproducibility, ask the assistant in the chat to "
    "run repeated k-shot trials on this run."
)


def finish_run(
    run_dir: str | Path,
    *,
    notes: str | None = None,
    rounds: int = REPEAT_DEFAULT_ROUNDS,
    points_per_class: int = REPEAT_DEFAULT_POINTS,
) -> dict[str, Any]:
    """Write the report, the standalone HTML copy, and the handoff record.

    Safe to call repeatedly: every output is rewritten in place and nothing is
    trained. Returns the paths and the command the conversation should run.
    """
    from .run_report import build_run_report

    directory = Path(run_dir).expanduser().resolve()
    result = build_run_report(directory, notes=notes, figures=True)

    html_path = directory / REPORT_HTML_NAME
    html_path.write_text(
        render_report_html(result.report_path, result.figure_paths),
        encoding="utf-8",
    )

    handoff = _handoff_record(
        directory,
        result,
        html_path,
        rounds=rounds,
        points_per_class=points_per_class,
    )
    handoff_path = directory / HANDOFF_NAME
    _write_text_atomic(handoff_path, json.dumps(handoff, indent=2, ensure_ascii=False))

    return {
        "status": "ok",
        "run_id": result.run_id,
        "run_directory": str(directory),
        "report": str(result.report_path),
        "report_html": str(html_path),
        "handoff": str(handoff_path),
        "figures": [str(path) for path in result.figure_paths],
        "suggested_command": handoff["suggested_command"],
        "next_step": NEXT_STEP,
    }


def suggested_repeat_command(
    directory: str | Path,
    *,
    rounds: int = REPEAT_DEFAULT_ROUNDS,
    points_per_class: int = REPEAT_DEFAULT_POINTS,
) -> str:
    """The one line that replicates this run k times, ready to paste or run."""
    return (
        f'symmetry repeat --run "{Path(directory).expanduser()}" '
        f"--rounds {int(rounds)} --points-per-class {int(points_per_class)} "
        "--no-keep-checkpoint --no-stop-on-alarm"
    )


def _handoff_record(
    directory: Path,
    result,
    html_path: Path,
    *,
    rounds: int,
    points_per_class: int,
) -> dict[str, Any]:
    """Name the run precisely enough that a new conversation needs no guessing."""
    record = _read_json(directory / "run_record.json") or {}
    session = _read_json(directory / "annotation_session.json") or {}
    raw_input = (record.get("input") or {}).get("path")
    image_name = Path(str(raw_input)).name if raw_input else directory.name
    image = session.get("image") or {}
    classes = [
        {
            "name": entry.get("name"),
            "points": len(entry.get("points_xy") or []),
            "color": entry.get("color"),
        }
        for entry in session.get("classes") or []
    ]
    configuration = record.get("configuration") or {}
    return {
        "schema_version": FINISH_SCHEMA_VERSION,
        "run_id": result.run_id,
        "run_directory": str(directory),
        "image": image_name,
        "input": str(directory / "input.npy"),
        "shape_hw": image.get("shape_hw"),
        "annotation_session": str(directory / "annotation_session.json"),
        "classes": classes,
        "classifier_patch_size": (configuration.get("model") or {}).get(
            "classifier_patch_size"
        ),
        "report": str(result.report_path),
        "report_html": str(html_path),
        "figures": [str(path) for path in result.figure_paths],
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "suggested_command": suggested_repeat_command(
            directory, rounds=rounds, points_per_class=points_per_class
        ),
        "next_step": NEXT_STEP,
    }


def render_report_html(
    report_path: str | Path, figures: Sequence[str | Path] = ()
) -> str:
    """Turn the Markdown report into one self-contained HTML document.

    Images are inlined as data URIs so the file keeps working after it is moved
    out of the run directory, mailed, or attached to a note.
    """
    report = Path(report_path).expanduser()
    text = report.read_text(encoding="utf-8")
    body = _markdown_to_html(text, base_dir=report.parent)
    # The report embeds only the figures its template calls for; the rest of the
    # figure set is still part of the run and belongs in the archive copy.
    referenced = {
        Path(target).name
        for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    }
    remaining = [Path(path) for path in figures if Path(path).name not in referenced]
    if remaining:
        body += "\n<h2>Full figure set</h2>\n" + "\n".join(
            _figure_html(path, base_dir=report.parent) for path in remaining
        )
    title = _first_heading(text) or report.stem
    return _HTML_TEMPLATE.format(title=html_module.escape(title), body=body)


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin: 0 auto; max-width: 46rem; padding: 2.5rem 1.5rem 4rem;
    font: 16px/1.6 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1c1c1e; background: #ffffff; }}
  h1 {{ font-size: 1.6rem; line-height: 1.25; margin: 0 0 1.5rem; }}
  h2 {{ font-size: 1.15rem; margin: 2.2rem 0 .6rem; padding-bottom: .3rem;
    border-bottom: 1px solid #e5e5ea; }}
  h3 {{ font-size: 1rem; margin: 1.4rem 0 .4rem; }}
  p, li {{ margin: .5rem 0; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: .88em; background: #f2f2f7; padding: .1em .35em; border-radius: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: .8rem 0; font-size: .9rem; }}
  th, td {{ border: 1px solid #d1d1d6; padding: .35rem .5rem; text-align: left;
    vertical-align: top; }}
  th {{ background: #f7f7fa; }}
  blockquote {{ margin: 1rem 0; padding: .5rem .9rem; border-left: 3px solid #d1d1d6;
    color: #3a3a3c; }}
  figure {{ margin: 1.2rem 0; }}
  figure img {{ width: 100%; height: auto; border: 1px solid #e5e5ea; border-radius: 6px; }}
  figcaption {{ font-size: .82rem; color: #6b6b70; margin-top: .35rem; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def _markdown_to_html(text: str, *, base_dir: Path) -> str:
    """Convert the report's Markdown subset to HTML.

    The report template is fixed and uses a small subset - headings, bullet
    lists, pipe tables, block quotes, inline code, bold, images - so a full
    Markdown dependency would buy nothing and cost an install.
    """
    lines = text.splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    index = 0

    def flush_paragraph() -> None:
        if paragraph:
            out.append("<p>" + _inline(" ".join(paragraph), base_dir) + "</p>")
            paragraph.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            out.append(
                f"<h{level}>{_inline(heading.group(2), base_dir)}</h{level}>"
            )
            index += 1
            continue

        if stripped.startswith("|") and "|" in stripped[1:]:
            flush_paragraph()
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            out.append(_table_html(table_lines, base_dir))
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            items: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("- "):
                items.append(lines[index].strip()[2:])
                index += 1
            out.append(
                "<ul>" + "".join(f"<li>{_inline(item, base_dir)}</li>" for item in items) + "</ul>"
            )
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            quoted: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quoted.append(lines[index].strip().lstrip(">").strip())
                index += 1
            out.append(f"<blockquote>{_inline(' '.join(quoted), base_dir)}</blockquote>")
            continue

        paragraph.append(stripped)
        index += 1

    flush_paragraph()
    return "\n".join(out)


def _table_html(rows: Sequence[str], base_dir: Path) -> str:
    cells = [[cell.strip() for cell in row.strip("|").split("|")] for row in rows]
    if len(cells) > 1 and all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells[1]):
        header, body = cells[0], cells[2:]
    else:
        header, body = None, cells
    parts = ["<table>"]
    if header:
        parts.append(
            "<thead><tr>"
            + "".join(f"<th>{_inline(cell, base_dir)}</th>" for cell in header)
            + "</tr></thead>"
        )
    parts.append("<tbody>")
    for row in body:
        parts.append(
            "<tr>" + "".join(f"<td>{_inline(cell, base_dir)}</td>" for cell in row) + "</tr>"
        )
    parts.append("</tbody></table>")
    return "".join(parts)


def _inline(text: str, base_dir: Path) -> str:
    """Render inline Markdown, inlining images as data URIs."""
    images: dict[str, str] = {}

    def stash(match: re.Match[str]) -> str:
        key = f"\x00IMG{len(images)}\x00"
        images[key] = _figure_html_from_match(match, base_dir)
        return key

    stashed = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", stash, text)
    escaped = html_module.escape(stashed)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)
    for key, markup in images.items():
        escaped = escaped.replace(key, markup)
    return escaped


def _figure_html_from_match(match: re.Match[str], base_dir: Path) -> str:
    caption = match.group(1).strip()
    target = match.group(2).strip()
    return _figure_html(base_dir / target, base_dir=base_dir, caption=caption)


def _figure_html(path: Path, *, base_dir: Path, caption: str = "") -> str:
    candidate = path if path.is_absolute() else (base_dir / path)
    label = html_module.escape(caption or candidate.stem.replace("_", " "))
    data = _data_uri(candidate)
    if data is None:
        return f'<p class="missing figure">[{label}]</p>'
    return (
        "<figure>"
        f'<img src="{data}" alt="{label}">'
        f"<figcaption>{label}</figcaption>"
        "</figure>"
    )


def _data_uri(path: Path) -> str | None:
    try:
        payload = path.read_bytes()
    except OSError:
        return None
    suffix = path.suffix.lower().lstrip(".") or "png"
    if suffix == "jpg":
        suffix = "jpeg"
    return f"data:image/{suffix};base64,{base64.b64encode(payload).decode('ascii')}"


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        match = re.match(r"^#\s+(.*)$", line.strip())
        if match:
            return match.group(1).strip()
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_text_atomic(path: Path, text: str) -> None:
    """Write through a temporary file so a reader never sees half a document.

    The conversation may read the handoff while the interface is writing it;
    an interrupted write must leave the previous copy intact rather than a
    truncated JSON that fails to parse.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
