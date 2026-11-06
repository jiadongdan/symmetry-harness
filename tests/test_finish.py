"""Tests for closing a run out and handing it to the conversation."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from symmetry_harness.finish import (
    HANDOFF_NAME,
    REPORT_HTML_NAME,
    finish_run,
    render_report_html,
    suggested_repeat_command,
)

# A 1x1 PNG, enough for the inliner to have real bytes to encode.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _report(tmp_path: Path, *, reference: str = "fig.png") -> Path:
    report = tmp_path / "report_summary.md"
    report.write_text(
        "\n".join(
            [
                "# Symmetry Analysis Run Report",
                "",
                "**Run:** `symmetry-test` | **Status:** `completed`",
                "",
                "## 1. Result at a Glance",
                "",
                "- Class A IoU: `0.91`",
                "- Class B IoU: `0.88`",
                "",
                "| metric | value |",
                "| --- | --- |",
                "| accuracy | `97.2%` |",
                "",
                f"![Support points]({reference})",
                "",
                "> Support points sit inside each phase.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report


def test_report_html_inlines_figures_and_keeps_structure(tmp_path: Path) -> None:
    report = _report(tmp_path)
    (tmp_path / "fig.png").write_bytes(PNG_BYTES)

    markup = render_report_html(report, [tmp_path / "fig.png"])

    assert markup.startswith("<!doctype html>")
    assert "<h1>Symmetry Analysis Run Report</h1>" in markup
    assert "<h2>1. Result at a Glance</h2>" in markup
    assert "<ul>" in markup and "<li>" in markup
    assert "<table>" in markup and "<th>metric</th>" in markup
    assert "<blockquote>" in markup
    assert "data:image/png;base64," in markup
    # The referenced figure is embedded once, not duplicated by the appendix.
    assert markup.count("<img ") == 1
    assert "Full figure set" not in markup


def test_report_html_appends_figures_the_report_never_mentions(tmp_path: Path) -> None:
    report = _report(tmp_path)
    (tmp_path / "fig.png").write_bytes(PNG_BYTES)
    extra = tmp_path / "02_support_points.png"
    extra.write_bytes(PNG_BYTES)

    markup = render_report_html(report, [tmp_path / "fig.png", extra])

    assert "Full figure set" in markup
    assert markup.count("<img ") == 2
    assert markup.count("data:image/png;base64,") == 2


def test_suggested_command_names_the_run_exactly(tmp_path: Path) -> None:
    command = suggested_repeat_command(tmp_path, rounds=20, points_per_class=5)

    assert command.startswith("symmetry repeat --run ")
    assert f'"{tmp_path}"' in command
    assert "--rounds 20" in command
    assert "--points-per-class 5" in command
    assert "--no-stop-on-alarm" in command


def test_finish_run_writes_a_handoff_and_a_standalone_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "symmetry-20260101T000000Z-abcdef"
    run_dir.mkdir()
    np.save(run_dir / "input.npy", np.zeros((32, 32), dtype=np.float32))
    figure = run_dir / "figures" / "01_input.png"
    figure.parent.mkdir()
    figure.write_bytes(PNG_BYTES)
    report = _report(run_dir)
    (run_dir / "annotation_session.json").write_text(
        json.dumps(
            {
                "image": {"shape_hw": [32, 32], "path": str(run_dir / "input.npy")},
                "classes": [
                    {"name": "Class A", "color": "#e41a1c", "points_xy": [[4, 4], [8, 8]]},
                    {"name": "Class B", "color": "#377eb8", "points_xy": [[20, 20]]},
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_record.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "input": {"path": "D:/data/example_image.npy"},
                "configuration": {"model": {"classifier_patch_size": 64}},
            }
        ),
        encoding="utf-8",
    )

    class _Result:
        run_id = run_dir.name
        report_path = report
        figure_paths = (figure,)

    monkeypatch.setattr(
        "symmetry_harness.run_report.build_run_report",
        lambda *args, **kwargs: _Result(),
    )

    outcome = finish_run(run_dir)

    assert outcome["status"] == "ok"
    handoff_path = run_dir / HANDOFF_NAME
    html_path = run_dir / REPORT_HTML_NAME
    assert handoff_path.is_file() and html_path.is_file()
    # The write is atomic: no temporary file survives a completed call.
    assert not list(run_dir.glob("*.tmp"))

    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    assert handoff["run_id"] == run_dir.name
    assert handoff["image"] == "example_image.npy"
    assert handoff["shape_hw"] == [32, 32]
    assert [entry["points"] for entry in handoff["classes"]] == [2, 1]
    assert handoff["classifier_patch_size"] == 64
    assert handoff["report_html"] == str(html_path)
    # The handoff names the next step but never claims the assistant was told.
    assert "--run" in handoff["suggested_command"]
    assert "ask the assistant" in handoff["next_step"]
    assert "repeat_requested" not in handoff

    assert "data:image/png;base64," in html_path.read_text(encoding="utf-8")


def test_finish_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "symmetry-20260101T000000Z-abcdef"
    run_dir.mkdir()
    (run_dir / "input.npy").write_bytes(b"not-an-image")
    figure = run_dir / "fig.png"
    figure.write_bytes(PNG_BYTES)
    report = _report(run_dir)

    class _Result:
        run_id = run_dir.name
        report_path = report
        figure_paths = (figure,)

    monkeypatch.setattr(
        "symmetry_harness.run_report.build_run_report",
        lambda *args, **kwargs: _Result(),
    )

    first = finish_run(run_dir)
    second = finish_run(run_dir)

    assert first["report_html"] == second["report_html"]
    assert (run_dir / HANDOFF_NAME).is_file()
    assert not list(run_dir.glob("*.tmp"))
