"""Markdown summary and figures for a repeat loop.

The charts are drawn with Pillow, matching the rest of the Harness: the package
depends on ``numpy`` and ``Pillow`` only, so no plotting library is introduced
for four small line charts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from .image_io import unit_to_uint8

BACKGROUND = (255, 255, 255)
INK = (28, 32, 38)
MUTED = (120, 128, 138)
GRID = (226, 230, 235)
BAND = (232, 238, 246)
SERIES = ((214, 39, 40), (31, 119, 180), (44, 160, 44))
CHART_SIZE = (960, 520)
PAD_LEFT = 84
PAD_RIGHT = 28
PAD_TOP = 56
PAD_BOTTOM = 64


def _font(size: int = 13) -> Any:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - Pillow < 10.1
        return ImageFont.load_default()


def _text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, *, size: int = 13, fill: tuple[int, int, int] = INK) -> None:
    draw.text(xy, value, font=_font(size), fill=fill)


def _finite(values: Sequence[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            result.append(float("nan"))
            continue
        result.append(number if np.isfinite(number) else float("nan"))
    return result


def _line_chart(
    path: Path,
    *,
    title: str,
    x_values: Sequence[int],
    series: Sequence[tuple[str, Sequence[float], tuple[int, int, int]]],
    y_range: tuple[float, float] | None = None,
    horizontal: Sequence[tuple[float, str, tuple[int, int, int]]] = (),
    band: tuple[Sequence[float], Sequence[float]] | None = None,
) -> Path:
    """Render one line chart with an optional shaded band and reference lines."""
    width, height = CHART_SIZE
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    plot_left, plot_top = PAD_LEFT, PAD_TOP
    plot_right, plot_bottom = width - PAD_RIGHT, height - PAD_BOTTOM

    cleaned = [_finite(values) for _, values, _ in series]
    candidates = [value for values in cleaned for value in values if np.isfinite(value)]
    for value, _, _ in horizontal:
        if np.isfinite(value):
            candidates.append(float(value))
    if y_range is None:
        if candidates:
            low, high = min(candidates), max(candidates)
        else:
            low, high = 0.0, 1.0
        span = max(high - low, 1e-6)
        low -= span * 0.12
        high += span * 0.12
    else:
        low, high = y_range
    if high <= low:
        high = low + 1e-6

    x_min = min(x_values) if x_values else 0
    x_max = max(x_values) if x_values else 1
    x_span = max(x_max - x_min, 1)

    def to_px(x: float, y: float) -> tuple[float, float]:
        px = plot_left + (x - x_min) / x_span * (plot_right - plot_left)
        py = plot_bottom - (y - low) / (high - low) * (plot_bottom - plot_top)
        return px, min(max(py, float(plot_top)), float(plot_bottom))

    # Grid and axis labels.
    for step in range(5):
        value = low + (high - low) * step / 4
        py = to_px(x_min, value)[1]
        draw.line([(plot_left, py), (plot_right, py)], fill=GRID, width=1)
        _text(draw, (12, py - 8), f"{value:.3f}", size=12, fill=MUTED)
    tick_count = min(10, max(2, len(x_values)))
    seen: set[int] = set()
    for index in range(tick_count):
        if not x_values:
            break
        position = round(index * (len(x_values) - 1) / max(tick_count - 1, 1))
        position = min(position, len(x_values) - 1)
        if position in seen:
            continue
        seen.add(position)
        px = to_px(x_values[position], low)[0]
        _text(draw, (px - 8, plot_bottom + 8), str(x_values[position]), size=12, fill=MUTED)
    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill=INK, width=1)
    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill=INK, width=1)

    if band is not None:
        lower, upper = (_finite(values) for values in band)
        points: list[tuple[float, float]] = []
        for x, value in zip(x_values, upper):
            if np.isfinite(value):
                points.append(to_px(x, value))
        for x, value in zip(reversed(list(x_values)), reversed(lower)):
            if np.isfinite(value):
                points.append(to_px(x, value))
        if len(points) >= 3:
            draw.polygon(points, fill=BAND)

    for value, label, color in horizontal:
        if not np.isfinite(value):
            continue
        py = to_px(x_min, value)[1]
        draw.line([(plot_left, py), (plot_right, py)], fill=color, width=1)
        _text(draw, (plot_right - 190, py - 18), label, size=12, fill=color)

    for (label, values, color), cleaned_values in zip(series, cleaned):
        points = [
            to_px(x, value)
            for x, value in zip(x_values, cleaned_values)
            if np.isfinite(value)
        ]
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
        for px, py in points:
            draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=color)

    legend_y = plot_top - 26
    legend_x = plot_right - 12
    for label, _, color in reversed(list(series)):
        text_width = draw.textlength(label, font=_font(12))
        legend_x -= int(text_width) + 26
        draw.line([(legend_x, legend_y + 6), (legend_x + 14, legend_y + 6)], fill=color, width=2)
        _text(draw, (legend_x + 18, legend_y - 1), label, size=12, fill=INK)

    _text(draw, (PAD_LEFT, 16), title, size=17)

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def build_accuracy_figure(
    path: Path, rounds: Sequence[int], accuracies: Sequence[float], *, floor: float, sigma: float
) -> Path:
    values = _finite(accuracies)
    finite = [value for value in values if np.isfinite(value)]
    mean = float(np.mean(finite)) if finite else float("nan")
    deviation = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    # The shaded band is one standard deviation: it shows how much consecutive
    # rounds actually move. The alarm bound (mean - sigma x std) is drawn as its
    # own reference line, because a filled band out to it swallows the plot.
    band = (
        [mean - deviation for _ in values],
        [mean + deviation for _ in values],
    )
    return _line_chart(
        path,
        title="Accuracy per repeated round",
        x_values=list(rounds),
        series=[("accuracy", values, SERIES[0])],
        horizontal=[
            (floor, f"floor {floor:.2f}", MUTED),
            (mean, f"mean {mean:.3f}" if np.isfinite(mean) else "mean", SERIES[1]),
            (
                mean - sigma * deviation,
                f"alarm {mean - sigma * deviation:.3f}",
                SERIES[1],
            ),
        ],
        band=band,
    )


def build_iou_figure(path: Path, rounds: Sequence[int], iou_a: Sequence[float], iou_b: Sequence[float]) -> Path:
    return _line_chart(
        path,
        title="Per-class IoU",
        x_values=list(rounds),
        series=[
            ("class A", _finite(iou_a), SERIES[0]),
            ("class B", _finite(iou_b), SERIES[1]),
        ],
        y_range=(0.0, 1.0),
    )


def build_area_figure(
    path: Path, rounds: Sequence[int], predicted: Sequence[float], truth: Sequence[float]
) -> Path:
    return _line_chart(
        path,
        title="Predicted vs ground-truth class-B area fraction",
        x_values=list(rounds),
        series=[
            ("predicted", _finite(predicted), SERIES[2]),
            ("truth", _finite(truth), SERIES[1]),
        ],
        y_range=(0.0, 1.0),
    )


def build_points_figure(
    path: Path,
    image: np.ndarray,
    points_by_round: Sequence[tuple[int, Sequence[Sequence[int]]]],
) -> Path:
    """Draw every sampled support point on the source image, coloured by round."""
    base = np.repeat(unit_to_uint8(image)[..., None], 3, axis=2)
    canvas = Image.fromarray(base.astype(np.uint8))
    scale = max(1, int(round(720 / max(canvas.size))))
    if scale > 1:
        canvas = canvas.resize((canvas.width * scale, canvas.height * scale), Image.NEAREST)
    draw = ImageDraw.Draw(canvas)
    palette = [
        ImageColor.getrgb("#e41a1c"),
        ImageColor.getrgb("#377eb8"),
        ImageColor.getrgb("#4daf4a"),
        ImageColor.getrgb("#ff7f00"),
        ImageColor.getrgb("#984ea3"),
        ImageColor.getrgb("#a65628"),
    ]
    for round_index, points in points_by_round:
        color = palette[round_index % len(palette)]
        for x, y in points:
            px = float(x) * scale
            py = float(y) * scale
            radius = 5
            draw.ellipse(
                [px - radius, py - radius, px + radius, py + radius],
                outline=color,
                width=2,
            )
            draw.ellipse([px - 1, py - 1, px + 1, py + 1], fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def _percent(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number * 100:.{digits}f}%"


def _number(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def write_loop_summary(
    loop_dir: Path,
    *,
    loop_id: str,
    manifest: dict[str, Any],
    options,
    history: Sequence[dict[str, Any]],
    status: str,
    alarm: str | None = None,
) -> Path:
    """Write ``loop_summary.md`` plus the four loop figures."""
    figures_dir = loop_dir / "figures"
    rounds = [int(float(entry.get("round", index))) for index, entry in enumerate(history)]
    accuracies = _finite([entry.get("acc", "nan") for entry in history])
    iou_a = _finite([entry.get("iou_class_a", "nan") for entry in history])
    iou_b = _finite([entry.get("iou_class_b", "nan") for entry in history])
    area_pred = _finite([entry.get("area_frac_pred", "nan") for entry in history])
    area_gt = _finite([entry.get("area_frac_gt", "nan") for entry in history])

    figures: list[tuple[str, Path]] = []
    if rounds:
        figures.append(
            (
                "Accuracy per round, with the historical mean ± sigma band",
                build_accuracy_figure(
                    figures_dir / "acc_vs_round.png",
                    rounds,
                    accuracies,
                    floor=float(options.alarm_floor),
                    sigma=float(options.alarm_drop),
                ),
            )
        )
        figures.append(
            (
                "Per-class IoU",
                build_iou_figure(figures_dir / "iou_vs_round.png", rounds, iou_a, iou_b),
            )
        )
        figures.append(
            (
                "Class-B area fraction: prediction against ground truth",
                build_area_figure(figures_dir / "area_frac_vs_round.png", rounds, area_pred, area_gt),
            )
        )

    points_path = None
    input_path = (manifest.get("input") or {}).get("path")
    if input_path and Path(input_path).is_file():
        from .image_io import inspect_input

        try:
            image, _ = inspect_input(input_path, "minmax_0_1")
        except Exception:  # noqa: BLE001 - the map is display-only
            image = None
        points_by_round: list[tuple[int, list[tuple[int, int]]]] = []
        for entry in history:
            raw = entry.get("points_by_class")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    raw = None
            if not raw:
                continue
            flat = [tuple(point) for group in raw for point in group]
            points_by_round.append((int(float(entry.get("round", 0))), flat))
        if image is not None and points_by_round:
            points_path = build_points_figure(
                figures_dir / "points_map.png", image, points_by_round
            )
            figures.append(("Every sampled support point, coloured by round", points_path))

    finite = [value for value in accuracies if np.isfinite(value)]
    mean = float(np.mean(finite)) if finite else float("nan")
    deviation = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    minimum = float(np.min(finite)) if finite else float("nan")
    maximum = float(np.max(finite)) if finite else float("nan")

    lines: list[str] = [
        f"# Repeat Loop Summary: `{loop_id}`",
        "",
        f"- Status: `{status}`",
        f"- Rounds completed: `{len(history)}` of `{int(options.rounds) + 1}` (round 0 is the seed round)",
        f"- Points per class per round: `{int(options.points_per_class)}`",
        f"- Confidence threshold `tau`: `{options.tau}`",
        f"- Training seed: `{options.base_seed} + round_index`",
        f"- Alarm: floor `{options.alarm_floor}`, band `mean - {options.alarm_drop:g} sigma`",
        "",
    ]
    if alarm:
        lines += ["## Alarm", "", f"> {alarm}", ""]
    lines += [
        "## Accuracy Distribution",
        "",
        f"- Mean: `{_percent(mean)}`",
        f"- Standard deviation (sample): `{_percent(deviation)}`",
        f"- Minimum / maximum: `{_percent(minimum)}` / `{_percent(maximum)}`",
        f"- Spread: `{_percent(maximum - minimum) if finite else 'n/a'}`",
        "",
        "Every round trains a fresh adapter from the backbone with the same number of",
        "points, so the spread above measures procedural reproducibility, not",
        "improvement. The support set never accumulates.",
        "",
        "## Rounds",
        "",
        "| round | acc | class A acc | class B acc | IoU A | IoU B | pred area | agree prev | support acc | seed | seconds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in history:
        lines.append(
            "| {round} | {acc} | {a} | {b} | {iou_a} | {iou_b} | {area} | {agree} | {support} | {seed} | {seconds} |".format(
                round=entry.get("round", ""),
                acc=_percent(entry.get("acc")),
                a=_percent(entry.get("acc_class_a")),
                b=_percent(entry.get("acc_class_b")),
                iou_a=_number(entry.get("iou_class_a"), 3),
                iou_b=_number(entry.get("iou_class_b"), 3),
                area=_percent(entry.get("area_frac_pred")),
                agree=_percent(entry.get("agree_prev")),
                support=_percent(entry.get("support_acc")),
                seed=entry.get("train_seed", ""),
                seconds=entry.get("seconds", ""),
            )
        )
    lines += ["", "## Figures", ""]
    for caption, figure_path in figures:
        relative = figure_path.relative_to(loop_dir).as_posix()
        lines += [f"**{caption}**", "", f"![{caption}]({relative})", ""]
    if points_path is None:
        lines += ["_The source image is unavailable, so the point map was not rendered._", ""]

    lines += [
        "## Reading This Loop",
        "",
        "- A round is an independent replicate: same backbone, same patch size, same",
        "  epoch count, fresh support points drawn only from cells whose accumulated",
        "  confidence is at least `tau`.",
        "- `agree prev` is informational. It measures how much the map moved between two",
        "  rounds; it is not part of the stop rule.",
        "- Sampling stays inside the confident region by design, so this loop measures",
        "  reproducibility under one sampling rule. It does not measure robustness to a",
        "  human who clicks near a phase boundary.",
        "",
    ]
    summary_path = loop_dir / "loop_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path
