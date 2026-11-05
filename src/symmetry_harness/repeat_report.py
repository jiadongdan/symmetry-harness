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
SHEET_PANEL = 300
SHEET_GAP = 10
SHEET_MARGIN = 24
SHEET_HEADER = 62
SHEET_ROW_TITLE = 26
SHEET_ROW_GAP = 16
SHEET_COLUMNS = ("support points", "predicted mask", "wrong cells vs truth")
SHEET_MAX_ROWS = 12
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


def _point_groups(entry: dict[str, Any]) -> list[list[Any]]:
    """Return the round's support points grouped by class.

    ``loop_rounds.csv`` stores them under ``points_xy`` while in-memory rows use
    ``points_by_class``; both may arrive as a JSON string after a resume.
    """
    raw = entry.get("points_by_class")
    if raw in (None, ""):
        raw = entry.get("points_xy")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list) or not raw:
        return []
    first = raw[0]
    if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple)):
        return [list(group) for group in raw]
    return [list(raw)]


def _evenly(values: Sequence[Any], limit: int) -> list[Any]:
    """Keep at most ``limit`` entries, always including the first and the last."""
    if len(values) <= limit or limit < 2:
        return list(values)
    picks = {round(position * (len(values) - 1) / (limit - 1)) for position in range(limit)}
    return [values[position] for position in sorted(picks)]


def _nearest_resize(grid: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Upscale a categorical grid by nearest neighbour so labels never blend."""
    return np.asarray(
        Image.fromarray(np.asarray(grid, dtype=np.uint8)).resize(size, Image.NEAREST)
    )


def _marker(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    color: tuple[int, int, int],
    *,
    radius: int = 6,
    patch_half: float | None = None,
) -> None:
    if patch_half is not None:
        draw.rectangle(
            [x - patch_half, y - patch_half, x + patch_half, y + patch_half],
            outline=color,
            width=1,
        )
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(255, 255, 255))
    inner = radius - 2
    draw.ellipse(
        [x - inner, y - inner, x + inner, y + inner],
        fill=color,
    )


def _round_sheet(
    path: Path,
    image: np.ndarray,
    manifest: dict[str, Any],
    history: Sequence[dict[str, Any]],
    loop_dir: Path,
) -> Path | None:
    """Resolve the ground truth and run options, then draw the round sheet."""
    from .repeat import DEFAULT_CLASS_COLORS, class_map_from_labels, load_label_map

    ground_truth = manifest.get("ground_truth") or {}
    run_options = manifest.get("run_options") or {}
    codes = [int(code) for code in (ground_truth.get("label_codes") or [])]
    class_map = None
    label_path = ground_truth.get("path")
    if label_path and len(codes) == 2 and Path(label_path).is_file():
        try:
            class_map = class_map_from_labels(load_label_map(label_path), codes)
        except Exception:  # noqa: BLE001 - the sheet is display-only
            class_map = None
    return build_round_sheet(
        path,
        image,
        class_map,
        history,
        rounds_dir=loop_dir / "rounds",
        class_colors=list(DEFAULT_CLASS_COLORS),
        patch_size=int(run_options.get("classifier_patch_size") or 64),
        stride=int(run_options.get("stride") or 4),
    )


def build_round_sheet(
    path: Path,
    image: np.ndarray,
    class_map: np.ndarray | None,
    rounds: Sequence[dict[str, Any]],
    *,
    rounds_dir: Path,
    class_colors: Sequence[str],
    patch_size: int,
    stride: int,
) -> Path | None:
    """Draw one row per round: support points, predicted mask, and wrong cells.

    The sheet answers the two questions a stability loop raises at a glance:
    where did each round put its support points, and what mask did that round
    produce. The third column marks cells that disagree with the ground truth,
    so a round that drifts shows up as a red band rather than a number.
    """
    from .analysis import dense_prediction_from_arrays, prediction_display_bounds

    palette = np.asarray([ImageColor.getrgb(color) for color in class_colors], dtype=np.uint8)
    base = np.repeat(unit_to_uint8(image)[..., None], 3, axis=2).astype(np.uint8)
    height, width = base.shape[:2]
    panel_h = max(1, int(round(SHEET_PANEL * height / max(width, 1))))
    scale = SHEET_PANEL / max(width, 1)

    rendered: list[tuple[dict[str, Any], Image.Image, Image.Image, Image.Image]] = []
    for entry in _evenly(list(rounds), SHEET_MAX_ROWS):
        index = int(float(entry.get("round", len(rendered))))
        source = rounds_dir / f"round-{index:03d}" / "kept" / "prediction.npz"
        if not source.is_file():
            continue
        try:
            with np.load(source, allow_pickle=False) as payload:
                arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
            prediction = dense_prediction_from_arrays(arrays)
            left, top, right, bottom = prediction_display_bounds(
                prediction, (height, width), stride=stride
            )
            classes = _nearest_resize(prediction.prediction_grid, (right - left, bottom - top))
        except Exception:  # noqa: BLE001 - the sheet is display-only
            continue

        colors = palette[classes]
        mask_rgb = np.full((height, width, 3), 244, dtype=np.uint8)
        mask_rgb[top:bottom, left:right] = colors
        # The error panel drops the class colours on purpose: a wrong cell painted
        # in the class colour would be invisible against that same class.
        wrong_rgb = np.full((height, width, 3), 238, dtype=np.uint8)
        if class_map is not None and class_map.shape == (height, width):
            truth = class_map[top:bottom, left:right]
            wrong_rgb[top:bottom, left:right] = (232, 232, 232)
            wrong_rgb[top:bottom, left:right][classes != truth] = (214, 39, 40)

        raw_panel = Image.fromarray(base).resize((SHEET_PANEL, panel_h), Image.LANCZOS)
        mask_panel = Image.fromarray(mask_rgb).resize((SHEET_PANEL, panel_h), Image.NEAREST)
        wrong_panel = Image.fromarray(wrong_rgb).resize((SHEET_PANEL, panel_h), Image.NEAREST)

        groups = _point_groups(entry)
        for target, hollow, boxes in (
            (raw_panel, False, True),
            (mask_panel, True, False),
        ):
            draw = ImageDraw.Draw(target)
            for class_index, group in enumerate(groups):
                color = tuple(int(channel) for channel in palette[class_index % len(palette)])
                for point in group:
                    _marker(
                        draw,
                        float(point[0]) * scale,
                        float(point[1]) * scale,
                        color,
                        radius=4 if hollow else 6,
                        patch_half=None if not boxes else patch_size * scale / 2.0,
                    )
        rendered.append((entry, raw_panel, mask_panel, wrong_panel))

    if not rendered:
        return None

    sheet_w = SHEET_MARGIN * 2 + 3 * SHEET_PANEL + 2 * SHEET_GAP
    sheet_h = SHEET_HEADER + len(rendered) * (SHEET_ROW_TITLE + panel_h + SHEET_ROW_GAP)
    canvas = Image.new("RGB", (sheet_w, sheet_h), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((SHEET_MARGIN, 16), "Support points and predicted mask, one row per round", font=_font(17), fill=INK)
    for column, label in enumerate(SHEET_COLUMNS):
        x = SHEET_MARGIN + column * (SHEET_PANEL + SHEET_GAP)
        draw.text((x, SHEET_HEADER - 20), label, font=_font(12), fill=MUTED)

    y = SHEET_HEADER
    for entry, raw_panel, mask_panel, wrong_panel in rendered:
        index = int(float(entry.get("round", 0)))
        counts = [len(group) for group in _point_groups(entry)]
        detail = " + ".join(str(count) for count in counts) if counts else "?"
        try:
            accuracy = f"{float(entry.get('acc')):.2%}"
        except (TypeError, ValueError):
            accuracy = "n/a"
        draw.text(
            (SHEET_MARGIN, y + 4),
            f"round {index}   accuracy {accuracy}   points {detail}",
            font=_font(13),
            fill=INK,
        )
        for column, panel in enumerate((raw_panel, mask_panel, wrong_panel)):
            x = SHEET_MARGIN + column * (SHEET_PANEL + SHEET_GAP)
            canvas.paste(panel, (x, y + SHEET_ROW_TITLE))
        y += SHEET_ROW_TITLE + panel_h + SHEET_ROW_GAP

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
    """Write ``loop_summary.md`` plus the loop figures."""
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
            groups = _point_groups(entry)
            if not groups:
                continue
            flat = [tuple(point) for group in groups for point in group]
            points_by_round.append((int(float(entry.get("round", 0))), flat))
        if image is not None and points_by_round:
            points_path = build_points_figure(
                figures_dir / "points_map.png", image, points_by_round
            )
            figures.append(("Every sampled support point, coloured by round", points_path))
            sheet_path = _round_sheet(
                figures_dir / "round_sheet.png", image, manifest, history, loop_dir
            )
            if sheet_path is not None:
                figures.append(
                    (
                        "One row per round: where the support points landed, the mask "
                        "that round produced, and the cells that disagree with the truth",
                        sheet_path,
                    )
                )

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
        "- Minimum support-point spacing: `"
        + (
            "auto (one classifier patch)"
            if options.min_point_distance is None
            else f"{options.min_point_distance:g} px"
        )
        + "`",
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
        "| round | acc | class A acc | class B acc | IoU A | IoU B | pred area | agree prev | support acc | spacing px | seed | seconds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in history:
        lines.append(
            "| {round} | {acc} | {a} | {b} | {iou_a} | {iou_b} | {area} | {agree} | {support} | {spread} | {seed} | {seconds} |".format(
                round=entry.get("round", ""),
                acc=_percent(entry.get("acc")),
                a=_percent(entry.get("acc_class_a")),
                b=_percent(entry.get("acc_class_b")),
                iou_a=_number(entry.get("iou_class_a"), 3),
                iou_b=_number(entry.get("iou_class_b"), 3),
                area=_percent(entry.get("area_frac_pred")),
                agree=_percent(entry.get("agree_prev")),
                support=_percent(entry.get("support_acc")),
                spread=_number(entry.get("spread"), 1),
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
        "- Within one round the support points keep a minimum distance (Poisson-disk",
        "  spacing, across both classes). The `spacing px` column is what actually",
        "  fit: a value below the requested spacing means the pool was too small for",
        "  that separation, and the round is slightly less spread than the protocol.",
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
