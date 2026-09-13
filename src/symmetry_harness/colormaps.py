"""Deterministic 256-entry RGB lookup tables for fixed scalar-map rendering.

The Harness renders confidence and entropy maps against fixed, interpretable
value ranges without importing Matplotlib, PyTorch, or ``symmetry-learn``. This
module embeds two canonical ``256 x 3`` ``uint8`` lookup tables as literal data,
sampled once from Matplotlib's canonical colormaps and frozen here. The tables
are pure data; Matplotlib is never imported at runtime.

Only ``numpy`` is imported. :func:`lut_index` fixes the shared indexing contract
``index = clip(rint(normalized * 255), 0, 255)`` so that ``0.0 -> 0``,
``1.0 -> 255`` and ``0.5 -> 128``.

``legacy_rainbow``
    Deliberately has **no** lookup table. It is the historical red/green/blue
    ramp that :func:`analysis.render_scalar_map` used before this revision, and
    it is retained only so the standalone traditional-ML validation page (an
    explicit non-goal of this revision) keeps producing byte-identical images.
    Because quantizing the normalized value to an 8-bit index shifts pixels by
    up to one unit, a 256-entry table can never reproduce the historical bytes.
    The ramp is therefore rendered through a direct arithmetic path
    (:func:`map_scalar_to_rgb` / :func:`map_normalized_to_rgb` with
    ``colormap="legacy_rainbow"``). :func:`lut` rejects ``"legacy_rainbow"``
    with a loud error rather than exposing a subtly wrong table.
"""

from __future__ import annotations

import numpy as np


LUT_SIZE: int = 256

VIRIDIS_NAME: str = "viridis"
MAGMA_NAME: str = "magma"
LEGACY_RAINBOW_NAME: str = "legacy_rainbow"

#: Colormaps that are backed by an embedded lookup table.
LUT_COLORMAP_NAMES: tuple[str, ...] = (
    VIRIDIS_NAME,
    MAGMA_NAME,
)

__all__ = (
    "LUT_SIZE",
    "VIRIDIS_NAME",
    "MAGMA_NAME",
    "LEGACY_RAINBOW_NAME",
    "LUT_COLORMAP_NAMES",
    "lut_index",
    "lut",
    "viridis_lut",
    "magma_lut",
    "map_normalized_to_rgb",
    "map_scalar_to_rgb",
)


_VIRIDIS: tuple[tuple[int, int, int], ...] = (
    (68, 1, 84),
    (68, 2, 86),
    (69, 4, 87),
    (69, 5, 89),
    (70, 7, 90),
    (70, 8, 92),
    (70, 10, 93),
    (70, 11, 94),
    (71, 13, 96),
    (71, 14, 97),
    (71, 16, 99),
    (71, 17, 100),
    (71, 19, 101),
    (72, 20, 103),
    (72, 22, 104),
    (72, 23, 105),
    (72, 24, 106),
    (72, 26, 108),
    (72, 27, 109),
    (72, 28, 110),
    (72, 29, 111),
    (72, 31, 112),
    (72, 32, 113),
    (72, 33, 115),
    (72, 35, 116),
    (72, 36, 117),
    (72, 37, 118),
    (72, 38, 119),
    (72, 40, 120),
    (72, 41, 121),
    (71, 42, 122),
    (71, 44, 122),
    (71, 45, 123),
    (71, 46, 124),
    (71, 47, 125),
    (70, 48, 126),
    (70, 50, 126),
    (70, 51, 127),
    (70, 52, 128),
    (69, 53, 129),
    (69, 55, 129),
    (69, 56, 130),
    (68, 57, 131),
    (68, 58, 131),
    (68, 59, 132),
    (67, 61, 132),
    (67, 62, 133),
    (66, 63, 133),
    (66, 64, 134),
    (66, 65, 134),
    (65, 66, 135),
    (65, 68, 135),
    (64, 69, 136),
    (64, 70, 136),
    (63, 71, 136),
    (63, 72, 137),
    (62, 73, 137),
    (62, 74, 137),
    (62, 76, 138),
    (61, 77, 138),
    (61, 78, 138),
    (60, 79, 138),
    (60, 80, 139),
    (59, 81, 139),
    (59, 82, 139),
    (58, 83, 139),
    (58, 84, 140),
    (57, 85, 140),
    (57, 86, 140),
    (56, 88, 140),
    (56, 89, 140),
    (55, 90, 140),
    (55, 91, 141),
    (54, 92, 141),
    (54, 93, 141),
    (53, 94, 141),
    (53, 95, 141),
    (52, 96, 141),
    (52, 97, 141),
    (51, 98, 141),
    (51, 99, 141),
    (50, 100, 142),
    (50, 101, 142),
    (49, 102, 142),
    (49, 103, 142),
    (49, 104, 142),
    (48, 105, 142),
    (48, 106, 142),
    (47, 107, 142),
    (47, 108, 142),
    (46, 109, 142),
    (46, 110, 142),
    (46, 111, 142),
    (45, 112, 142),
    (45, 113, 142),
    (44, 113, 142),
    (44, 114, 142),
    (44, 115, 142),
    (43, 116, 142),
    (43, 117, 142),
    (42, 118, 142),
    (42, 119, 142),
    (42, 120, 142),
    (41, 121, 142),
    (41, 122, 142),
    (41, 123, 142),
    (40, 124, 142),
    (40, 125, 142),
    (39, 126, 142),
    (39, 127, 142),
    (39, 128, 142),
    (38, 129, 142),
    (38, 130, 142),
    (38, 130, 142),
    (37, 131, 142),
    (37, 132, 142),
    (37, 133, 142),
    (36, 134, 142),
    (36, 135, 142),
    (35, 136, 142),
    (35, 137, 142),
    (35, 138, 141),
    (34, 139, 141),
    (34, 140, 141),
    (34, 141, 141),
    (33, 142, 141),
    (33, 143, 141),
    (33, 144, 141),
    (33, 145, 140),
    (32, 146, 140),
    (32, 146, 140),
    (32, 147, 140),
    (31, 148, 140),
    (31, 149, 139),
    (31, 150, 139),
    (31, 151, 139),
    (31, 152, 139),
    (31, 153, 138),
    (31, 154, 138),
    (30, 155, 138),
    (30, 156, 137),
    (30, 157, 137),
    (31, 158, 137),
    (31, 159, 136),
    (31, 160, 136),
    (31, 161, 136),
    (31, 161, 135),
    (31, 162, 135),
    (32, 163, 134),
    (32, 164, 134),
    (33, 165, 133),
    (33, 166, 133),
    (34, 167, 133),
    (34, 168, 132),
    (35, 169, 131),
    (36, 170, 131),
    (37, 171, 130),
    (37, 172, 130),
    (38, 173, 129),
    (39, 173, 129),
    (40, 174, 128),
    (41, 175, 127),
    (42, 176, 127),
    (44, 177, 126),
    (45, 178, 125),
    (46, 179, 124),
    (47, 180, 124),
    (49, 181, 123),
    (50, 182, 122),
    (52, 182, 121),
    (53, 183, 121),
    (55, 184, 120),
    (56, 185, 119),
    (58, 186, 118),
    (59, 187, 117),
    (61, 188, 116),
    (63, 188, 115),
    (64, 189, 114),
    (66, 190, 113),
    (68, 191, 112),
    (70, 192, 111),
    (72, 193, 110),
    (74, 193, 109),
    (76, 194, 108),
    (78, 195, 107),
    (80, 196, 106),
    (82, 197, 105),
    (84, 197, 104),
    (86, 198, 103),
    (88, 199, 101),
    (90, 200, 100),
    (92, 200, 99),
    (94, 201, 98),
    (96, 202, 96),
    (99, 203, 95),
    (101, 203, 94),
    (103, 204, 92),
    (105, 205, 91),
    (108, 205, 90),
    (110, 206, 88),
    (112, 207, 87),
    (115, 208, 86),
    (117, 208, 84),
    (119, 209, 83),
    (122, 209, 81),
    (124, 210, 80),
    (127, 211, 78),
    (129, 211, 77),
    (132, 212, 75),
    (134, 213, 73),
    (137, 213, 72),
    (139, 214, 70),
    (142, 214, 69),
    (144, 215, 67),
    (147, 215, 65),
    (149, 216, 64),
    (152, 216, 62),
    (155, 217, 60),
    (157, 217, 59),
    (160, 218, 57),
    (162, 218, 55),
    (165, 219, 54),
    (168, 219, 52),
    (170, 220, 50),
    (173, 220, 48),
    (176, 221, 47),
    (178, 221, 45),
    (181, 222, 43),
    (184, 222, 41),
    (186, 222, 40),
    (189, 223, 38),
    (192, 223, 37),
    (194, 223, 35),
    (197, 224, 33),
    (200, 224, 32),
    (202, 225, 31),
    (205, 225, 29),
    (208, 225, 28),
    (210, 226, 27),
    (213, 226, 26),
    (216, 226, 25),
    (218, 227, 25),
    (221, 227, 24),
    (223, 227, 24),
    (226, 228, 24),
    (229, 228, 25),
    (231, 228, 25),
    (234, 229, 26),
    (236, 229, 27),
    (239, 229, 28),
    (241, 229, 29),
    (244, 230, 30),
    (246, 230, 32),
    (248, 230, 33),
    (251, 231, 35),
    (253, 231, 37),
)

_MAGMA: tuple[tuple[int, int, int], ...] = (
    (0, 0, 4),
    (1, 0, 5),
    (1, 1, 6),
    (1, 1, 8),
    (2, 1, 9),
    (2, 2, 11),
    (2, 2, 13),
    (3, 3, 15),
    (3, 3, 18),
    (4, 4, 20),
    (5, 4, 22),
    (6, 5, 24),
    (6, 5, 26),
    (7, 6, 28),
    (8, 7, 30),
    (9, 7, 32),
    (10, 8, 34),
    (11, 9, 36),
    (12, 9, 38),
    (13, 10, 41),
    (14, 11, 43),
    (16, 11, 45),
    (17, 12, 47),
    (18, 13, 49),
    (19, 13, 52),
    (20, 14, 54),
    (21, 14, 56),
    (22, 15, 59),
    (24, 15, 61),
    (25, 16, 63),
    (26, 16, 66),
    (28, 16, 68),
    (29, 17, 71),
    (30, 17, 73),
    (32, 17, 75),
    (33, 17, 78),
    (34, 17, 80),
    (36, 18, 83),
    (37, 18, 85),
    (39, 18, 88),
    (41, 17, 90),
    (42, 17, 92),
    (44, 17, 95),
    (45, 17, 97),
    (47, 17, 99),
    (49, 17, 101),
    (51, 16, 103),
    (52, 16, 105),
    (54, 16, 107),
    (56, 16, 108),
    (57, 15, 110),
    (59, 15, 112),
    (61, 15, 113),
    (63, 15, 114),
    (64, 15, 116),
    (66, 15, 117),
    (68, 15, 118),
    (69, 16, 119),
    (71, 16, 120),
    (73, 16, 120),
    (74, 16, 121),
    (76, 17, 122),
    (78, 17, 123),
    (79, 18, 123),
    (81, 18, 124),
    (82, 19, 124),
    (84, 19, 125),
    (86, 20, 125),
    (87, 21, 126),
    (89, 21, 126),
    (90, 22, 126),
    (92, 22, 127),
    (93, 23, 127),
    (95, 24, 127),
    (96, 24, 128),
    (98, 25, 128),
    (100, 26, 128),
    (101, 26, 128),
    (103, 27, 128),
    (104, 28, 129),
    (106, 28, 129),
    (107, 29, 129),
    (109, 29, 129),
    (110, 30, 129),
    (112, 31, 129),
    (114, 31, 129),
    (115, 32, 129),
    (117, 33, 129),
    (118, 33, 129),
    (120, 34, 129),
    (121, 34, 130),
    (123, 35, 130),
    (124, 35, 130),
    (126, 36, 130),
    (128, 37, 130),
    (129, 37, 129),
    (131, 38, 129),
    (132, 38, 129),
    (134, 39, 129),
    (136, 39, 129),
    (137, 40, 129),
    (139, 41, 129),
    (140, 41, 129),
    (142, 42, 129),
    (144, 42, 129),
    (145, 43, 129),
    (147, 43, 128),
    (148, 44, 128),
    (150, 44, 128),
    (152, 45, 128),
    (153, 45, 128),
    (155, 46, 127),
    (156, 46, 127),
    (158, 47, 127),
    (160, 47, 127),
    (161, 48, 126),
    (163, 48, 126),
    (165, 49, 126),
    (166, 49, 125),
    (168, 50, 125),
    (170, 51, 125),
    (171, 51, 124),
    (173, 52, 124),
    (174, 52, 123),
    (176, 53, 123),
    (178, 53, 123),
    (179, 54, 122),
    (181, 54, 122),
    (183, 55, 121),
    (184, 55, 121),
    (186, 56, 120),
    (188, 57, 120),
    (189, 57, 119),
    (191, 58, 119),
    (192, 58, 118),
    (194, 59, 117),
    (196, 60, 117),
    (197, 60, 116),
    (199, 61, 115),
    (200, 62, 115),
    (202, 62, 114),
    (204, 63, 113),
    (205, 64, 113),
    (207, 64, 112),
    (208, 65, 111),
    (210, 66, 111),
    (211, 67, 110),
    (213, 68, 109),
    (214, 69, 108),
    (216, 69, 108),
    (217, 70, 107),
    (219, 71, 106),
    (220, 72, 105),
    (222, 73, 104),
    (223, 74, 104),
    (224, 76, 103),
    (226, 77, 102),
    (227, 78, 101),
    (228, 79, 100),
    (229, 80, 100),
    (231, 82, 99),
    (232, 83, 98),
    (233, 84, 98),
    (234, 86, 97),
    (235, 87, 96),
    (236, 88, 96),
    (237, 90, 95),
    (238, 91, 94),
    (239, 93, 94),
    (240, 95, 94),
    (241, 96, 93),
    (242, 98, 93),
    (242, 100, 92),
    (243, 101, 92),
    (244, 103, 92),
    (244, 105, 92),
    (245, 107, 92),
    (246, 108, 92),
    (246, 110, 92),
    (247, 112, 92),
    (247, 114, 92),
    (248, 116, 92),
    (248, 118, 92),
    (249, 120, 93),
    (249, 121, 93),
    (249, 123, 93),
    (250, 125, 94),
    (250, 127, 94),
    (250, 129, 95),
    (251, 131, 95),
    (251, 133, 96),
    (251, 135, 97),
    (252, 137, 97),
    (252, 138, 98),
    (252, 140, 99),
    (252, 142, 100),
    (252, 144, 101),
    (253, 146, 102),
    (253, 148, 103),
    (253, 150, 104),
    (253, 152, 105),
    (253, 154, 106),
    (253, 155, 107),
    (254, 157, 108),
    (254, 159, 109),
    (254, 161, 110),
    (254, 163, 111),
    (254, 165, 113),
    (254, 167, 114),
    (254, 169, 115),
    (254, 170, 116),
    (254, 172, 118),
    (254, 174, 119),
    (254, 176, 120),
    (254, 178, 122),
    (254, 180, 123),
    (254, 182, 124),
    (254, 183, 126),
    (254, 185, 127),
    (254, 187, 129),
    (254, 189, 130),
    (254, 191, 132),
    (254, 193, 133),
    (254, 194, 135),
    (254, 196, 136),
    (254, 198, 138),
    (254, 200, 140),
    (254, 202, 141),
    (254, 204, 143),
    (254, 205, 144),
    (254, 207, 146),
    (254, 209, 148),
    (254, 211, 149),
    (254, 213, 151),
    (254, 215, 153),
    (254, 216, 154),
    (253, 218, 156),
    (253, 220, 158),
    (253, 222, 160),
    (253, 224, 161),
    (253, 226, 163),
    (253, 227, 165),
    (253, 229, 167),
    (253, 231, 169),
    (253, 233, 170),
    (253, 235, 172),
    (252, 236, 174),
    (252, 238, 176),
    (252, 240, 178),
    (252, 242, 180),
    (252, 244, 182),
    (252, 246, 184),
    (252, 247, 185),
    (252, 249, 187),
    (252, 251, 189),
    (252, 253, 191),
)


_TABLES: dict[str, tuple[tuple[int, int, int], ...]] = {
    VIRIDIS_NAME: _VIRIDIS,
    MAGMA_NAME: _MAGMA,
}


def lut_index(normalized: np.ndarray) -> np.ndarray:
    """Map normalized ``[0, 1]`` values to LUT indices.

    The convention is ``index = clip(rint(normalized * 255), 0, 255)`` so that
    ``0.0 -> 0``, ``1.0 -> 255`` and ``0.5 -> 128``. Values outside ``[0, 1]``
    are clipped to the nearest table entry.
    """
    values = np.asarray(normalized, dtype=np.float64)
    scaled = np.rint(values * (LUT_SIZE - 1))
    return np.clip(scaled, 0, LUT_SIZE - 1).astype(np.intp)


def _table(name: str) -> np.ndarray:
    return np.asarray(_TABLES[name], dtype=np.uint8).reshape(LUT_SIZE, 3)


def viridis_lut() -> np.ndarray:
    """Return the frozen ``(256, 3)`` uint8 viridis table."""
    return _table(VIRIDIS_NAME)


def magma_lut() -> np.ndarray:
    """Return the frozen ``(256, 3)`` uint8 magma table."""
    return _table(MAGMA_NAME)


def lut(name: str) -> np.ndarray:
    """Return the named ``(256, 3)`` uint8 lookup table.

    Only :data:`VIRIDIS_NAME` and :data:`MAGMA_NAME` are backed by a table.
    ``"legacy_rainbow"`` is rejected on purpose: quantizing the normalized value
    to an 8-bit index would shift pixels away from the historical output, so the
    legacy ramp must go through the direct arithmetic path in
    :func:`map_scalar_to_rgb` / :func:`map_normalized_to_rgb`.
    """
    key = str(name).strip().lower()
    if key == LEGACY_RAINBOW_NAME:
        raise ValueError(
            "The 'legacy_rainbow' colormap has no lookup table because 8-bit "
            "index quantization does not reproduce the historical pixels. Use "
            "map_scalar_to_rgb(..., colormap='legacy_rainbow') instead."
        )
    if key not in _TABLES:
        raise ValueError(
            f"Unknown colormap {name!r}; expected one of {sorted(_TABLES)}."
        )
    return _table(key)


def _legacy_rainbow_rgb(normalized: np.ndarray) -> np.ndarray:
    """Reproduce the historical rainbow bytes exactly.

    The standalone traditional-ML page must stay byte-identical, so this ramp
    is computed directly on the float32 normalized values instead of being
    resampled through a 256-entry table. See the module docstring.
    """
    n = np.asarray(normalized, dtype=np.float32)
    red = n
    green = 1.0 - np.abs(2.0 * n - 1.0)
    blue = 1.0 - n
    rgb = np.stack((red, green, blue), axis=-1)
    return np.rint(rgb * 255.0).astype(np.uint8)


def map_normalized_to_rgb(normalized: np.ndarray, *, colormap: str) -> np.ndarray:
    """Map already-normalized ``[0, 1]`` values to ``uint8`` RGB.

    Returns an array whose shape is ``values.shape + (3,)``.
    """
    key = str(colormap).strip().lower()
    if key == LEGACY_RAINBOW_NAME:
        return _legacy_rainbow_rgb(normalized)
    return lut(key)[lut_index(normalized)]


def map_scalar_to_rgb(
    values: np.ndarray,
    *,
    value_range: tuple[float, float],
    colormap: str,
) -> np.ndarray:
    """Map raw scalar values through a fixed declared value range.

    Values are normalized with the declared ``value_range`` (never with the
    array's own extrema) and then mapped through the named colormap.
    """
    array = np.asarray(values, dtype=np.float32)
    minimum, maximum = (float(value) for value in value_range)
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError("Scalar map value_range must be finite and increasing.")
    normalized = np.clip((array - minimum) / (maximum - minimum), 0.0, 1.0)
    return map_normalized_to_rgb(normalized, colormap=colormap)
