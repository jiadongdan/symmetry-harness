"""Serializable point annotations for user-guided few-shot training."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import ANNOTATION_SCHEMA_VERSION


DEFAULT_CLASS_COLORS = (
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
)


@dataclass(frozen=True)
class SupportClass:
    """One user-defined local class and its source-image coordinates."""

    index: int
    name: str
    color: str
    points: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "color": self.color,
            "points_xy": [list(point) for point in self.points],
        }


@dataclass(frozen=True)
class AnnotationSession:
    """A versioned immutable annotation session in source-image coordinates."""

    image_path: str
    image_sha256: str
    image_shape: tuple[int, int]
    classifier_patch_size: int
    classes: tuple[SupportClass, ...]
    created_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "coordinate_convention": "points are (x, y); arrays are indexed [y, x]",
            "image": {
                "path": self.image_path,
                "sha256": self.image_sha256,
                "shape_hw": list(self.image_shape),
            },
            "classifier_patch_size": self.classifier_patch_size,
            "classes": [entry.to_dict() for entry in self.classes],
            "created_utc": self.created_utc,
        }


def create_annotation_session(
    *,
    image_path: str,
    image_sha256: str,
    image_shape: tuple[int, int],
    classifier_patch_size: int,
    class_names: Iterable[str],
    points_by_class: Iterable[Iterable[tuple[int, int]]],
    colors: Iterable[str] | None = None,
) -> AnnotationSession:
    """Build a session from UI state and validate its static structure."""
    names = tuple(str(name).strip() for name in class_names)
    grouped_points = tuple(tuple(group) for group in points_by_class)
    if colors is None:
        color_values = tuple(
            DEFAULT_CLASS_COLORS[index % len(DEFAULT_CLASS_COLORS)]
            for index in range(len(names))
        )
    else:
        color_values = tuple(colors)
    if not len(names) == len(grouped_points) == len(color_values):
        raise ValueError("Class names, colors, and point groups must have equal length.")
    classes = tuple(
        SupportClass(
            index=index,
            name=name,
            color=str(color_values[index]),
            points=tuple((int(point[0]), int(point[1])) for point in grouped_points[index]),
        )
        for index, name in enumerate(names)
    )
    session = AnnotationSession(
        image_path=str(Path(image_path).expanduser().resolve()),
        image_sha256=str(image_sha256).lower(),
        image_shape=(int(image_shape[0]), int(image_shape[1])),
        classifier_patch_size=int(classifier_patch_size),
        classes=classes,
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    validate_annotation_session(session, minimum_shots=0, maximum_shots=None)
    return session


def _parse_session(payload: dict[str, Any]) -> AnnotationSession:
    if payload.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
        raise ValueError("Unsupported annotation session schema version.")
    image = payload.get("image")
    if not isinstance(image, dict):
        raise TypeError("Annotation image metadata must be an object.")
    raw_classes = payload.get("classes")
    if not isinstance(raw_classes, list):
        raise TypeError("Annotation classes must be an array.")
    classes = []
    for raw in raw_classes:
        if not isinstance(raw, dict):
            raise TypeError("Each annotation class must be an object.")
        raw_points = raw.get("points_xy", [])
        if not isinstance(raw_points, list):
            raise TypeError("points_xy must be an array.")
        points = []
        for point in raw_points:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("Every point must be a two-element [x, y] array.")
            points.append((int(point[0]), int(point[1])))
        classes.append(
            SupportClass(
                index=int(raw.get("index")),
                name=str(raw.get("name", "")).strip(),
                color=str(raw.get("color", "#ffffff")),
                points=tuple(points),
            )
        )
    shape = image.get("shape_hw")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError("image.shape_hw must contain height and width.")
    return AnnotationSession(
        image_path=str(Path(str(image.get("path"))).expanduser().resolve()),
        image_sha256=str(image.get("sha256", "")).lower(),
        image_shape=(int(shape[0]), int(shape[1])),
        classifier_patch_size=int(payload.get("classifier_patch_size")),
        classes=tuple(classes),
        created_utc=str(payload.get("created_utc", "")),
    )


def load_annotation_session(path: str | Path) -> AnnotationSession:
    """Load and statically validate an annotation session."""
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    session = _parse_session(payload)
    validate_annotation_session(session, minimum_shots=0, maximum_shots=None)
    return session


def save_annotation_session(path: str | Path, session: AnnotationSession) -> None:
    """Write an annotation session as stable, human-readable JSON."""
    Path(path).write_text(
        json.dumps(session.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def valid_center_bounds(
    image_shape: tuple[int, int], patch_size: int
) -> tuple[int, int, int, int]:
    """Return inclusive valid center bounds as min_x, max_x, min_y, max_y."""
    height, width = image_shape
    if patch_size <= 0:
        raise ValueError("The classifier patch size must be positive.")
    before = patch_size // 2
    after = patch_size - before
    if height < patch_size or width < patch_size:
        raise ValueError("The image is smaller than the classifier patch.")
    return before, width - after, before, height - after


def validate_annotation_session(
    session: AnnotationSession,
    *,
    minimum_shots: int,
    maximum_shots: int | None,
) -> None:
    """Validate class identity, shot counts, uniqueness, and border safety."""
    if len(session.image_sha256) != 64:
        raise ValueError("The annotation session requires a valid image SHA-256.")
    if len(session.classes) < 2:
        raise ValueError("At least two local classes are required.")
    if [entry.index for entry in session.classes] != list(range(len(session.classes))):
        raise ValueError("Class indices must be contiguous and zero-based.")
    names = [entry.name for entry in session.classes]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Class names must be nonempty and unique.")
    min_x, max_x, min_y, max_y = valid_center_bounds(
        session.image_shape, session.classifier_patch_size
    )
    seen: set[tuple[int, int]] = set()
    for entry in session.classes:
        count = len(entry.points)
        if count < minimum_shots:
            raise ValueError(
                f"Class {entry.name!r} has {count} points; at least {minimum_shots} are required."
            )
        if maximum_shots is not None and count > maximum_shots:
            raise ValueError(
                f"Class {entry.name!r} has {count} points; at most {maximum_shots} are allowed."
            )
        if len(set(entry.points)) != count:
            raise ValueError(f"Class {entry.name!r} contains duplicate points.")
        for x, y in entry.points:
            if not min_x <= x <= max_x or not min_y <= y <= max_y:
                raise ValueError(
                    f"Point ({x}, {y}) for class {entry.name!r} cannot provide a full patch."
                )
            if (x, y) in seen:
                raise ValueError(f"Point ({x}, {y}) is assigned to multiple classes.")
            seen.add((x, y))
