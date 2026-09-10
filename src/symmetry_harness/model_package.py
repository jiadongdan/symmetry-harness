"""Safe inspection of portable fine-tuned symmetry model packages.

This module never deserializes PyTorch data. Torch belongs to the Provider
process, which loads ``model_state.pt`` with ``weights_only=True``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any
from zipfile import ZipFile

from .config import NORMALIZATION_POLICIES
from .contracts import FINE_TUNED_MODEL_PACKAGE_SCHEMA_VERSION


MANDATORY_MEMBERS = ("manifest.json", "model_state.pt", "training_summary.json")
ALLOWED_OPTIONAL_MEMBERS: tuple[str, ...] = ()

MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_TRAINING_SUMMARY_BYTES = 8 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024

_JSON_MEMBER_LIMITS = {
    "manifest.json": MAX_MANIFEST_BYTES,
    "training_summary.json": MAX_TRAINING_SUMMARY_BYTES,
}

_COLOR_PATTERN = re.compile(r"#[0-9a-fA-F]{6}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class ModelPackageError(ValueError):
    """Raised when a model package is missing, unsafe, or structurally invalid."""


@dataclass(frozen=True)
class FineTunedModelPackage:
    """One validated portable fine-tuned model package."""

    source_path: Path
    package_sha256: str
    manifest: dict[str, Any]
    training_summary: dict[str, Any]
    model_state_sha256: str

    @property
    def model_identifier(self) -> str:
        return str(self.manifest["model"]["identifier"])

    @property
    def task_classes(self) -> int:
        return int(self.manifest["model"]["task_classes"])

    @property
    def classes(self) -> list[dict[str, Any]]:
        return list(self.manifest["classes"])

    @property
    def class_names(self) -> list[str]:
        return [str(entry["name"]) for entry in self.classes]

    @property
    def class_colors(self) -> list[str]:
        return [str(entry["color"]) for entry in self.classes]

    @property
    def feature_options(self) -> dict[str, Any]:
        """Return the immutable feature options recorded in the package."""
        return dict(self.manifest["features"])

    @property
    def input_normalization(self) -> str:
        return str(self.manifest["features"]["input_normalization"])

    @property
    def prediction_defaults(self) -> dict[str, Any]:
        return dict(self.manifest["prediction_defaults"])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ModelPackageError(message)


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelPackageError(f"{label} must be an integer.")
    if value <= 0:
        raise ModelPackageError(f"{label} must be positive.")
    return value


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelPackageError(f"{label} must be a number.")
    resolved = float(value)
    if resolved <= 0:
        raise ModelPackageError(f"{label} must be positive.")
    return resolved


def _read_json_member(archive: ZipFile, name: str) -> dict[str, Any]:
    limit = _JSON_MEMBER_LIMITS[name]
    try:
        info = archive.getinfo(name)
    except KeyError as error:
        raise ModelPackageError(f"The model package is missing {name}.") from error
    if info.file_size > limit:
        raise ModelPackageError(f"{name} exceeds the maximum allowed size.")
    raw = archive.read(name)
    if len(raw) > limit:
        raise ModelPackageError(f"{name} exceeds the maximum allowed size.")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ModelPackageError(f"{name} is not valid UTF-8 text.") from error
    except json.JSONDecodeError as error:
        raise ModelPackageError(f"{name} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ModelPackageError(f"{name} must contain a JSON object.")
    return payload


def _validate_classes(manifest: dict[str, Any]) -> None:
    classes = manifest.get("classes")
    _require(isinstance(classes, list) and classes, "The package records no classes.")
    task_classes = _positive_integer(
        manifest.get("model", {}).get("task_classes"), "model.task_classes"
    )
    _require(
        len(classes) == task_classes,
        "The recorded class count does not match model.task_classes.",
    )
    for index, entry in enumerate(classes):
        _require(isinstance(entry, dict), f"Class {index} must be an object.")
        _require(entry.get("index") == index, "Class indices must be contiguous and zero-based.")
        name = entry.get("name")
        _require(
            isinstance(name, str) and bool(name.strip()),
            f"Class {index} has an invalid name.",
        )
        color = entry.get("color")
        _require(
            isinstance(color, str) and bool(_COLOR_PATTERN.match(color)),
            f"Class {index} has an invalid display color.",
        )
    names = [str(entry["name"]).strip() for entry in classes]
    _require(len(set(names)) == len(names), "Recorded class names must be unique.")


def _validate_model_section(manifest: dict[str, Any]) -> None:
    model = manifest.get("model")
    _require(isinstance(model, dict), "The package has no model section.")
    identifier = model.get("identifier")
    _require(
        isinstance(identifier, str) and bool(identifier.strip()),
        "The package records no model identifier.",
    )
    _positive_integer(model.get("input_channels"), "model.input_channels")
    _positive_integer(model.get("classifier_patch_size"), "model.classifier_patch_size")
    _positive_integer(model.get("adapter_bottleneck"), "model.adapter_bottleneck")


def _validate_feature_section(manifest: dict[str, Any]) -> None:
    features = manifest.get("features")
    _require(isinstance(features, dict), "The package has no features section.")
    pipeline = features.get("pipeline")
    _require(
        isinstance(pipeline, str) and bool(pipeline.strip()),
        "The package records no feature pipeline.",
    )
    channel_names = features.get("channel_names")
    _require(
        isinstance(channel_names, list) and channel_names
        and all(isinstance(name, str) and name.strip() for name in channel_names),
        "The package records invalid feature channel names.",
    )
    _positive_integer(features.get("n_max"), "features.n_max")
    symmetry_patch_size = _positive_integer(
        features.get("symmetry_patch_size"), "features.symmetry_patch_size"
    )
    _require(
        symmetry_patch_size % 2 == 1,
        "features.symmetry_patch_size must be odd.",
    )
    _positive_float(features.get("reflection_p"), "features.reflection_p")
    _require(
        isinstance(features.get("normalize_rotation_maps"), bool),
        "features.normalize_rotation_maps must be boolean.",
    )
    folds = features.get("rotation_folds")
    _require(isinstance(folds, list) and bool(folds), "features.rotation_folds is invalid.")
    for fold in folds:
        _require(
            isinstance(fold, int) and not isinstance(fold, bool) and fold > 0,
            "features.rotation_folds must contain positive integers.",
        )
    normalization = features.get("input_normalization")
    _require(
        normalization in NORMALIZATION_POLICIES,
        f"Unsupported recorded input normalization policy: {normalization!r}",
    )


def _validate_manifest(manifest: dict[str, Any]) -> None:
    _require(
        manifest.get("schema_version") == FINE_TUNED_MODEL_PACKAGE_SCHEMA_VERSION,
        "Unsupported fine-tuned model package schema.",
    )
    _validate_model_section(manifest)
    _validate_classes(manifest)
    _validate_feature_section(manifest)
    defaults = manifest.get("prediction_defaults")
    _require(
        isinstance(defaults, dict), "The package has no prediction defaults section."
    )
    _positive_integer(defaults.get("stride"), "prediction_defaults.stride")
    _positive_integer(defaults.get("batch_size"), "prediction_defaults.batch_size")


def inspect_model_package(path: str | Path) -> FineTunedModelPackage:
    """Validate one local ``.symmodel`` package without deserializing weights."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ModelPackageError(f"The model package does not exist: {source}")

    package_sha256 = _file_sha256(source)
    with ZipFile(source) as archive:
        names = archive.namelist()
        duplicates = sorted(
            name for name, count in Counter(names).items() if count > 1
        )
        _require(
            not duplicates,
            f"The model package contains duplicate members: {duplicates}",
        )
        for name in names:
            _require(
                not name.startswith(("/", "\\")) and ".." not in Path(name).parts,
                f"The model package contains an unsafe member name: {name!r}",
            )
            _require(
                name in MANDATORY_MEMBERS or name in ALLOWED_OPTIONAL_MEMBERS,
                f"The model package contains an unexpected member: {name!r}",
            )
        missing = sorted(set(MANDATORY_MEMBERS).difference(names))
        _require(not missing, f"The model package is missing members: {missing}")

        total = sum(info.file_size for info in archive.infolist())
        _require(
            total <= MAX_TOTAL_UNCOMPRESSED_BYTES,
            "The model package contents exceed the maximum allowed size.",
        )
        manifest = _read_json_member(archive, "manifest.json")
        training_summary = _read_json_member(archive, "training_summary.json")
        model_state_sha256 = _member_sha256(archive, "model_state.pt")

    _validate_manifest(manifest)
    recorded = manifest.get("provenance", {}).get("model_state_sha256")
    _require(
        isinstance(recorded, str) and bool(_SHA256_PATTERN.match(recorded.lower())),
        "The package records a malformed model_state_sha256.",
    )
    if recorded.lower() != model_state_sha256:
        raise ModelPackageError(
            "The model state checksum does not match the package manifest. "
            "The package may be incomplete or modified."
        )
    return FineTunedModelPackage(
        source_path=source,
        package_sha256=package_sha256,
        manifest=manifest,
        training_summary=training_summary,
        model_state_sha256=model_state_sha256,
    )


def materialize_model_state(
    package: FineTunedModelPackage,
    destination: str | Path,
) -> Path:
    """Write the verified ``model_state.pt`` into a private caller-owned path."""
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(package.source_path) as archive:
        with archive.open("model_state.pt") as source, target.open("wb") as handle:
            while True:
                block = source.read(4 * 1024 * 1024)
                if not block:
                    break
                handle.write(block)
    if _file_sha256(target) != package.model_state_sha256:
        target.unlink(missing_ok=True)
        raise ModelPackageError(
            "The materialized model state does not match the validated checksum."
        )
    return target


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member_sha256(archive: ZipFile, name: str) -> str:
    digest = sha256()
    with archive.open(name) as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
