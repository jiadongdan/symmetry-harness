"""Load and validate portable symmetry-harness configuration files."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import shutil
import sys
from typing import Any


CONFIG_SCHEMA_VERSION = "symmetry-harness-config-v1"
NORMALIZATION_POLICIES = {"minmax_0_1", "already_0_1", "dtype_unit"}


@dataclass(frozen=True)
class ProviderConfig:
    """External symmetry execution provider used by the Harness."""

    name: str
    module: str
    distribution: str
    minimum_version: str | None
    source_root: Path | None
    install_hint: str | None
    timeout_seconds: int


@dataclass(frozen=True)
class ModelConfig:
    """Pretrained model identity and fixed input contract."""

    identifier: str
    python_executable: str
    weight_identifier: str | None
    checkpoint_path: Path | None
    checkpoint_sha256: str | None
    device: str
    input_channels: int
    pretrained_classes: int
    classifier_patch_size: int


@dataclass(frozen=True)
class FeatureConfig:
    """Parameters for building the eight-channel symmetry representation."""

    n_max: int
    symmetry_patch_size: int
    rotation_folds: tuple[int, ...]
    reflection_p: float
    normalize_rotation_maps: bool
    input_normalization: str


@dataclass(frozen=True)
class FineTuneConfig:
    """Bounds and defaults for user-guided adapter fine-tuning."""

    minimum_shots_per_class: int
    recommended_shots_per_class: int
    maximum_shots_per_class: int
    adapter_bottleneck: int
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int


@dataclass(frozen=True)
class PredictionConfig:
    """Dense-prediction defaults."""

    stride: int
    batch_size: int


@dataclass(frozen=True)
class HarnessConfig:
    """Resolved and validated symmetry-harness configuration."""

    source_path: Path
    project_root: Path
    output_root: Path
    provider: ProviderConfig
    model: ModelConfig
    features: FeatureConfig
    fine_tuning: FineTuneConfig
    prediction: PredictionConfig


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object.")
    return dict(value)


def _resolve_path(base: Path, value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _resolve_python(base: Path, value: Any) -> Path:
    requested = "auto" if value is None else str(value).strip()
    if requested in {"", "auto"}:
        return Path(sys.executable).resolve()
    path = Path(requested).expanduser()
    if path.is_absolute():
        return path.resolve()
    discovered = shutil.which(requested)
    if discovered is not None:
        return Path(discovered).resolve()
    return (base / path).resolve()


def _positive_integer(value: Any, label: str) -> int:
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{label} must be positive.")
    return resolved


def _positive_float(value: Any, label: str, *, allow_zero: bool = False) -> float:
    resolved = float(value)
    invalid = resolved < 0 if allow_zero else resolved <= 0
    if invalid:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{label} must be {qualifier}.")
    return resolved


def load_harness_config(path: str | Path) -> HarnessConfig:
    """Load one explicit configuration without searching private locations."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Harness configuration does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported symmetry-harness configuration schema.")

    config_dir = source.parent
    project_root = _resolve_path(config_dir, payload.get("project_root", "."))
    assert project_root is not None
    output_value = payload.get("output_root", "symmetry-runs")
    output_root = _resolve_path(project_root, output_value)
    assert output_root is not None

    provider_raw = _mapping(payload.get("provider", {}), "provider")
    provider = ProviderConfig(
        name=str(provider_raw.get("name", "symmetry-learn")),
        module=str(provider_raw.get("module", "symmlearn.provider.worker")),
        distribution=str(provider_raw.get("distribution", "symmetry-learn")),
        minimum_version=(
            None
            if provider_raw.get("minimum_version") is None
            else str(provider_raw["minimum_version"])
        ),
        install_hint=(
            None
            if provider_raw.get("install_hint") is None
            else str(provider_raw["install_hint"])
        ),
        source_root=_resolve_path(config_dir, provider_raw.get("source_root")),
        timeout_seconds=_positive_integer(
            provider_raw.get("timeout_seconds", 3600), "provider.timeout_seconds"
        ),
    )

    model_raw = _mapping(payload.get("model", {}), "model")
    raw_weight_identifier = model_raw.get("weight_identifier")
    weight_identifier = (
        None
        if raw_weight_identifier is None or not str(raw_weight_identifier).strip()
        else str(raw_weight_identifier).strip()
    )
    checkpoint_path = _resolve_path(project_root, model_raw.get("checkpoint_path"))
    checkpoint_sha256 = model_raw.get("checkpoint_sha256")
    if checkpoint_sha256 is not None:
        checkpoint_sha256 = str(checkpoint_sha256).lower()
        if len(checkpoint_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in checkpoint_sha256
        ):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest.")
    device = str(model_raw.get("device", "auto"))
    if device != "auto" and device != "cpu" and not device.startswith("cuda"):
        raise ValueError("model.device must be auto, cpu, or a CUDA device.")
    if checkpoint_path is not None and weight_identifier is not None:
        raise ValueError(
            "Select either model.weight_identifier or model.checkpoint_path, not both."
        )
    model = ModelConfig(
        identifier=str(model_raw.get("identifier", "cnn_8ch_pg17")),
        python_executable=str(
            _resolve_python(project_root, model_raw.get("python_executable"))
        ),
        weight_identifier=weight_identifier,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        device=device,
        input_channels=_positive_integer(model_raw.get("input_channels", 8), "input_channels"),
        pretrained_classes=_positive_integer(
            model_raw.get("pretrained_classes", 17), "pretrained_classes"
        ),
        classifier_patch_size=_positive_integer(
            model_raw.get("classifier_patch_size", 64), "classifier_patch_size"
        ),
    )
    feature_raw = _mapping(payload.get("features", {}), "features")
    folds_raw = feature_raw.get("rotation_folds", [2, 3, 4, 6])
    if not isinstance(folds_raw, list):
        raise TypeError("rotation_folds must be a JSON array.")
    folds = tuple(int(value) for value in folds_raw)
    if not folds or any(value <= 0 for value in folds) or len(set(folds)) != len(folds):
        raise ValueError("rotation_folds must contain unique positive integers.")
    symmetry_patch_size = _positive_integer(
        feature_raw.get("symmetry_patch_size", 51), "symmetry_patch_size"
    )
    if symmetry_patch_size % 2 == 0:
        raise ValueError("symmetry_patch_size must be odd.")
    normalization = str(feature_raw.get("input_normalization", "minmax_0_1"))
    if normalization not in NORMALIZATION_POLICIES:
        raise ValueError(f"Unsupported input normalization policy: {normalization}")
    features = FeatureConfig(
        n_max=_positive_integer(feature_raw.get("n_max", 20), "n_max"),
        symmetry_patch_size=symmetry_patch_size,
        rotation_folds=folds,
        reflection_p=_positive_float(feature_raw.get("reflection_p", 2), "reflection_p"),
        normalize_rotation_maps=bool(
            feature_raw.get("normalize_rotation_maps", False)
        ),
        input_normalization=normalization,
    )

    tune_raw = _mapping(payload.get("fine_tuning", {}), "fine_tuning")
    minimum = _positive_integer(
        tune_raw.get("minimum_shots_per_class", 3), "minimum_shots_per_class"
    )
    recommended = _positive_integer(
        tune_raw.get("recommended_shots_per_class", 5),
        "recommended_shots_per_class",
    )
    maximum = _positive_integer(
        tune_raw.get("maximum_shots_per_class", 50), "maximum_shots_per_class"
    )
    if not minimum <= recommended <= maximum:
        raise ValueError("Shot bounds must satisfy minimum <= recommended <= maximum.")
    fine_tuning = FineTuneConfig(
        minimum_shots_per_class=minimum,
        recommended_shots_per_class=recommended,
        maximum_shots_per_class=maximum,
        adapter_bottleneck=_positive_integer(
            tune_raw.get("adapter_bottleneck", 16), "adapter_bottleneck"
        ),
        epochs=_positive_integer(tune_raw.get("epochs", 150), "epochs"),
        learning_rate=_positive_float(
            tune_raw.get("learning_rate", 0.0005), "learning_rate"
        ),
        weight_decay=_positive_float(
            tune_raw.get("weight_decay", 0.0), "weight_decay", allow_zero=True
        ),
        seed=int(tune_raw.get("seed", 42)),
    )

    prediction_raw = _mapping(payload.get("prediction", {}), "prediction")
    prediction = PredictionConfig(
        stride=_positive_integer(prediction_raw.get("stride", 4), "stride"),
        batch_size=_positive_integer(
            prediction_raw.get("batch_size", 512), "batch_size"
        ),
    )
    return HarnessConfig(
        source_path=source,
        project_root=project_root,
        output_root=output_root,
        provider=provider,
        model=model,
        features=features,
        fine_tuning=fine_tuning,
        prediction=prediction,
    )


def config_snapshot(config: HarnessConfig) -> dict[str, Any]:
    """Return a JSON-safe snapshot for a run record."""
    payload = asdict(config)
    payload["source_path"] = str(config.source_path)
    payload["project_root"] = str(config.project_root)
    payload["output_root"] = str(config.output_root)
    payload["provider"]["source_root"] = (
        None
        if config.provider.source_root is None
        else str(config.provider.source_root)
    )
    payload["model"]["checkpoint_path"] = (
        None if config.model.checkpoint_path is None else str(config.model.checkpoint_path)
    )
    payload["features"]["rotation_folds"] = list(config.features.rotation_folds)
    payload["schema_version"] = CONFIG_SCHEMA_VERSION
    return payload


def config_for_model_selection(
    config: HarnessConfig,
    capability: dict[str, Any],
    *,
    weight_identifier: str | None,
    checkpoint_path: str | Path | None = None,
    checkpoint_sha256: str | None = None,
) -> HarnessConfig:
    """Return a runtime configuration for one catalog-backed model selection."""
    identifier = str(capability.get("identifier", "")).strip()
    if not identifier:
        raise ValueError("The selected model capability has no identifier.")
    defaults = dict(capability.get("defaults", {}))
    resolved_checkpoint = (
        None
        if checkpoint_path is None
        else Path(checkpoint_path).expanduser().resolve()
    )
    if resolved_checkpoint is not None and weight_identifier is not None:
        raise ValueError("A custom checkpoint cannot also select a registered weight.")
    if resolved_checkpoint is None and checkpoint_sha256 is not None:
        raise ValueError("A checkpoint SHA-256 requires a custom checkpoint path.")

    model = replace(
        config.model,
        identifier=identifier,
        weight_identifier=weight_identifier,
        checkpoint_path=resolved_checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        input_channels=_positive_integer(
            capability.get("input_channels", config.model.input_channels),
            "input_channels",
        ),
        pretrained_classes=_positive_integer(
            capability.get("pretrained_classes", config.model.pretrained_classes),
            "pretrained_classes",
        ),
        classifier_patch_size=_positive_integer(
            capability.get(
                "classifier_patch_size", config.model.classifier_patch_size
            ),
            "classifier_patch_size",
        ),
    )
    folds_raw = defaults.get("rotation_folds", config.features.rotation_folds)
    folds = tuple(int(value) for value in folds_raw)
    if not folds or any(value <= 0 for value in folds) or len(set(folds)) != len(folds):
        raise ValueError("Provider rotation_folds must contain unique positive integers.")
    symmetry_patch_size = _positive_integer(
        defaults.get("symmetry_patch_size", config.features.symmetry_patch_size),
        "symmetry_patch_size",
    )
    if symmetry_patch_size % 2 == 0:
        raise ValueError("Provider symmetry_patch_size must be odd.")
    features = replace(
        config.features,
        n_max=_positive_integer(
            defaults.get("n_max", config.features.n_max), "n_max"
        ),
        symmetry_patch_size=symmetry_patch_size,
        rotation_folds=folds,
        reflection_p=_positive_float(
            defaults.get("reflection_p", config.features.reflection_p),
            "reflection_p",
        ),
        normalize_rotation_maps=bool(
            defaults.get(
                "normalize_rotation_maps", config.features.normalize_rotation_maps
            )
        ),
    )
    minimum = _positive_integer(
        capability.get(
            "minimum_shots_per_class",
            defaults.get(
                "minimum_shots_per_class",
                config.fine_tuning.minimum_shots_per_class,
            ),
        ),
        "minimum_shots_per_class",
    )
    recommended = _positive_integer(
        capability.get(
            "recommended_shots_per_class",
            defaults.get(
                "recommended_shots_per_class",
                config.fine_tuning.recommended_shots_per_class,
            ),
        ),
        "recommended_shots_per_class",
    )
    maximum = _positive_integer(
        capability.get(
            "maximum_shots_per_class",
            defaults.get(
                "maximum_shots_per_class",
                config.fine_tuning.maximum_shots_per_class,
            ),
        ),
        "maximum_shots_per_class",
    )
    if not minimum <= recommended <= maximum:
        raise ValueError("Provider shot bounds must satisfy minimum <= recommended <= maximum.")
    fine_tuning = replace(
        config.fine_tuning,
        minimum_shots_per_class=minimum,
        recommended_shots_per_class=recommended,
        maximum_shots_per_class=maximum,
        adapter_bottleneck=_positive_integer(
            defaults.get(
                "adapter_bottleneck", config.fine_tuning.adapter_bottleneck
            ),
            "adapter_bottleneck",
        ),
        epochs=_positive_integer(
            defaults.get("epochs", config.fine_tuning.epochs), "epochs"
        ),
        learning_rate=_positive_float(
            defaults.get("learning_rate", config.fine_tuning.learning_rate),
            "learning_rate",
        ),
        weight_decay=_positive_float(
            defaults.get("weight_decay", config.fine_tuning.weight_decay),
            "weight_decay",
            allow_zero=True,
        ),
        seed=int(defaults.get("seed", config.fine_tuning.seed)),
    )
    prediction = replace(
        config.prediction,
        stride=_positive_integer(
            defaults.get("stride", config.prediction.stride), "stride"
        ),
        batch_size=_positive_integer(
            defaults.get("batch_size", config.prediction.batch_size), "batch_size"
        ),
    )
    return replace(
        config,
        model=model,
        features=features,
        fine_tuning=fine_tuning,
        prediction=prediction,
    )
