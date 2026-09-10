"""Versioned public contracts for symmetry-harness runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


HARNESS_CONTRACT_VERSION = "symmetry-harness-run-v1"
ANNOTATION_SCHEMA_VERSION = "symmetry-annotation-session-v1"
ADAPTER_SCHEMA_VERSION = "symmetry-adapter-head-v1"
FINE_TUNED_MODEL_PACKAGE_SCHEMA_VERSION = "symmetry-fine-tuned-model-package-v1"
PREDICTION_RUN_CONTRACT_VERSION = "symmetry-harness-prediction-run-v1"
TRADITIONAL_RUN_CONTRACT_VERSION = "symmetry-harness-traditional-ml-run-v1"

# Shared traditional-ML settings. Classifier parameter defaults are never
# duplicated here: they come from the Provider capability block so the Provider
# registry stays the single source of truth.
TRADITIONAL_DEFAULT_CLASSIFIER_PATCH_SIZE = 64
TRADITIONAL_DEFAULT_SEED = 42
TRADITIONAL_MINIMUM_PATCHES_PER_CLASS = 2
TRADITIONAL_MAXIMUM_PATCHES_PER_CLASS = 5000

TRADITIONAL_FEATURE_MODES = ("raw_image", "image_plus_symmetry_maps")
TRADITIONAL_CLASSIFIER_IDENTIFIERS = ("logistic_regression", "random_forest")


@dataclass(frozen=True)
class RunOptions:
    """Validated options that fully determine one few-shot run."""

    n_max: int
    symmetry_patch_size: int
    rotation_folds: tuple[int, ...]
    reflection_p: float
    normalize_rotation_maps: bool
    input_normalization: str
    classifier_patch_size: int
    minimum_shots_per_class: int
    maximum_shots_per_class: int
    adapter_bottleneck: int
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int
    stride: int
    batch_size: int
    device: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraditionalMLSettings:
    """Validated settings that fully determine one traditional-ML run.

    ``classifier_patch_size`` belongs to this validation experiment rather than
    to a pretrained model, so it is a user-editable field. The provider applies
    its own strict validation on top of these values.
    """

    classifier_patch_size: int = TRADITIONAL_DEFAULT_CLASSIFIER_PATCH_SIZE
    seed: int = TRADITIONAL_DEFAULT_SEED
    stride: int = 4
    batch_size: int = 512

    def __post_init__(self) -> None:
        for name in ("classifier_patch_size", "stride", "batch_size"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, value)
        seed = int(self.seed)
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be an integer between 0 and 4294967295.")
        object.__setattr__(self, "seed", seed)

    def to_provider_options(self) -> dict[str, int]:
        """Return the exact option mapping accepted by the Provider worker."""
        return {
            "classifier_patch_size": self.classifier_patch_size,
            "seed": self.seed,
            "stride": self.stride,
            "batch_size": self.batch_size,
            "minimum_patches_per_class": TRADITIONAL_MINIMUM_PATCHES_PER_CLASS,
            "maximum_patches_per_class": TRADITIONAL_MAXIMUM_PATCHES_PER_CLASS,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

