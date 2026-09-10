"""Versioned public contracts for symmetry-harness runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


HARNESS_CONTRACT_VERSION = "symmetry-harness-run-v1"
ANNOTATION_SCHEMA_VERSION = "symmetry-annotation-session-v1"
ADAPTER_SCHEMA_VERSION = "symmetry-adapter-head-v1"
FINE_TUNED_MODEL_PACKAGE_SCHEMA_VERSION = "symmetry-fine-tuned-model-package-v1"
PREDICTION_RUN_CONTRACT_VERSION = "symmetry-harness-prediction-run-v1"


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
