"""Legacy feature entry points retained for an explicit migration error."""

from __future__ import annotations

CHANNEL_NAMES = (
    "image",
    "reflection_strength",
    "reflection_sin_2theta",
    "reflection_cos_2theta",
    "rotation_2_fold",
    "rotation_3_fold",
    "rotation_4_fold",
    "rotation_6_fold",
)


def compute_symmetry_features(*args, **kwargs):
    """Reject the retired in-process numerical path with actionable guidance."""
    raise RuntimeError(
        "Feature computation moved to the symmetry-learn Provider. Use "
        "symmetry_harness.provider.run_provider_analysis or the public workflow."
    )
