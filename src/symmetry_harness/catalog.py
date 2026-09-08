"""Model and weight catalog helpers for Provider capability documents."""

from __future__ import annotations

import re
from typing import Any


_DISTRIBUTION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+!-]*\Z")


def provider_models(capabilities: dict[str, Any]) -> list[dict[str, Any]]:
    """Return validated model records in Provider order."""
    raw_models = capabilities.get("models", [])
    if not isinstance(raw_models, list):
        raise TypeError("Provider capabilities models must be an array.")
    models = []
    for raw in raw_models:
        if not isinstance(raw, dict):
            raise TypeError("Each Provider model capability must be an object.")
        identifier = str(raw.get("identifier", "")).strip()
        if not identifier:
            raise ValueError("Every Provider model requires an identifier.")
        models.append(dict(raw))
    identifiers = [str(model["identifier"]) for model in models]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Provider model identifiers must be unique.")
    return models


def model_capability(
    capabilities: dict[str, Any], identifier: str
) -> dict[str, Any]:
    """Return one model capability or raise an actionable selection error."""
    for model in provider_models(capabilities):
        if model.get("identifier") == identifier:
            return model
    available = ", ".join(str(item["identifier"]) for item in provider_models(capabilities))
    raise ValueError(
        f"Provider does not expose model {identifier!r}; available models: "
        f"{available or 'none'}."
    )


def choose_model(
    capabilities: dict[str, Any], requested_identifier: str | None
) -> dict[str, Any]:
    """Choose an explicit model, or the sole model when discovery is unambiguous."""
    if requested_identifier is not None and str(requested_identifier).strip():
        return model_capability(capabilities, str(requested_identifier).strip())
    models = provider_models(capabilities)
    if len(models) == 1:
        return models[0]
    if not models:
        raise RuntimeError("The symmetry-learn Provider exposes no models.")
    identifiers = ", ".join(str(item["identifier"]) for item in models)
    raise ValueError(
        "The Provider exposes multiple models. Select one with --model from: "
        + identifiers
    )


def model_weights(model: dict[str, Any]) -> list[dict[str, Any]]:
    """Return validated registered-weight records for one model."""
    raw_weights = model.get("weights", [])
    if raw_weights is None:
        raw_weights = []
    if not isinstance(raw_weights, list):
        raise TypeError("Provider model weights must be an array.")
    weights = []
    for raw in raw_weights:
        if not isinstance(raw, dict):
            raise TypeError("Each Provider weight capability must be an object.")
        identifier = str(raw.get("identifier", "")).strip()
        if not identifier:
            raise ValueError("Every registered weight requires an identifier.")
        weights.append(dict(raw))
    identifiers = [str(weight["identifier"]) for weight in weights]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Provider weight identifiers must be unique within a model.")
    return weights


def weight_capability(
    model: dict[str, Any], identifier: str | None
) -> dict[str, Any]:
    """Resolve an explicit registered weight or the model default."""
    requested = None if identifier is None else str(identifier).strip()
    if not requested:
        default = model.get("default_weight")
        if not isinstance(default, dict) or not str(default.get("identifier", "")).strip():
            raise ValueError(
                f"Model {model.get('identifier')!r} does not declare a default weight."
            )
        requested = str(default["identifier"])
    for weight in model_weights(model):
        if weight.get("identifier") == requested:
            return weight
    default = model.get("default_weight")
    if isinstance(default, dict) and default.get("identifier") == requested:
        return dict(default)
    available = ", ".join(str(item["identifier"]) for item in model_weights(model))
    raise ValueError(
        f"Model {model.get('identifier')!r} does not expose weight {requested!r}; "
        f"available weights: {available or 'none'}."
    )


def weight_install_requirement(weight: dict[str, Any]) -> str:
    """Build a safe exact Python distribution requirement for one weight package."""
    distribution = str(weight.get("distribution", "")).strip()
    version = str(weight.get("version", "")).strip()
    if not _DISTRIBUTION_PATTERN.fullmatch(distribution):
        raise ValueError("The registered weight has an invalid distribution name.")
    if not _VERSION_PATTERN.fullmatch(version):
        raise ValueError("The registered weight has an invalid package version.")
    return f"{distribution}=={version}"


def model_catalog_report(
    capabilities: dict[str, Any],
    *,
    selected_model: str,
    selected_weight: str | None,
    uses_custom_checkpoint: bool,
) -> list[dict[str, Any]]:
    """Return a JSON-safe catalog annotated with the current selection."""
    report = []
    for raw_model in provider_models(capabilities):
        model = dict(raw_model)
        identifier = str(model["identifier"])
        weights = []
        for raw_weight in model_weights(model):
            weight = dict(raw_weight)
            weight["selected"] = bool(
                not uses_custom_checkpoint
                and identifier == selected_model
                and weight.get("identifier") == selected_weight
            )
            weights.append(weight)
        model["weights"] = weights
        model["selected"] = identifier == selected_model
        model["uses_custom_checkpoint"] = bool(
            identifier == selected_model and uses_custom_checkpoint
        )
        report.append(model)
    return report
