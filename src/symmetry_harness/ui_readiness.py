"""Pure readiness evaluation for the fine-tune and predict workspaces.

Everything here is a deterministic function over a plain dictionary snapshot.
Nothing imports Gradio, so the whole module is unit-testable without starting a
server. The UI layer assembles a snapshot from its widgets, calls
:func:`evaluate_fine_tune_readiness` / :func:`evaluate_predict_readiness`, and
renders the result with :func:`readiness_panel_html`. When a primary action is
disabled, the checklist always states the immediate next action -- it never
relies on button color alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import Any

STATUS_READY: str = "ready"
STATUS_RECOMMENDATION: str = "recommendation"
STATUS_BLOCKER: str = "blocker"

STATUS_ICONS: dict[str, str] = {
    STATUS_READY: "\u2713",
    STATUS_RECOMMENDATION: "!",
    STATUS_BLOCKER: "\u00d7",
}

DEFAULT_MINIMUM_SHOTS: int = 3
DEFAULT_RECOMMENDED_SHOTS: int = 5

_ALLOWED_DEVICES: tuple[str, ...] = ("auto", "cpu")


@dataclass(frozen=True)
class ReadinessItem:
    """One readiness check with a status icon, detail and next action."""

    key: str
    label: str
    status: str
    detail: str = ""
    hint: str = ""


def has_blocker(items: Sequence[ReadinessItem]) -> bool:
    """Return whether any item blocks the associated primary action."""
    return any(item.status == STATUS_BLOCKER for item in items)


# ---------------------------------------------------------------------------
# Snapshot coercion helpers
# ---------------------------------------------------------------------------
def _as_bool(snapshot: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = snapshot.get(key, default)
    return bool(value)


def _as_int(snapshot: Mapping[str, Any], key: str, default: int | None) -> int | None:
    value = snapshot.get(key, default)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(snapshot: Mapping[str, Any], key: str, default: float | None) -> float | None:
    value = snapshot.get(key, default)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_names(snapshot: Mapping[str, Any]) -> list[str]:
    raw = snapshot.get("class_names", []) or []
    return [str(name).strip() for name in raw if str(name).strip()]


def _as_counts(snapshot: Mapping[str, Any]) -> list[int]:
    raw = snapshot.get("class_counts", []) or []
    counts: list[int] = []
    for value in raw:
        try:
            counts.append(int(value))
        except (TypeError, ValueError):
            counts.append(0)
    return counts


def _device_is_valid(device: Any) -> bool:
    if device is None:
        return True
    text = str(device)
    return text in _ALLOWED_DEVICES or text.startswith("cuda")


# ---------------------------------------------------------------------------
# Fine-tune readiness
# ---------------------------------------------------------------------------
def evaluate_fine_tune_readiness(
    snapshot: Mapping[str, Any],
) -> list[ReadinessItem]:
    """Evaluate the ten fine-tune readiness checks from a snapshot dictionary.

    Recognized snapshot keys: ``model_supported``, ``model_name``,
    ``checkpoint_ready``, ``checkpoint_detail``, ``image_valid``,
    ``image_name``, ``features_current``, ``features_detail``, ``class_names``,
    ``class_counts``, ``minimum_shots``, ``recommended_shots``, ``maximum_shots``,
    ``symmetry_patch_size``, ``epochs``, ``learning_rate``, ``stride``,
    ``batch_size``, ``device``, ``stale_state``, ``stale_detail``.
    """
    items: list[ReadinessItem] = []

    # 1. Selected model is supported by the Provider.
    model_supported = _as_bool(snapshot, "model_supported")
    model_name = str(snapshot.get("model_name", "") or "the selected model")
    items.append(
        ReadinessItem(
            key="model_supported",
            label="Model is supported",
            status=STATUS_READY if model_supported else STATUS_BLOCKER,
            detail=(
                f"{model_name} is available."
                if model_supported
                else f"{model_name} is not available."
            ),
            hint="" if model_supported else "Install or select a supported model in the Provider.",
        )
    )

    # 2. Registered checkpoint installed (or a valid custom checkpoint).
    checkpoint_ready = _as_bool(snapshot, "checkpoint_ready")
    checkpoint_detail = str(snapshot.get("checkpoint_detail", "") or "")
    items.append(
        ReadinessItem(
            key="checkpoint_ready",
            label="Checkpoint is ready",
            status=STATUS_READY if checkpoint_ready else STATUS_BLOCKER,
            detail=checkpoint_detail
            or ("Checkpoint installed." if checkpoint_ready else "Checkpoint is missing."),
            hint="" if checkpoint_ready else "Click Install selected weight to fetch the registered checkpoint.",
        )
    )

    # 3. Source image is valid for the current model.
    image_valid = _as_bool(snapshot, "image_valid")
    image_name = str(snapshot.get("image_name", "") or "the source image")
    items.append(
        ReadinessItem(
            key="image_valid",
            label="Source image is valid",
            status=STATUS_READY if image_valid else STATUS_BLOCKER,
            detail=(
                f"{image_name} matches the model contract."
                if image_valid
                else f"{image_name} has not been validated for this model."
            ),
            hint="" if image_valid else "Upload a valid single-channel image for the selected model.",
        )
    )

    # 4. Symmetry maps exist and match the current image/model/settings/device.
    features_current = _as_bool(snapshot, "features_current")
    features_detail = str(snapshot.get("features_detail", "") or "")
    items.append(
        ReadinessItem(
            key="features_current",
            label="Symmetry maps are current",
            status=STATUS_READY if features_current else STATUS_BLOCKER,
            detail=features_detail
            or ("Symmetry maps match the current settings." if features_current else "Symmetry maps are missing or stale."),
            hint="" if features_current else "Compute or update the symmetry maps for the current image and model.",
        )
    )

    # 5. At least two unique classes are configured.
    names = _as_names(snapshot)
    unique_names = list(dict.fromkeys(names))
    classes_ok = len(unique_names) >= 2
    items.append(
        ReadinessItem(
            key="classes_configured",
            label="At least two classes are configured",
            status=STATUS_READY if classes_ok else STATUS_BLOCKER,
            detail=f"{len(unique_names)} unique class(es).",
            hint="" if classes_ok else "Configure at least two comma-separated, unique class names.",
        )
    )

    # 6. Each class satisfies the minimum support-point count.
    counts = _as_counts(snapshot)
    minimum = _as_int(snapshot, "minimum_shots", DEFAULT_MINIMUM_SHOTS)
    minimum = DEFAULT_MINIMUM_SHOTS if minimum is None else minimum
    counts_detail = ", ".join(
        f"{name}: {count}" for name, count in zip(names, counts)
    ) or "no support points yet"
    if not names or not counts:
        min_status = STATUS_BLOCKER
        min_hint = "Configure classes and add support points first."
    else:
        deficits = [
            (index, minimum - count)
            for index, count in enumerate(counts[: len(names)])
            if minimum - count > 0
        ]
        if deficits:
            min_status = STATUS_BLOCKER
            worst = deficits[0]
            label = names[worst[0]]
            min_hint = f"Add {worst[1]} more support point(s) to {label}."
        else:
            min_status = STATUS_READY
            min_hint = ""
    items.append(
        ReadinessItem(
            key="min_support",
            label=f"Every class has at least {minimum} support points",
            status=min_status,
            detail=counts_detail,
            hint=min_hint,
        )
    )

    # 7. Count versus minimum/recommended/maximum. The recommendation is
    # advisory; exceeding the hard maximum blocks the run.
    recommended = _as_int(snapshot, "recommended_shots", DEFAULT_RECOMMENDED_SHOTS)
    recommended = DEFAULT_RECOMMENDED_SHOTS if recommended is None else recommended
    maximum = _as_int(snapshot, "maximum_shots", None)
    if not counts:
        advice_status = STATUS_RECOMMENDATION
        advice_hint = f"{recommended} support points per class give more stable results."
    elif maximum is not None and any(count > maximum for count in counts):
        advice_status = STATUS_BLOCKER
        advice_hint = f"Remove support points until every class has at most {maximum}."
    elif all(count >= recommended for count in counts):
        advice_status = STATUS_READY
        advice_hint = ""
    else:
        advice_status = STATUS_RECOMMENDATION
        advice_hint = f"{recommended} support points per class give more stable results."
    items.append(
        ReadinessItem(
            key="support_advice",
            label="Support-point counts versus recommended range",
            status=advice_status,
            detail=(
                f"{counts_detail}. Minimum {minimum}; recommended {recommended}"
                + (f"; maximum {maximum}." if maximum is not None else ".")
            ),
            hint=advice_hint,
        )
    )

    # 8. Symmetry patch size is a positive odd integer.
    patch = _as_int(snapshot, "symmetry_patch_size", None)
    patch_ok = patch is not None and patch > 0 and patch % 2 == 1
    items.append(
        ReadinessItem(
            key="symmetry_patch",
            label="Symmetry patch size is a positive odd integer",
            status=STATUS_READY if patch_ok else STATUS_BLOCKER,
            detail=f"Symmetry patch size: {patch}." if patch is not None else "No patch size set.",
            hint="" if patch_ok else "Enter a positive odd integer for the symmetry patch size.",
        )
    )

    # 9. Runtime parameters are valid.
    runtime_problems: list[str] = []
    epochs = _as_int(snapshot, "epochs", None)
    if epochs is None or epochs <= 0:
        runtime_problems.append("epochs")
    learning_rate = _as_float(snapshot, "learning_rate", None)
    if learning_rate is None or learning_rate <= 0:
        runtime_problems.append("learning rate")
    stride = _as_int(snapshot, "stride", None)
    if stride is None or stride <= 0:
        runtime_problems.append("stride")
    batch_size = _as_int(snapshot, "batch_size", None)
    if batch_size is None or batch_size <= 0:
        runtime_problems.append("batch size")
    device = snapshot.get("device")
    if not _device_is_valid(device):
        runtime_problems.append("device")
    runtime_ok = not runtime_problems
    items.append(
        ReadinessItem(
            key="runtime_params",
            label="Runtime parameters are valid",
            status=STATUS_READY if runtime_ok else STATUS_BLOCKER,
            detail=(
                "epochs, learning rate, stride, batch size and device are valid."
                if runtime_ok
                else "Invalid: " + ", ".join(runtime_problems) + "."
            ),
            hint="" if runtime_ok else f"Fix these options: {', '.join(runtime_problems)}.",
        )
    )

    # 10. No stale downstream state remains.
    stale = _as_bool(snapshot, "stale_state")
    stale_detail = str(snapshot.get("stale_detail", "") or "")
    items.append(
        ReadinessItem(
            key="no_stale_state",
            label="No stale downstream state",
            status=STATUS_BLOCKER if stale else STATUS_READY,
            detail=stale_detail or ("Stale state detected." if stale else "No stale state."),
            hint=(
                "Recompute the symmetry maps or reconfigure classes to clear the stale result."
                if stale
                else ""
            ),
        )
    )

    return items


# ---------------------------------------------------------------------------
# Predict readiness
# ---------------------------------------------------------------------------
def _predict_extension_ok(snapshot: Mapping[str, Any]) -> bool:
    explicit = snapshot.get("extension_ok")
    if explicit is not None:
        return bool(explicit)
    path = snapshot.get("package_path")
    if not path:
        return False
    return str(path).strip().lower().endswith(".symmodel")


def evaluate_predict_readiness(
    snapshot: Mapping[str, Any],
) -> list[ReadinessItem]:
    """Evaluate the seven predict readiness checks from a snapshot dictionary.

    Recognized snapshot keys: ``provider_supported``, ``package_path``,
    ``extension_ok``, ``package_valid``, ``package_compatible``,
    ``compatibility_warnings``, ``package_detail``, ``valid_images``,
    ``invalid_images``, ``device``, ``stride``, ``batch_size``.
    """
    items: list[ReadinessItem] = []

    # 1. Provider supports saved-model prediction.
    provider_supported = _as_bool(snapshot, "provider_supported")
    items.append(
        ReadinessItem(
            key="provider_support",
            label="Provider supports saved-model prediction",
            status=STATUS_READY if provider_supported else STATUS_BLOCKER,
            detail=(
                "The Provider supports saved-model prediction."
                if provider_supported
                else "The installed Provider does not support saved-model prediction."
            ),
            hint="" if provider_supported else "Update symmetry-learn to a Provider that supports this workflow.",
        )
    )

    # 2. Selected path has the .symmodel extension.
    ext_ok = _predict_extension_ok(snapshot)
    package_path = str(snapshot.get("package_path", "") or "")
    items.append(
        ReadinessItem(
            key="extension",
            label="Selected path ends with .symmodel",
            status=STATUS_READY if ext_ok else STATUS_BLOCKER,
            detail=f"Selected: {package_path}." if package_path else "No package selected.",
            hint="" if ext_ok else "Select a file with the .symmodel extension.",
        )
    )

    # 3. Package content and checksum validation passed.
    package_valid = _as_bool(snapshot, "package_valid")
    package_detail = str(snapshot.get("package_detail", "") or "")
    items.append(
        ReadinessItem(
            key="package_valid",
            label="Package content and checksum are valid",
            status=STATUS_READY if package_valid else STATUS_BLOCKER,
            detail=package_detail or ("Package validated." if package_valid else "Package not validated."),
            hint="" if package_valid else "The package is invalid or was modified; re-export it.",
        )
    )

    # 4. Saved model/feature contract is compatible with the Provider.
    compatible = _as_bool(snapshot, "package_compatible")
    warnings = [str(w) for w in (snapshot.get("compatibility_warnings", []) or [])]
    if not package_valid:
        compat_status = STATUS_BLOCKER
        compat_detail = "Resolve the package validation issue first."
        compat_hint = "Select a valid .symmodel package."
    elif not compatible:
        compat_status = STATUS_BLOCKER
        compat_detail = package_detail or "The saved model is incompatible with the Provider."
        compat_hint = package_detail or "Load a package compatible with the installed Provider."
    elif warnings:
        compat_status = STATUS_RECOMMENDATION
        compat_detail = " ".join(warnings)
        compat_hint = "Review the compatibility warnings before running."
    else:
        compat_status = STATUS_READY
        compat_detail = "The saved model matches the installed Provider."
        compat_hint = ""
    items.append(
        ReadinessItem(
            key="compatible",
            label="Model contract is compatible with the Provider",
            status=compat_status,
            detail=compat_detail,
            hint=compat_hint,
        )
    )

    # 5. Valid versus invalid selected images.
    valid_images = _as_int(snapshot, "valid_images", 0) or 0
    invalid_images = _as_int(snapshot, "invalid_images", 0) or 0
    images_ok = valid_images >= 1
    items.append(
        ReadinessItem(
            key="images",
            label="At least one valid image is selected",
            status=STATUS_READY if images_ok else STATUS_BLOCKER,
            detail=f"{valid_images} valid / {invalid_images} invalid.",
            hint="" if images_ok else "Select at least one valid prediction image.",
        )
    )

    # 6. Device, stride and batch size are valid.
    runtime_problems: list[str] = []
    device = snapshot.get("device")
    if not _device_is_valid(device):
        runtime_problems.append("device")
    stride = _as_int(snapshot, "stride", None)
    if stride is None or stride <= 0:
        runtime_problems.append("stride")
    batch_size = _as_int(snapshot, "batch_size", None)
    if batch_size is None or batch_size <= 0:
        runtime_problems.append("batch size")
    runtime_ok = not runtime_problems
    items.append(
        ReadinessItem(
            key="runtime_params",
            label="Device, stride and batch size are valid",
            status=STATUS_READY if runtime_ok else STATUS_BLOCKER,
            detail=(
                "device, stride and batch size are valid."
                if runtime_ok
                else "Invalid: " + ", ".join(runtime_problems) + "."
            ),
            hint="" if runtime_ok else f"Fix these overrides: {', '.join(runtime_problems)}.",
        )
    )

    # 7. At least one image can be predicted.
    can_predict = (
        provider_supported
        and package_valid
        and compatible
        and images_ok
        and runtime_ok
    )
    items.append(
        ReadinessItem(
            key="can_predict",
            label="At least one image can be predicted",
            status=STATUS_READY if can_predict else STATUS_BLOCKER,
            detail=(
                "Ready to predict."
                if can_predict
                else "Resolve the blocking items above."
            ),
            hint="" if can_predict else "Resolve the blocking items above.",
        )
    )

    return items


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------
def readiness_panel_html(
    items: Sequence[ReadinessItem],
    *,
    elem_id: str | None = "readiness-panel",
) -> str:
    """Render the single shared HTML readiness panel.

    ``elem_id`` controls the wrapper's ``id`` attribute. The default keeps the
    sticky-panel id used by the Predict workspace. Pass ``elem_id=None`` when an
    outer container already owns that id (the Fine-tune sticky panel keeps the
    identically styled ``readiness-panel`` class without duplicating the id).

    Blockers always carry their immediate next action so disabling the primary
    button never leaves the user without guidance.
    """
    rows = "".join(
        (
            f'<div class="readiness-item readiness-{escape(item.status)}">'
            f'<span class="readiness-icon">{escape(STATUS_ICONS.get(item.status, "?"))}</span>'
            f'<span class="readiness-body">'
            f'<span class="readiness-label">{escape(item.label)}</span>'
            + (
                f'<span class="readiness-detail">{escape(item.detail)}</span>'
                if item.detail
                else ""
            )
            + (
                f'<span class="readiness-hint">'
                f'{"Next" if item.status == STATUS_BLOCKER else "Recommended"}: '
                f'{escape(item.hint)}</span>'
                if item.status in {STATUS_BLOCKER, STATUS_RECOMMENDATION}
                and item.hint
                else ""
            )
            + "</span></div>"
        )
        for item in items
    )
    id_attribute = "" if elem_id is None else f' id="{escape(elem_id)}"'
    return (
        f'<div{id_attribute} class="readiness-panel">'
        '<div class="readiness-title">Readiness</div>'
        f"{rows}</div>"
    )
