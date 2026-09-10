"""Orchestrate saved-model prediction without any fine-tuning step."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import tempfile
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image

from . import __version__ as harness_version
from .analysis import (
    dense_prediction_from_arrays,
    prediction_display_bounds,
    render_prediction_overlay,
    render_scalar_map,
)
from .catalog import model_capability
from .config import HarnessConfig, config_for_model_selection
from .contracts import PREDICTION_RUN_CONTRACT_VERSION
from .image_io import file_sha256, inspect_input, save_preview
from .model_package import FineTunedModelPackage, inspect_model_package, materialize_model_state
from .provider import (
    provider_capabilities,
    run_provider_prediction_batch,
)


PREDICTION_OPERATION = "predict_with_fine_tuned_model"


class PredictionError(RuntimeError):
    """Raised when a saved-model prediction request cannot be executed."""


@dataclass(frozen=True)
class PredictionRequest:
    """One validated saved-model prediction request."""

    package: FineTunedModelPackage
    config: HarnessConfig
    image_paths: tuple[Path, ...]
    device: str
    stride: int
    batch_size: int
    compatibility_warnings: tuple[str, ...] = ()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _resolve_runtime_value(
    name: str, override: Any, saved: Any, *, minimum: int = 1
) -> int:
    """Resolve one runtime-only override, falling back to the saved default."""
    value = saved if override is None else override
    try:
        resolved = int(value)
    except (TypeError, ValueError) as error:
        raise PredictionError(f"{name} must be an integer.") from error
    if resolved < minimum:
        raise PredictionError(f"{name} must be at least {minimum}.")
    return resolved


def _version_core(value: Any) -> tuple[int, int, int] | None:
    """Return a numeric semantic-version core without adding a dependency."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(value or "").strip())
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def _check_provider_compatibility(
    config: HarnessConfig,
    capabilities: dict[str, Any],
    package: FineTunedModelPackage,
) -> tuple[dict[str, Any], list[str]]:
    """Return model capability and warnings, blocking real incompatibilities."""
    operations = capabilities.get("operations", [])
    if PREDICTION_OPERATION not in operations:
        raise PredictionError(
            "The installed symmetry Provider does not support saved-model "
            "prediction. Update symmetry-learn to use this workflow."
        )
    identifier = package.model_identifier
    available = {
        str(entry.get("identifier"))
        for entry in capabilities.get("models", [])
        if isinstance(entry, dict)
    }
    if identifier not in available:
        raise PredictionError(
            f"The saved model {identifier!r} is not installed in this Provider."
        )
    capability = model_capability(capabilities, identifier)
    if not bool(capability.get("available", False)):
        raise PredictionError(
            f"The Provider reports saved model {identifier!r} as unavailable."
        )
    if int(capability.get("input_channels", -1)) != int(
        package.manifest["model"]["input_channels"]
    ):
        raise PredictionError(
            "The saved model input-channel contract does not match the Provider."
        )
    if int(capability.get("classifier_patch_size", -1)) != int(
        package.manifest["model"]["classifier_patch_size"]
    ):
        raise PredictionError(
            "The saved model patch-size contract does not match the Provider."
        )
    if str(capability.get("feature_pipeline", "")) != str(
        package.manifest["features"]["pipeline"]
    ):
        raise PredictionError(
            "The saved feature pipeline does not match the installed Provider."
        )
    if list(capability.get("feature_channels", [])) != list(
        package.manifest["features"]["channel_names"]
    ):
        raise PredictionError(
            "The saved feature channel order does not match the installed Provider."
        )
    warnings: list[str] = []
    software = package.manifest.get("software", {})
    recorded_provider = software.get("symmetry_learn_version")
    current_provider = capabilities.get("provider_version")
    recorded_core = _version_core(recorded_provider)
    current_core = _version_core(current_provider)
    if recorded_core is not None and current_core is not None:
        incompatible = recorded_core[0] != current_core[0] or (
            recorded_core[0] == 0 and recorded_core[1] != current_core[1]
        )
        if incompatible:
            raise PredictionError(
                "The saved model was created by an incompatible symmetry-learn "
                f"version ({recorded_provider}); the installed Provider is "
                f"{current_provider}."
            )
        if recorded_core != current_core:
            warnings.append(
                "The saved model and installed symmetry-learn patch versions differ "
                f"({recorded_provider} vs {current_provider})."
            )
    recorded_harness = software.get("symmetry_harness_version")
    if recorded_harness and str(recorded_harness) != harness_version:
        warnings.append(
            "The saved model was created by symmetry-harness "
            f"{recorded_harness}; the current version is {harness_version}."
        )
    recorded_torch = software.get("torch_version")
    current_torch = dict(capabilities.get("runtime", {}) or {}).get(
        "torch_version"
    )
    recorded_torch_core = _version_core(recorded_torch)
    current_torch_core = _version_core(current_torch)
    if recorded_torch_core is not None and current_torch_core is not None:
        if recorded_torch_core[0] != current_torch_core[0]:
            raise PredictionError(
                "The saved model was created with an incompatible PyTorch major "
                f"version ({recorded_torch}); the Provider has {current_torch}."
            )
        if recorded_torch_core != current_torch_core:
            warnings.append(
                "The saved model and Provider PyTorch versions differ "
                f"({recorded_torch} vs {current_torch})."
            )
    return capability, warnings


def validate_prediction_request(
    config: HarnessConfig,
    *,
    model_package: str | Path,
    image_paths: list[str | Path],
    device: str | None = None,
    stride: int | None = None,
    batch_size: int | None = None,
    allow_partial: bool = False,
) -> PredictionRequest:
    """Validate one package and its inputs before any Provider job is submitted."""
    if not image_paths:
        raise PredictionError("Select at least one image to predict.")
    package = inspect_model_package(model_package)
    capabilities = provider_capabilities(config)
    capability, compatibility_warnings = _check_provider_compatibility(
        config, capabilities, package
    )
    config = config_for_model_selection(
        config, capability, weight_identifier=None
    )

    defaults = package.prediction_defaults
    resolved_device = "auto" if device is None else str(device)
    if (
        resolved_device != "auto"
        and resolved_device != "cpu"
        and not resolved_device.startswith("cuda")
    ):
        raise PredictionError("device must be auto, cpu, or a CUDA device.")
    resolved_stride = _resolve_runtime_value("stride", stride, defaults.get("stride", 4))
    resolved_batch = _resolve_runtime_value(
        "batch_size", batch_size, defaults.get("batch_size", 512)
    )

    resolved_images: list[Path] = []
    for raw in image_paths:
        candidate = Path(raw).expanduser().resolve()
        if not allow_partial and not candidate.is_file():
            raise PredictionError(f"Input image does not exist: {candidate}")
        resolved_images.append(candidate)
    if not allow_partial:
        minimum_size = int(package.manifest["model"]["classifier_patch_size"])
        too_small: list[str] = []
        for path in resolved_images:
            try:
                image, _ = inspect_input(path, package.input_normalization)
            except FileNotFoundError:
                raise
            except (ValueError, TypeError) as error:
                # Surface every input-validation failure as one workflow error so
                # callers always receive an actionable prediction message.
                raise PredictionError(f"{path.name}: {error}") from error
            if min(image.shape) < minimum_size:
                too_small.append(path.name)
        if too_small:
            raise PredictionError(
                "These images are smaller than the saved classifier patch "
                f"({minimum_size} pixels): {', '.join(too_small)}."
            )
    return PredictionRequest(
        package=package,
        config=config,
        image_paths=tuple(resolved_images),
        device=resolved_device,
        stride=resolved_stride,
        batch_size=resolved_batch,
        compatibility_warnings=tuple(compatibility_warnings),
    )


def _new_prediction_directory(root: Path) -> tuple[str, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"prediction-{timestamp}-{uuid4().hex[:8]}"
    directory = root.expanduser().resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    return run_id, directory


def _class_statistics(
    prediction_grid: np.ndarray, class_count: int
) -> dict[str, Any]:
    """Return per-class counts and fractions over the predicted grid."""
    values = np.asarray(prediction_grid, dtype=np.int64)
    total = int(values.size)
    counts = np.bincount(values.ravel(), minlength=class_count)
    return {
        "grid_cell_count": total,
        "classes": [
            {
                "index": index,
                "count": int(counts[index]),
                "fraction": float(counts[index] / total) if total else 0.0,
            }
            for index in range(class_count)
        ],
    }


def _render_item_artifacts(
    directory: Path,
    *,
    image: np.ndarray,
    arrays: dict[str, np.ndarray],
    colors: list[str],
    stride: int,
) -> dict[str, Any]:
    prediction = dense_prediction_from_arrays(arrays)
    np.savez_compressed(
        directory / "prediction.npz",
        coordinates_xy=prediction.coordinates_xy,
        x_coordinates=prediction.x_coordinates,
        y_coordinates=prediction.y_coordinates,
        logits=prediction.logits,
        probabilities=prediction.probabilities,
        predictions=prediction.predictions,
        confidence=prediction.confidence,
        entropy=prediction.entropy,
        prediction_grid=prediction.prediction_grid,
        confidence_grid=prediction.confidence_grid,
        entropy_grid=prediction.entropy_grid,
    )
    overlay = render_prediction_overlay(
        image, prediction, colors, stride=stride
    )
    Image.fromarray(overlay).save(directory / "prediction_overlay.png")
    left, top, right, bottom = prediction_display_bounds(
        prediction, image.shape, stride=stride
    )
    map_size = (right - left, bottom - top)
    Image.fromarray(
        render_scalar_map(
            prediction.confidence_grid, map_size, value_range=(0.0, 1.0)
        ),
    ).save(directory / "confidence.png")
    Image.fromarray(
        render_scalar_map(
            prediction.entropy_grid,
            map_size,
            value_range=(0.0, float(np.log(max(len(colors), 2)))),
        ),
    ).save(directory / "entropy.png")
    return {
        "prediction": directory / "prediction.npz",
        "prediction_overlay": directory / "prediction_overlay.png",
        "confidence": directory / "confidence.png",
        "entropy": directory / "entropy.png",
        "grid_shape": list(prediction.prediction_grid.shape),
        "bounds": {"left": left, "top": top, "right": right, "bottom": bottom},
    }


def _write_batch_report(
    path: Path,
    *,
    run_id: str,
    request: PredictionRequest,
    items: list[dict[str, Any]],
) -> None:
    lines = "\n".join(
        f"- `{entry['item_id']}`: {entry['status']}"
        + (f" — {entry['error']}" if entry.get("error") else "")
        for entry in items
    )
    content = f"""# Saved-Model Prediction Run

- Run ID: `{run_id}`
- Model package: `{request.package.source_path}`
- Model identifier: `{request.package.model_identifier}`
- Dense prediction stride: `{request.stride}`
- Classes: {", ".join(request.package.class_names)}

## Images

{lines}

## Interpretation

The confidence and entropy artifacts describe model output, not physical
correctness. Review important boundaries and rare structures against the source
image and relevant domain evidence.

Saved-model prediction does not fine-tune. Feature parameters and the input
normalization policy are restored from the model package and are read-only.
"""
    path.write_text(content, encoding="utf-8")


def _batch_archive(directory: Path) -> Path:
    """Package root records plus every available per-image artifact."""
    archive_path = directory / "results.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for root_name in ("batch_record.json", "report.md"):
            artifact = directory / root_name
            if artifact.is_file():
                archive.write(artifact, root_name)
        items_root = directory / "items"
        if items_root.is_dir():
            for artifact in sorted(items_root.rglob("*")):
                if artifact.is_file():
                    archive.write(artifact, str(artifact.relative_to(directory)))
    return archive_path


def _downloadable_copy(archive_path: Path, run_id: str) -> Path:
    """Copy one artifact into the system temporary directory for downloads.

    Gradio only serves files inside the current working directory, the system
    temporary directory, or an explicitly allowed path. Prediction run
    directories live under the Harness state directory and match none of those,
    so a browser download needs a temporary copy. The run directory stays the
    durable record and is never moved or deleted.
    """
    download_directory = Path(tempfile.gettempdir()) / "symmetry-harness-downloads"
    download_directory.mkdir(parents=True, exist_ok=True)
    destination = download_directory / f"{run_id}-results.zip"
    shutil.copyfile(archive_path, destination)
    return destination


def run_saved_model_prediction_batch(
    config: HarnessConfig,
    *,
    model_package: str | Path,
    image_paths: list[str | Path],
    output_root: str | Path | None = None,
    device: str | None = None,
    stride: int | None = None,
    batch_size: int | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Predict one or more images with a saved fine-tuned model package."""
    started = perf_counter()
    request = validate_prediction_request(
        config,
        model_package=model_package,
        image_paths=image_paths,
        device=device,
        stride=stride,
        batch_size=batch_size,
        allow_partial=True,
    )
    package = request.package
    root = config.output_root if output_root is None else Path(output_root)
    run_id, run_dir = _new_prediction_directory(root)
    items_root = run_dir / "items"
    items_root.mkdir(parents=True, exist_ok=True)

    prepared: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    minimum_size = int(package.manifest["model"]["classifier_patch_size"])
    for index, image_path in enumerate(request.image_paths, start=1):
        item_id = f"image-{index:04d}"
        item_directory = items_root / item_id
        item_directory.mkdir(parents=True, exist_ok=True)
        try:
            image, input_record = inspect_input(
                image_path, package.input_normalization
            )
            if min(image.shape) < minimum_size:
                raise PredictionError(
                    f"Image is smaller than the saved {minimum_size}-pixel "
                    "classifier patch."
                )
        except Exception as error:
            failure = {
                "contract_version": PREDICTION_RUN_CONTRACT_VERSION,
                "run_id": run_id,
                "item_id": item_id,
                "status": "failed",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "phase": "input_validation",
                "source_path": str(image_path),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _write_json(item_directory / "prediction_record.json", failure)
            results.append(
                {
                    **failure,
                    "item_directory": str(item_directory),
                    "prediction_record": str(
                        item_directory / "prediction_record.json"
                    ),
                }
            )
            continue
        np.save(item_directory / "input.npy", image)
        save_preview(item_directory / "input_preview.png", image)
        prepared.append(
            {
                "item_id": item_id,
                "item_directory": item_directory,
                "input_record": input_record,
                "source_path": image_path,
            }
        )

    temporary = Path(tempfile.mkdtemp(prefix="symmetry-harness-model-state-"))
    try:
        model_state_path = materialize_model_state(
            package, temporary / "model_state.pt"
        )
        feature_options = {
            name: package.feature_options[name]
            for name in (
                "n_max",
                "symmetry_patch_size",
                "rotation_folds",
                "reflection_p",
                "normalize_rotation_maps",
            )
        }
        job_items = [
            {
                "item_id": entry["item_id"],
                "input_path": str(entry["item_directory"] / "input.npy"),
                "output_path": str(entry["item_directory"] / "prediction.npz"),
                "record_path": str(
                    entry["item_directory"] / "provider_record.json"
                ),
            }
            for entry in prepared
        ]
        if job_items:
            batch = run_provider_prediction_batch(
                request.config,
                model_state_path=model_state_path,
                expected_model={
                    "identifier": package.model_identifier,
                    "task_classes": package.task_classes,
                    "adapter_bottleneck": package.manifest["model"][
                        "adapter_bottleneck"
                    ],
                    "class_names": package.class_names,
                    "base_checkpoint_sha256": package.manifest.get(
                        "provenance", {}
                    ).get("base_checkpoint_sha256"),
                    "model_state_sha256": package.model_state_sha256,
                },
                feature_options=feature_options,
                prediction_options={
                    "device": request.device,
                    "stride": request.stride,
                    "batch_size": request.batch_size,
                },
                items=job_items,
                progress_path=run_dir / "progress.json",
                progress_callback=progress_callback,
            )
            batch_summary = batch.summary
        else:
            batch_summary = {}
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    statuses = {
        str(entry.get("item_id")): entry
        for entry in batch_summary.get("items", [])
    }
    colors = package.class_colors
    for entry in prepared:
        item_id = entry["item_id"]
        item_directory: Path = entry["item_directory"]
        status = statuses.get(item_id, {})
        if status.get("status") != "completed":
            failure = {
                "contract_version": PREDICTION_RUN_CONTRACT_VERSION,
                "run_id": run_id,
                "item_id": item_id,
                "status": "failed",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "phase": "prediction",
                "source_path": str(entry["source_path"]),
                "error_type": str(status.get("error_type", "PredictionError")),
                "error": str(
                    status.get("error", "The Provider did not report this item.")
                ),
            }
            _write_json(item_directory / "prediction_record.json", failure)
            results.append(
                {
                    **failure,
                    "item_directory": str(item_directory),
                    "prediction_record": str(
                        item_directory / "prediction_record.json"
                    ),
                }
            )
            continue
        with np.load(item_directory / "prediction.npz", allow_pickle=False) as archive:
            arrays = {
                name: np.asarray(archive[name]).copy() for name in archive.files
            }
        if "features" not in arrays or "channel_names" not in arrays:
            raise RuntimeError(
                "Provider prediction output is missing reusable feature arrays."
            )
        np.savez_compressed(
            item_directory / "features.npz",
            features=np.asarray(arrays["features"], dtype=np.float32),
            channel_names=np.asarray(arrays["channel_names"]),
        )
        artifacts = _render_item_artifacts(
            item_directory,
            image=np.asarray(
                np.load(item_directory / "input.npy", allow_pickle=False),
                dtype=np.float32,
            ),
            arrays=arrays,
            colors=colors,
            stride=request.stride,
        )
        provider_record = json.loads(
            (item_directory / "provider_record.json").read_text(encoding="utf-8")
        )
        statistics = _class_statistics(
            arrays["prediction_grid"], len(colors)
        )
        record = {
            "contract_version": PREDICTION_RUN_CONTRACT_VERSION,
            "run_id": run_id,
            "item_id": item_id,
            "status": "completed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "package": {
                "path": str(package.source_path),
                "package_sha256": package.package_sha256,
                "schema_version": package.manifest["schema_version"],
                "model_state_sha256": package.model_state_sha256,
            },
            "model": {
                "identifier": package.model_identifier,
                "task_classes": package.task_classes,
                "class_names": package.class_names,
                "class_colors": colors,
                "classifier_patch_size": package.manifest["model"][
                    "classifier_patch_size"
                ],
                "adapter_bottleneck": package.manifest["model"]["adapter_bottleneck"],
            },
            "input": {
                "path": str(entry["source_path"]),
                **entry["input_record"],
            },
            "features": feature_options,
            "prediction_options": {
                "device": request.device,
                "stride": request.stride,
                "batch_size": request.batch_size,
            },
            "provider": {
                "name": provider_record.get("provider"),
                "version": provider_record.get("provider_version"),
                "contract_version": provider_record.get(
                    "provider_contract_version"
                ),
                "record_path": str(item_directory / "provider_record.json"),
            },
            "output": {
                "grid_shape": artifacts["grid_shape"],
                "bounds": artifacts["bounds"],
                "class_statistics": statistics,
            },
            "artifacts": {
                "prediction": str(artifacts["prediction"]),
                "features": str(item_directory / "features.npz"),
                "prediction_overlay": str(artifacts["prediction_overlay"]),
                "confidence": str(artifacts["confidence"]),
                "entropy": str(artifacts["entropy"]),
                "input": str(item_directory / "input.npy"),
                "input_preview": str(item_directory / "input_preview.png"),
            },
            "warnings": [
                "Confidence and entropy do not establish physical correctness."
            ],
        }
        _write_json(item_directory / "prediction_record.json", record)
        results.append(
            {
                "item_id": item_id,
                "item_directory": str(item_directory),
                "status": "completed",
                "source_path": str(entry["source_path"]),
                **record["output"],
                **{
                    f"artifact_{key}": value
                    for key, value in record["artifacts"].items()
                },
            }
        )

    results.sort(key=lambda entry: str(entry["item_id"]))
    completed = [entry for entry in results if entry["status"] == "completed"]
    failed = [entry for entry in results if entry["status"] != "completed"]
    record_items = [
        {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in entry.items()
        }
        for entry in results
    ]
    batch_record = {
        "contract_version": PREDICTION_RUN_CONTRACT_VERSION,
        "run_id": run_id,
        "status": "completed" if completed else "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "symmetry_harness_version": harness_version,
        "package": {
            "path": str(package.source_path),
            "package_sha256": package.package_sha256,
            "schema_version": package.manifest["schema_version"],
            "model_state_sha256": package.model_state_sha256,
            "training_run_id": package.manifest.get("provenance", {}).get(
                "training_run_id"
            ),
        },
        "model": {
            "identifier": package.model_identifier,
            "task_classes": package.task_classes,
            "class_names": package.class_names,
            "class_colors": colors,
        },
        "features": feature_options,
        "prediction_options": {
            "device": request.device,
            "stride": request.stride,
            "batch_size": request.batch_size,
        },
        "input_normalization": package.input_normalization,
        "provider": {
            "name": batch_summary.get("provider"),
            "version": batch_summary.get("provider_version"),
            "contract_version": batch_summary.get(
                "provider_contract_version"
            ),
            "transport": batch_summary.get("transport", {}),
        },
        "counts": {
            "requested": len(results),
            "completed": len(completed),
            "failed": len(failed),
        },
        "items": record_items,
        "artifacts": {
            "batch_record": str(run_dir / "batch_record.json"),
            "report": str(run_dir / "report.md"),
            "results_archive": str(run_dir / "results.zip"),
        },
        "warnings": list(
            dict.fromkeys(
                [
                    *request.compatibility_warnings,
                    "Confidence and entropy do not establish physical correctness.",
                ]
            )
        ),
        "runtime_seconds": perf_counter() - started,
    }
    _write_json(run_dir / "batch_record.json", batch_record)
    _write_batch_report(
        run_dir / "report.md", run_id=run_id, request=request, items=results
    )
    archive_path = _batch_archive(run_dir)
    # Browser downloads can only read files Gradio is allowed to serve, so the
    # durable run archive stays put and a temporary copy is offered instead.
    download_path = _downloadable_copy(archive_path, run_id)
    return {
        "status": batch_record["status"],
        "run_id": run_id,
        "run_directory": str(run_dir),
        "batch_record": str(run_dir / "batch_record.json"),
        "report": str(run_dir / "report.md"),
        "results_archive": str(archive_path),
        "download_archive": str(download_path),
        "counts": batch_record["counts"],
        "items": record_items,
        "class_names": package.class_names,
        "class_colors": colors,
        "warnings": batch_record["warnings"],
    }


def run_saved_model_prediction(
    config: HarnessConfig,
    *,
    model_package: str | Path,
    image_path: str | Path,
    output_root: str | Path | None = None,
    device: str | None = None,
    stride: int | None = None,
    batch_size: int | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Predict one image with a saved fine-tuned model package."""
    batch = run_saved_model_prediction_batch(
        config,
        model_package=model_package,
        image_paths=[image_path],
        output_root=output_root,
        device=device,
        stride=stride,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )
    first = batch["items"][0] if batch["items"] else {}
    return {
        **batch,
        "item": first,
        "prediction_overlay": first.get("artifact_prediction_overlay"),
        "confidence": first.get("artifact_confidence"),
        "entropy": first.get("artifact_entropy"),
    }
