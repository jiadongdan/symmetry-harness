"""End-to-end few-shot execution and exact-record reproduction."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from .catalog import model_capability
from .analysis import (
    dense_prediction_from_arrays,
    render_prediction_overlay,
    render_scalar_map,
)
from .annotations import (
    AnnotationSession,
    load_annotation_session,
    save_annotation_session,
    validate_annotation_session,
)
from .config import HarnessConfig, config_for_model_selection, config_snapshot
from .contracts import HARNESS_CONTRACT_VERSION, RunOptions
from .image_io import file_sha256, inspect_input, save_preview
from .provider import (
    doctor,
    probe_model,
    provider_capabilities,
    run_provider_analysis,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_run_options(
    config: HarnessConfig, overrides: dict[str, Any] | None = None
) -> RunOptions:
    """Resolve run options from configuration and a bounded public override set."""
    options = RunOptions(
        n_max=config.features.n_max,
        symmetry_patch_size=config.features.symmetry_patch_size,
        rotation_folds=config.features.rotation_folds,
        reflection_p=config.features.reflection_p,
        normalize_rotation_maps=config.features.normalize_rotation_maps,
        input_normalization=config.features.input_normalization,
        classifier_patch_size=config.model.classifier_patch_size,
        minimum_shots_per_class=config.fine_tuning.minimum_shots_per_class,
        maximum_shots_per_class=config.fine_tuning.maximum_shots_per_class,
        adapter_bottleneck=config.fine_tuning.adapter_bottleneck,
        epochs=config.fine_tuning.epochs,
        learning_rate=config.fine_tuning.learning_rate,
        weight_decay=config.fine_tuning.weight_decay,
        seed=config.fine_tuning.seed,
        stride=config.prediction.stride,
        batch_size=config.prediction.batch_size,
        device=config.model.device,
    )
    allowed = {
        "symmetry_patch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "seed",
        "stride",
        "batch_size",
        "device",
    }
    raw = dict(overrides or {})
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unsupported run option overrides: {unknown}")
    typed: dict[str, Any] = {}
    for name, value in raw.items():
        if value is None:
            continue
        if name in {
            "symmetry_patch_size",
            "epochs",
            "seed",
            "stride",
            "batch_size",
        }:
            typed[name] = int(value)
        elif name in {"learning_rate", "weight_decay"}:
            typed[name] = float(value)
        else:
            typed[name] = str(value)
    options = replace(options, **typed)
    return _validate_run_options(options)


def _validate_run_options(options: RunOptions) -> RunOptions:
    """Validate every option that can affect execution or support selection."""
    if options.n_max <= 0:
        raise ValueError("n_max must be positive.")
    if (
        not options.rotation_folds
        or any(value <= 0 for value in options.rotation_folds)
        or len(set(options.rotation_folds)) != len(options.rotation_folds)
    ):
        raise ValueError("rotation_folds must contain unique positive integers.")
    if options.reflection_p <= 0:
        raise ValueError("reflection_p must be positive.")
    if not isinstance(options.normalize_rotation_maps, bool):
        raise ValueError("normalize_rotation_maps must be boolean.")
    if options.input_normalization not in {"minmax_0_1", "already_0_1", "dtype_unit"}:
        raise ValueError("The recorded input normalization policy is unsupported.")
    if options.classifier_patch_size <= 0:
        raise ValueError("classifier_patch_size must be positive.")
    if options.minimum_shots_per_class <= 0:
        raise ValueError("minimum_shots_per_class must be positive.")
    if options.maximum_shots_per_class < options.minimum_shots_per_class:
        raise ValueError("maximum_shots_per_class must not be smaller than the minimum.")
    if options.symmetry_patch_size <= 0 or options.symmetry_patch_size % 2 == 0:
        raise ValueError("symmetry_patch_size must be a positive odd number.")
    for name in ("epochs", "stride", "batch_size", "adapter_bottleneck"):
        if int(getattr(options, name)) <= 0:
            raise ValueError(f"{name} must be positive.")
    if options.learning_rate <= 0 or options.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative.")
    if not isinstance(options.device, str):
        raise ValueError("device must be a string.")
    if (
        options.device != "auto"
        and options.device != "cpu"
        and not options.device.startswith("cuda")
    ):
        raise ValueError("device must be auto, cpu, or a CUDA device.")
    return options


def recorded_run_options(payload: dict[str, Any]) -> RunOptions:
    """Restore the complete execution options stored in a versioned run record."""
    raw = dict(payload)
    raw["rotation_folds"] = tuple(int(value) for value in raw["rotation_folds"])
    try:
        options = RunOptions(**raw)
    except TypeError as error:
        raise ValueError("The run record contains invalid or incomplete options.") from error
    return _validate_run_options(options)


def _new_run_directory(root: Path) -> tuple[str, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"symmetry-{timestamp}-{uuid4().hex[:8]}"
    directory = root.expanduser().resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    return run_id, directory


def _save_prediction(path: Path, prediction) -> None:
    np.savez_compressed(
        path,
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


def _artifact_paths(run_dir: Path) -> dict[str, str]:
    names = {
        "annotation_session": "annotation_session.json",
        "input_array": "input.npy",
        "input_preview": "input_preview.png",
        "features": "features.npz",
        "support_patches": "support_patches.npz",
        "adapter_head": "adapter_head.pt",
        "provider_record": "provider_record.json",
        "training_history": "training_history.json",
        "prediction": "prediction.npz",
        "prediction_overlay": "prediction_overlay.png",
        "confidence": "confidence.png",
        "entropy": "entropy.png",
        "run_record": "run_record.json",
        "report": "report.md",
    }
    return {key: str((run_dir / name).resolve()) for key, name in names.items()}


def _write_report(
    path: Path,
    *,
    run_id: str,
    session: AnnotationSession,
    options: RunOptions,
    training: dict[str, Any],
    artifacts: dict[str, str],
) -> None:
    counts = "\n".join(
        f"- {entry.name}: {len(entry.points)} support points"
        for entry in session.classes
    )
    content = f"""# Symmetry Analysis Run

- Run ID: `{run_id}`
- Status: `completed`
- Classifier patch size: `{options.classifier_patch_size}`
- Symmetry patch size: `{options.symmetry_patch_size}`
- Dense prediction stride: `{options.stride}`
- Fine-tuning epochs: `{options.epochs}`

## Support Data

{counts}

## Training Diagnostics

- Best support loss: `{training['best_support_loss']:.8g}`
- Final support accuracy: `{training['final_support_accuracy']:.8g}`
- Trainable parameters: `{training['trainable_parameters']}`

Support loss and support accuracy describe the training examples only. They are
not independent estimates of generalization accuracy.

## Interpretation

The confidence and entropy artifacts describe model output, not physical
correctness. Review important boundaries and rare structures against the source
image and relevant domain evidence.

## Primary Artifacts

- Prediction overlay: `{artifacts['prediction_overlay']}`
- Prediction arrays: `{artifacts['prediction']}`
- Adapter and local head: `{artifacts['adapter_head']}`
- Provider record: `{artifacts['provider_record']}`
- Run record: `{artifacts['run_record']}`
"""
    path.write_text(content, encoding="utf-8")


def run_analysis(
    config: HarnessConfig,
    *,
    image_path: str | Path,
    annotation_session: AnnotationSession | str | Path,
    overrides: dict[str, Any] | None = None,
    output_root: str | Path | None = None,
    resolved_options: RunOptions | None = None,
) -> dict[str, Any]:
    """Validate user input and delegate the complete numerical run to the provider."""
    readiness = doctor(config)
    if readiness["status"] != "ready":
        raise RuntimeError(
            "The configured runtime is blocked: " + "; ".join(readiness["issues"])
        )
    session = (
        annotation_session
        if isinstance(annotation_session, AnnotationSession)
        else load_annotation_session(annotation_session)
    )
    if resolved_options is not None and overrides:
        raise ValueError("Use either resolved_options or overrides, not both.")
    options = (
        build_run_options(config, overrides)
        if resolved_options is None
        else _validate_run_options(resolved_options)
    )
    if session.classifier_patch_size != options.classifier_patch_size:
        raise ValueError("The annotation and configured classifier patch sizes differ.")
    validate_annotation_session(
        session,
        minimum_shots=options.minimum_shots_per_class,
        maximum_shots=options.maximum_shots_per_class,
    )

    image, input_record = inspect_input(image_path, options.input_normalization)
    if tuple(input_record["shape"]) != session.image_shape:
        raise ValueError("The annotation session image shape does not match the input.")
    if input_record["sha256"] != session.image_sha256:
        raise RuntimeError("The annotation session image checksum does not match the input.")

    root = config.output_root if output_root is None else Path(output_root)
    run_id, run_dir = _new_run_directory(root)
    artifacts = _artifact_paths(run_dir)
    started = perf_counter()
    np.save(artifacts["input_array"], image)
    portable_session = replace(
        session,
        image_path=artifacts["input_array"],
        image_sha256=file_sha256(artifacts["input_array"]),
    )
    save_annotation_session(artifacts["annotation_session"], portable_session)
    save_preview(artifacts["input_preview"], image)

    support_coordinates = []
    support_labels = []
    for entry in session.classes:
        for point in entry.points:
            support_coordinates.append(point)
            support_labels.append(entry.index)
    provider_result = run_provider_analysis(
        config,
        image,
        coordinates_xy=np.asarray(support_coordinates, dtype=np.int32),
        labels=np.asarray(support_labels, dtype=np.int64),
        class_names=[entry.name for entry in session.classes],
        options=options.to_dict(),
    )
    arrays = provider_result.arrays
    features = np.asarray(arrays.get("features"), dtype=np.float32)
    if features.shape != (
        config.model.input_channels,
        image.shape[0],
        image.shape[1],
    ):
        raise RuntimeError("Provider features do not satisfy the selected model contract.")
    channel_names = np.asarray(arrays.get("channel_names"))
    if channel_names.shape != (config.model.input_channels,):
        raise RuntimeError("Provider feature channel names are incomplete.")
    support_patches = np.asarray(arrays.get("support_patches"), dtype=np.float32)
    returned_labels = np.asarray(arrays.get("support_labels"), dtype=np.int64)
    returned_coordinates = np.asarray(
        arrays.get("support_coordinates_xy"), dtype=np.int32
    )
    if returned_labels.shape != (len(support_coordinates),):
        raise RuntimeError("Provider support labels do not match the annotation session.")
    if returned_coordinates.shape != (len(support_coordinates), 2):
        raise RuntimeError("Provider support coordinates do not match the annotation session.")
    if not np.array_equal(returned_labels, np.asarray(support_labels, dtype=np.int64)):
        raise RuntimeError("Provider changed the annotation label order.")
    if not np.array_equal(
        returned_coordinates, np.asarray(support_coordinates, dtype=np.int32)
    ):
        raise RuntimeError("Provider changed the annotation coordinate order.")
    if support_patches.shape != (
        len(support_coordinates),
        config.model.input_channels,
        options.classifier_patch_size,
        options.classifier_patch_size,
    ):
        raise RuntimeError("Provider support patches do not satisfy the model contract.")

    np.savez_compressed(
        artifacts["features"], features=features, channel_names=channel_names
    )
    np.savez_compressed(
        artifacts["support_patches"],
        patches=support_patches,
        labels=returned_labels,
        coordinates_xy=returned_coordinates,
    )
    Path(artifacts["adapter_head"]).write_bytes(provider_result.adapter_checkpoint)
    _write_json(Path(artifacts["provider_record"]), provider_result.record)
    training = dict(provider_result.record["training"])
    _write_json(Path(artifacts["training_history"]), training)

    prediction = dense_prediction_from_arrays(arrays)
    _save_prediction(Path(artifacts["prediction"]), prediction)
    colors = [entry.color for entry in session.classes]
    overlay = render_prediction_overlay(
        image, prediction, colors, stride=options.stride
    )
    Image.fromarray(overlay, mode="RGB").save(artifacts["prediction_overlay"])
    map_size = (image.shape[1], image.shape[0])
    Image.fromarray(
        render_scalar_map(
            prediction.confidence_grid,
            map_size,
            value_range=(0.0, 1.0),
        ),
        mode="RGB",
    ).save(artifacts["confidence"])
    Image.fromarray(
        render_scalar_map(
            prediction.entropy_grid,
            map_size,
            value_range=(0.0, float(np.log(len(session.classes)))),
        ),
        mode="RGB",
    ).save(artifacts["entropy"])

    class_records = [
        {
            "index": entry.index,
            "name": entry.name,
            "color": entry.color,
            "support_count": len(entry.points),
        }
        for entry in session.classes
    ]
    provider_record = provider_result.record
    runtime = dict(provider_record["runtime"])
    runtime["harness_total_seconds"] = perf_counter() - started
    warnings = list(
        dict.fromkeys(
            [
                *provider_record.get("warnings", []),
                "Confidence and entropy do not establish physical correctness.",
            ]
        )
    )
    record = {
        "contract_version": HARNESS_CONTRACT_VERSION,
        "run_id": run_id,
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": input_record,
        "annotations": {
            "path": artifacts["annotation_session"],
            "schema_version": session.to_dict()["schema_version"],
            "classes": class_records,
        },
        "provider": {
            "name": provider_record["provider"],
            "version": provider_record["provider_version"],
            "contract_version": provider_record["provider_contract_version"],
            "record_path": artifacts["provider_record"],
        },
        "model": dict(provider_record["model"]),
        "options": options.to_dict(),
        "features": dict(provider_record["features"]),
        "training": training,
        "prediction": dict(provider_record["prediction"]),
        "runtime": runtime,
        "configuration": config_snapshot(config),
        "artifacts": artifacts,
        "warnings": warnings,
    }
    _write_json(Path(artifacts["run_record"]), record)
    _write_report(
        Path(artifacts["report"]),
        run_id=run_id,
        session=session,
        options=options,
        training=training,
        artifacts=artifacts,
    )
    return {
        "status": "completed",
        "run_id": run_id,
        "run_directory": str(run_dir),
        "run_record": artifacts["run_record"],
        "prediction_overlay": artifacts["prediction_overlay"],
        "confidence": artifacts["confidence"],
        "entropy": artifacts["entropy"],
        "classes": class_records,
        "warnings": warnings,
    }


def reproduce_analysis(
    config: HarnessConfig,
    record_path: str | Path,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Repeat feature extraction and fine-tuning from a completed run record."""
    source = Path(record_path).expanduser().resolve()
    record = json.loads(source.read_text(encoding="utf-8"))
    if record.get("contract_version") != HARNESS_CONTRACT_VERSION:
        raise ValueError("Unsupported run record contract version.")
    if record.get("status") != "completed":
        raise ValueError("Only completed run records can be reproduced.")
    recorded_checkpoint = dict(record["model"]["checkpoint"])
    recorded_sha = recorded_checkpoint["sha256"]
    recorded_model_identifier = str(record["model"]["identifier"])
    capabilities = provider_capabilities(config)
    capability = model_capability(capabilities, recorded_model_identifier)
    recorded_source = recorded_checkpoint.get("source")
    uses_custom_checkpoint = recorded_source == "explicit" or (
        recorded_source is None and config.model.checkpoint_path is not None
    )
    if uses_custom_checkpoint:
        recorded_configuration = dict(record.get("configuration", {}).get("model", {}))
        candidate = config.model.checkpoint_path
        if candidate is None and recorded_configuration.get("checkpoint_path"):
            candidate = Path(recorded_configuration["checkpoint_path"])
        if candidate is None or not Path(candidate).expanduser().resolve().is_file():
            raise RuntimeError(
                "The recorded custom checkpoint is unavailable. Configure a local "
                "checkpoint with the recorded SHA-256."
            )
        config = config_for_model_selection(
            config,
            capability,
            weight_identifier=None,
            checkpoint_path=candidate,
            checkpoint_sha256=recorded_sha,
        )
    else:
        recorded_weight = recorded_checkpoint.get("weight_identifier")
        if recorded_weight in {None, "custom"}:
            recorded_weight = config.model.weight_identifier
        config = config_for_model_selection(
            config,
            capability,
            weight_identifier=recorded_weight,
        )
    current_probe = probe_model(config)
    current_checkpoint = dict(current_probe.get("details", {}).get("checkpoint", {}))
    if current_checkpoint.get("sha256") != recorded_sha:
        raise RuntimeError("The selected model weight does not match the recorded run.")
    annotation_path = Path(record["annotations"]["path"]).expanduser().resolve()
    session = load_annotation_session(annotation_path)
    candidate_paths = [
        record.get("artifacts", {}).get("input_array"),
        record.get("input", {}).get("path"),
    ]
    image_path = None
    for candidate in candidate_paths:
        if not candidate:
            continue
        candidate_path = Path(candidate).expanduser().resolve()
        if candidate_path.is_file() and file_sha256(candidate_path) == session.image_sha256:
            image_path = candidate_path
            break
    if image_path is None:
        raise RuntimeError(
            "Neither the bundled input nor the original input matches the annotation checksum."
        )
    raw_options = dict(record["options"])
    recorded_tuning = record.get("configuration", {}).get("fine_tuning", {})
    raw_options.setdefault(
        "minimum_shots_per_class",
        recorded_tuning.get("minimum_shots_per_class", 3),
    )
    raw_options.setdefault(
        "maximum_shots_per_class",
        recorded_tuning.get("maximum_shots_per_class", 50),
    )
    options = recorded_run_options(raw_options)
    return run_analysis(
        config,
        image_path=image_path,
        annotation_session=session,
        output_root=output_root,
        resolved_options=options,
    )
