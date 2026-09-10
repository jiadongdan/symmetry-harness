"""Standalone traditional-ML validation runs.

One run validates its inputs, persists the exact features and annotations it
used, delegates training plus dense prediction to the Provider, validates the
returned record, renders the display artifacts, and writes a versioned run
record. The workflow never resolves, loads, or probes a pretrained checkpoint,
adapter, or saved model, and it never serializes a scikit-learn estimator.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image

from . import __version__ as harness_version
from .analysis import (
    prediction_display_bounds,
    render_prediction_overlay,
    render_scalar_map,
    traditional_prediction_from_arrays,
)
from .annotations import (
    AnnotationSession,
    load_annotation_session,
    save_annotation_session,
    validate_annotation_session,
)
from .config import HarnessConfig, config_snapshot
from .contracts import (
    TRADITIONAL_FEATURE_MODES,
    TRADITIONAL_MAXIMUM_PATCHES_PER_CLASS,
    TRADITIONAL_MINIMUM_PATCHES_PER_CLASS,
    TRADITIONAL_RUN_CONTRACT_VERSION,
    TraditionalMLSettings,
)
from .image_io import file_sha256, inspect_input
from .provider import (
    TRADITIONAL_ML_OPERATION,
    run_provider_traditional_ml,
    traditional_classifier,
    traditional_readiness,
)


ProgressCallback = Callable[[str, int, int], None]

RUN_ID_PREFIX = "traditional-validation"

ARTIFACT_NAMES = {
    "input_array": "input.npy",
    "annotation_session": "annotation_session.json",
    "features": "features.npz",
    "features_record": "features_record.json",
    "progress": "progress.json",
    "traditional_prediction": "traditional_prediction.npz",
    "provider_record": "provider_record.json",
    "training_summary": "training_summary.json",
    "prediction_overlay": "prediction_overlay.png",
    "confidence": "confidence.png",
    "entropy": "entropy.png",
    "report": "report.md",
    "run_record": "run_record.json",
    "results_archive": "results.zip",
}


def artifact_paths(run_directory: Path) -> dict[str, str]:
    """Return the absolute artifact paths of one traditional validation run."""
    return {
        key: str((run_directory / name).resolve())
        for key, name in ARTIFACT_NAMES.items()
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def new_run_directory(root: Path) -> tuple[str, Path]:
    """Create a unique traditional validation run directory."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{RUN_ID_PREFIX}-{timestamp}-{uuid4().hex[:8]}"
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


def _class_records(session: AnnotationSession) -> list[dict[str, Any]]:
    return [
        {
            "index": entry.index,
            "name": entry.name,
            "color": entry.color,
            "support_count": len(entry.points),
            "support_points_xy": [list(point) for point in entry.points],
        }
        for entry in session.classes
    ]


def _load_features_artifact(
    features_path: Path,
    features_record_path: Path,
    *,
    expected_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and validate the eight-channel artifact used by one run."""
    with np.load(features_path, allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float32)
        channel_names = np.asarray(archive["channel_names"])
    if features.shape != expected_shape:
        raise ValueError(
            "The feature artifact does not match the loaded image: "
            f"{tuple(features.shape)} != {tuple(expected_shape)}."
        )
    if not np.isfinite(features).all():
        raise ValueError("The feature artifact contains nonfinite values.")
    if channel_names.shape != (expected_shape[0],):
        raise ValueError("The feature artifact is missing channel names.")
    record = json.loads(features_record_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not isinstance(record.get("features"), dict):
        raise ValueError("The feature record does not contain a feature contract.")
    return features, channel_names, record


def _write_report(
    path: Path,
    *,
    run_id: str,
    session: AnnotationSession,
    classifier: str,
    feature_mode: str,
    settings: TraditionalMLSettings,
    provider_record: dict[str, Any],
) -> None:
    counts = "\n".join(
        f"- {entry.name}: {len(entry.points)} support patches"
        for entry in session.classes
    )
    parameters = dict(provider_record["classifier"]["parameters"])
    parameter_lines = "\n".join(
        f"- `{name}`: `{value!r}`" for name, value in sorted(parameters.items())
    )
    content = f"""# Traditional ML Validation Run

- Run ID: `{run_id}`
- Classifier: `{classifier}`
- Feature mode: `{feature_mode}`
- Classifier patch size: `{settings.classifier_patch_size}`
- Dense prediction stride: `{settings.stride}`
- Seed: `{settings.seed}`

## Support Data

{counts}

## Resolved Parameters

{parameter_lines}

## Diagnostic Only

- Support training accuracy: `{provider_record['support_training_accuracy']:.6g}`
- Support training accuracy is measured on the user-selected support patches
  themselves. It is not held-out validation accuracy.

## Interpretation

This page is an exploratory validation tool. It compares a conventional
scikit-learn classifier trained on user-selected patches against nothing more
than itself, and it never loads, probes, or requires a pretrained neural-network
checkpoint. Conclusions drawn from it are exploratory; they are not formal proof
that a pretrained model is necessary or unnecessary.

Confidence and entropy describe classifier output, not physical correctness.
Review important boundaries and rare structures against the source image and
relevant domain evidence.
"""
    path.write_text(content, encoding="utf-8")


def _write_archive(run_directory: Path, archive_path: Path) -> Path:
    """Package every completed run artifact into a single ZIP."""
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for artifact in sorted(run_directory.rglob("*")):
            if not artifact.is_file() or artifact == archive_path:
                continue
            archive.write(artifact, str(artifact.relative_to(run_directory)))
    return archive_path


def _provider_failure_record(
    *,
    run_id: str,
    artifacts: dict[str, str],
    phase: str,
    error: BaseException,
    configuration: dict[str, Any],
) -> None:
    """Preserve a failed run record without presenting it as completed."""
    record = {
        "contract_version": TRADITIONAL_RUN_CONTRACT_VERSION,
        "run_id": run_id,
        "status": "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "harness_version": harness_version,
        "phase": phase,
        "error_type": type(error).__name__,
        "error": str(error),
        "configuration": configuration,
        "artifacts": {
            key: value
            for key, value in artifacts.items()
            if Path(value).is_file()
        },
    }
    _write_json(Path(artifacts["run_record"]), record)


def run_traditional_validation(
    config: HarnessConfig,
    *,
    image_path: str | Path,
    annotation_session: AnnotationSession | str | Path,
    classifier: str,
    feature_mode: str,
    settings: TraditionalMLSettings,
    parameters: dict[str, Any] | None = None,
    features_path: str | Path,
    features_record_path: str | Path,
    output_root: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Train one conventional classifier and densely predict one image."""
    if not isinstance(settings, TraditionalMLSettings):
        raise TypeError("settings must be a TraditionalMLSettings instance.")
    readiness = traditional_readiness(config, require_ui=False)
    if readiness["status"] != "ready":
        raise RuntimeError(
            "The traditional validation runtime is blocked: "
            + "; ".join(readiness["issues"])
        )
    capabilities = readiness["environment"]["capabilities"]
    classifier_record = traditional_classifier(capabilities, classifier)
    identifier = str(classifier_record["identifier"])
    if not bool(classifier_record.get("supports_predict_proba", False)):
        raise RuntimeError(
            f"The Provider does not guarantee predict_proba for {identifier!r}."
        )
    mode = str(feature_mode).strip()
    if mode not in TRADITIONAL_FEATURE_MODES:
        raise ValueError(
            f"Unsupported traditional ML feature mode: {feature_mode!r}. "
            f"Supported modes are {list(TRADITIONAL_FEATURE_MODES)}."
        )
    session = (
        annotation_session
        if isinstance(annotation_session, AnnotationSession)
        else load_annotation_session(annotation_session)
    )
    if session.classifier_patch_size != settings.classifier_patch_size:
        raise ValueError(
            "The annotation session and the requested classifier patch size differ."
        )
    validate_annotation_session(
        session,
        minimum_shots=TRADITIONAL_MINIMUM_PATCHES_PER_CLASS,
        maximum_shots=TRADITIONAL_MAXIMUM_PATCHES_PER_CLASS,
    )

    image, input_record = inspect_input(image_path, config.features.input_normalization)
    if tuple(input_record["shape"]) != session.image_shape:
        raise ValueError("The annotation session image shape does not match the input.")
    if input_record["sha256"] != session.image_sha256:
        raise RuntimeError(
            "The annotation session image checksum does not match the input."
        )

    source_features = Path(features_path).expanduser().resolve()
    source_feature_record = Path(features_record_path).expanduser().resolve()
    if not source_features.is_file() or not source_feature_record.is_file():
        raise ValueError(
            "The symmetry feature artifact is missing or stale. Compute it again."
        )
    expected_shape = (
        int(config.model.input_channels),
        int(image.shape[0]),
        int(image.shape[1]),
    )
    features, channel_names, feature_record = _load_features_artifact(
        source_features, source_feature_record, expected_shape=expected_shape
    )

    root = config.output_root if output_root is None else Path(output_root)
    run_id, run_directory = new_run_directory(root)
    artifacts = artifact_paths(run_directory)
    started = perf_counter()
    try:
        # 1. Persist the exact inputs this run consumed.
        np.save(artifacts["input_array"], image)
        portable_session = AnnotationSession(
            image_path=artifacts["input_array"],
            image_sha256=file_sha256(artifacts["input_array"]),
            image_shape=session.image_shape,
            classifier_patch_size=session.classifier_patch_size,
            classes=session.classes,
            created_utc=session.created_utc,
        )
        save_annotation_session(artifacts["annotation_session"], portable_session)
        shutil.copyfile(source_features, artifacts["features"])
        shutil.copyfile(source_feature_record, artifacts["features_record"])
        feature_artifact_sha256 = file_sha256(artifacts["features"])

        support_coordinates: list[tuple[int, int]] = []
        support_labels: list[int] = []
        for entry in session.classes:
            for point in entry.points:
                support_coordinates.append((int(point[0]), int(point[1])))
                support_labels.append(int(entry.index))

        provider_result = run_provider_traditional_ml(
            config,
            input_path=artifacts["input_array"],
            features_path=artifacts["features"],
            features_record_path=artifacts["features_record"],
            coordinates_xy=np.asarray(support_coordinates, dtype=np.int32),
            labels=np.asarray(support_labels, dtype=np.int64),
            class_names=[entry.name for entry in session.classes],
            classifier=identifier,
            parameters=parameters,
            feature_mode=mode,
            options=settings.to_provider_options(),
            output_path=artifacts["traditional_prediction"],
            record_path=artifacts["provider_record"],
            input_sha256=portable_session.image_sha256,
            progress_path=artifacts["progress"],
            progress_callback=progress_callback,
        )
        provider_record = provider_result.record
        returned_names = [str(name) for name in provider_record["class_names"]]
        if returned_names != [entry.name for entry in session.classes]:
            raise RuntimeError("The Provider changed the requested class order.")
        prediction = traditional_prediction_from_arrays(
            provider_result.arrays, class_count=len(session.classes)
        )
        if list(prediction.prediction_grid.shape) != [
            int(value) for value in provider_record["grid_shape"]
        ]:
            raise RuntimeError("The Provider grid shape disagrees with its record.")

        # 2. Render display artifacts from the validated arrays.
        colors = [entry.color for entry in session.classes]
        overlay = render_prediction_overlay(
            image, prediction, colors, stride=settings.stride
        )
        Image.fromarray(overlay).save(artifacts["prediction_overlay"])
        left, top, right, bottom = prediction_display_bounds(
            prediction, image.shape, stride=settings.stride
        )
        map_size = (right - left, bottom - top)
        Image.fromarray(
            render_scalar_map(
                prediction.confidence_grid, map_size, value_range=(0.0, 1.0)
            )
        ).save(artifacts["confidence"])
        Image.fromarray(
            render_scalar_map(
                prediction.entropy_grid,
                map_size,
                value_range=(0.0, float(np.log(len(session.classes)))),
            )
        ).save(artifacts["entropy"])

        class_records = _class_records(session)
        statistics = _class_statistics(
            prediction.prediction_grid, len(session.classes)
        )
        training_summary = {
            "contract_version": TRADITIONAL_RUN_CONTRACT_VERSION,
            "run_id": run_id,
            "classifier": identifier,
            "resolved_parameters": dict(provider_record["classifier"]["parameters"]),
            "feature_mode": mode,
            "seed": int(provider_record["seed"]),
            "classes": class_records,
            "training_matrix_shape": list(provider_record["training_matrix_shape"]),
            "support_training_accuracy": float(
                provider_record["support_training_accuracy"]
            ),
            "support_training_accuracy_is_validation": False,
            "support_training_accuracy_statement": (
                "support_training_accuracy is measured on the user-selected support "
                "patches themselves. It is not held-out validation accuracy and must "
                "not be reported as such."
            ),
            "training_duration_seconds": float(
                provider_record["timings_seconds"]["fit"]
            ),
            "prediction_duration_seconds": float(
                provider_record["timings_seconds"]["prediction"]
            ),
            "runtime": dict(provider_record["runtime"]),
        }
        _write_json(Path(artifacts["training_summary"]), training_summary)
        _write_report(
            Path(artifacts["report"]),
            run_id=run_id,
            session=session,
            classifier=identifier,
            feature_mode=mode,
            settings=settings,
            provider_record=provider_record,
        )

        warnings = list(
            dict.fromkeys(
                [
                    *provider_record.get("warnings", []),
                    "Traditional ML validation is exploratory; it is not a formal "
                    "paired benchmark against the pretrained model.",
                    "Support training accuracy is not held-out validation accuracy.",
                    "Confidence and entropy do not establish physical correctness.",
                ]
            )
        )
        runtime = dict(provider_record["runtime"])
        runtime["harness_total_seconds"] = perf_counter() - started
        run_record = {
            "contract_version": TRADITIONAL_RUN_CONTRACT_VERSION,
            "run_id": run_id,
            "status": "completed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "harness_version": harness_version,
            "pretrained_checkpoint_accessed": False,
            "input": input_record,
            "annotations": {
                "path": artifacts["annotation_session"],
                "schema_version": portable_session.to_dict()["schema_version"],
                "classes": class_records,
            },
            "provider": {
                "name": provider_record["provider"],
                "version": provider_record["provider_version"],
                "contract_version": provider_record["provider_contract_version"],
                "operation": TRADITIONAL_ML_OPERATION,
                "record_path": artifacts["provider_record"],
            },
            "traditional_ml": {
                "classifier": dict(provider_record["classifier"]),
                "feature_mode": provider_record["feature_mode"],
                "channel_count": int(provider_record["channel_count"]),
                "channel_names": list(provider_record["channel_names"]),
                "class_names": list(provider_record["class_names"]),
                "class_count": int(provider_record["class_count"]),
                "support_counts": dict(provider_record["support_counts"]),
                "support_sha256": provider_record["support"]["sha256"],
                "settings": settings.to_dict(),
                "options": dict(provider_record["options"]),
                "seed": int(provider_record["seed"]),
                "stride": int(provider_record["stride"]),
                "batch_size": int(provider_record["batch_size"]),
                "grid_shape": list(provider_record["grid_shape"]),
                "sample_count": int(provider_record["sample_count"]),
            },
            "features": {
                "path": artifacts["features"],
                "record_path": artifacts["features_record"],
                "shape": [int(value) for value in features.shape],
                "channel_names": [str(name) for name in channel_names],
                "artifact_sha256": feature_artifact_sha256,
                "feature_sha256": provider_record["feature_sha256"],
                "provider_feature_record": feature_record.get("features"),
            },
            "prediction": {
                "grid_shape": list(prediction.prediction_grid.shape),
                "sample_count": int(len(prediction.coordinates_xy)),
                "class_statistics": statistics,
                "probability_columns": int(prediction.probabilities.shape[1]),
                "confidence_range": [
                    float(prediction.confidence.min()),
                    float(prediction.confidence.max()),
                ],
                "entropy_range": [
                    float(prediction.entropy.min()),
                    float(prediction.entropy.max()),
                ],
            },
            "training": dict(training_summary),
            "runtime": runtime,
            "configuration": config_snapshot(config),
            "artifacts": artifacts,
            "warnings": warnings,
        }
        _write_json(Path(artifacts["run_record"]), run_record)
        _write_archive(run_directory, Path(artifacts["results_archive"]))
    except BaseException as error:  # noqa: BLE001 - recorded then re-raised
        _provider_failure_record(
            run_id=run_id,
            artifacts=artifacts,
            phase="traditional_validation",
            error=error,
            configuration=config_snapshot(config),
        )
        raise

    return {
        "status": "completed",
        "run_id": run_id,
        "run_directory": str(run_directory),
        "run_record": artifacts["run_record"],
        "training_summary": artifacts["training_summary"],
        "provider_record": artifacts["provider_record"],
        "traditional_prediction": artifacts["traditional_prediction"],
        "prediction_overlay": artifacts["prediction_overlay"],
        "confidence": artifacts["confidence"],
        "entropy": artifacts["entropy"],
        "report": artifacts["report"],
        "results_archive": artifacts["results_archive"],
        "classifier": identifier,
        "feature_mode": mode,
        "seed": int(provider_record["seed"]),
        "classifier_patch_size": int(provider_record["classifier_patch_size"]),
        "classifier_parameters": dict(provider_record["classifier"]["parameters"]),
        "classes": class_records,
        "class_statistics": statistics,
        "support_training_accuracy": float(
            provider_record["support_training_accuracy"]
        ),
        "support_training_accuracy_is_validation": False,
        "warnings": warnings,
    }


__all__ = [
    "ARTIFACT_NAMES",
    "RUN_ID_PREFIX",
    "artifact_paths",
    "new_run_directory",
    "run_traditional_validation",
]
