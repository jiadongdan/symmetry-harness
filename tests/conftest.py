from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from symmetry_harness.contracts import FINE_TUNED_MODEL_PACKAGE_SCHEMA_VERSION


MODEL_STATE = b"complete fine-tuned state"


def manifest(**overrides) -> dict:
    """Return one valid model package manifest."""
    payload = {
        "schema_version": FINE_TUNED_MODEL_PACKAGE_SCHEMA_VERSION,
        "model": {
            "identifier": "cnn_8ch_pg17",
            "input_channels": 8,
            "classifier_patch_size": 64,
            "adapter_bottleneck": 16,
            "task_classes": 2,
        },
        "classes": [
            {"index": 0, "name": "Phase A", "color": "#e41a1c"},
            {"index": 1, "name": "Phase B", "color": "#377eb8"},
        ],
        "features": {
            "pipeline": "eight_channel_v1",
            "channel_names": [f"channel_{index}" for index in range(8)],
            "n_max": 20,
            "symmetry_patch_size": 51,
            "rotation_folds": [2, 3, 4, 6],
            "reflection_p": 2.0,
            "normalize_rotation_maps": False,
            "input_normalization": "minmax_0_1",
        },
        "prediction_defaults": {"stride": 4, "batch_size": 512},
        "provenance": {
            "training_run_id": "symmetry-test",
            "model_state_sha256": None,
        },
    }
    payload.update(overrides)
    return payload


@pytest.fixture()
def model_package_factory(tmp_path):
    """Write a `.symmodel` package whose recorded checksum always matches."""
    from symmetry_harness.image_io import file_sha256

    def factory(
        name: str = "model.symmodel",
        *,
        manifest_overrides: dict | None = None,
        extra_members: dict[str, bytes] | None = None,
        omit: tuple[str, ...] = (),
        model_state: bytes = MODEL_STATE,
        break_checksum: bool = False,
    ) -> Path:
        checksum_path = tmp_path / f"{name}.digest"
        checksum_path.write_bytes(MODEL_STATE if break_checksum else model_state)
        payload = manifest(
            provenance={
                "training_run_id": "symmetry-test",
                "model_state_sha256": file_sha256(checksum_path),
            }
        )
        payload.update(manifest_overrides or {})
        members = {
            "manifest.json": json.dumps(payload).encode("utf-8"),
            "model_state.pt": model_state,
            "training_summary.json": b'{"training_run_id": "symmetry-test"}',
        }
        for member in omit:
            members.pop(member, None)
        members.update(extra_members or {})
        destination = tmp_path / name
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for member, content in members.items():
                archive.writestr(member, content)
        return destination

    return factory


@pytest.fixture()
def unit_image():
    """Return a deterministic single-channel image writer."""

    def writer(path: Path, size: int = 96) -> Path:
        y, x = np.mgrid[:size, :size]
        values = np.sin(x / 6.0) + np.cos(y / 7.0)
        np.save(path, ((values - values.min()) / np.ptp(values)).astype(np.float32))
        return path

    return writer
