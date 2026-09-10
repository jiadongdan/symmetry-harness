from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from symmetry_harness.model_package import (
    FineTunedModelPackage,
    ModelPackageError,
    inspect_model_package,
    materialize_model_state,
)


def _features(**overrides) -> dict:
    features = {
        "pipeline": "eight_channel_v1",
        "channel_names": [f"channel_{index}" for index in range(8)],
        "n_max": 20,
        "symmetry_patch_size": 51,
        "rotation_folds": [2, 3, 4, 6],
        "reflection_p": 2.0,
        "normalize_rotation_maps": False,
        "input_normalization": "minmax_0_1",
    }
    features.update(overrides)
    return features


def test_valid_package_inspection_exposes_the_saved_contract(
    model_package_factory,
) -> None:
    package = inspect_model_package(model_package_factory())

    assert isinstance(package, FineTunedModelPackage)
    assert package.model_identifier == "cnn_8ch_pg17"
    assert package.task_classes == 2
    assert package.class_names == ["Phase A", "Phase B"]
    assert package.class_colors == ["#e41a1c", "#377eb8"]
    assert package.input_normalization == "minmax_0_1"
    assert package.prediction_defaults == {"stride": 4, "batch_size": 512}
    assert package.feature_options["symmetry_patch_size"] == 51
    assert len(package.package_sha256) == 64
    assert len(package.model_state_sha256) == 64


def test_materialized_model_state_matches_the_validated_checksum(
    model_package_factory, tmp_path
) -> None:
    package = inspect_model_package(model_package_factory())

    target = materialize_model_state(package, tmp_path / "private" / "model_state.pt")

    assert target.is_file()
    with zipfile.ZipFile(package.source_path) as archive:
        assert target.read_bytes() == archive.read("model_state.pt")


def test_package_inspection_rejects_missing_and_unexpected_members(
    model_package_factory,
) -> None:
    with pytest.raises(ModelPackageError, match="missing"):
        inspect_model_package(
            model_package_factory("a.symmodel", omit=("training_summary.json",))
        )


def test_package_inspection_rejects_duplicate_members(
    model_package_factory,
) -> None:
    package = model_package_factory("duplicate.symmodel")
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(package, "a") as archive:
            archive.writestr("manifest.json", b"{}")

    with pytest.raises(ModelPackageError, match="duplicate members"):
        inspect_model_package(package)

    with pytest.raises(ModelPackageError, match="unexpected member"):
        inspect_model_package(
            model_package_factory(
                "b.symmodel", extra_members={"notes.txt": b"unexpected"}
            )
        )


def test_package_inspection_rejects_unsafe_member_names(model_package_factory) -> None:
    cases = {
        "traversal.symmodel": "../../escape.json",
        "absolute.symmodel": "/etc/absolute.json",
    }
    for name, member in cases.items():
        path = model_package_factory(name)
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr(member, b"{}")
        with pytest.raises(ModelPackageError, match="unsafe member"):
            inspect_model_package(path)


def test_package_inspection_rejects_a_corrupt_model_state(model_package_factory) -> None:
    corrupt = model_package_factory(
        "corrupt.symmodel", model_state=b"tampered", break_checksum=True
    )

    with pytest.raises(ModelPackageError, match="checksum"):
        inspect_model_package(corrupt)


def test_package_inspection_rejects_invalid_manifest_content(
    model_package_factory,
) -> None:
    with pytest.raises(ModelPackageError, match="schema"):
        inspect_model_package(
            model_package_factory(
                "a.symmodel",
                manifest_overrides={
                    "schema_version": "symmetry-fine-tuned-model-package-v2"
                },
            )
        )

    with pytest.raises(ModelPackageError, match="contiguous"):
        inspect_model_package(
            model_package_factory(
                "b.symmodel",
                manifest_overrides={
                    "classes": [
                        {"index": 1, "name": "Phase A", "color": "#e41a1c"},
                        {"index": 2, "name": "Phase B", "color": "#377eb8"},
                    ]
                },
            )
        )

    with pytest.raises(ModelPackageError, match="unique"):
        inspect_model_package(
            model_package_factory(
                "c.symmodel",
                manifest_overrides={
                    "classes": [
                        {"index": 0, "name": "Same", "color": "#e41a1c"},
                        {"index": 1, "name": "Same", "color": "#377eb8"},
                    ]
                },
            )
        )

    with pytest.raises(ModelPackageError, match="normalization"):
        inspect_model_package(
            model_package_factory(
                "d.symmodel",
                manifest_overrides={
                    "features": _features(input_normalization="linear_0_255")
                },
            )
        )

    with pytest.raises(ModelPackageError, match="odd"):
        inspect_model_package(
            model_package_factory(
                "e.symmodel",
                manifest_overrides={"features": _features(symmetry_patch_size=50)},
            )
        )


def test_package_inspection_rejects_invalid_json_and_missing_file(
    model_package_factory, tmp_path
) -> None:
    with pytest.raises(ModelPackageError, match="valid JSON"):
        inspect_model_package(
            model_package_factory(
                "bad.symmodel", extra_members={"manifest.json": b"{not json"}
            )
        )

    with pytest.raises(ModelPackageError, match="UTF-8"):
        inspect_model_package(
            model_package_factory(
                "encoding.symmodel",
                extra_members={"manifest.json": b"\xff\xfe\x00"},
            )
        )

    with pytest.raises(ModelPackageError, match="does not exist"):
        inspect_model_package(tmp_path / "absent.symmodel")


def test_source_package_is_never_modified_by_inspection(model_package_factory) -> None:
    path = model_package_factory()
    before = path.read_bytes()

    inspect_model_package(path)

    assert path.read_bytes() == before
