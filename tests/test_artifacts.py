"""Train/serve manifest checks for statistical calibrator artefacts."""

from __future__ import annotations

import json

import joblib
import pytest

from downscaling.utils.artifacts import load_validated_artifact, write_artifact_metadata


class FakeQDM:
    kind = "delta"
    by_month = True
    n_quantiles = 10

    def transform(self, value):
        return value


def _write_qdm(tmp_path):
    artifact = tmp_path / "qdm.joblib"
    joblib.dump(FakeQDM(), artifact)
    manifest = write_artifact_metadata(
        artifact,
        {
            "artifact_type": "qdm",
            "fit_protocol": {"kind": "delta", "by_month": True, "n_quantiles": 10},
            "fit_years": [2022, 2023],
        },
    )
    return artifact, manifest


def test_qdm_manifest_round_trip(tmp_path) -> None:
    artifact, manifest_path = _write_qdm(tmp_path)

    loaded, manifest = load_validated_artifact(artifact, expected_type="qdm")

    assert isinstance(loaded, FakeQDM)
    assert manifest_path.name == "qdm.metadata.json"
    assert manifest["artifact_sha256"]
    assert manifest["fit_years"] == [2022, 2023]


def test_qdm_manifest_rejects_tampering(tmp_path) -> None:
    artifact, manifest_path = _write_qdm(tmp_path)
    artifact.write_bytes(artifact.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="Checksum"):
        load_validated_artifact(artifact, expected_type="qdm")

    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 1
