from __future__ import annotations

import hashlib
import json

import pytest

from downscaling.utils.registry import (
    load_registered_artifact,
    promote_artifact,
    register_artifact,
    resolve_artifact,
    resolve_production,
)


def test_registry_is_content_addressed_and_requires_approval(tmp_path):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"immutable model")
    manifest = register_artifact(
        source,
        tmp_path / "registry",
        artifact_type="model",
        metadata={"metrics": {"rmse": 0.8}},
        api_version="2",
        output_schema="temperature.v1",
    )
    digest = manifest["digest"]
    assert manifest["status"] == "candidate"
    with pytest.raises(ValueError, match="non production"):
        resolve_production(tmp_path / "registry", digest)
    promote_artifact(
        tmp_path / "registry",
        digest,
        "validated",
        approved_by="qa@example.test",
        reason="holdout validé",
    )
    promote_artifact(
        tmp_path / "registry",
        digest,
        "production",
        approved_by="ops@example.test",
        reason="go-live",
    )
    artifact, resolved = resolve_production(
        tmp_path / "registry",
        digest,
        expected_api_version="2",
        expected_output_schema="temperature.v1",
    )
    assert artifact.read_bytes() == source.read_bytes()
    assert resolved["digest"] == digest
    with pytest.raises(ValueError, match="Transition interdite"):
        promote_artifact(
            tmp_path / "registry",
            digest,
            "validated",
            approved_by="qa@example.test",
            reason="rollback invalide",
        )


def test_registry_detects_tampering(tmp_path):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"weights")
    root = tmp_path / "registry"
    digest = register_artifact(source, root, artifact_type="model")["digest"]
    promote_artifact(root, digest, "validated", approved_by="qa", reason="ok")
    promote_artifact(root, digest, "production", approved_by="ops", reason="ok")
    object_path, _ = resolve_artifact(root, digest)
    object_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="corrompu"):
        resolve_production(root, digest)


class FakeCalibrator:
    kind = "delta"
    by_month = True
    n_quantiles = 10

    def transform(self, values):
        return values


def test_joblib_loader_requires_production_and_matching_sidecar(tmp_path):
    joblib = pytest.importorskip("joblib")
    source = tmp_path / "qdm.joblib"
    joblib.dump(FakeCalibrator(), source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    source.with_suffix(".metadata.json").write_text(
        json.dumps({"artifact_type": "qdm", "artifact_sha256": digest})
    )
    root = tmp_path / "registry"
    manifest = register_artifact(source, root, artifact_type="qdm")
    promote_artifact(root, manifest["digest"], "validated", approved_by="qa", reason="ok")
    promote_artifact(root, manifest["digest"], "production", approved_by="ops", reason="ok")
    obj, _ = load_registered_artifact(root, manifest["digest"], expected_type="qdm")
    assert obj.n_quantiles == 10
