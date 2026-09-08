"""Versioned model/calibrator artefacts for train/serve handoff."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACT_SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a local artefact."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_path(artifact_path: str | Path) -> Path:
    """Use the stable ``<artifact>.metadata.json`` sidecar convention."""
    return Path(artifact_path).with_suffix(".metadata.json")


def write_artifact_metadata(
    artifact_path: str | Path,
    metadata: dict[str, Any],
    *,
    output_path: str | Path | None = None,
) -> Path:
    """Write an atomic manifest containing the artefact checksum and protocol."""
    artifact = Path(artifact_path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Artefact absent : {artifact}")
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_filename": artifact.name,
        "artifact_sha256": sha256_file(artifact),
        "artifact_size_bytes": artifact.stat().st_size,
        "created_at": datetime.now(UTC).isoformat(),
        **metadata,
    }
    target = Path(output_path) if output_path else metadata_path(artifact)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, target)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def load_validated_artifact(
    artifact_path: str | Path,
    *,
    expected_type: str | None = None,
    metadata_file: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a joblib artefact only after validating its manifest and checksum."""
    import joblib

    artifact = Path(artifact_path)
    manifest_file = Path(metadata_file) if metadata_file else metadata_path(artifact)
    if not artifact.is_file():
        raise FileNotFoundError(f"Artefact absent : {artifact}")
    if not manifest_file.is_file():
        raise ValueError(f"Manifest artefact absent : {manifest_file}")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest JSON invalide : {manifest_file}") from exc
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Version de manifest incompatible : {manifest.get('schema_version')!r}"
        )
    if expected_type and manifest.get("artifact_type") != expected_type:
        raise ValueError(
            f"Type d'artefact inattendu : {manifest.get('artifact_type')!r}"
        )
    expected_sha = manifest.get("artifact_sha256")
    actual_sha = sha256_file(artifact)
    if not expected_sha or expected_sha != actual_sha:
        raise ValueError(f"Checksum artefact invalide : attendu={expected_sha}, réel={actual_sha}")
    obj = joblib.load(artifact)
    if expected_type == "qdm" and not all(
        hasattr(obj, attr) for attr in ("transform", "n_quantiles", "by_month")
    ):
        raise ValueError("Artefact QDM incompatible : protocole de transform absent")
    protocol = manifest.get("fit_protocol", {})
    for key in ("kind", "n_quantiles", "by_month"):
        if key in protocol and getattr(obj, key, None) != protocol[key]:
            raise ValueError(f"Artefact incompatible avec son manifest : {key}")
    return obj, manifest
