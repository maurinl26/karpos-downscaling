"""Small content-addressed registry for production model artefacts.

The registry is deliberately storage-agnostic: ``root`` may be a mounted
Scaleway Object Storage bucket (or a local directory in CI).  Consumers use
the digest returned by :func:`register_artifact`, never a mutable ``latest``
pointer.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .artifacts import load_validated_artifact, sha256_file

ArtifactStatus = Literal["candidate", "validated", "production", "retired"]
_STATUSES: tuple[ArtifactStatus, ...] = (
    "candidate",
    "validated",
    "production",
    "retired",
)
_TRANSITIONS: dict[ArtifactStatus, frozenset[ArtifactStatus]] = {
    "candidate": frozenset({"validated", "retired"}),
    "validated": frozenset({"production", "retired"}),
    "production": frozenset({"retired"}),
    "retired": frozenset(),
}


def _manifest_path(root: Path, digest: str) -> Path:
    return root / "manifests" / f"{digest}.json"


def _artifact_path(root: Path, digest: str, filename: str) -> Path:
    return root / "objects" / digest / filename


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def register_artifact(
    artifact_path: str | Path,
    root: str | Path,
    *,
    artifact_type: str,
    metadata: dict[str, Any] | None = None,
    api_version: str = "1",
    output_schema: str = "1",
) -> dict[str, Any]:
    """Register an artefact by SHA-256 and return its immutable manifest.

    Re-registering identical bytes is idempotent.  A digest collision with a
    different filename or metadata is rejected instead of overwriting data.
    """
    source = Path(artifact_path)
    if not source.is_file():
        raise FileNotFoundError(f"Artefact absent : {source}")
    if not artifact_type:
        raise ValueError("artifact_type est obligatoire")
    registry = Path(root)
    digest = sha256_file(source)
    manifest_file = _manifest_path(registry, digest)
    object_file = _artifact_path(registry, digest, source.name)
    manifest: dict[str, Any] = {
        "registry_schema_version": 1,
        "digest": digest,
        "artifact_type": artifact_type,
        "artifact_filename": source.name,
        "artifact_size_bytes": source.stat().st_size,
        "status": "candidate",
        "api_version": api_version,
        "output_schema": output_schema,
        "created_at": datetime.now(UTC).isoformat(),
        "metadata": metadata or {},
        "history": [{"status": "candidate", "at": datetime.now(UTC).isoformat()}],
    }
    if manifest_file.exists():
        existing = json.loads(manifest_file.read_text())
        if existing.get("digest") != digest or existing.get("artifact_type") != artifact_type:
            raise ValueError(f"Collision de digest dans le registre : {digest}")
        if not object_file.is_file() or sha256_file(object_file) != digest:
            raise ValueError(f"Objet du registre absent ou corrompu : {digest}")
        return existing
    object_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, object_file)
    source_sidecar = source.with_suffix(".metadata.json")
    if source_sidecar.is_file():
        shutil.copyfile(source_sidecar, object_file.with_suffix(".metadata.json"))
    if sha256_file(object_file) != digest:
        object_file.unlink(missing_ok=True)
        raise ValueError("Echec de vérification du checksum lors de l'enregistrement")
    _write_json(manifest_file, manifest)
    return manifest


def read_manifest(root: str | Path, digest: str) -> dict[str, Any]:
    """Read and validate a manifest identified by its digest."""
    if not digest or "/" in digest or ".." in digest:
        raise ValueError("Digest invalide")
    path = _manifest_path(Path(root), digest)
    if not path.is_file():
        raise FileNotFoundError(f"Digest inconnu : {digest}")
    manifest = json.loads(path.read_text())
    if manifest.get("digest") != digest or manifest.get("status") not in _STATUSES:
        raise ValueError(f"Manifeste invalide : {path}")
    return manifest


def promote_artifact(
    root: str | Path,
    digest: str,
    status: ArtifactStatus,
    *,
    approved_by: str,
    reason: str,
) -> dict[str, Any]:
    """Perform an explicit, auditable status transition."""
    if status not in _STATUSES:
        raise ValueError(f"Statut inconnu : {status}")
    if not approved_by or not reason:
        raise ValueError("approved_by et reason sont obligatoires")
    registry = Path(root)
    manifest = read_manifest(registry, digest)
    current = manifest["status"]
    if status not in _TRANSITIONS[current]:
        raise ValueError(f"Transition interdite : {current} -> {status}")
    manifest["status"] = status
    manifest["approved_by"] = approved_by
    manifest["approval_reason"] = reason
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    manifest.setdefault("history", []).append(
        {
            "from": current,
            "status": status,
            "at": manifest["updated_at"],
            "approved_by": approved_by,
            "reason": reason,
        }
    )
    _write_json(_manifest_path(registry, digest), manifest)
    return manifest


def resolve_artifact(
    root: str | Path,
    digest: str,
    *,
    expected_type: str | None = None,
    expected_api_version: str | None = None,
    expected_output_schema: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve and checksum-verify an artefact by digest."""
    registry = Path(root)
    manifest = read_manifest(registry, digest)
    if expected_type and manifest.get("artifact_type") != expected_type:
        raise ValueError("Type d'artefact incompatible")
    if expected_api_version and manifest.get("api_version") != expected_api_version:
        raise ValueError("Version d'API incompatible")
    if expected_output_schema and manifest.get("output_schema") != expected_output_schema:
        raise ValueError("Schéma de sortie incompatible")
    artifact = _artifact_path(registry, digest, manifest["artifact_filename"])
    if not artifact.is_file() or sha256_file(artifact) != digest:
        raise ValueError(f"Objet du registre absent ou corrompu : {digest}")
    return artifact, manifest


def resolve_production(
    root: str | Path,
    digest: str,
    *,
    expected_type: str | None = None,
    expected_api_version: str | None = None,
    expected_output_schema: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve a digest only when it has explicitly reached production."""
    artifact, manifest = resolve_artifact(
        root,
        digest,
        expected_type=expected_type,
        expected_api_version=expected_api_version,
        expected_output_schema=expected_output_schema,
    )
    if manifest["status"] != "production":
        raise ValueError(f"Artefact non production : {digest} ({manifest['status']})")
    return artifact, manifest


def load_registered_artifact(
    root: str | Path,
    digest: str,
    *,
    expected_type: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a registered joblib artefact after registry and checksum checks."""
    artifact, manifest = resolve_production(root, digest, expected_type=expected_type)
    obj, sidecar = load_validated_artifact(artifact, expected_type=expected_type)
    if sidecar.get("artifact_sha256") != digest:
        raise ValueError("Le sidecar de l'artefact ne correspond pas au digest du registre")
    return obj, manifest
