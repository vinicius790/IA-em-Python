from __future__ import annotations

from typing import Any


def build_manifest(record: dict[str, Any], artifact_file: str, environment: str) -> dict[str, Any]:
    """Manifesto de implantação montado a partir de um registro real do registry."""
    return {
        "apiVersion": "ml.platform/v1",
        "kind": "ModelDeployment",
        "metadata": {"name": record["name"], "version": record["version"], "stage": record["stage"]},
        "spec": {
            "artifact": artifact_file,
            "checksum_sha256": record["checksum_sha256"],
            "environment": environment,
            "health": {"liveness": "/healthz", "readiness": "/readyz"},
        },
    }
