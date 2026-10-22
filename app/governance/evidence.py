from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..platform.hash_utils import canonical_hash


def build_evidence_pack(record: dict[str, Any], gate: dict[str, Any], output: str | Path) -> dict[str, Any]:
    """Pacote de evidências de uma versão registrada: registro, release gate e hash do conjunto."""
    pack: dict[str, Any] = {
        "schema": "iris-ai-governance-evidence/v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "model": record,
        "release_gate": gate,
    }
    pack["evidence_hash"] = canonical_hash(pack)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pack, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return {"path": str(path), "evidence_hash": pack["evidence_hash"], "schema": pack["schema"]}
