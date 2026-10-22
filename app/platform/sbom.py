from __future__ import annotations

import importlib.metadata
from typing import Any


def installed_software(include: set[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name") or "unknown"
        if include and name.lower() not in {item.lower() for item in include}:
            continue
        rows.append(
            {"name": name, "version": dist.version, "license": dist.metadata.get("License") or "Unknown"}
        )
    return sorted(rows, key=lambda row: row["name"].lower())


def minimal_cyclonedx() -> dict[str, Any]:
    components = [
        {
            "type": "library",
            "name": item["name"],
            "version": item["version"],
            "licenses": [{"license": {"name": item["license"]}}],
        }
        for item in installed_software()
    ]
    return {"bomFormat": "CycloneDX", "specVersion": "1.6", "version": 1, "components": components}
