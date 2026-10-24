"""Issue #9: routers por domínio sem mudar o contrato HTTP público."""

from __future__ import annotations

import hashlib
import json

import pytest
from fastapi.routing import APIRoute

from app import explainability as legacy_explainability
from app.api import DOMAINS, VERSIONED_ROUTERS, build_domain_router, domain_of
from app.api.routers import DOMAIN_ROUTERS
from app.main import app
from app.modeling import explainability

# sha256 do JSON canônico {"paths", "schemas"} do OpenAPI antes da reorganização.
# Ao mudar o contrato de propósito, atualize com `_contract_digest()`.
OPENAPI_CONTRACT_SHA256 = "9b8afcccab96aa1d94e3b1d53e068d87b81d6f06dc9b0d48f2049483ff908389"


def _contract_digest() -> str:
    spec = app.openapi()
    contract = {"paths": spec["paths"], "schemas": spec["components"]["schemas"]}
    canonical = json.dumps(contract, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _operations(routes) -> list[tuple[str, str]]:
    return sorted((m, r.path) for r in routes if isinstance(r, APIRoute) for m in r.methods)


def test_openapi_paths_and_schemas_unchanged() -> None:
    assert _contract_digest() == OPENAPI_CONTRACT_SHA256


def test_every_versioned_route_belongs_to_exactly_one_domain() -> None:
    versioned = _operations(r for router in VERSIONED_ROUTERS for r in router.routes)
    by_domain = _operations(r for module in DOMAIN_ROUTERS for r in module.router.routes)
    assert by_domain == versioned
    assert len(set(by_domain)) == len(by_domain)


def test_domain_routers_cover_all_domains_in_order() -> None:
    assert tuple(m.__name__.rsplit(".", 1)[1] for m in DOMAIN_ROUTERS) == DOMAINS
    assert all(m.router.routes for m in DOMAIN_ROUTERS)


@pytest.mark.parametrize(
    ("path", "domain"),
    [
        ("/api/v1/iris/{iris_id}", "data"),
        ("/api/v4/pipelines/train", "training"),
        ("/api/v8/canary/{canary_id}/rollback", "registry"),
        ("/api/v8/explain", "inference"),
        ("/api/v2/monitoring/drift/{model_name}", "monitoring"),
        ("/api/v8/users/me", "governance"),
        ("/api/v7/labs/automl", "labs"),
    ],
)
def test_domain_of(path: str, domain: str) -> None:
    assert domain_of(path) == domain


@pytest.mark.parametrize("path", ["/health", "/api/v9/unknown"])
def test_domain_of_rejects_unmapped_paths(path: str) -> None:
    with pytest.raises(ValueError):
        domain_of(path)


def test_build_domain_router_rejects_unknown_domain() -> None:
    with pytest.raises(ValueError):
        build_domain_router("billing")


def test_legacy_explainability_module_reexports_unified_implementation() -> None:
    assert legacy_explainability.shapley_explanation is explainability.shapley_explanation
    assert legacy_explainability.permutation_explanation is explainability.permutation_explanation
