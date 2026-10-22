"""Smoke test de ponta a ponta contra uma API em execução.

Uso:
    python scripts/smoke_test.py [URL_BASE] [--api-key CHAVE]

Sem URL usa http://127.0.0.1:8000. Só depende da biblioteca padrão. Treina um
modelo leve, prediz, explica e consulta o monitoramento; sai com código 1 se
qualquer passo falhar.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

SAMPLE = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
MODEL = "logistic_regression"


def call(base: str, method: str, path: str, api_key: str, body: dict | None = None) -> tuple[int, object]:
    headers = {"Content-Type": "application/json"} if body is not None else {}
    if api_key:
        headers["X-API-Key"] = api_key
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw, status = response.read(), response.status
    except urllib.error.HTTPError as error:
        raw, status = error.read(), error.code
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw.decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url", nargs="?", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    payload = {**SAMPLE, "model_name": MODEL}
    steps = [
        ("GET", "/healthz", None, lambda b: b["status"] == "ok"),
        ("GET", "/readyz", None, lambda b: b["database"] is True),
        ("GET", "/console/", None, lambda b: "Iris AI Platform" in b),
        ("GET", "/api/v1/dataset/summary", None, lambda b: b["rows"] == 150),
        ("GET", "/api/v2/data/quality", None, lambda b: b["passed"] is True),
        (
            "POST",
            "/api/v1/training/train",
            {"model_name": MODEL, "cv_folds": 3},
            lambda b: b["status"] == "completed",
        ),
        ("POST", "/api/v1/predict", payload, lambda b: b["predicted_class"] == "setosa"),
        ("POST", "/api/v2/explain", payload, lambda b: b["explanation"]["method"] == "exact_shapley"),
        ("GET", f"/api/v2/monitoring/{MODEL}", None, lambda b: b["predictions"] >= 1),
        ("POST", "/api/v4/pipelines/dry-run", {"model_name": MODEL}, lambda b: len(b["events"]) == 5),
        ("GET", "/api/v7/labs/lineage-graph", None, lambda b: any(MODEL in node for node in b["nodes"])),
    ]
    failures = 0
    for method, path, body, check in steps:
        status, response = call(base, method, path, args.api_key, body)
        try:
            ok = status == 200 and bool(check(response))
        except (KeyError, TypeError):
            ok = False
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {status} {method} {path}")
    print("Smoke test passou." if not failures else f"Smoke test falhou em {failures} passo(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
