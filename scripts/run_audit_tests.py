"""Executor direto de testes unitários para o módulo de auditoria, drift, explicabilidade e CLI."""

import os
import sys
import tempfile
from pathlib import Path

# Configurar ambiente antes dos imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RUNTIME_DIR = Path(tempfile.mkdtemp(prefix="iris-ai-script-tests-"))

os.environ.update(
    {
        "ENVIRONMENT": "development",
        "API_KEY": "",
        "ARTIFACT_SIGNING_KEY": "",
        "MLFLOW_TRACKING_URI": "",
        "DATABASE_URL": f"sqlite:///{(RUNTIME_DIR / 'test.db').as_posix()}",
        "MODEL_DIR": str(RUNTIME_DIR / "models"),
        "ARTIFACT_DIR": str(RUNTIME_DIR / "artifacts"),
        "DATA_DIR": str(RUNTIME_DIR / "data"),
        "RATE_LIMIT_REQUESTS": "100000",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
)

from tests.test_audit_and_advanced_pipelines import (  # noqa: E402
    iris_data,
    test_cli_audit_export,
    test_cli_drift_check,
    test_cli_explain_pipeline,
    test_cli_kfold,
    test_explainability_pipeline_and_waterfall,
    test_explainability_pipeline_tree_models,
    test_kfold_cross_validation_stratified,
    test_kfold_cross_validation_unstratified_and_validation_errors,
    test_model_audit_report_generation_and_export,
    test_psi_and_ks_test_statistical_properties,
    test_statistical_drift_analysis_multivariate,
)


def main():
    print("=" * 70)
    print("INICIANDO BATERIA DE TESTES DE AUDITORIA, XAI, DRIFT, KFOLD E CLI")
    print("=" * 70)

    data = iris_data()
    tmp_path = RUNTIME_DIR / "tmp"
    tmp_path.mkdir(parents=True, exist_ok=True)

    tests = [
        ("K-Fold Stratified & IC 95%", lambda: test_kfold_cross_validation_stratified(data)),
        ("K-Fold Unstratified & Validation Errors", lambda: test_kfold_cross_validation_unstratified_and_validation_errors(data)),
        ("PSI & KS-test Statistical Properties", lambda: test_psi_and_ks_test_statistical_properties()),
        ("Multivariate Statistical Drift Analysis", lambda: test_statistical_drift_analysis_multivariate(data)),
        ("Explainability Pipeline & Waterfall", lambda: test_explainability_pipeline_and_waterfall(data)),
        ("Explainability Pipeline Tree Models", lambda: test_explainability_pipeline_tree_models(data)),
        ("Model Audit Report Generation & Export", lambda: test_model_audit_report_generation_and_export(data, tmp_path)),
        ("CLI Explain Pipeline", lambda: test_cli_explain_pipeline()),
        ("CLI K-Fold", lambda: test_cli_kfold()),
        ("CLI Drift Check", lambda: test_cli_drift_check()),
        ("CLI Audit Export", lambda: test_cli_audit_export(tmp_path)),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 70)
    print(f"RESULTADO: {passed} APROVADOS | {failed} FALHAS (Total: {len(tests)})")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
