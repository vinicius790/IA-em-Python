from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

_STYLE = "body{font-family:system-ui;margin:2rem}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.5rem}"


def training_report_html(payload: dict[str, Any], path: str | Path) -> Path:
    """Relatório HTML estático de um treino; todo valor é escapado antes de ir para a página."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    scalars = {k: v for k, v in (payload.get("metrics") or {}).items() if isinstance(v, int | float | str)}
    rows = "".join(
        f"<tr><td>{html.escape(str(name))}</td><td>{html.escape(str(value))}</td></tr>"
        for name, value in scalars.items()
    )
    model = html.escape(str(payload.get("model_name")))
    details = html.escape(json.dumps(payload, indent=2, default=str))
    page = (
        f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Relatório de treino: {model}</title><style>{_STYLE}</style></head><body>"
        f"<h1>Relatório de treino</h1><p>Modelo: <b>{model}</b></p>"
        f"<table><tr><th>Métrica</th><th>Valor</th></tr>{rows}</table>"
        f"<h2>Detalhes</h2><pre>{details}</pre></body></html>"
    )
    target.write_text(page, encoding="utf-8")
    return target


def comparison_report_html(payload: dict[str, Any], path: str | Path) -> Path:
    """Relatório HTML de uma comparação de modelos (média e intervalo de confiança por modelo)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    protocol = payload.get("protocol") or {}

    def interval(value: dict[str, Any]) -> str:
        return html.escape(f"{value['mean']:.4f} [{value['lower']:.4f}; {value['upper']:.4f}]")

    rows = "".join(
        f"<tr><td>{row['rank']}</td><td>{html.escape(str(row['model']))}</td>"
        f"<td>{interval(row['f1_macro'])}</td><td>{interval(row['accuracy'])}</td>"
        f"<td>{interval(row['f1_macro_delta_to_best'])}</td>"
        f"<td>{'sim' if row['differs_from_best'] else 'não'}</td>"
        f"<td>{html.escape(str(row['fit_time_mean']))}</td></tr>"
        for row in payload.get("results", [])
    )
    title = html.escape(f"Comparação de modelos: {payload.get('run_id')}")
    summary = html.escape(
        f"Dataset {payload.get('dataset')}, {protocol.get('rows')} linhas, {protocol.get('folds')} dobras "
        f"estratificadas, confiança {protocol.get('confidence')}, semente {protocol.get('random_state')}."
    )
    details = html.escape(json.dumps(payload, indent=2, default=str))
    page = (
        f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{title}</title><style>{_STYLE}</style></head><body>"
        f"<h1>{title}</h1><p>{summary}</p>"
        f"<p>Melhor modelo: <b>{html.escape(str(payload.get('best_model')))}</b></p>"
        f"<table><tr><th>#</th><th>Modelo</th><th>F1 macro (IC)</th><th>Acurácia (IC)</th>"
        f"<th>Diferença de F1 para o melhor (IC)</th><th>Difere do melhor</th>"
        f"<th>Treino médio (s)</th></tr>"
        f"{rows}</table><h2>Detalhes</h2><pre>{details}</pre></body></html>"
    )
    target.write_text(page, encoding="utf-8")
    return target
