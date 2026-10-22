"""Perfil de um CSV usando só a biblioteca padrão (`csv` e `statistics`).

É o que valida um upload antes de ele entrar no catálogo: tipos, nulos, estatísticas,
coluna-alvo e features. Não depende de pandas, então roda (e é testado) em qualquer Python.
"""

from __future__ import annotations

import csv
import io
import math
import re
import statistics
from collections import Counter
from typing import Any

NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "?"}
MAX_TARGET_CLASSES = 50
# Os mesmos limites dos gates de qualidade do treino (`app/quality.py` importa daqui).
MIN_ROWS = 30
MAX_DUPLICATE_RATE = 0.01
_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
# Só a notação decimal comum: `float()` também aceitaria `1_000`, `inf` e dígitos não ASCII.
_NUMBER = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def valid_dataset_name(name: str) -> bool:
    """Nome usado em caminho de arquivo e em URL: minúsculas, dígitos, `_` e `-`."""
    return bool(_NAME.fullmatch(name))


def is_null(value: str | None) -> bool:
    return value is None or value.strip().lower() in NULL_TOKENS


def to_float(value: str) -> float | None:
    if not _NUMBER.fullmatch(value.strip()):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def read_rows(text: str) -> tuple[list[str], list[list[str]]]:
    """Cabeçalho (sem espaços nas pontas) e linhas de um CSV, já validados.

    É o único leitor de CSV enviado: o perfil e o treino veem exatamente as mesmas linhas.
    Levanta `ValueError` para arquivo vazio, cabeçalho inválido, CSV malformado ou
    linhas com número errado de campos.
    """
    reader = csv.reader(io.StringIO(text.lstrip("﻿")))
    try:
        header = next(reader, None)
        if not header or not any(cell.strip() for cell in header):
            raise ValueError("CSV is empty or has no header row")
        header = [cell.strip() for cell in header]
        if any(not cell for cell in header) or len(set(header)) != len(header):
            raise ValueError("CSV header must have unique, non-empty column names")
        rows = [row for row in reader if row]
    except csv.Error as exc:
        raise ValueError(f"Malformed CSV (line {reader.line_num}): {exc}") from exc
    for line, row in enumerate(rows, start=2):
        if len(row) != len(header):
            raise ValueError(f"CSV line {line} has {len(row)} fields; expected {len(header)}")
    if not rows:
        raise ValueError("CSV has a header but no data rows")
    return header, rows


def _column_profile(name: str, values: list[str | None]) -> dict[str, Any]:
    present = [value.strip() for value in values if not is_null(value)]
    numbers = [to_float(value) for value in present]
    numeric = bool(present) and all(number is not None for number in numbers)
    item: dict[str, Any] = {
        "name": name,
        "semantic_type": "numeric" if numeric else ("categorical" if present else "empty"),
        "nulls": len(values) - len(present),
        "null_rate": round((len(values) - len(present)) / len(values), 6) if values else 0.0,
        "unique": len(set(present)),
    }
    if numeric:
        item["numeric"] = {
            "min": min(numbers),
            "max": max(numbers),
            "mean": round(statistics.fmean(numbers), 8),
            "std": round(statistics.pstdev(numbers), 8),
            "p50": round(statistics.median(numbers), 8),
        }
        item["integer_like"] = all(number == int(number) for number in numbers)
    elif present:
        item["top_values"] = dict(Counter(present).most_common(10))
    return item


def _is_class_column(column: dict[str, Any], rows: int) -> bool:
    """Coluna que serve de alvo de classificação: poucas classes, sem nulos."""
    if column["nulls"] or not 2 <= column["unique"] <= min(MAX_TARGET_CLASSES, max(2, rows // 2)):
        return False
    return column["semantic_type"] == "categorical" or column.get("integer_like", False)


def infer_target(columns: list[dict[str, Any]], rows: int) -> str | None:
    """Alvo provável: a última coluna se for de classes; senão a categórica de menor cardinalidade."""
    if not columns:
        return None
    if _is_class_column(columns[-1], rows):
        return columns[-1]["name"]
    candidates = [c for c in columns if c["semantic_type"] == "categorical" and _is_class_column(c, rows)]
    return min(candidates, key=lambda c: c["unique"])["name"] if candidates else None


def profile_csv(text: str, target: str | None = None) -> dict[str, Any]:
    """Perfil completo de um CSV com cabeçalho.

    Levanta `ValueError` (vira 422 na API) para arquivo vazio, cabeçalho inválido,
    CSV malformado, linhas com número errado de campos ou alvo inexistente.
    """
    header, rows = read_rows(text)
    if target is not None and target not in header:
        raise ValueError(f"Target column '{target}' not found. Columns: {', '.join(header)}")

    columns = [_column_profile(name, [row[i] for row in rows]) for i, name in enumerate(header)]
    chosen = target or infer_target(columns, len(rows))
    features = [c["name"] for c in columns if c["semantic_type"] == "numeric" and c["name"] != chosen]
    target_profile = next((c for c in columns if c["name"] == chosen), None)
    classes: dict[str, int] = {}
    training_duplicates = 0
    if target_profile is not None:
        position = header.index(chosen)
        classes = dict(Counter(row[position].strip() for row in rows if not is_null(row[position])))
        # Duplicatas como o gate do treino as vê: só features e alvo, com os números já convertidos.
        indexes = [header.index(name) for name in features]
        keys = {
            (*(None if is_null(row[i]) else to_float(row[i]) for i in indexes), row[position].strip())
            for row in rows
        }
        training_duplicates = len(rows) - len(keys)
    return {
        "rows": len(rows),
        "columns": len(header),
        "duplicate_rows": len(rows) - len({tuple(row) for row in rows}),
        "duplicate_training_rows": training_duplicates,
        "column_profiles": columns,
        "target": chosen,
        "target_inferred": target is None and chosen is not None,
        "features": features,
        "classes": classes,
    }


def training_problems(profile: dict[str, Any]) -> list[str]:
    """Motivos pelos quais o dataset não serve para treino de classificação (lista vazia = serve)."""
    problems = []
    if profile["rows"] < MIN_ROWS:
        problems.append(f"at least {MIN_ROWS} rows are required (got {profile['rows']})")
    if profile["target"] is None:
        problems.append("no target column could be inferred; pass `target` explicitly")
    elif len(profile["classes"]) < 2:
        problems.append("target must have at least two classes")
    elif len(profile["classes"]) > MAX_TARGET_CLASSES:
        problems.append(f"target has more than {MAX_TARGET_CLASSES} classes; this is not classification")
    elif min(profile["classes"].values()) < 2:
        problems.append("every target class needs at least two rows")
    if not profile["features"]:
        problems.append("at least one numeric feature column is required")
    target = next((c for c in profile["column_profiles"] if c["name"] == profile["target"]), None)
    if target is not None and target["nulls"]:
        problems.append("target column has missing values")
    with_nulls = [
        c["name"] for c in profile["column_profiles"] if c["name"] in profile["features"] and c["nulls"]
    ]
    if with_nulls:
        problems.append(f"feature columns have missing values: {', '.join(with_nulls)}")
    duplicates = profile.get("duplicate_training_rows", 0)
    if profile["rows"] and duplicates / profile["rows"] > MAX_DUPLICATE_RATE:
        problems.append(
            f"{duplicates} rows repeat the same features and target "
            f"(more than {MAX_DUPLICATE_RATE:.0%} of the rows)"
        )
    return problems
