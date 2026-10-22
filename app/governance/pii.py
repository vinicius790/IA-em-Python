from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_PATTERNS = {
    # O lookbehind só deixa a parte local começar no início de uma sequência; com `\b`
    # cada posição de "a.a.a..." virava um início e o custo ficava quadrático (ReDoS).
    "email": re.compile(r"(?<![A-Z0-9._%+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)"),
    "cpf_like": re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"),
    "credit_card_like": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
}


@dataclass(frozen=True, slots=True)
class PIIFinding:
    category: str
    count: int


def scan_text(text: str) -> dict[str, Any]:
    findings = [PIIFinding(name, len(pattern.findall(text))) for name, pattern in _PATTERNS.items()]
    findings = [finding for finding in findings if finding.count]
    return {
        "contains_pii": bool(findings),
        "findings": [{"category": f.category, "count": f.count} for f in findings],
    }


def redact_text(text: str) -> str:
    redacted = text
    for name, pattern in _PATTERNS.items():
        redacted = pattern.sub(f"[REDACTED:{name}]", redacted)
    return redacted
