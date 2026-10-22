from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .hybrid import HybridSearch, SearchDocument

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def documentation_files(root: Path) -> list[Path]:
    """Só `README.md` e `docs/**/*.md`: nunca varre o repositório inteiro."""
    files = sorted((root / "docs").rglob("*.md")) if (root / "docs").is_dir() else []
    readme = root / "README.md"
    return ([readme] if readme.is_file() else []) + files


@lru_cache(maxsize=1)
def documentation_index(root: Path = PROJECT_ROOT) -> HybridSearch:
    """Índice TF-IDF da documentação do projeto, montado uma vez por processo."""
    documents = [
        SearchDocument(
            path.relative_to(root).as_posix(),
            path.read_text(encoding="utf-8", errors="ignore"),
            {"path": path.relative_to(root).as_posix()},
        )
        for path in documentation_files(root)
    ]
    return HybridSearch(documents)
