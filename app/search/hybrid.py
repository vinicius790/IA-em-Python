from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(slots=True)
class SearchDocument:
    id: str
    text: str
    metadata: dict[str, Any]


class HybridSearch:
    """In-process lexical + TF-IDF search, deliberately replaceable by a vector DB adapter."""

    def __init__(self, documents: Iterable[SearchDocument] = ()):
        self.documents = list(documents)
        self._fit()

    def _fit(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True) if self.documents else None
        self.matrix = (
            self.vectorizer.fit_transform([d.text for d in self.documents]) if self.vectorizer else None
        )

    def add(self, doc: SearchDocument):
        self.documents.append(doc)
        self._fit()

    def search(self, query: str, k=5):
        if not self.documents:
            return []
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        lexical = []
        terms = set(re.findall(r"\w+", query.lower()))
        for doc, sim in zip(self.documents, sims, strict=False):
            lexical.append((float(sim) + 0.05 * sum(t in doc.text.lower() for t in terms), doc))
        return [doc for _, doc in sorted(lexical, key=lambda x: x[0], reverse=True)[:k]]
