from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


@dataclass(slots=True)
class TextClassifier:
    max_features: int = 10000
    ngram_range: tuple[int, int] = (1, 2)
    encoder: LabelEncoder = field(init=False)
    model: Pipeline = field(init=False)

    def __post_init__(self):
        self.encoder = LabelEncoder()
        self.model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=self.max_features, ngram_range=self.ngram_range, sublinear_tf=True
                    ),
                ),
                ("classifier", LogisticRegression(max_iter=2500)),
            ]
        )

    def fit(self, texts: list[str], labels: list[str]) -> TextClassifier:
        y = self.encoder.fit_transform(labels)
        self.model.fit(texts, y)
        return self

    def predict(self, texts: list[str]) -> list[str]:
        values = self.model.predict(texts)
        return [str(value) for value in self.encoder.inverse_transform(values)]

    def probabilities(self, texts: list[str]) -> list[dict[str, float]]:
        values = self.model.predict_proba(texts)
        return [
            {
                str(label): round(float(prob), 6)
                for label, prob in zip(self.encoder.classes_, row, strict=False)
            }
            for row in values
        ]

    def manifest(self) -> dict[str, Any]:
        return {
            "kind": "tfidf-logistic-regression",
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "classes": [str(c) for c in self.encoder.classes_],
        }


def frame_from_records(texts: list[str], labels: list[str] | None = None) -> pd.DataFrame:
    data = {"text": texts}
    if labels is not None:
        data["label"] = labels
    return pd.DataFrame(data)
