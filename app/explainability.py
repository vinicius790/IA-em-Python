"""Compatibilidade: a explicabilidade foi unificada em ``app.modeling.explainability``."""

from .modeling.explainability import MAX_SHAPLEY_FEATURES, permutation_explanation, shapley_explanation

__all__ = ["MAX_SHAPLEY_FEATURES", "permutation_explanation", "shapley_explanation"]
