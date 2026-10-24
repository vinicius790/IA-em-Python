"""Compatibilidade: a explicabilidade foi unificada em ``app.modeling.shapley``."""

from .modeling.shapley import MAX_SHAPLEY_FEATURES, permutation_explanation, shapley_explanation

__all__ = ["MAX_SHAPLEY_FEATURES", "permutation_explanation", "shapley_explanation"]
