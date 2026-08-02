"""
ハイブリッド(GA+ML)モジュール

機械学習と遺伝的アルゴリズムを統合したハイブリッドアプローチを提供します。
各モジュール（numpy/pandas 依存）は属性アクセスまで遅延インポートされます。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hybrid_feature_adapter import (
        HybridFeatureAdapter,
        WaveletFeatureTransformer,
    )
    from .hybrid_individual_evaluator import HybridIndividualEvaluator
    from .hybrid_predictor import HybridPredictor

_ATTRIBUTE_EXPORTS = {
    "HybridFeatureAdapter": ".hybrid_feature_adapter",
    "WaveletFeatureTransformer": ".hybrid_feature_adapter",
    "HybridIndividualEvaluator": ".hybrid_individual_evaluator",
    "HybridPredictor": ".hybrid_predictor",
}

__all__ = [
    "HybridFeatureAdapter",
    "HybridIndividualEvaluator",
    "HybridPredictor",
    "WaveletFeatureTransformer",
]

from app.utils.lazy_import import setup_lazy_import  # noqa: E402

setup_lazy_import(globals(), _ATTRIBUTE_EXPORTS)
