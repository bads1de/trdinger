"""最適化されたハイブリッドGA個体評価器。"""

import logging
from typing import Any, Dict, Optional

from app.services.auto_strategy.core.hybrid.hybrid_predictor import HybridPredictor
from app.services.auto_strategy.core.hybrid.hybrid_individual_evaluator import (
    HybridIndividualEvaluator,
)
from app.services.backtest.services.backtest_service import BacktestService

logger = logging.getLogger(__name__)

class OptimizedHybridIndividualEvaluator(HybridIndividualEvaluator):
    """
    最適化されたハイブリッド個体評価器

    主な最適化ポイント:
    1. ML予測結果のキャッシング
    2. 特徴量生成のキャッシング
    3. モデルの再利用
    """

    def __init__(
        self,
        backtest_service: BacktestService,
        predictor: Optional[HybridPredictor] = None,
        feature_adapter: Optional[Any] = None,
        cache_size: int = 1000,
    ):
        """初期化。"""
        super().__init__(
            backtest_service=backtest_service,
            predictor=predictor,
            feature_adapter=feature_adapter,
        )

        self._prediction_cache: Dict[str, Any] = {}
        self._feature_cache: Dict[str, Any] = {}
        self._cache_size = cache_size

    def _get_evaluation_context(self, gene, backtest_config, config) -> Dict[str, Any]:
        """
        最適化版でも fitness へ ML 予測を加点しない。

        ボラ回帰化後の ML は runtime の volatility gate 用であり、
        個体評価コンテキストには載せない。
        """
        return {}

    def clear_caches(self):
        """キャッシュをクリア"""
        self._prediction_cache.clear()
        self._feature_cache.clear()

    def get_cache_statistics(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        return {
            "prediction_cache_size": len(self._prediction_cache),
            "feature_cache_size": len(self._feature_cache),
            "cache_limit": self._cache_size,
        }
