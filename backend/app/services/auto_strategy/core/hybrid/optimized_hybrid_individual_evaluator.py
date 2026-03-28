"""
最適化されたハイブリッドGA個体評価器

ML予測のキャッシング、特徴量生成のキャッシングなどの最適化を提供します。
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from app.services.auto_strategy.config.ga import GAConfig
from app.services.auto_strategy.core.hybrid.hybrid_predictor import HybridPredictor
from app.services.auto_strategy.core.evaluation.individual_evaluator import IndividualEvaluator
from app.services.backtest.services.backtest_service import BacktestService
from app.services.ml.common.exceptions import MLPredictionError, MLTrainingError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.services.auto_strategy.core.hybrid.hybrid_feature_adapter import (
        HybridFeatureAdapter,
    )


class OptimizedHybridIndividualEvaluator(IndividualEvaluator):
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
        feature_adapter: Optional["HybridFeatureAdapter"] = None,
        cache_size: int = 1000,
    ):
        """
        初期化

        Args:
            backtest_service: バックテストサービス
            predictor: ハイブリッド予測器（オプション）
            feature_adapter: 特徴量アダプタ（オプション）
            cache_size: キャッシュサイズ
        """
        super().__init__(backtest_service)
        self.predictor = predictor
        self.feature_adapter = feature_adapter or self._create_feature_adapter()

        # 最適化: キャッシュ
        self._prediction_cache: Dict[str, Any] = {}
        self._feature_cache: Dict[str, Any] = {}
        self._cache_size = cache_size

    def _create_feature_adapter(self):
        """特徴量アダプタを作成"""
        try:
            from app.services.auto_strategy.core.hybrid.hybrid_feature_adapter import (
                HybridFeatureAdapter,
            )
            return HybridFeatureAdapter()
        except ImportError:
            logger.warning("HybridFeatureAdapterが見つかりません")
            return None

    def _get_evaluation_context(
        self, gene, backtest_config: Dict[str, Any], config: GAConfig
    ) -> Dict[str, Any]:
        """
        評価計算に必要な追加コンテキスト（ML予測シグナル等）を取得します（最適化版）。

        最適化:
        - 予測結果のキャッシング
        - 特徴量生成のキャッシング
        """
        if not self.predictor:
            return {}

        try:
            # キャッシュキーを生成
            cache_key = self._generate_cache_key(gene, backtest_config)

            # 予測結果のキャッシュチェック
            if cache_key in self._prediction_cache:
                logger.debug("ML予測キャッシュヒット: %s", cache_key)
                return {"prediction_signals": self._prediction_cache[cache_key]}

            ohlcv_data = self._fetch_ohlcv_data(backtest_config, config)
            if ohlcv_data is not None and not ohlcv_data.empty:
                # 特徴量生成のキャッシュチェック
                feature_cache_key = f"features_{cache_key}"
                if feature_cache_key in self._feature_cache:
                    features_df = self._feature_cache[feature_cache_key]
                    logger.debug("特徴量キャッシュヒット: %s", feature_cache_key)
                else:
                    features_df = self.feature_adapter.gene_to_features(
                        gene,
                        ohlcv_data,
                        apply_preprocessing=self._should_apply_preprocessing(config),
                    )
                    # 特徴量をキャッシュ
                    if len(self._feature_cache) < self._cache_size:
                        self._feature_cache[feature_cache_key] = features_df

                prediction_signals = self.predictor.predict(features_df)

                # 予測結果をキャッシュ
                if len(self._prediction_cache) < self._cache_size:
                    self._prediction_cache[cache_key] = prediction_signals

                logger.debug("ML予測: %s", prediction_signals)
                return {"prediction_signals": prediction_signals}

        except (MLTrainingError, MLPredictionError) as e:
            logger.warning(f"ML予測エラー: {e}")
        except Exception as e:
            logger.error(f"ML予測中の予期しないエラー: {e}", exc_info=True)

        return {}

    def _generate_cache_key(self, gene, backtest_config: Dict[str, Any]) -> str:
        """キャッシュキーを生成"""
        try:
            gene_id = getattr(gene, "id", "") or str(id(gene))
            symbol = backtest_config.get("symbol", "")
            timeframe = backtest_config.get("timeframe", "")
            start_date = backtest_config.get("start_date", "")
            end_date = backtest_config.get("end_date", "")
            return f"{gene_id}_{symbol}_{timeframe}_{start_date}_{end_date}"
        except Exception:
            return str(id(gene))

    def _should_apply_preprocessing(self, config: GAConfig) -> bool:
        """前処理を適用すべきか判定"""
        return getattr(config, "hybrid_preprocessing", True)

    def _fetch_ohlcv_data(
        self, backtest_config: Dict[str, Any], config: GAConfig
    ):
        """OHLCVデータを取得"""
        try:
            symbol = backtest_config.get("symbol")
            timeframe = backtest_config.get("timeframe")
            start_date = backtest_config.get("start_date")
            end_date = backtest_config.get("end_date")

            if not all([symbol, timeframe, start_date, end_date]):
                return None

            data_service = getattr(self.backtest_service, "data_service", None)
            if data_service is None:
                return None

            return data_service.get_ohlcv_data(
                symbol, timeframe, start_date, end_date
            )
        except Exception as e:
            logger.debug(f"OHLCVデータ取得エラー: {e}")
            return None

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
