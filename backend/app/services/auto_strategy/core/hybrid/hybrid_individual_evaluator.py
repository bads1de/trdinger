"""ML統合を伴うハイブリッドGA個体評価器"""

import importlib
import logging
from typing import Any, Dict, Optional, Type

from app.services.auto_strategy.config.ga import GAConfig
from app.services.auto_strategy.config.helpers import (
    resolve_ml_gate_settings,
)
from app.services.auto_strategy.core.evaluation.individual_evaluator import (
    IndividualEvaluator,
)
from app.services.auto_strategy.core.hybrid.hybrid_feature_adapter import (
    HybridFeatureAdapter,
)
from app.services.auto_strategy.core.hybrid.hybrid_predictor import (
    HybridPredictor,
    RuntimeModelPredictorAdapter,
)
from app.services.backtest.services.backtest_service import BacktestService
from app.services.backtest.shared import normalize_ohlcv_columns
from app.services.ml.models.model_manager import model_manager

logger = logging.getLogger(__name__)


class HybridIndividualEvaluator(IndividualEvaluator):
    """
    ハイブリッド個体評価器

    IndividualEvaluatorを継承し、ML予測スコアをフィットネス計算に統合します。
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
        """
        super().__init__(backtest_service)
        self.predictor = predictor
        self.feature_adapter = feature_adapter or self._create_feature_adapter()
        self._prediction_cache: Dict[str, Any] = {}
        self._feature_cache: Dict[str, Any] = {}
        self._cache_size = cache_size

    def _prepare_run_config(
        self, gene, backtest_config: Dict[str, Any], config: GAConfig
    ) -> Optional[Dict[str, Any]]:
        """
        ハイブリッド評価用のバックテスト実行設定を構築します。

        デフォルト値の補完を行った後、基底クラスの構築処理を呼び出します。

        Args:
            gene: 評価対象の遺伝子
            backtest_config: 基本的なバックテスト設定
            config: GA設定

        Returns:
            準備された実行設定辞書
        """
        defaults: Dict[str, Any] = {
            "symbol": (
                getattr(config, "target_symbol", None)
                or getattr(config, "base_symbol", None)
                or "BTCUSDT"
            ),
            "timeframe": (
                getattr(config, "target_timeframe", None)
                or getattr(config, "timeframe", None)
                or "1h"
            ),
        }

        fallback_start = getattr(config, "fallback_start_date", None)
        fallback_end = getattr(config, "fallback_end_date", None)
        if fallback_start:
            defaults["start_date"] = fallback_start
        if fallback_end:
            defaults["end_date"] = fallback_end

        return self._run_config_builder.build_run_config(
            gene,
            backtest_config,
            config,
            defaults=defaults,
        )

    def _get_evaluation_context(
        self, gene, backtest_config: Dict[str, Any], config: GAConfig
    ) -> Dict[str, Any]:
        """
        評価計算に必要な追加コンテキスト（ML予測シグナル等）を取得します。

        遺伝子から特徴量を生成し、予測器を用いて将来の予測シグナルを取得します。

        Args:
            gene: 評価対象の遺伝子
            backtest_config: バックテスト設定
            config: GA設定

        Returns:
            予測シグナルを含むコンテキスト辞書
        """
        return {}

    def _inject_external_objects(
        self,
        run_config: Dict[str, Any],
        backtest_config: Dict[str, Any],
        config: GAConfig,
    ) -> None:
        """実行設定への外部オブジェクト注入（1分足データ、MLモデル）"""
        # 基底クラスの処理（1分足データ注入）
        super()._inject_external_objects(run_config, backtest_config, config)

        enabled, predictor = self._resolve_ml_gate_runtime_state(config)
        self._set_ml_gate_state(
            run_config,
            enabled=enabled,
            predictor=predictor,
        )

    def _create_feature_adapter(
        self,
        wavelet_config: Optional[Dict[str, Any]] = None,
        use_derived_features: bool = True,
    ) -> "HybridFeatureAdapter":
        adapter_cls = self._resolve_feature_adapter_cls()
        return adapter_cls(
            wavelet_config=wavelet_config,
            use_derived_features=use_derived_features,
        )

    def _resolve_feature_adapter_cls(self) -> Type["HybridFeatureAdapter"]:
        module = importlib.import_module(
            "app.services.auto_strategy.core.hybrid.hybrid_feature_adapter"
        )
        return getattr(module, "HybridFeatureAdapter")

    def _should_apply_preprocessing(self, ga_config: GAConfig) -> bool:
        """前処理を適用するか判定"""
        hybrid_config = getattr(ga_config, "hybrid_config", None)
        if hybrid_config is None:
            return True

        return bool(getattr(hybrid_config, "preprocess_features", True))

    def _fetch_ohlcv_data(
        self,
        backtest_config: Dict[str, Any],
        ga_config: GAConfig,
    ):
        """
        バックテスト設定に基づきOHLCVデータを取得（キャッシュ対応）

        基底クラスの _get_cached_ohlcv_data を使用して DB アクセスを最小化します。
        """
        ohlcv_data = self._get_cached_ohlcv_data(
            symbol=str(backtest_config.get("symbol", "")),
            timeframe=str(backtest_config.get("timeframe", "")),
            start_date=backtest_config.get("start_date"),
            end_date=backtest_config.get("end_date"),
            cache_prefix="ohlcv",
        )

        if ohlcv_data is None:
            return None

        return normalize_ohlcv_columns(ohlcv_data, lowercase=True)

    def clear_caches(self) -> None:
        """ハイブリッド評価用のキャッシュをクリアする。"""
        self._prediction_cache.clear()
        self._feature_cache.clear()

    def get_cache_statistics(self) -> Dict[str, Any]:
        """ハイブリッド評価用のキャッシュ統計を返す。"""
        return {
            "prediction_cache_size": len(self._prediction_cache),
            "feature_cache_size": len(self._feature_cache),
            "cache_limit": self._cache_size,
        }

    @staticmethod
    def _set_ml_gate_state(
        run_config: Dict[str, Any],
        *,
        enabled: bool,
        predictor: Optional[Any] = None,
    ) -> None:
        """MLゲート関連フラグをまとめて更新する。"""
        parameters = run_config["strategy_config"]["parameters"]
        parameters["volatility_gate_enabled"] = enabled

        if predictor is not None:
            parameters["ml_predictor"] = predictor
        else:
            parameters.pop("ml_predictor", None)

    def _resolve_ml_gate_runtime_state(
        self, config: GAConfig
    ) -> tuple[bool, Optional[Any]]:
        """ML gate の実行時状態を解決する。"""
        ml_gate_settings = resolve_ml_gate_settings(config)
        if not ml_gate_settings.enabled:
            return False, None

        model_path = ml_gate_settings.model_path
        if model_path:
            try:
                ml_filter_model = RuntimeModelPredictorAdapter(
                    model_manager.load_model(model_path)
                )
                if ml_filter_model.is_trained():
                    return True, ml_filter_model
            except Exception:
                return False, None

        if self.predictor and self.predictor.is_trained():
            return True, self.predictor

        return False, None
