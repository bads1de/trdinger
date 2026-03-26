"""ML統合を伴うハイブリッドGA個体評価器"""

import importlib
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from app.services.auto_strategy.config.ga import GAConfig
from app.services.auto_strategy.core.hybrid.hybrid_predictor import HybridPredictor
from app.services.auto_strategy.core.evaluation.individual_evaluator import IndividualEvaluator
from app.services.backtest.services.backtest_service import BacktestService
from app.services.ml.common.exceptions import MLPredictionError, MLTrainingError
from app.services.ml.models.model_manager import model_manager

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - 型チェック専用
    from app.services.auto_strategy.core.hybrid.hybrid_feature_adapter import (
        HybridFeatureAdapter,
    )


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
        ensured_config = self._ensure_backtest_defaults(backtest_config, config)
        return super()._prepare_run_config(gene, ensured_config, config)

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
        if not self.predictor:
            return {}

        try:
            ohlcv_data = self._fetch_ohlcv_data(backtest_config, config)
            if ohlcv_data is not None and not ohlcv_data.empty:
                features_df = self.feature_adapter.gene_to_features(
                    gene,
                    ohlcv_data,
                    apply_preprocessing=self._should_apply_preprocessing(config),
                )

                prediction_signals = self.predictor.predict(features_df)
                logger.debug("ML予測: %s", prediction_signals)
                return {"prediction_signals": prediction_signals}

        except (MLTrainingError, MLPredictionError) as e:
            logger.warning(f"ML予測エラー: {e}")
        except Exception as e:
            logger.error(f"ML予測エラー（予期しない）: {e}")

        return {}

    def _calculate_fitness(
        self, backtest_result: Dict[str, Any], config: GAConfig, **kwargs
    ) -> float:
        """
        フィットネス計算（ML予測スコア統合版）

        Args:
            backtest_result: バックテスト結果
            config: GA設定
            **kwargs: 追加のコンテキスト情報（prediction_signalsを含む）

        Returns:
            フィットネス値
        """
        base_fitness = super()._calculate_fitness(
            backtest_result, config, **kwargs
        )

        prediction_signals = kwargs.get("prediction_signals")

        try:
            metrics = self._extract_performance_metrics(backtest_result)
            total_trades = metrics.get("total_trades", 0)
        except Exception:
            total_trades = 0

        if total_trades == 0:
            return max(0.0, base_fitness)

        if prediction_signals:
            prediction_score = self._extract_prediction_score(prediction_signals)
            prediction_weight = config.fitness_weights.get("prediction_score", 0.1)

            fitness = base_fitness + prediction_weight * prediction_score

            logger.debug(
                f"Fitness: base={base_fitness:.4f}, "
                f"pred_score={prediction_score:.4f}, "
                f"final={fitness:.4f}"
            )

            return max(0.0, fitness)

        return base_fitness

    def _calculate_multi_objective_fitness(
        self, backtest_result: Dict[str, Any], config: GAConfig, **kwargs
    ) -> tuple:
        """
        多目的最適化用フィットネス計算（ML予測スコア統合版）

        Args:
            backtest_result: バックテスト結果
            config: GA設定
            **kwargs: 追加のコンテキスト情報（prediction_signalsを含む）

        Returns:
            各目的の評価値のタプル
        """
        base_fitness_values = super()._calculate_multi_objective_fitness(
            backtest_result, config, **kwargs
        )

        fitness_list = list(base_fitness_values)
        prediction_signals = kwargs.get("prediction_signals")

        if "prediction_score" in config.objectives:
            pred_score_index = config.objectives.index("prediction_score")

            if prediction_signals:
                fitness_list[pred_score_index] = self._extract_prediction_score(
                    prediction_signals
                )
            else:
                fitness_list[pred_score_index] = 0.0

        return tuple(fitness_list)

    def _inject_external_objects(
        self,
        run_config: Dict[str, Any],
        backtest_config: Dict[str, Any],
        config: GAConfig,
    ) -> None:
        """実行設定への外部オブジェクト注入（1分足データ、MLモデル）"""
        # 基底クラスの処理（1分足データ注入）
        super()._inject_external_objects(run_config, backtest_config, config)

        # MLフィルター設定
        if config.ml_filter_enabled and config.ml_model_path:
            try:
                ml_filter_model = model_manager.load_model(config.ml_model_path)
                if ml_filter_model:
                    run_config["strategy_config"]["parameters"][
                        "ml_predictor"
                    ] = ml_filter_model
                    run_config["strategy_config"]["parameters"][
                        "ml_filter_threshold"
                    ] = 0.5
            except Exception:
                run_config["strategy_config"]["parameters"]["ml_filter_enabled"] = False
        elif config.ml_filter_enabled:
            run_config["strategy_config"]["parameters"]["ml_filter_enabled"] = False

    @staticmethod
    def _extract_prediction_score(prediction_signals: Dict[str, Any]) -> float:
        """予測シグナルから中心化されたスコアを抽出"""
        if "is_valid" in prediction_signals:
            return prediction_signals["is_valid"] - 0.5
        if "trend" in prediction_signals:
            return prediction_signals["trend"] - 0.5
        return prediction_signals.get("up", 0.0) - prediction_signals.get("down", 0.0)

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

    def _ensure_backtest_defaults(
        self,
        backtest_config: Dict[str, Any],
        ga_config: GAConfig,
    ) -> Dict[str, Any]:
        """バックテスト設定に欠損している基本情報を補完"""

        ensured = backtest_config.copy()

        default_symbol = (
            getattr(ga_config, "target_symbol", None)
            or getattr(ga_config, "base_symbol", None)
            or "BTCUSDT"
        )
        default_timeframe = (
            getattr(ga_config, "target_timeframe", None)
            or getattr(ga_config, "timeframe", None)
            or "1h"
        )

        ensured.setdefault("symbol", default_symbol)
        ensured.setdefault("timeframe", default_timeframe)

        fallback_start = getattr(ga_config, "fallback_start_date", None)
        fallback_end = getattr(ga_config, "fallback_end_date", None)
        if fallback_start:
            ensured.setdefault("start_date", fallback_start)
        if fallback_end:
            ensured.setdefault("end_date", fallback_end)

        return ensured

    def _should_apply_preprocessing(self, ga_config: GAConfig) -> bool:
        """前処理を適用するか判定"""
        return getattr(ga_config, "preprocess_features", True)

    def _fetch_ohlcv_data(
        self,
        backtest_config: Dict[str, Any],
        ga_config: GAConfig,
    ):
        """
        バックテスト設定に基づきOHLCVデータを取得（キャッシュ対応）

        基底クラスの _get_cached_ohlcv_data を使用して DB アクセスを最小化します。
        """
        return self._get_cached_ohlcv_data(
            symbol=backtest_config.get("symbol"),
            timeframe=backtest_config.get("timeframe"),
            start_date=backtest_config.get("start_date"),
            end_date=backtest_config.get("end_date"),
            cache_prefix="ohlcv",
        )
