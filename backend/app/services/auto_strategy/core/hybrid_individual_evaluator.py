"""Hybrid GA individual evaluator with ML integration."""

import importlib
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

import numpy as np
import pandas as pd

from app.services.auto_strategy.config.ga_runtime import GAConfig
from app.services.auto_strategy.core.individual_evaluator import IndividualEvaluator
from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor
from app.services.auto_strategy.models.strategy_gene import StrategyGene
from app.services.auto_strategy.serializers.gene_serialization import GeneSerializer
from app.services.backtest.backtest_service import BacktestService
from app.services.ml.exceptions import MLTrainingError, MLPredictionError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - 型チェック専用
    from app.services.auto_strategy.utils.hybrid_feature_adapter import (
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
        regime_detector: Optional[Any] = None,
    ):
        """
        初期化

        Args:
            backtest_service: バックテストサービス
            predictor: ハイブリッド予測器（オプション）
            feature_adapter: 特徴量アダプタ（オプション）
            regime_detector: レジーム検知器（オプション）
        """
        super().__init__(backtest_service, regime_detector)
        self.predictor = predictor
        self.feature_adapter = feature_adapter or self._create_feature_adapter()

    def evaluate_individual(self, individual, config: GAConfig):
        """
        個体評価（ML予測統合版）

        Args:
            individual: 評価する個体
            config: GA設定

        Returns:
            フィットネス値のタプル
        """
        try:
            serializer = GeneSerializer()
            gene = serializer.from_list(individual, StrategyGene)

            # バックテスト実行用の設定を構築
            backtest_config = (
                self._fixed_backtest_config.copy()
                if self._fixed_backtest_config
                else {}
            )
            backtest_config = self._ensure_backtest_defaults(backtest_config, config)

            # 戦略設定を追加
            strategy_payload = serializer.strategy_gene_to_dict(gene)
            backtest_config["strategy_config"] = {
                "strategy_type": "GENERATED_GA",
                "parameters": {"strategy_gene": strategy_payload},
            }
            gene_identifier = getattr(gene, "id", "GENE")[:8]
            backtest_config["strategy_name"] = f"GA_Individual_{gene_identifier}"

            # レジーム適応が有効な場合、レジーム検知を行う
            regime_labels = None
            if config.regime_adaptation_enabled and self.regime_detector:
                try:
                    symbol = backtest_config.get("symbol")
                    timeframe = backtest_config.get("timeframe")
                    start_date = backtest_config.get("start_date")
                    end_date = backtest_config.get("end_date")

                    if symbol and timeframe and start_date and end_date:
                        self.backtest_service._ensure_data_service_initialized()
                        ohlcv_data = self.backtest_service.data_service.get_ohlcv_data(
                            symbol, timeframe, start_date, end_date
                        )

                        if not ohlcv_data.empty:
                            regime_labels = self.regime_detector.detect_regimes(
                                ohlcv_data
                            )
                            logger.info(
                                f"レジーム検知完了: {len(regime_labels)} サンプル"
                            )
                except Exception as e:
                    logger.error(f"レジーム検知エラー: {e}")
                    regime_labels = None

            # バックテスト実行
            result = self.backtest_service.run_backtest(backtest_config)

            # ML予測スコアを取得（predictorが設定されている場合）
            prediction_signals = None
            if self.predictor:
                try:
                    # Gene → 特徴量変換
                    symbol = backtest_config.get("symbol")
                    timeframe = backtest_config.get("timeframe")
                    start_date = backtest_config.get("start_date")
                    end_date = backtest_config.get("end_date")

                    ohlcv_data = self._fetch_ohlcv_data(backtest_config, config)
                    if ohlcv_data is not None and not ohlcv_data.empty:
                        features_df = self.feature_adapter.gene_to_features(
                            gene,
                            ohlcv_data,
                            apply_preprocessing=self._should_apply_preprocessing(
                                config
                            ),
                        )

                        prediction_signals = self.predictor.predict(features_df)
                        logger.debug("ML予測: %s", prediction_signals)
                except (MLTrainingError, MLPredictionError) as e:
                    logger.warning(f"ML予測エラー: {e}")
                    prediction_signals = None
                except Exception as e:
                    logger.error(f"ML予測エラー（予期しない）: {e}")
                    prediction_signals = None

            # フィットネス計算（単一目的・多目的対応、ML予測スコア統合）
            if config.enable_multi_objective:
                fitness_values = self._calculate_multi_objective_fitness(
                    result, config, regime_labels, prediction_signals
                )
                return fitness_values
            else:
                fitness = self._calculate_fitness(
                    result, config, regime_labels, prediction_signals
                )
                return (fitness,)

        except Exception as e:
            logger.error(f"個体評価エラー: {e}")
            if config.enable_multi_objective:
                return tuple(0.0 for _ in config.objectives)
            else:
                return (0.0,)

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
            "app.services.auto_strategy.utils.hybrid_feature_adapter"
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
        # デフォルトではFalseを返す（必要に応じてGAConfigに設定を追加）
        return False

    def _fetch_ohlcv_data(
        self,
        backtest_config: Dict[str, Any],
        ga_config: GAConfig,
    ) -> pd.DataFrame:
        """バックテスト設定に基づきOHLCVデータを取得"""

        symbol = backtest_config.get("symbol")
        timeframe = backtest_config.get("timeframe")
        start_date = backtest_config.get("start_date")
        end_date = backtest_config.get("end_date")

        data_service = getattr(self.backtest_service, "data_service", None)

        if (
            symbol
            and timeframe
            and start_date
            and end_date
            and data_service is not None
        ):
            try:
                if hasattr(self.backtest_service, "_ensure_data_service_initialized"):
                    self.backtest_service._ensure_data_service_initialized()

                ohlcv_data = data_service.get_ohlcv_data(
                    symbol, timeframe, start_date, end_date
                )
                if isinstance(ohlcv_data, pd.DataFrame) and not ohlcv_data.empty:
                    return ohlcv_data
            except Exception as exc:
                logger.warning(f"OHLCVデータ取得エラー: {exc}")

        return self._generate_fallback_ohlcv(ga_config)

    def _generate_fallback_ohlcv(self, ga_config: GAConfig) -> pd.DataFrame:
        """データ取得に失敗した際のフォールバックOHLCVデータを生成"""

        start = getattr(ga_config, "fallback_start_date", None)
        end = getattr(ga_config, "fallback_end_date", None)

        try:
            start_ts = pd.to_datetime(start) if start else None
        except Exception:
            start_ts = None
        try:
            end_ts = pd.to_datetime(end) if end else None
        except Exception:
            end_ts = None

        if start_ts is None or end_ts is None or start_ts >= end_ts:
            end_ts = pd.Timestamp.utcnow().floor("H")
            start_ts = end_ts - pd.Timedelta(days=7)

        index = pd.date_range(start=start_ts, end=end_ts, freq="1H")
        if index.empty:
            index = pd.date_range(end=end_ts, periods=72, freq="1H")

        base_price = np.linspace(100.0, 105.0, len(index))
        fallback_df = pd.DataFrame(
            {
                "timestamp": index,
                "open": base_price,
                "high": base_price + 0.5,
                "low": base_price - 0.5,
                "close": base_price + 0.1,
                "volume": np.linspace(1000.0, 1500.0, len(index)),
            }
        )

        return fallback_df.reset_index(drop=True)

    def _calculate_fitness(
        self,
        backtest_result: Dict[str, Any],
        config: GAConfig,
        regime_labels: Optional[list] = None,
        prediction_signals: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        フィットネス計算（ML予測スコア統合版）

        Args:
            backtest_result: バックテスト結果
            config: GA設定
            regime_labels: レジームラベル（オプション）
            prediction_signals: ML予測信号（オプション）

        Returns:
            フィットネス値
        """
        # 基底クラスのフィットネス計算を呼び出し
        base_fitness = super()._calculate_fitness(
            backtest_result, config, regime_labels
        )

        # 取引が発生していない場合はMLスコアを統合しない
        try:
            metrics = self._extract_performance_metrics(backtest_result)
            total_trades = metrics.get("total_trades", 0)
        except Exception:
            total_trades = 0

        if total_trades == 0:
            return max(0.0, base_fitness)

        # ML予測スコアを追加（predictorが設定され、予測が成功した場合）
        if prediction_signals:
            # prediction_score = up確率 - down確率
            prediction_score = prediction_signals["up"] - prediction_signals["down"]

            # 予測重みを取得（デフォルト0.1）
            prediction_weight = config.fitness_weights.get("prediction_score", 0.1)

            # フィットネスに予測スコアを加算
            fitness = base_fitness + prediction_weight * prediction_score

            logger.debug(
                f"Fitness: base={base_fitness:.4f}, "
                f"pred_score={prediction_score:.4f}, "
                f"final={fitness:.4f}"
            )

            return max(0.0, fitness)

        # 予測がない場合はベースフィットネスのみ
        return base_fitness

    def _calculate_multi_objective_fitness(
        self,
        backtest_result: Dict[str, Any],
        config: GAConfig,
        regime_labels: Optional[list] = None,
        prediction_signals: Optional[Dict[str, float]] = None,
    ) -> tuple:
        """
        多目的最適化用フィットネス計算（ML予測スコア統合版）

        Args:
            backtest_result: バックテスト結果
            config: GA設定
            regime_labels: レジームラベル（オプション）
            prediction_signals: ML予測信号（オプション）

        Returns:
            各目的の評価値のタプル
        """
        # 基底クラスの多目的フィットネス計算を呼び出し
        base_fitness_values = super()._calculate_multi_objective_fitness(
            backtest_result, config, regime_labels
        )

        # 目的にprediction_scoreが含まれている場合
        if "prediction_score" in config.objectives:
            fitness_list = list(base_fitness_values)

            # prediction_scoreの位置を見つける
            pred_score_index = config.objectives.index("prediction_score")

            # ML予測スコアを計算
            if prediction_signals:
                prediction_score = prediction_signals["up"] - prediction_signals["down"]
                fitness_list[pred_score_index] = prediction_score
            else:
                # 予測がない場合はデフォルト値
                fitness_list[pred_score_index] = 0.0

            return tuple(fitness_list)

        # prediction_scoreが目的に含まれていない場合はベースのみ
        return base_fitness_values
