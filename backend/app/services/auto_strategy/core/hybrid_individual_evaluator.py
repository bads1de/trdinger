"""ML統合を伴うハイブリッドGA個体評価器"""

import importlib
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

import pandas as pd

from app.services.auto_strategy.config.ga import GAConfig
from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor
from app.services.auto_strategy.core.individual_evaluator import IndividualEvaluator
from app.services.backtest.backtest_service import BacktestService
from app.services.ml.exceptions import MLPredictionError, MLTrainingError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - 型チェック専用
    from app.services.auto_strategy.core.hybrid_feature_adapter import (
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
        # バックテスト設定の補完
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
            # Gene → 特徴量変換
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
        # 基底クラスのフィットネス計算を呼び出し
        base_fitness = super()._calculate_fitness(backtest_result, config, **kwargs)

        # 予測信号を取得
        prediction_signals = kwargs.get("prediction_signals")

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
            # prediction_score計算
            if "is_valid" in prediction_signals:
                # ダマシ予測の場合: 有効確率 - 0.5 (中心化)
                prediction_score = prediction_signals["is_valid"] - 0.5
            elif "trend" in prediction_signals:
                # ボラティリティ予測の場合: トレンド確率 - 0.5 (中心化)
                prediction_score = prediction_signals["trend"] - 0.5
            else:
                # 方向予測の場合: up確率 - down確率
                prediction_score = prediction_signals.get(
                    "up", 0.0
                ) - prediction_signals.get("down", 0.0)

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
        # 基底クラスの多目的フィットネス計算を呼び出し
        # kwargsにはprediction_signalsが含まれており、self._calculate_fitness(Hybrid)に正しく伝播する
        base_fitness_values = super()._calculate_multi_objective_fitness(
            backtest_result, config, **kwargs
        )

        fitness_list = list(base_fitness_values)
        prediction_signals = kwargs.get("prediction_signals")

        # prediction_scoreが独立した目的として含まれている場合
        if "prediction_score" in config.objectives:
            pred_score_index = config.objectives.index("prediction_score")

            if prediction_signals:
                if "is_valid" in prediction_signals:
                    # ダマシ予測の場合: 有効確率 - 0.5 (中心化)
                    prediction_score = prediction_signals["is_valid"] - 0.5
                elif "trend" in prediction_signals:
                    prediction_score = prediction_signals["trend"] - 0.5
                else:
                    prediction_score = prediction_signals.get(
                        "up", 0.0
                    ) - prediction_signals.get("down", 0.0)
                fitness_list[pred_score_index] = prediction_score
            else:
                fitness_list[pred_score_index] = 0.0

        return tuple(fitness_list)

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
            "app.services.auto_strategy.core.hybrid_feature_adapter"
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
        # GAConfigの設定に従う（デフォルトはTrue）
        return getattr(ga_config, "preprocess_features", True)

    def _fetch_ohlcv_data(
        self,
        backtest_config: Dict[str, Any],
        ga_config: GAConfig,
    ) -> Optional[pd.DataFrame]:
        """
        バックテスト設定に基づきOHLCVデータを取得（キャッシュ対応）

        親クラスの _data_cache（LRUCache）を活用して DB アクセスを最小化します。
        データ取得に失敗した場合は None を返し、ML 予測をスキップします。
        これにより偽データによる学習汚染を防止します。
        """
        symbol = backtest_config.get("symbol")
        timeframe = backtest_config.get("timeframe")
        start_date = backtest_config.get("start_date")
        end_date = backtest_config.get("end_date")

        # 必須パラメータのチェック
        if not all([symbol, timeframe, start_date, end_date]):
            logger.warning(
                "ML予測用OHLCVデータ取得: 必須パラメータが不足しています "
                f"(symbol={symbol}, timeframe={timeframe}, "
                f"start_date={start_date}, end_date={end_date})"
            )
            return None

        # キャッシュキーを作成（"ohlcv:" プレフィックスで基底クラスのキーと区別）
        cache_key = ("ohlcv", symbol, timeframe, str(start_date), str(end_date))

        # 親クラスのキャッシュを確認（LRUCache）
        with self._lock:
            if cache_key in self._data_cache:
                cached_data = self._data_cache[cache_key]
                if isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
                    logger.debug(f"OHLCVデータ: キャッシュヒット (key={cache_key})")
                    return cached_data

        # キャッシュミス: DB からデータを取得
        data_service = getattr(self.backtest_service, "data_service", None)
        if data_service is None:
            logger.warning("data_service が利用できません。ML予測をスキップします。")
            return None

        try:
            if hasattr(self.backtest_service, "ensure_data_service_initialized"):
                self.backtest_service.ensure_data_service_initialized()

            ohlcv_data = data_service.get_ohlcv_data(
                symbol, timeframe, start_date, end_date
            )

            if isinstance(ohlcv_data, pd.DataFrame) and not ohlcv_data.empty:
                # キャッシュに保存
                with self._lock:
                    self._data_cache[cache_key] = ohlcv_data
                logger.debug(
                    f"OHLCVデータ: DB から取得・キャッシュ保存 (key={cache_key})"
                )
                return ohlcv_data
            else:
                logger.warning(
                    f"OHLCVデータが空または無効です: symbol={symbol}, "
                    f"timeframe={timeframe}"
                )
                return None

        except Exception as exc:
            logger.warning(f"OHLCVデータ取得エラー: {exc}")
            return None
