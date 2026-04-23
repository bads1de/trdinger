"""
戦略初期化モジュール

UniversalStrategy.init() に集約されていた初期化責務を分離する。
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pandas as pd

from ..genes import IndicatorGene, TPSLMethod

logger = logging.getLogger(__name__)


class StrategyInitializer:
    """UniversalStrategy の初期化フローを担当するクラス。"""

    def __init__(self, strategy):
        self.strategy = strategy

    def initialize(self) -> None:
        """指標初期化と各種事前計算を実行する。"""
        try:
            if not self.strategy.gene:
                return

            self._initialize_indicators()
            self._precompute_ml_features()
            self._precompute_condition_signals()
            self._precompute_position_sizing_atr()
            self._precompute_tpsl_atr()
        except Exception as e:
            logger.error("戦略初期化エラー: %s", e, exc_info=True)
            raise

    def init_indicator(self, indicator_gene: IndicatorGene) -> None:
        """単一指標の初期化を委譲する。"""
        try:
            self.strategy.indicator_calculator.init_indicator(
                indicator_gene, self.strategy
            )
        except Exception as e:
            logger.error(
                "指標初期化エラー %s: %s",
                indicator_gene.type,
                e,
                exc_info=True,
            )
            raise

    def _initialize_indicators(self) -> None:
        enabled_indicators = [
            ind for ind in self.strategy.gene.indicators if ind.enabled
        ]
        for indicator_gene in enabled_indicators:
            self.init_indicator(indicator_gene)

    def _precompute_ml_features(self) -> None:
        if (
            self.strategy.volatility_gate_enabled
            and self.strategy.ml_predictor
        ):
            self.strategy.ml_filter.precompute_ml_features()

    def _precompute_condition_signals(self) -> None:
        try:
            precomputed_signals = self._get_or_create_signal_cache(
                "_precomputed_signals"
            )
            precomputed_exit_signals = self._get_or_create_signal_cache(
                "_precomputed_exit_signals"
            )

            # 同一 strategy インスタンスの再初期化でも古い方向別キャッシュを残さない。
            precomputed_signals.clear()
            precomputed_exit_signals.clear()

            # Entry条件
            long_conds = getattr(
                self.strategy.gene, "long_entry_conditions", []
            )
            if long_conds:
                self._cache_vectorized_signal(
                    precomputed_signals,
                    1.0,
                    self.strategy.condition_evaluator.calculate_conditions_vectorized(
                        long_conds,
                        self.strategy,
                    ),
                )

            short_conds = getattr(
                self.strategy.gene, "short_entry_conditions", []
            )
            if short_conds:
                self._cache_vectorized_signal(
                    precomputed_signals,
                    -1.0,
                    self.strategy.condition_evaluator.calculate_conditions_vectorized(
                        short_conds,
                        self.strategy,
                    ),
                )

            # Exit条件
            long_exit_conds = getattr(
                self.strategy.gene, "long_exit_conditions", []
            )
            if long_exit_conds:
                self._cache_vectorized_signal(
                    precomputed_exit_signals,
                    1.0,
                    self.strategy.condition_evaluator.calculate_conditions_vectorized(
                        long_exit_conds,
                        self.strategy,
                    ),
                )

            short_exit_conds = getattr(
                self.strategy.gene, "short_exit_conditions", []
            )
            if short_exit_conds:
                self._cache_vectorized_signal(
                    precomputed_exit_signals,
                    -1.0,
                    self.strategy.condition_evaluator.calculate_conditions_vectorized(
                        short_exit_conds,
                        self.strategy,
                    ),
                )
        except Exception as e:
            logger.debug("ベクトル化事前計算失敗（フォールバック使用）: %s", e)

    def _get_or_create_signal_cache(self, attr_name: str) -> dict:
        """シグナルキャッシュを取得または初期化する。"""
        cache = getattr(self.strategy, attr_name, None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self.strategy, attr_name, cache)
        return cache

    def _cache_vectorized_signal(
        self,
        cache: dict,
        direction: float,
        signal,
    ) -> None:
        """ベクトル化できたシグナルだけをキャッシュする。"""
        if signal is None:
            return

        if isinstance(signal, pd.Series):
            cache[direction] = signal
            return

        if isinstance(signal, np.ndarray) and signal.ndim > 0:
            cache[direction] = signal
            return

        logger.debug(
            "スカラー結果のためベクトル化シグナルをキャッシュしません: direction=%s, type=%s",
            direction,
            type(signal).__name__,
        )

    def _precompute_position_sizing_atr(self) -> None:
        self.strategy._precomputed_atr = None
        if not (
            self.strategy.gene.position_sizing_gene
            and self.strategy.gene.position_sizing_gene.enabled
        ):
            return

        try:
            lookback = getattr(
                self.strategy.gene.position_sizing_gene,
                "lookback_period",
                14,
            )
            try:
                import pandas_ta_classic as ta

                temp_df = pd.DataFrame(
                    {
                        "high": self.strategy.data.High,
                        "low": self.strategy.data.Low,
                        "close": self.strategy.data.Close,
                    }
                )
                atr_series = ta.atr(
                    cast(pd.Series, temp_df["high"]),
                    cast(pd.Series, temp_df["low"]),
                    cast(pd.Series, temp_df["close"]),
                    length=lookback,
                )
                if atr_series is not None:
                    self.strategy._precomputed_atr = cast(
                        pd.Series,
                        atr_series,
                    ).values
                    logger.debug("ATR事前計算完了")
                else:
                    logger.warning(
                        "ATR事前計算失敗: ta.atr が None を返しました"
                    )
            except ImportError:
                logger.warning(
                    "pandas-taが見つからないためATR事前計算をスキップ"
                )
            except Exception as e:
                logger.debug(
                    "ATR事前計算中のエラー（フォールバック使用）: %s", e
                )
        except Exception as e:
            logger.debug("ATR事前計算失敗: %s", e)

    def _precompute_tpsl_atr(self) -> None:
        self.strategy._precomputed_tpsl_atr = {}
        for direction in [1.0, -1.0]:
            tpsl_gene = self.strategy._get_effective_tpsl_gene(direction)
            if tpsl_gene and getattr(tpsl_gene, "method", None) in (
                TPSLMethod.VOLATILITY_BASED,
                TPSLMethod.ADAPTIVE,
                TPSLMethod.STATISTICAL,
            ):
                try:
                    atr_period = getattr(tpsl_gene, "atr_period", 14)
                    if atr_period not in self.strategy._precomputed_tpsl_atr:
                        if hasattr(self.strategy.data, "df"):
                            import pandas_ta_classic as ta

                            high = self.strategy.data.df["High"]
                            low = self.strategy.data.df["Low"]
                            close = self.strategy.data.df["Close"]
                            atr_result = ta.atr(
                                high, low, close, length=atr_period
                            )
                            if atr_result is not None:
                                self.strategy._precomputed_tpsl_atr[
                                    atr_period
                                ] = cast(
                                    pd.Series,
                                    atr_result,
                                ).values
                except Exception as e:
                    logger.debug("TP/SL ATR事前計算失敗: %s", e)
