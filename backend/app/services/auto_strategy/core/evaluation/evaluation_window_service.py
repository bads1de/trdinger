"""
評価期間の warmup と評価窓トリミングを扱う補助サービス
"""

from __future__ import annotations

import logging
from datetime import datetime
from math import ceil
from typing import Any, Dict, cast

import numpy as np
import pandas as pd

from app.services.backtest.conversion.backtest_result_converter import (
    BacktestResultConverter,
)
from app.services.backtest.shared import normalize_ohlcv_columns

from .time_alignment import align_timestamp_to_index

logger = logging.getLogger(__name__)


class EvaluationWindowService:
    """評価窓の前処理と統計再計算を提供する。"""

    def prepare_backtest_config_for_evaluation(
        self, gene: object, backtest_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """指標 warmup 用に評価開始前の履歴を含む実行設定へ変換する。"""
        prepared_config = backtest_config.copy()
        start_date = backtest_config.get("start_date")
        timeframe = str(backtest_config.get("timeframe", ""))

        if not start_date or not timeframe:
            return prepared_config

        warmup_bars = self.estimate_required_warmup_bars(gene, timeframe)
        if warmup_bars <= 0:
            return prepared_config

        start_timestamp = pd.Timestamp(start_date)
        base_minutes = self.timeframe_to_minutes(timeframe)
        adjusted_start = start_timestamp - pd.to_timedelta(
            warmup_bars * base_minutes, unit="m"
        )

        prepared_config["_evaluation_start"] = start_date
        prepared_config["start_date"] = self.format_datetime_like(
            start_date, cast(pd.Timestamp, adjusted_start)
        )
        return prepared_config

    _prepare_backtest_config_for_evaluation = prepare_backtest_config_for_evaluation

    def estimate_required_warmup_bars(self, gene: object, base_timeframe: str) -> int:
        """戦略実行前に必要な warmup バー数を推定する。"""
        base_minutes = self.timeframe_to_minutes(base_timeframe)
        max_bars = 0

        indicators = getattr(gene, "indicators", []) or []
        for indicator in indicators:
            if not getattr(indicator, "enabled", True):
                continue

            lookback = self.extract_lookback_from_parameters(
                getattr(indicator, "parameters", {}) or {}
            )
            indicator_timeframe = (
                getattr(indicator, "timeframe", None) or base_timeframe
            )
            timeframe_scale = max(
                1, ceil(self.timeframe_to_minutes(indicator_timeframe) / base_minutes)
            )
            max_bars = max(max_bars, (lookback + 1) * timeframe_scale)

        position_sizing_gene = getattr(gene, "position_sizing_gene", None)
        if position_sizing_gene and getattr(position_sizing_gene, "enabled", True):
            max_bars = max(
                max_bars,
                int(
                    max(
                        getattr(position_sizing_gene, "lookback_period", 0),
                        getattr(position_sizing_gene, "atr_period", 0),
                    )
                )
                + 1,
            )

        for tpsl_attr in ("tpsl_gene", "long_tpsl_gene", "short_tpsl_gene"):
            tpsl_gene = getattr(gene, tpsl_attr, None)
            if tpsl_gene and getattr(tpsl_gene, "enabled", True):
                max_bars = max(
                    max_bars,
                    int(
                        max(
                            getattr(tpsl_gene, "lookback_period", 0),
                            getattr(tpsl_gene, "atr_period", 0),
                        )
                    )
                    + 1,
                )

        return int(max_bars)

    _estimate_required_warmup_bars = estimate_required_warmup_bars

    @staticmethod
    def extract_lookback_from_parameters(parameters: Dict[str, Any]) -> int:
        """インディケーターパラメータから lookback 長を推定する。"""
        if not isinstance(parameters, dict):
            return 0

        excluded_tokens = ("multiplier", "threshold", "offset", "shift", "std")
        lookback_tokens = ("length", "period", "window", "lookback", "span")
        lookback = 0

        for key, value in parameters.items():
            key_str = str(key).lower()
            if any(token in key_str for token in excluded_tokens):
                continue
            if not any(token in key_str for token in lookback_tokens):
                continue
            if isinstance(value, (int, float)) and value > 0:
                lookback = max(lookback, int(ceil(float(value))))

        return lookback

    _extract_lookback_from_parameters = extract_lookback_from_parameters

    @staticmethod
    def timeframe_to_minutes(timeframe: str) -> int:
        """timeframe 文字列を分単位へ変換する。"""
        timeframe_map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }
        return timeframe_map.get(str(timeframe), 60)

    _timeframe_to_minutes = timeframe_to_minutes

    @staticmethod
    def format_datetime_like(original_value: object, timestamp: pd.Timestamp) -> object:
        """元の入力型に合わせて Timestamp を整形する。"""
        if isinstance(original_value, pd.Timestamp):
            return timestamp
        if isinstance(original_value, datetime):
            return timestamp.to_pydatetime()
        if isinstance(original_value, str):
            return str(timestamp)
        return timestamp

    _format_datetime_like = format_datetime_like

    def apply_evaluation_window_to_result(
        self,
        backtest_result: Dict[str, Any],
        raw_stats: object,
        market_data: pd.DataFrame,
        evaluation_start: object,
        evaluation_end: object,
    ) -> Dict[str, Any]:
        """warmup を除外した評価窓だけでバックテスト結果を再計算する。"""
        if raw_stats is None or market_data is None or market_data.empty:
            return backtest_result

        raw_equity_curve = getattr(raw_stats, "_equity_curve", None)
        if raw_equity_curve is None or getattr(raw_equity_curve, "empty", True):
            return backtest_result

        market_df = market_data.copy()
        market_df = market_df.sort_index()

        start_ts = self.normalize_timestamp_to_index(evaluation_start, market_df.index)
        end_ts = self.normalize_timestamp_to_index(evaluation_end, market_df.index)

        start_pos = int(cast(pd.Index, market_df.index).searchsorted(cast(Any, start_ts), side="left"))
        end_pos = int(cast(pd.Index, market_df.index).searchsorted(cast(Any, end_ts), side="right"))

        if start_pos >= end_pos:
            logger.warning(
                "評価窓のトリミングに失敗しました: start=%s end=%s",
                evaluation_start,
                evaluation_end,
            )
            return backtest_result

        trimmed_market_data = market_df.iloc[start_pos:end_pos].copy()
        if trimmed_market_data.empty:
            return backtest_result

        trimmed_equity_curve: pd.DataFrame = self.slice_equity_curve_for_window(
            raw_equity_curve,
            trimmed_market_data.index,
            start_pos,
            end_pos,
            backtest_result.get("initial_capital", 0.0),
        )
        equity_values = (
            cast(
                pd.Series,
                pd.to_numeric(trimmed_equity_curve["Equity"], errors="coerce"),
            )
            .fillna(float(backtest_result.get("initial_capital", 0.0)))
            .to_numpy()
        )

        trimmed_trades = self.slice_trades_for_window(
            getattr(raw_stats, "_trades", None),
            start_pos,
            end_pos,
        )

        normalized_market_data = self.normalize_ohlc_data_for_stats(trimmed_market_data)
        window_stats = self._compute_window_stats(
            trimmed_trades,
            equity_values,
            normalized_market_data,
        )

        converter = BacktestResultConverter()
        adjusted_result = converter.convert_backtest_results(
            stats=window_stats,
            strategy_name=str(backtest_result.get("strategy_name", "")),
            symbol=str(backtest_result.get("symbol", "")),
            timeframe=str(backtest_result.get("timeframe", "")),
            initial_capital=float(backtest_result.get("initial_capital", 0.0)),
            start_date=start_ts.to_pydatetime(),
            end_date=end_ts.to_pydatetime(),
            config_json=backtest_result.get("config_json", {}),
        )
        adjusted_result["_raw_stats"] = window_stats
        return adjusted_result

    _apply_evaluation_window_to_result = apply_evaluation_window_to_result

    @staticmethod
    def normalize_timestamp_to_index(value: object, index: pd.Index) -> pd.Timestamp:
        """インデックスのタイムゾーンに合わせて Timestamp を正規化する。"""
        return align_timestamp_to_index(value, index)

    _normalize_timestamp_to_index = normalize_timestamp_to_index

    @staticmethod
    def normalize_ohlc_data_for_stats(market_data: pd.DataFrame) -> pd.DataFrame:
        """backtesting.py が期待する大文字 OHLCV カラムへ正規化する。"""
        return normalize_ohlcv_columns(market_data, ensure_volume=True)

    _normalize_ohlc_data_for_stats = normalize_ohlc_data_for_stats

    @staticmethod
    def slice_equity_curve_for_window(
        raw_equity_curve: pd.DataFrame,
        target_index: pd.Index,
        start_pos: int,
        end_pos: int,
        initial_capital: float,
    ) -> pd.DataFrame:
        """評価窓に対応するエクイティカーブを切り出す。"""
        if not isinstance(raw_equity_curve, pd.DataFrame) or raw_equity_curve.empty:
            return pd.DataFrame(
                {"Equity": [initial_capital] * len(target_index)}, index=target_index
            )

        if len(raw_equity_curve) >= end_pos:
            trimmed = raw_equity_curve.iloc[start_pos:end_pos].copy()
        else:
            trimmed = raw_equity_curve.reindex(target_index).copy()

        trimmed = trimmed.reindex(target_index).ffill()
        if "Equity" in trimmed.columns:
            trimmed["Equity"] = cast(
                pd.Series, pd.to_numeric(trimmed["Equity"], errors="coerce")
            ).fillna(float(initial_capital))
        else:
            trimmed["Equity"] = float(initial_capital)
        if "DrawdownPct" in trimmed.columns:
            trimmed["DrawdownPct"] = cast(
                pd.Series, pd.to_numeric(trimmed["DrawdownPct"], errors="coerce")
            ).fillna(0.0)
        else:
            trimmed["DrawdownPct"] = 0.0
        return trimmed

    _slice_equity_curve_for_window = slice_equity_curve_for_window

    @staticmethod
    def slice_trades_for_window(
        raw_trades: object,
        start_pos: int,
        end_pos: int,
    ) -> pd.DataFrame:
        """評価窓内のトレードだけを抽出し、バー番号を窓内基準へ補正する。"""
        if not isinstance(raw_trades, pd.DataFrame):
            return pd.DataFrame()

        trades_df = raw_trades.copy()
        if trades_df.empty:
            return trades_df

        if {"EntryBar", "ExitBar"}.issubset(trades_df.columns):
            entry_bars: pd.Series = cast(
                pd.Series, pd.to_numeric(trades_df["EntryBar"], errors="coerce")
            )
            exit_bars: pd.Series = cast(
                pd.Series, pd.to_numeric(trades_df["ExitBar"], errors="coerce")
            )
            mask = (
                entry_bars.notna()
                & exit_bars.notna()
                & (entry_bars >= start_pos)
                & (exit_bars < end_pos)
            )
            trades_df = trades_df.loc[mask].copy()
            if trades_df.empty:
                return trades_df
            trades_df["EntryBar"] = (
                pd.to_numeric(trades_df["EntryBar"], errors="coerce").astype(int)  # type: ignore[reportAttributeAccessIssue]
                - start_pos
            )
            trades_df["ExitBar"] = (
                pd.to_numeric(trades_df["ExitBar"], errors="coerce").astype(int)  # type: ignore[reportAttributeAccessIssue]
                - start_pos
            )
        return trades_df

    _slice_trades_for_window = slice_trades_for_window

    def _compute_window_stats(
        self,
        trades_df: pd.DataFrame,
        equity_values: np.ndarray,
        ohlc_data: pd.DataFrame,
    ) -> object:
        """評価窓だけを対象に backtesting.py の統計を再計算する。"""
        from backtesting._stats import compute_stats  # type: ignore

        return compute_stats(
            trades=trades_df,
            equity=equity_values,
            ohlc_data=ohlc_data,
            strategy_instance=None,
        )
