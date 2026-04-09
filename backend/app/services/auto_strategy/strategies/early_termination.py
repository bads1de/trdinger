"""
早期終了評価モジュール

UniversalStrategy に集約されていた評価進捗・早期打ち切り判定の責務を分離する。
"""

from __future__ import annotations

import logging
from math import ceil
from typing import Any, Optional

import pandas as pd

from app.services.auto_strategy.config.ga_nested_configs import (
    resolve_early_termination_settings,
)
from app.services.auto_strategy.core.evaluation.time_alignment import (
    align_timestamp_to_index,
    align_timestamp_to_reference,
)

logger = logging.getLogger(__name__)


class StrategyEarlyTermination(RuntimeError):
    """戦略が早期打ち切り条件に達したことを示す例外。"""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class StrategyEarlyTerminationController:
    """評価進捗と早期打ち切り判定を担当するクラス。"""

    def __init__(self, strategy):
        self.strategy = strategy

    def normalize_evaluation_start(self, value: Any) -> Optional[pd.Timestamp]:
        """評価開始時刻を pandas.Timestamp に正規化する。"""
        if value is None or value == "":
            return None

        try:
            return pd.Timestamp(value)
        except Exception:
            logger.warning("evaluation_start の解析に失敗しました: %s", value)
            return None

    def is_evaluation_bar(self) -> bool:
        """現在バーが評価開始時刻以降かを返す。"""
        evaluation_start_raw = getattr(self.strategy, "_evaluation_start", None)
        if evaluation_start_raw is None:
            return True

        if (
            not hasattr(self.strategy.data, "index")
            or len(self.strategy.data.index) == 0
        ):
            return True

        current_time = pd.Timestamp(self.strategy.data.index[-1])
        evaluation_start = align_timestamp_to_reference(
            evaluation_start_raw,
            current_time,
        )

        return current_time >= evaluation_start

    def initialize_evaluation_progress_bounds(
        self,
        data: Any,
    ) -> tuple[Optional[pd.DatetimeIndex], int, int]:
        """評価進捗計算に使う評価窓の境界を初期化する。"""
        raw_index = getattr(data, "index", None)
        total_bars = max(1, int(getattr(self.strategy, "_total_bars", 1) or 1))
        if raw_index is None or len(raw_index) == 0:
            return None, 0, total_bars

        try:
            full_index = pd.DatetimeIndex(raw_index)
        except Exception:
            return None, 0, total_bars

        start_index = 0
        evaluation_start = getattr(self.strategy, "_evaluation_start", None)
        if evaluation_start is not None:
            aligned_start = align_timestamp_to_index(evaluation_start, full_index)
            start_index = int(full_index.searchsorted(aligned_start, side="left"))

        evaluation_total_bars = max(1, len(full_index) - start_index)
        return full_index, start_index, evaluation_total_bars

    @staticmethod
    def align_timestamp_to_index_tz(
        value: pd.Timestamp,
        index: pd.DatetimeIndex,
    ) -> pd.Timestamp:
        """DatetimeIndex に合わせて Timestamp の timezone をそろえる。"""
        return align_timestamp_to_index(value, index)

    def get_current_equity(self, default: float = 0.0) -> float:
        """現在資産を安全に取得する。"""
        try:
            return float(getattr(self.strategy, "equity", default) or default)
        except Exception:
            return float(default)

    def get_progress_ratio(self) -> float:
        """現在までの評価進捗を返す。"""
        evaluation_index = getattr(self.strategy, "_evaluation_index", None)
        if isinstance(evaluation_index, pd.DatetimeIndex) and len(evaluation_index) > 0:
            current_index = getattr(self.strategy.data, "index", None)
            if current_index is not None and len(current_index) > 0:
                try:
                    current_time = self.align_timestamp_to_index_tz(
                        pd.Timestamp(current_index[-1]),
                        evaluation_index,
                    )
                    current_position = int(
                        evaluation_index.searchsorted(current_time, side="right")
                    )
                    evaluation_start_index = int(
                        getattr(self.strategy, "_evaluation_start_index", 0) or 0
                    )
                    evaluation_total_bars = max(
                        1,
                        int(getattr(self.strategy, "_evaluation_total_bars", 1) or 1),
                    )
                    evaluated_bars = max(0, current_position - evaluation_start_index)
                    return min(1.0, evaluated_bars / evaluation_total_bars)
                except Exception:
                    logger.debug(
                        "評価窓ベースの進捗計算に失敗したためフォールバックします"
                    )

        total_bars = max(1, int(getattr(self.strategy, "_total_bars", 1) or 1))
        current_bar = max(
            0,
            int(getattr(self.strategy, "_current_bar_index", 0) or 0),
        )
        return min(1.0, current_bar / total_bars)

    def calculate_closed_trade_expectancy(self) -> Optional[float]:
        """クローズ済みトレードの平均期待値を返す。"""
        try:
            trades = list(getattr(self.strategy, "closed_trades", []) or [])
        except Exception:
            return None

        if not trades:
            return None

        values = []
        for trade in trades:
            for attr_name in ("pl_pct", "pl", "pnl", "return_pct"):
                value = getattr(trade, attr_name, None)
                if value is None:
                    continue
                try:
                    values.append(float(value))
                    break
                except Exception:
                    continue

        if not values:
            return None

        return float(sum(values) / len(values))

    def should_terminate_early(self) -> Optional[str]:
        """早期打ち切りすべき理由を返す。"""
        settings = resolve_early_termination_settings(self.strategy)
        if not settings.enabled:
            return None

        current_equity = self.get_current_equity(
            default=float(getattr(self.strategy, "_starting_equity", 0.0) or 0.0)
        )
        self.strategy._max_equity_seen = max(
            float(getattr(self.strategy, "_max_equity_seen", current_equity)),
            current_equity,
        )

        max_drawdown = settings.max_drawdown
        if max_drawdown is not None and self.strategy._max_equity_seen > 0:
            drawdown = max(
                0.0,
                (self.strategy._max_equity_seen - current_equity)
                / self.strategy._max_equity_seen,
            )
            if drawdown >= float(max_drawdown):
                return "max_drawdown"

        progress = self.get_progress_ratio()

        min_trades = settings.min_trades
        if min_trades is not None and progress >= float(
            settings.min_trade_check_progress
        ):
            closed_trade_count = len(getattr(self.strategy, "closed_trades", []) or [])
            required_trade_count = max(
                1,
                int(
                    ceil(
                        float(min_trades)
                        * progress
                        * float(settings.trade_pace_tolerance)
                    )
                ),
            )
            if closed_trade_count < required_trade_count:
                return "trade_pace"

        min_expectancy = settings.min_expectancy
        if min_expectancy is not None and progress >= float(
            settings.expectancy_progress
        ):
            closed_trade_count = len(getattr(self.strategy, "closed_trades", []) or [])
            expectancy_min_trades = int(settings.expectancy_min_trades)
            if closed_trade_count >= expectancy_min_trades:
                expectancy = self.calculate_closed_trade_expectancy()
                if expectancy is not None and expectancy < float(min_expectancy):
                    return "expectancy"

        return None

    def check_early_termination(self) -> None:
        """早期打ち切り条件を満たした場合に例外を送出する。"""
        reason = self.should_terminate_early()
        if reason:
            raise StrategyEarlyTermination(reason)
