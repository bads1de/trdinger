"""
早期終了評価モジュール

UniversalStrategy に集約されていた評価進捗・早期打ち切り判定の責務を分離する。
"""

from __future__ import annotations

import logging
from math import ceil
from typing import Any, Optional, Union, cast

import pandas as pd

from app.services.auto_strategy.config.ga.nested_configs import EarlyTerminationSettings
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
    """評価進捗と早期打ち切り判定を担当するクラス。

    バックテスト実行中に、設定された条件（最小取引数、最大損失など）
    を満たした場合に評価を早期に終了し、リソースを節約します。

    主な機能:
    - 評価開始時刻の管理
    - 進捗状況のトラッキング
    - 早期終了条件の判定
    - 取引統計の収集
    """

    def __init__(self, strategy):
        """早期終了コントローラを初期化する。

        Args:
            strategy: 戦略インスタンス。評価対象のバックテスト戦略。
        """
        self.strategy = strategy

    def normalize_evaluation_start(self, value: Union[str, pd.Timestamp, None]) -> Optional[pd.Timestamp]:
        """評価開始時刻をpandas.Timestampに正規化する。

        文字列やその他の形式で指定された評価開始時刻を、
        pandas.Timestampオブジェクトに変換します。

        Args:
            value: 正規化する評価開始時刻。
                文字列、数値、またはNone。

        Returns:
            Optional[pd.Timestamp]: 正規化された評価開始時刻。
                変換に失敗した場合はNone。
        """
        if value is None or value == "":
            return None

        try:
            return pd.Timestamp(value)  # type: ignore[return-value]
        except Exception as e:
            logger.warning(
                "evaluation_start の解析に失敗しました: %s, エラー: %s", value, e
            )
            return None

    def is_evaluation_bar(self) -> bool:
        """現在バーが評価開始時刻以降かを判定する。

        評価開始時刻が設定されている場合、現在処理中のバーが
        その時刻以降かどうかを確認します。設定されていない場合は
        常にTrueを返します。

        Returns:
            bool: 現在バーが評価開始時刻以降の場合はTrue。
                評価開始時刻が未設定の場合もTrue。
        """
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

        return current_time >= evaluation_start  # type: ignore[operator]

    def initialize_evaluation_progress_bounds(
        self,
        data: object,
    ) -> tuple[Optional[pd.DatetimeIndex], int, int]:
        """評価進捗計算に使う評価窓の境界を初期化する。

        評価開始時刻から現在時刻までの評価対象バー数を計算し、
        進捗率の算出に使用します。

        Args:
            data: 評価対象データ。index属性を持つ必要があります。

        Returns:
            tuple[Optional[pd.DatetimeIndex], int, int]:
                - 評価窓のDatetimeIndex（失敗時はNone）
                - 評価開始インデックス
                - 評価対象総バー数
        """
        raw_index = getattr(data, "index", None)
        total_bars = max(1, int(getattr(self.strategy, "_total_bars", 1) or 1))
        if raw_index is None or len(raw_index) == 0:
            return None, 0, total_bars

        try:
            full_index = pd.DatetimeIndex(raw_index)
        except Exception as e:
            logger.debug("評価用インデックスの解析に失敗しました: %s", e)
            return None, 0, total_bars

        start_index = 0
        evaluation_start = getattr(self.strategy, "_evaluation_start", None)
        if evaluation_start is not None:
            aligned_start = align_timestamp_to_index(evaluation_start, full_index)
            start_index = int(cast(pd.Index, full_index).searchsorted(cast(Any, aligned_start), side="left"))

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
        except Exception as e:
            logger.debug("現在資産の取得に失敗しました: %s", e)
            return float(default)

    def get_progress_ratio(self) -> float:
        """現在までの評価進捗を返す。"""
        evaluation_index = getattr(self.strategy, "_evaluation_index", None)
        eval_len = int(len(evaluation_index)) if evaluation_index is not None else 0
        if isinstance(evaluation_index, pd.DatetimeIndex) and eval_len > 0:
            current_index = getattr(self.strategy.data, "index", None)
            current_len = int(len(current_index)) if current_index is not None else 0
            if current_index is not None and current_len > 0:
                try:
                    current_ts = pd.Timestamp(current_index[-1])
                    if bool(pd.isna(current_ts)):
                        raise ValueError("NaT timestamp in index")
                    current_time = self.align_timestamp_to_index_tz(
                        cast(pd.Timestamp, current_ts),
                        evaluation_index,
                    )
                    current_position = int(
                        cast(pd.Index, evaluation_index).searchsorted(cast(Any, current_time), side="right")
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
                except Exception as e:
                    logger.debug(
                        "評価窓ベースの進捗計算に失敗したためフォールバックします: %s",
                        e,
                    )

        total_bars = max(1, int(getattr(self.strategy, "_total_bars", 1) or 1))
        current_bar = max(
            0,
            int(getattr(self.strategy, "_current_bar_index", 0) or 0),
        )
        return min(1.0, current_bar / total_bars)

    def calculate_closed_trade_expectancy(self) -> Optional[float]:
        """クローズ済みトレードの平均期待値（期待リターン）を計算する。

        確定済みのトレードから平均リターン（%）を算出し、
        戦略の期待収益性を評価します。

        Returns:
            Optional[float]: 平均トレード期待値（%）。
                トレードが存在しない場合はNone。
        """
        try:
            trades = list(getattr(self.strategy, "closed_trades", []) or [])
        except Exception as e:
            logger.debug("トレード履歴の取得に失敗しました: %s", e)
            return None

        if not trades:
            return None

        values = []
        for trade in trades:
            # パーセントベースの属性を優先（pl_pct, return_pct）
            # 絶対値（pl, pnl）は通貨単位であり、期待値閾値と比較できない
            for attr_name in ("pl_pct", "return_pct"):
                value = getattr(trade, attr_name, None)
                if value is None:
                    continue
                try:
                    values.append(float(value))
                    break
                except Exception as e:
                    logger.debug("トレードP&L値の変換に失敗しました: %s", e)
                    continue
            else:
                # パーセント値がない場合、絶対値から換算を試みる
                for attr_name in ("pl", "pnl"):
                    value = getattr(trade, attr_name, None)
                    if value is None:
                        continue
                    try:
                        initial_capital = getattr(
                            self.strategy, "initial_capital", None
                        )
                        if initial_capital and initial_capital > 0:
                            values.append(float(value) / initial_capital * 100.0)
                        else:
                            logger.debug(
                                f"トレードP&Lのパーセント変換をスキップ: "
                                f"initial_capital={initial_capital}"
                            )
                        break
                    except Exception as e:
                        logger.debug("トレードP&Lのパーセント変換に失敗しました: %s", e)
                        continue

        if not values:
            return None

        return float(sum(values) / len(values))

    def should_terminate_early(self) -> Optional[str]:
        """早期打ち切りすべきかどうかを判定する。

        設定された早期終了条件（最小取引数、最大ドローダウン、
        進捗状況など）をチェックし、条件を満たす場合は
        終了理由を返します。

        Returns:
            Optional[str]: 早期終了理由。条件を満たさない場合はNone。
        """
        raw_settings = getattr(self.strategy, "early_termination_settings", None)
        if raw_settings is None:
            settings = EarlyTerminationSettings()
        elif isinstance(raw_settings, EarlyTerminationSettings):
            settings = raw_settings
        else:
            settings = EarlyTerminationSettings.from_source(raw_settings)
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
        """早期打ち切り条件を満たした場合に例外を送出する。

        should_terminate_early() を呼び出して早期終了理由をチェックし、
        条件が満たされている場合はStrategyEarlyTermination例外を
        送出して評価を中断します。

        Raises:
            StrategyEarlyTermination: 早期終了条件が満たされた場合。
        """
        reason = self.should_terminate_early()
        if reason:
            raise StrategyEarlyTermination(reason)
