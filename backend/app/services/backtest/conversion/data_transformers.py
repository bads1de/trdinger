"""
バックテストデータ変換

backtesting.py の統計結果から取引履歴やエクイティカーブを
データベース保存用の形式に変換します。
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class TradeHistoryTransformer:
    """取引履歴変換専門クラス"""

    def transform(self, stats: Any) -> List[Dict[str, Any]]:
        """
        取引履歴を辞書リストに変換

        Args:
            stats: backtesting.pyの統計結果

        Returns:
            取引履歴の辞書リスト
        """
        try:
            trades_df = getattr(stats, "_trades", None)

            if trades_df is None or (hasattr(trades_df, "empty") and trades_df.empty):
                logger.debug("バックテストで取引が発生しませんでした")
                return []

            df = trades_df.copy()

            conversions = [
                (
                    "entry_time",
                    "EntryTime",
                    lambda s: pd.to_datetime(s, errors="coerce").dt.to_pydatetime(),
                ),
                (
                    "exit_time",
                    "ExitTime",
                    lambda s: pd.to_datetime(s, errors="coerce").dt.to_pydatetime(),
                ),
                (
                    "entry_price",
                    "EntryPrice",
                    lambda s: pd.to_numeric(s, errors="coerce"),
                ),
                (
                    "exit_price",
                    "ExitPrice",
                    lambda s: pd.to_numeric(s, errors="coerce"),
                ),
                ("size", "Size", lambda s: pd.to_numeric(s, errors="coerce")),
                ("pnl", "PnL", lambda s: pd.to_numeric(s, errors="coerce")),
                (
                    "return_pct",
                    "ReturnPct",
                    lambda s: pd.to_numeric(s, errors="coerce"),
                ),
                (
                    "duration",
                    "Duration",
                    lambda s: pd.to_numeric(s, errors="coerce").fillna(0).astype(int),  # type: ignore[reportAttributeAccessIssue]
                ),
            ]

            for target_col, source_col, func in conversions:
                if source_col in df.columns:
                    try:
                        df[target_col] = func(df[source_col])
                    except Exception:
                        df[target_col] = None if "time" in target_col else 0.0
                else:
                    df[target_col] = None if "time" in target_col else 0.0

            num_cols = ["entry_price", "exit_price", "size", "pnl", "return_pct"]
            for col in num_cols:
                df[col] = df[col].fillna(0.0)

            result_cols = [
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "size",
                "pnl",
                "return_pct",
                "duration",
            ]
            return df[result_cols].to_dict("records")
        except Exception as e:
            logger.warning(f"取引履歴の変換中にエラー: {e}")
            return []


class EquityCurveTransformer:
    """エクイティカーブ変換専門クラス"""

    def transform(self, stats: Any, max_points: int = 1000) -> List[Dict[str, Any]]:
        """
        エクイティカーブを辞書リストに変換

        Args:
            stats: backtesting.pyの統計結果
            max_points: 最大データポイント数

        Returns:
            エクイティカーブの辞書リスト
        """
        try:
            equity_df = getattr(stats, "_equity_curve", None)
            if equity_df is None or equity_df.empty:
                return []

            df = equity_df.copy()
            if len(df) > max_points:
                step = len(df) // max_points
                df = df.iloc[::step]

            df["timestamp"] = [self._safe_timestamp_conversion(t) for t in df.index]
            df["equity"] = pd.to_numeric(df["Equity"], errors="coerce").fillna(0.0)  # type: ignore[reportAttributeAccessIssue]
            df["drawdown"] = pd.to_numeric(
                df.get("DrawdownPct", 0), errors="coerce"
            ).fillna(
                0.0
            )  # type: ignore[reportAttributeAccessIssue]

            result_cols = ["timestamp", "equity", "drawdown"]
            return df[result_cols].to_dict("records")
        except Exception as e:
            logger.warning(f"エクイティカーブの変換中にエラー: {e}")
            return []

    @staticmethod
    def _safe_timestamp_conversion(value: Any) -> Optional[datetime]:
        """安全なtimestamp変換"""
        if value is None or pd.isna(value):
            return None
        try:
            if isinstance(value, pd.Timestamp):
                return value.to_pydatetime()
            elif isinstance(value, datetime):
                return value
            else:
                return pd.to_datetime(value).to_pydatetime()
        except Exception:
            return None
