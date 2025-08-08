"""
バックテスト結果変換サービス

backtesting.pyの結果をデータベース保存用形式に変換します。
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class BacktestResultConversionError(Exception):
    """バックテスト結果変換エラー"""


class BacktestResultConverter:
    """
    バックテスト結果変換サービス

    backtesting.pyの統計結果をデータベース保存用の形式に変換します。
    """

    def convert_backtest_results(
        self,
        stats: Any,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        initial_capital: float,
        start_date: Any,
        end_date: Any,
        config_json: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        バックテスト結果をデータベース形式に変換

        Args:
            stats: backtesting.pyの統計結果
            strategy_name: 戦略名
            symbol: 取引ペア
            timeframe: 時間軸
            initial_capital: 初期資金
            start_date: 開始日時
            end_date: 終了日時
            config_json: 設定JSON

        Returns:
            データベース保存用の結果辞書

        Raises:
            BacktestResultConversionError: 変換に失敗した場合
        """
        try:
            # 基本情報
            result = {
                "strategy_name": strategy_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": self._normalize_date(start_date),
                "end_date": self._normalize_date(end_date),
                "initial_capital": float(initial_capital),
                "config_json": config_json,
                "created_at": datetime.now(),
            }

            # 統計情報を追加
            result.update(self._extract_statistics(stats))

            # 取引履歴を追加
            result["trade_history"] = self._convert_trade_history(stats)

            # エクイティカーブを追加
            result["equity_curve"] = self._convert_equity_curve(stats)

            return result

        except Exception as e:
            logger.error(f"バックテスト結果変換エラー: {e}")
            raise BacktestResultConversionError(f"結果の変換に失敗しました: {e}")

    def _extract_statistics(self, stats: Any) -> Dict[str, Any]:
        """統計情報を抽出"""
        try:
            return {
                "total_return": self._safe_float_conversion(stats.get("Return [%]", 0)),
                "total_trades": int(stats.get("# Trades", 0)),
                "win_rate": self._safe_float_conversion(stats.get("Win Rate [%]", 0)),
                "best_trade": self._safe_float_conversion(
                    stats.get("Best Trade [%]", 0)
                ),
                "worst_trade": self._safe_float_conversion(
                    stats.get("Worst Trade [%]", 0)
                ),
                "avg_trade": self._safe_float_conversion(
                    stats.get("Avg. Trade [%]", 0)
                ),
                "max_drawdown": self._safe_float_conversion(
                    stats.get("Max. Drawdown [%]", 0)
                ),
                "avg_drawdown": self._safe_float_conversion(
                    stats.get("Avg. Drawdown [%]", 0)
                ),
                "max_drawdown_duration": self._safe_int_conversion(
                    stats.get("Max. Drawdown Duration", 0)
                ),
                "avg_drawdown_duration": self._safe_float_conversion(
                    stats.get("Avg. Drawdown Duration", 0)
                ),
                "sharpe_ratio": self._safe_float_conversion(
                    stats.get("Sharpe Ratio", 0)
                ),
                "sortino_ratio": self._safe_float_conversion(
                    stats.get("Sortino Ratio", 0)
                ),
                "calmar_ratio": self._safe_float_conversion(
                    stats.get("Calmar Ratio", 0)
                ),
                "final_equity": self._safe_float_conversion(
                    stats.get("Equity Final [$]", 0)
                ),
                "equity_peak": self._safe_float_conversion(
                    stats.get("Equity Peak [$]", 0)
                ),
                "buy_hold_return": self._safe_float_conversion(
                    stats.get("Buy & Hold Return [%]", 0)
                ),
            }
        except Exception as e:
            logger.warning(f"統計情報の抽出中にエラー: {e}")
            return {}

    def _convert_trade_history(self, stats: Any) -> List[Dict[str, Any]]:
        """取引履歴を変換"""
        try:
            trades_df = getattr(stats, "_trades", None)
            if trades_df is None or trades_df.empty:
                return []

            trades = []
            for _, trade in trades_df.iterrows():
                trade_dict = {
                    "entry_time": self._safe_timestamp_conversion(
                        trade.get("EntryTime")
                    ),
                    "exit_time": self._safe_timestamp_conversion(trade.get("ExitTime")),
                    "entry_price": self._safe_float_conversion(trade.get("EntryPrice")),
                    "exit_price": self._safe_float_conversion(trade.get("ExitPrice")),
                    "size": self._safe_float_conversion(trade.get("Size")),
                    "pnl": self._safe_float_conversion(trade.get("PnL")),
                    "return_pct": self._safe_float_conversion(trade.get("ReturnPct")),
                    "duration": self._safe_int_conversion(trade.get("Duration")),
                }
                trades.append(trade_dict)

            return trades

        except Exception as e:
            logger.warning(f"取引履歴の変換中にエラー: {e}")
            return []

    def _convert_equity_curve(self, stats: Any) -> List[Dict[str, Any]]:
        """エクイティカーブを変換"""
        try:
            equity_df = getattr(stats, "_equity_curve", None)
            if equity_df is None or equity_df.empty:
                return []

            # データ量を制限（最大1000ポイント）
            if len(equity_df) > 1000:
                step = len(equity_df) // 1000
                equity_df = equity_df.iloc[::step]

            equity_curve = []
            for timestamp, row in equity_df.iterrows():
                equity_point = {
                    "timestamp": self._safe_timestamp_conversion(timestamp),
                    "equity": self._safe_float_conversion(row.get("Equity")),
                    "drawdown": self._safe_float_conversion(row.get("DrawdownPct", 0)),
                }
                equity_curve.append(equity_point)

            return equity_curve

        except Exception as e:
            logger.warning(f"エクイティカーブの変換中にエラー: {e}")
            return []

    def _normalize_date(self, date_value: Any) -> datetime:
        """日付値を正規化"""
        if isinstance(date_value, datetime):
            return date_value
        elif isinstance(date_value, str):
            return datetime.fromisoformat(date_value.replace("Z", "+00:00"))
        else:
            raise ValueError(f"サポートされていない日付形式: {type(date_value)}")

    def _safe_float_conversion(self, value: Any) -> float:
        """安全なfloat変換"""
        if value is None or pd.isna(value):
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _safe_int_conversion(self, value: Any) -> int:
        """安全なint変換"""
        if value is None or pd.isna(value):
            return 0
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0

    def _safe_timestamp_conversion(self, value: Any) -> Optional[datetime]:
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
