"""
バックテスト結果変換サービス

backtesting.pyの統計結果をデータベース保存用の形式に変換します。
統計計算とデータ変換は専門クラスに委譲します。
"""

import logging
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)


class BacktestResultConversionError(Exception):
    """バックテスト結果変換エラー"""


class BacktestResultConverter:
    """
    バックテスト結果変換サービス（Facade）

    統計計算とデータ変換を専門クラスに委譲し、
    結果の組み立てのみを担当します。
    """

    def __init__(self) -> None:
        from .data_transformers import EquityCurveTransformer, TradeHistoryTransformer
        from .statistics_calculator import BacktestStatisticsCalculator

        self._stats_calculator = BacktestStatisticsCalculator()
        self._trade_transformer = TradeHistoryTransformer()
        self._equity_transformer = EquityCurveTransformer()

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
            # statsオブジェクトの実体を取得
            actual_stats = self._resolve_stats_object(stats)

            result = {
                "strategy_name": strategy_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": self._normalize_date(start_date),
                "end_date": self._normalize_date(end_date),
                "initial_capital": float(initial_capital),
                "commission_rate": config_json.get("commission_rate", 0.001),
                "config_json": config_json,
                "execution_time": None,
                "status": "completed",
                "error_message": None,
                "created_at": datetime.now(),
                "performance_metrics": self._stats_calculator.calculate_statistics(
                    actual_stats
                ),
                "trade_history": self._trade_transformer.transform(actual_stats),
                "equity_curve": self._equity_transformer.transform(actual_stats),
            }

            return result
        except Exception as e:
            logger.error(f"バックテスト結果変換エラー: {e}")
            raise BacktestResultConversionError(f"結果の変換に失敗しました: {e}")

    def _resolve_stats_object(self, stats: Any) -> Any:
        """statsオブジェクトの実体を取得（callableなら呼び出す）"""
        if hasattr(stats, "__call__"):
            try:
                return stats()
            except Exception as e:
                logger.warning(f"statsの呼び出しに失敗: {e}")
                return stats
        return stats

    def _normalize_date(self, date_value: Any) -> datetime:
        """日付値を正規化"""
        if isinstance(date_value, datetime):
            return date_value
        elif isinstance(date_value, str):
            return datetime.fromisoformat(date_value.replace("Z", "+00:00"))
        else:
            raise ValueError(f"サポートされていない日付形式: {type(date_value)}")
