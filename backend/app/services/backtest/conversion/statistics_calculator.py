"""
バックテスト統計計算

backtesting.py の統計結果から各種パフォーマンス指標を計算・補完します。
"""

import logging
from typing import Any, Callable, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class BacktestStatisticsCalculator:
    """バックテスト統計計算専門クラス"""

    def calculate_statistics(self, stats: Any) -> Dict[str, Any]:
        """
        統計情報の計算メインメソッド

        Args:
            stats: backtesting.pyの統計結果

        Returns:
            統計情報辞書
        """
        try:
            actual_stats = self._resolve_stats_object(stats)
            statistics: Dict[str, Any] = {}

            is_series = isinstance(actual_stats, pd.Series)
            if is_series:
                statistics = self._extract_metrics_from_series(actual_stats)
            elif hasattr(actual_stats, "keys") and hasattr(actual_stats, "get"):
                statistics = self._extract_metrics_from_dict(actual_stats)

            statistics = self._enrich_metrics_from_trades(statistics, actual_stats)
            statistics = self._enrich_metrics_from_equity(statistics, actual_stats)
            statistics = self._validate_and_fill_defaults(statistics)

            return statistics
        except Exception as e:
            logger.error(f"統計情報の抽出中にエラー: {e}")
            return {}

    def _resolve_stats_object(self, stats: Any) -> Any:
        """statsオブジェクトの実体を取得（callableなら呼び出す）"""
        if hasattr(stats, "__call__"):
            try:
                return stats()
            except Exception as e:
                logger.warning(f"statsの呼び出しに失敗: {e}")
                return stats
        return stats

    def _extract_metrics_from_series(self, stats: Any) -> Dict[str, Any]:
        """pandas.Seriesから基本統計情報を抽出"""
        statistics = self._extract_common_metrics(stats.get)
        statistics["total_trades"] = self._safe_int_conversion(stats.get("# Trades", 0))
        statistics["avg_win"] = 0.0
        statistics["avg_loss"] = 0.0
        return statistics

    def _extract_metrics_from_dict(self, stats: Any) -> Dict[str, Any]:
        """辞書ライクなオブジェクトから基本統計情報を抽出"""
        statistics = self._extract_common_metrics(stats.get)
        statistics["total_trades"] = self._safe_int_conversion(stats.get("# Trades", 0))
        statistics["avg_win"] = 0.0
        statistics["avg_loss"] = 0.0
        return statistics

    def _extract_common_metrics(
        self, getter: Callable[[str, Any], Any]
    ) -> Dict[str, Any]:
        """Series/Dict 共通の統計指標を抽出"""
        statistics: Dict[str, Any] = {}

        statistics["total_return"] = self._safe_float_conversion(
            getter("Return [%]", 0.0)
        )
        statistics["win_rate"] = self._safe_float_conversion(
            getter("Win Rate [%]", 0.0)
        )
        statistics["profit_factor"] = self._safe_float_conversion(
            getter("Profit Factor", 0.0)
        )
        statistics["best_trade"] = self._safe_float_conversion(
            getter("Best Trade [%]", 0.0)
        )
        statistics["worst_trade"] = self._safe_float_conversion(
            getter("Worst Trade [%]", 0.0)
        )
        statistics["avg_trade"] = self._safe_float_conversion(
            getter("Avg. Trade [%]", 0.0)
        )
        statistics["max_drawdown"] = self._safe_float_conversion(
            getter("Max. Drawdown [%]", 0.0)
        )
        statistics["avg_drawdown"] = self._safe_float_conversion(
            getter("Avg. Drawdown [%]", 0.0)
        )
        statistics["max_drawdown_duration"] = self._safe_int_conversion(
            getter("Max. Drawdown Duration", 0)
        )
        statistics["avg_drawdown_duration"] = self._safe_float_conversion(
            getter("Avg. Drawdown Duration", 0)
        )
        statistics["sharpe_ratio"] = self._safe_float_conversion(
            getter("Sharpe Ratio", 0.0)
        )
        statistics["sortino_ratio"] = self._safe_float_conversion(
            getter("Sortino Ratio", 0.0)
        )
        statistics["calmar_ratio"] = self._safe_float_conversion(
            getter("Calmar Ratio", 0.0)
        )
        statistics["final_equity"] = self._safe_float_conversion(
            getter("Equity Final [$]", 0.0)
        )
        statistics["equity_peak"] = self._safe_float_conversion(
            getter("Equity Peak [$]", 0.0)
        )
        statistics["buy_hold_return"] = self._safe_float_conversion(
            getter("Buy & Hold Return [%]", 0.0)
        )

        return statistics

    def _enrich_metrics_from_trades(
        self, statistics: Dict[str, Any], stats: Any
    ) -> Dict[str, Any]:
        """取引データから詳細指標を再計算・補完"""
        try:
            trades_df = getattr(stats, "_trades", None)

            if trades_df is not None and len(trades_df) > 0:
                inferred_trades = len(trades_df)

                if inferred_trades > 0:
                    statistics["total_trades"] = int(inferred_trades)

                    pnl_col = None
                    for col in ["PnL", "Pnl", "Profit", "ProfitLoss"]:
                        if col in trades_df.columns:
                            pnl_col = col
                            break

                    if pnl_col is not None:
                        self._calculate_trade_metrics(
                            statistics, trades_df, pnl_col, inferred_trades
                        )
                    else:
                        logger.warning(
                            "取引データにPnL列が見つからないため、詳細指標の再計算をスキップします"
                        )
            else:
                if statistics.get("total_trades", 0) == 0:
                    logger.info(
                        "バックテストで0件の取引が発生しました。戦略が市場条件を満たさなかった可能性があります。"
                    )
                    statistics["total_trades"] = 0
                    statistics["win_rate"] = 0.0
                    statistics["profit_factor"] = 0.0
                    statistics["avg_win"] = 0.0
                    statistics["avg_loss"] = 0.0

            return statistics
        except Exception as e:
            logger.error(f"取引データからの指標再計算エラー: {e}")
            return statistics

    def _calculate_trade_metrics(
        self,
        statistics: Dict[str, Any],
        trades_df: Any,
        pnl_col: str,
        total_trades: int,
    ) -> None:
        """個別の取引データから指標を計算してstatisticsを更新"""
        pnl_series = pd.to_numeric(trades_df[pnl_col], errors="coerce").fillna(
            0.0
        )  # pyright: ignore[reportAttributeAccessIssue]

        wins_mask = pnl_series > 0  # pyright: ignore[reportOperatorIssue]
        losses_mask = pnl_series < 0  # pyright: ignore[reportOperatorIssue]

        win_count = int(wins_mask.sum())  # pyright: ignore[reportAttributeAccessIssue]
        loss_count = int(
            losses_mask.sum()
        )  # pyright: ignore[reportAttributeAccessIssue]
        winning_pnl = float(
            pnl_series[wins_mask].sum()
        )  # pyright: ignore[reportAttributeAccessIssue]
        losing_pnl = float(
            pnl_series[losses_mask].sum()
        )  # pyright: ignore[reportAttributeAccessIssue]

        calculated_win_rate = (
            float(win_count) / float(total_trades) * 100.0 if total_trades > 0 else 0.0
        )

        if losing_pnl < 0:
            calculated_profit_factor = winning_pnl / abs(losing_pnl)
        elif winning_pnl > 0:
            calculated_profit_factor = 999.99
        else:
            calculated_profit_factor = 0.0

        statistics["win_rate"] = calculated_win_rate
        statistics["profit_factor"] = calculated_profit_factor
        statistics["avg_win"] = winning_pnl / win_count if win_count > 0 else 0.0
        statistics["avg_loss"] = abs(losing_pnl) / loss_count if loss_count > 0 else 0.0

    def _enrich_metrics_from_equity(
        self, statistics: Dict[str, Any], stats: Any
    ) -> Dict[str, Any]:
        """エクイティカーブから指標を補完"""
        try:
            current_total_return = statistics.get("total_return", 0)
            equity_df = getattr(stats, "_equity_curve", None)

            if equity_df is not None and len(equity_df) > 0:
                last_equity = self._get_equity_value(equity_df, -1)

                if current_total_return == 0 or current_total_return is None:
                    first_equity = self._get_equity_value(equity_df, 0)

                    if (
                        first_equity is not None
                        and last_equity is not None
                        and first_equity > 0
                    ):
                        computed_return = float(
                            (last_equity - first_equity) / float(first_equity) * 100.0
                        )
                        statistics["total_return"] = computed_return
                        statistics["final_equity"] = float(last_equity)

                current_final_equity = statistics.get("final_equity", 0)
                if current_final_equity == 0 and last_equity is not None:
                    statistics["final_equity"] = float(last_equity)

            return statistics
        except Exception as e:
            logger.error(f"エクイティカーブからの指標補完エラー: {e}")
            return statistics

    def _get_equity_value(self, equity_df: Any, index: int) -> Optional[float]:
        """エクイティカーブの特定インデックスの値を取得"""
        try:
            if hasattr(equity_df, "columns") and "Equity" in equity_df.columns:
                return equity_df.iloc[index]["Equity"]
            elif hasattr(equity_df, "iloc"):
                return equity_df.iloc[index]
            return None
        except Exception as e:
            logger.warning(f"エクイティ値の取得失敗 (index={index}): {e}")
            return None

    def _validate_and_fill_defaults(self, statistics: Dict[str, Any]) -> Dict[str, Any]:
        """整合性チェックとデフォルト値設定"""
        final_total_trades = statistics.get("total_trades", 0)
        final_total_return = statistics.get("total_return", 0)

        if final_total_trades == 0 and final_total_return != 0:
            logger.warning(
                f"不整合検出: 取引数が0なのにリターンが{final_total_return}%です。"
            )
            statistics["win_rate"] = 0.0
            statistics["profit_factor"] = 0.0
            statistics["avg_win"] = 0.0
            statistics["avg_loss"] = 0.0
            statistics["best_trade"] = 0.0
            statistics["worst_trade"] = 0.0
            statistics["avg_trade"] = 0.0

        return statistics

    @staticmethod
    def _safe_float_conversion(value: Any) -> float:
        """安全なfloat変換"""
        if value is None or pd.isna(value):
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _safe_int_conversion(value: Any) -> int:
        """安全なint変換"""
        if value is None or pd.isna(value):
            return 0
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0
