"""
バックテスト結果変換サービス

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
                "commission_rate": config_json.get("commission_rate", 0.001),
                "config_json": config_json,
                "execution_time": None,  # バックテスト実行時間は後で設定される
                "status": "completed",
                "error_message": None,
                "created_at": datetime.now(),
            }

            # 統計情報を追加
            result["performance_metrics"] = self._extract_statistics(stats)

            # 取引履歴を追加
            result["trade_history"] = self._convert_trade_history(stats)

            # エクイティカーブを追加
            result["equity_curve"] = self._convert_equity_curve(stats)

            return result

        except Exception as e:
            logger.error(f"バックテスト結果変換エラー: {e}")
            raise BacktestResultConversionError(f"結果の変換に失敗しました: {e}")

    # ここからリファクタリングしたメソッド群

    def _extract_statistics(self, stats: Any) -> Dict[str, Any]:
        """統計情報を抽出"""
        try:
            # 1. statsオブジェクトの実体を取得
            actual_stats = self._resolve_stats_object(stats)
            statistics = {}

            # 2. 基本的な統計情報を抽出
            is_series = isinstance(actual_stats, pd.Series)
            if is_series:
                statistics = self._extract_metrics_from_series(actual_stats)
            elif hasattr(actual_stats, "keys") and hasattr(actual_stats, "get"):
                # Dict-like object
                statistics = self._extract_metrics_from_dict(actual_stats)

            # 3. 取引データから詳細指標を再計算・補完
            statistics = self._enrich_metrics_from_trades(statistics, actual_stats)

            # 4. エクイティカーブから指標を補完
            statistics = self._enrich_metrics_from_equity(statistics, actual_stats)

            # 5. 整合性チェックとデフォルト値設定
            statistics = self._validate_and_fill_defaults(statistics)

            return statistics

        except Exception as e:
            logger.error(f"統計情報の抽出中にエラー: {e}")
            logger.error(f"Statsの型: {type(stats)}")
            # actual_stats変数が存在しない場合のハンドリング
            actual_stats_type = "未取得"
            try:
                actual_stats = self._resolve_stats_object(stats)
                actual_stats_type = type(actual_stats)
            except Exception:
                pass
            logger.error(f"実際のstatsの型: {actual_stats_type}")
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
        statistics = {}

        # 基本指標
        statistics["total_return"] = self._safe_float_conversion(
            stats.get("Return [%]", 0.0)
        )
        statistics["total_trades"] = self._safe_int_conversion(stats.get("# Trades", 0))
        statistics["win_rate"] = self._safe_float_conversion(
            stats.get("Win Rate [%]", 0.0)
        )
        statistics["profit_factor"] = self._safe_float_conversion(
            stats.get("Profit Factor", 0.0)
        )

        # 追加の指標
        statistics["best_trade"] = self._safe_float_conversion(
            stats.get("Best Trade [%]", 0.0)
        )
        statistics["worst_trade"] = self._safe_float_conversion(
            stats.get("Worst Trade [%]", 0.0)
        )
        statistics["avg_trade"] = self._safe_float_conversion(
            stats.get("Avg. Trade [%]", 0.0)
        )
        statistics["max_drawdown"] = self._safe_float_conversion(
            stats.get("Max. Drawdown [%]", 0.0)
        )
        statistics["avg_drawdown"] = self._safe_float_conversion(
            stats.get("Avg. Drawdown [%]", 0.0)
        )

        # 期間関連
        statistics["max_drawdown_duration"] = self._safe_int_conversion(
            stats.get("Max. Drawdown Duration", 0)
        )
        statistics["avg_drawdown_duration"] = self._safe_float_conversion(
            stats.get("Avg. Drawdown Duration", 0)
        )

        # リスク指標
        statistics["sharpe_ratio"] = self._safe_float_conversion(
            stats.get("Sharpe Ratio", 0.0)
        )
        statistics["sortino_ratio"] = self._safe_float_conversion(
            stats.get("Sortino Ratio", 0.0)
        )
        statistics["calmar_ratio"] = self._safe_float_conversion(
            stats.get("Calmar Ratio", 0.0)
        )

        # 資産関連
        statistics["final_equity"] = self._safe_float_conversion(
            stats.get("Equity Final [$]", 0.0)
        )
        statistics["equity_peak"] = self._safe_float_conversion(
            stats.get("Equity Peak [$]", 0.0)
        )
        statistics["buy_hold_return"] = self._safe_float_conversion(
            stats.get("Buy & Hold Return [%]", 0.0)
        )

        # 平均利益・平均損失（初期化）
        statistics["avg_win"] = 0.0
        statistics["avg_loss"] = 0.0

        return statistics

    def _extract_metrics_from_dict(self, stats: Any) -> Dict[str, Any]:
        """辞書ライクなオブジェクトから基本統計情報を抽出"""
        statistics = {}

        # 基本的なパフォーマンス指標
        statistics["total_return"] = self._safe_float_conversion(
            stats.get("Return [%]", 0.0)
        )
        statistics["total_trades"] = int(stats.get("# Trades", 0))
        statistics["win_rate"] = self._safe_float_conversion(
            stats.get("Win Rate [%]", 0.0)
        )

        # 追加の指標
        statistics["best_trade"] = self._safe_float_conversion(
            stats.get("Best Trade [%]", 0.0)
        )
        statistics["worst_trade"] = self._safe_float_conversion(
            stats.get("Worst Trade [%]", 0.0)
        )
        statistics["avg_trade"] = self._safe_float_conversion(
            stats.get("Avg. Trade [%]", 0.0)
        )
        statistics["max_drawdown"] = self._safe_float_conversion(
            stats.get("Max. Drawdown [%]", 0.0)
        )
        statistics["avg_drawdown"] = self._safe_float_conversion(
            stats.get("Avg. Drawdown [%]", 0.0)
        )

        # 期間関連
        statistics["max_drawdown_duration"] = self._safe_int_conversion(
            stats.get("Max. Drawdown Duration", 0)
        )
        statistics["avg_drawdown_duration"] = self._safe_float_conversion(
            stats.get("Avg. Drawdown Duration", 0)
        )

        # リスク指標
        statistics["sharpe_ratio"] = self._safe_float_conversion(
            stats.get("Sharpe Ratio", 0.0)
        )
        statistics["sortino_ratio"] = self._safe_float_conversion(
            stats.get("Sortino Ratio", 0.0)
        )
        statistics["calmar_ratio"] = self._safe_float_conversion(
            stats.get("Calmar Ratio", 0.0)
        )

        # 資産関連
        statistics["final_equity"] = self._safe_float_conversion(
            stats.get("Equity Final [$]", 0.0)
        )
        statistics["equity_peak"] = self._safe_float_conversion(
            stats.get("Equity Peak [$]", 0.0)
        )
        statistics["buy_hold_return"] = self._safe_float_conversion(
            stats.get("Buy & Hold Return [%]", 0.0)
        )

        # Profit Factor
        statistics["profit_factor"] = self._safe_float_conversion(
            stats.get("Profit Factor", 0.0)
        )

        # 平均利益・平均損失（初期化）
        statistics["avg_win"] = 0.0
        statistics["avg_loss"] = 0.0

        return statistics

    def _enrich_metrics_from_trades(
        self, statistics: Dict[str, Any], stats: Any
    ) -> Dict[str, Any]:
        """取引データから詳細指標を再計算・補完"""
        try:
            # 取引データ取得
            trades_df = getattr(stats, "_trades", None)

            if trades_df is not None and len(trades_df) > 0:
                inferred_trades = len(trades_df)

                # total_tradesを常に更新（Seriesの値が0の場合が多いため）
                if inferred_trades > 0:
                    statistics["total_trades"] = int(inferred_trades)

                    # PnL列を探す
                    pnl_col = None
                    for col in ["PnL", "Pnl", "Profit", "ProfitLoss"]:
                        if col in trades_df.columns:
                            pnl_col = col
                            break

                    if pnl_col is not None:
                        # 詳細指標を計算
                        self._calculate_trade_metrics(
                            statistics, trades_df, pnl_col, inferred_trades
                        )
                    else:
                        logger.warning(
                            "取引データにPnL列が見つからないため、詳細指標の再計算をスキップします"
                        )
            else:
                # 取引がない場合の処理
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
            logger.error("エラー詳細", exc_info=True)
            return statistics

    def _calculate_trade_metrics(
        self,
        statistics: Dict[str, Any],
        trades_df: Any,
        pnl_col: str,
        total_trades: int,
    ) -> None:
        """個別の取引データから指標を計算してstatisticsを更新"""
        wins = 0
        winning_pnl = 0.0
        losing_pnl = 0.0
        win_count = 0
        loss_count = 0

        for _, trade in trades_df.iterrows():
            pnl_value = self._safe_float_conversion(trade.get(pnl_col, 0))
            if pnl_value > 0:
                wins += 1
                winning_pnl += pnl_value
                win_count += 1
            elif pnl_value < 0:
                losing_pnl += pnl_value
                loss_count += 1

        # 詳細指標を計算
        calculated_win_rate = (
            float(wins) / float(total_trades) * 100.0 if total_trades > 0 else 0.0
        )

        # profit_factor計算
        calculated_profit_factor = 0.0
        if losing_pnl < 0:
            calculated_profit_factor = winning_pnl / abs(losing_pnl)
        elif winning_pnl > 0:
            calculated_profit_factor = 999.99
        else:
            calculated_profit_factor = 0.0

        # avg_win, avg_loss計算
        calculated_avg_win = winning_pnl / win_count if win_count > 0 else 0.0
        calculated_avg_loss = abs(losing_pnl) / loss_count if loss_count > 0 else 0.0

        # 詳細指標を更新
        statistics["win_rate"] = calculated_win_rate
        statistics["profit_factor"] = calculated_profit_factor
        statistics["avg_win"] = calculated_avg_win
        statistics["avg_loss"] = calculated_avg_loss

    def _enrich_metrics_from_equity(
        self, statistics: Dict[str, Any], stats: Any
    ) -> Dict[str, Any]:
        """エクイティカーブから指標を補完"""
        try:
            # total_returnが0またはNoneの場合、エクイティカーブから計算
            current_total_return = statistics.get("total_return", 0)

            # エクイティカーブ取得
            equity_df = getattr(stats, "_equity_curve", None)

            if equity_df is not None and len(equity_df) > 0:
                # 最終エクイティの取得
                last_equity = self._get_equity_value(equity_df, -1)

                # リターンの計算
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

                # final_equityがまだ0の場合、エクイティカーブから設定
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
            # 取引関連指標を明示的に0に設定
            statistics["win_rate"] = 0.0
            statistics["profit_factor"] = 0.0
            statistics["avg_win"] = 0.0
            statistics["avg_loss"] = 0.0
            statistics["best_trade"] = 0.0
            statistics["worst_trade"] = 0.0
            statistics["avg_trade"] = 0.0

        return statistics

    def _convert_trade_history(self, stats: Any) -> List[Dict[str, Any]]:
        """取引履歴を変換"""
        try:
            # statsがpropertyの場合、実際の値を取得
            actual_stats = self._resolve_stats_object(stats)

            trades_df = getattr(actual_stats, "_trades", None)

            if trades_df is None or (hasattr(trades_df, "empty") and trades_df.empty):
                # ログレベルを下げてノイズを減らす
                logger.debug("バックテストで取引が発生しませんでした")
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
            # statsがpropertyの場合、実際の値を取得
            actual_stats = self._resolve_stats_object(stats)

            equity_df = getattr(actual_stats, "_equity_curve", None)
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
