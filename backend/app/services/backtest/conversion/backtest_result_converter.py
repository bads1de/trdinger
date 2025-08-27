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

    def _extract_statistics(self, stats: Any) -> Dict[str, Any]:
        """統計情報を抽出"""
        # statsが呼び出し可能な場合、実際の値を取得
        actual_stats = stats
        try:
            # backtesting.pyのResultオブジェクト（pandas.Series）から統計情報を抽出
            statistics = {}
            if hasattr(stats, "__call__"):
                try:
                    actual_stats = stats()
                except Exception as e:
                    logger.warning(f"statsの呼び出しに失敗: {e}")
                    # actual_stats はそのまま使用

            # backtesting.pyのResultオブジェクト（pandas.Series）の場合
            if hasattr(actual_stats, "index") and hasattr(actual_stats, "values"):
                logger.info(
                    "backtesting.pyのResultオブジェクト（pandas.Series）を処理中"
                )
                # pandas.Seriesから直接統計情報を取得
                series_total_return = self._safe_float_conversion(
                    actual_stats.get("Return [%]", 0.0)
                )
                series_total_trades = self._safe_int_conversion(
                    actual_stats.get("# Trades", 0)
                )
                series_win_rate = self._safe_float_conversion(
                    actual_stats.get("Win Rate [%]", 0.0)
                )
                series_profit_factor = self._safe_float_conversion(
                    actual_stats.get("Profit Factor", 0.0)
                )

                logger.info(
                    f"Seriesから取得した統計情報: total_return={series_total_return}, total_trades={series_total_trades}, win_rate={series_win_rate}, profit_factor={series_profit_factor}"
                )

                # Seriesからの統計情報を設定（取引関連は後で再計算される可能性がある）
                statistics["total_return"] = series_total_return
                statistics["total_trades"] = (
                    series_total_trades  # 後で取引データから更新される
                )
                statistics["win_rate"] = series_win_rate  # 後で取引データから更新される
                statistics["profit_factor"] = (
                    series_profit_factor  # 後で取引データから更新される
                )

                # 追加の指標
                statistics["best_trade"] = self._safe_float_conversion(
                    actual_stats.get("Best Trade [%]", 0.0)
                )
                statistics["worst_trade"] = self._safe_float_conversion(
                    actual_stats.get("Worst Trade [%]", 0.0)
                )
                statistics["avg_trade"] = self._safe_float_conversion(
                    actual_stats.get("Avg. Trade [%]", 0.0)
                )
                statistics["max_drawdown"] = self._safe_float_conversion(
                    actual_stats.get("Max. Drawdown [%]", 0.0)
                )
                statistics["avg_drawdown"] = self._safe_float_conversion(
                    actual_stats.get("Avg. Drawdown [%]", 0.0)
                )

                # 期間関連
                statistics["max_drawdown_duration"] = self._safe_int_conversion(
                    actual_stats.get("Max. Drawdown Duration", 0)
                )
                statistics["avg_drawdown_duration"] = self._safe_float_conversion(
                    actual_stats.get("Avg. Drawdown Duration", 0)
                )

                # リスク指標
                statistics["sharpe_ratio"] = self._safe_float_conversion(
                    actual_stats.get("Sharpe Ratio", 0.0)
                )
                statistics["sortino_ratio"] = self._safe_float_conversion(
                    actual_stats.get("Sortino Ratio", 0.0)
                )
                statistics["calmar_ratio"] = self._safe_float_conversion(
                    actual_stats.get("Calmar Ratio", 0.0)
                )

                # 資産関連
                statistics["final_equity"] = self._safe_float_conversion(
                    actual_stats.get("Equity Final [$]", 0.0)
                )
                statistics["equity_peak"] = self._safe_float_conversion(
                    actual_stats.get("Equity Peak [$]", 0.0)
                )
                statistics["buy_hold_return"] = self._safe_float_conversion(
                    actual_stats.get("Buy & Hold Return [%]", 0.0)
                )

                # 平均利益・平均損失（初期化）
                statistics["avg_win"] = 0.0
                statistics["avg_loss"] = 0.0

            # _Statsオブジェクトの場合、直接resultオブジェクトを使用
            elif hasattr(actual_stats, "keys") and hasattr(actual_stats, "get"):
                logger.info("_Statsオブジェクトを直接使用中")
                # 基本的なパフォーマンス指標
                statistics["total_return"] = self._safe_float_conversion(
                    actual_stats.get("Return [%]", 0.0)
                )
                statistics["total_trades"] = int(
                    actual_stats.get("# Trades", 0)
                )  # 後で取引データから更新される
                statistics["win_rate"] = self._safe_float_conversion(
                    actual_stats.get(
                        "Win Rate [%]", 0.0
                    )  # 後で取引データから更新される
                )

                # 追加の指標
                statistics["best_trade"] = self._safe_float_conversion(
                    actual_stats.get("Best Trade [%]", 0.0)
                )
                statistics["worst_trade"] = self._safe_float_conversion(
                    actual_stats.get("Worst Trade [%]", 0.0)
                )
                statistics["avg_trade"] = self._safe_float_conversion(
                    actual_stats.get("Avg. Trade [%]", 0.0)
                )
                statistics["max_drawdown"] = self._safe_float_conversion(
                    actual_stats.get("Max. Drawdown [%]", 0.0)
                )
                statistics["avg_drawdown"] = self._safe_float_conversion(
                    actual_stats.get("Avg. Drawdown [%]", 0.0)
                )

                # 期間関連
                statistics["max_drawdown_duration"] = self._safe_int_conversion(
                    actual_stats.get("Max. Drawdown Duration", 0)
                )
                statistics["avg_drawdown_duration"] = self._safe_float_conversion(
                    actual_stats.get("Avg. Drawdown Duration", 0)
                )

                # リスク指標
                statistics["sharpe_ratio"] = self._safe_float_conversion(
                    actual_stats.get("Sharpe Ratio", 0.0)
                )
                statistics["sortino_ratio"] = self._safe_float_conversion(
                    actual_stats.get("Sortino Ratio", 0.0)
                )
                statistics["calmar_ratio"] = self._safe_float_conversion(
                    actual_stats.get("Calmar Ratio", 0.0)
                )

                # 資産関連
                statistics["final_equity"] = self._safe_float_conversion(
                    actual_stats.get("Equity Final [$]", 0.0)
                )
                statistics["equity_peak"] = self._safe_float_conversion(
                    actual_stats.get("Equity Peak [$]", 0.0)
                )
                statistics["buy_hold_return"] = self._safe_float_conversion(
                    actual_stats.get("Buy & Hold Return [%]", 0.0)
                )

                # Profit Factor
                statistics["profit_factor"] = self._safe_float_conversion(
                    actual_stats.get(
                        "Profit Factor", 0.0
                    )  # 後で取引データから更新される
                )

                # 平均利益・平均損失（初期化 - 後で取引データから再計算される）
                statistics["avg_win"] = 0.0
                statistics["avg_loss"] = 0.0

            logger.debug(
                f"抽出された統計情報: total_return={statistics.get('total_return', 'N/A')}, total_trades={statistics.get('total_trades', 'N/A')}, win_rate={statistics.get('win_rate', 'N/A')}"
            )
            logger.debug(f"利用可能な統計情報キー: {list(statistics.keys())}")
            logger.debug(f"実際のstatsの型: {type(actual_stats)}")

            # デバッグログ: statsオブジェクトの詳細調査
            logger.info("=== statsオブジェクト詳細調査 ===")
            logger.info(f"statsの型: {type(actual_stats)}")
            logger.info(
                f"statsの属性: {[attr for attr in dir(actual_stats) if not attr.startswith('__')]}"
            )

            # _tradesの詳細確認
            if hasattr(actual_stats, "_trades"):
                trades_attr = getattr(actual_stats, "_trades")
                logger.info(f"_trades属性の型: {type(trades_attr)}")
                logger.info(f"_trades属性の値: {trades_attr}")
            else:
                logger.info("_trades属性が存在しません")

            # その他の重要な属性の確認
            for attr_name in ["_equity_curve", "_strategy", "_results"]:
                if hasattr(actual_stats, attr_name):
                    attr_value = getattr(actual_stats, attr_name)
                    logger.info(f"{attr_name}属性の型: {type(attr_value)}")
                    if hasattr(attr_value, "__len__"):
                        logger.info(f"{attr_name}属性の長さ: {len(attr_value)}")
                else:
                    logger.info(f"{attr_name}属性が存在しません")

            logger.info("=== statsオブジェクト調査終了 ===")

            # If key metrics are zero or missing, attempt to recompute from trades or equity curve
            try:
                logger.debug("統計情報のフォールバック計算を確認中...")

                # Seriesからの統計情報が有効かどうかチェック
                series_total_trades = statistics.get("total_trades", 0)
                series_win_rate = statistics.get("win_rate", 0)
                series_profit_factor = statistics.get("profit_factor", 0)

                # より柔軟な判定: 取引数が0でも他の指標があれば有効とみなす
                # ただし、取引関連の詳細指標（avg_win, avg_loss等）は再計算が必要
                series_has_basic_data = (
                    statistics.get("total_return", 0) != 0
                    or statistics.get("sharpe_ratio", 0) != 0
                    or statistics.get("max_drawdown", 0) != 0
                )

                logger.debug(
                    f"Series統計情報チェック: total_trades={series_total_trades}, win_rate={series_win_rate}, profit_factor={series_profit_factor}, 基本データ有効={series_has_basic_data}"
                )

                # 取引関連の詳細指標は常に再計算を試行
                # 基本的なパフォーマンス指標（リターン、シャープレシオ等）はSeriesから取得
                # 取引データから詳細指標を再計算
                trades_df = getattr(actual_stats, "_trades", None)

                # デバッグログ: 取引データの詳細確認
                logger.info(f"統計情報抽出: 取引データフレーム確認")
                logger.info(f"  _trades属性存在: {hasattr(actual_stats, '_trades')}")
                logger.info(f"  trades_df is not None: {trades_df is not None}")
                if trades_df is not None:
                    logger.info(f"  trades_dfの型: {type(trades_df)}")
                    logger.info(
                        f"  trades_dfの長さ: {len(trades_df) if hasattr(trades_df, '__len__') else 'N/A'}"
                    )
                    if hasattr(trades_df, "columns"):
                        logger.info(f"  trades_dfの列: {list(trades_df.columns)}")
                    if hasattr(trades_df, "empty"):
                        logger.info(f"  trades_dfが空: {trades_df.empty}")

                if trades_df is not None and len(trades_df) > 0:
                    inferred_trades = len(trades_df)
                    logger.info(
                        f"{inferred_trades}件の取引を含む取引データフレームが見つかりました"
                    )

                    # total_tradesを常に更新（Seriesの値が0の場合が多いため）
                    if inferred_trades > 0:
                        statistics["total_trades"] = int(inferred_trades)
                        logger.info(
                            f"total_tradesを{statistics['total_trades']}に更新しました"
                        )

                        # PnL列を探す
                        pnl_col = None
                        for col in ["PnL", "Pnl", "Profit", "ProfitLoss"]:
                            if col in trades_df.columns:
                                pnl_col = col
                                break

                        if pnl_col is not None:
                            logger.debug(
                                f"PnL列'{pnl_col}'が見つかりました。詳細指標を再計算します"
                            )

                            # 取引データから詳細指標を計算
                            wins = 0
                            total_pnl = 0.0
                            winning_pnl = 0.0
                            losing_pnl = 0.0
                            win_count = 0
                            loss_count = 0

                            for _, trade in trades_df.iterrows():
                                pnl_value = self._safe_float_conversion(
                                    trade.get(pnl_col, 0)
                                )
                                total_pnl += pnl_value
                                if pnl_value > 0:
                                    wins += 1
                                    winning_pnl += pnl_value
                                    win_count += 1
                                elif pnl_value < 0:
                                    losing_pnl += pnl_value
                                    loss_count += 1

                            # 詳細指標を計算
                            calculated_win_rate = (
                                float(wins) / float(inferred_trades) * 100.0
                                if inferred_trades > 0
                                else 0.0
                            )

                            # profit_factor計算（負の損失を正の値として扱う）
                            calculated_profit_factor = 0.0
                            if losing_pnl < 0:  # 損失がある場合
                                calculated_profit_factor = winning_pnl / abs(losing_pnl)
                            elif winning_pnl > 0:  # 利益のみの場合
                                calculated_profit_factor = (
                                    999.99  # 無限大の代わりに大きな値を使用
                                )
                            else:  # 利益も損失もない場合
                                calculated_profit_factor = 0.0

                            # avg_win, avg_loss計算
                            calculated_avg_win = (
                                winning_pnl / win_count if win_count > 0 else 0.0
                            )
                            calculated_avg_loss = (
                                abs(losing_pnl) / loss_count if loss_count > 0 else 0.0
                            )

                            # 計算結果をログ出力
                            pf_display = (
                                f"{calculated_profit_factor:.4f}"
                                if calculated_profit_factor < 999
                                else "999.99+"
                            )
                            logger.info(
                                f"取引データからの計算結果: wins={wins}, total={inferred_trades}, win_rate={calculated_win_rate:.2f}%, profit_factor={pf_display}"
                            )

                            # 詳細指標を更新（Seriesの値が0の場合が多いため常に更新）
                            statistics["win_rate"] = calculated_win_rate
                            statistics["profit_factor"] = calculated_profit_factor
                            statistics["avg_win"] = calculated_avg_win
                            statistics["avg_loss"] = calculated_avg_loss

                            logger.info(f"取引データから計算した詳細指標で更新しました")
                        else:
                            logger.warning(
                                "PnL列が見つからないため、詳細指標の再計算をスキップします"
                            )
                else:
                    logger.info(
                        "取引データフレームが見つからないか空のため、詳細指標の計算をスキップします"
                    )
                    # 取引がない場合でも基本的な指標は設定
                    if statistics.get("total_trades", 0) == 0:
                        logger.info("取引数が0のため、取引関連指標を0に設定します")
                        statistics["total_trades"] = 0
                        statistics["win_rate"] = 0.0
                        statistics["profit_factor"] = 0.0
                        statistics["avg_win"] = 0.0
                        statistics["avg_loss"] = 0.0

                # Recompute total_return from equity curve if available
                current_total_return = statistics.get("total_return", 0)
                logger.debug(f"現在のtotal_return: {current_total_return}")
                if current_total_return == 0 or current_total_return is None:
                    equity_df = getattr(actual_stats, "_equity_curve", None)
                    logger.info(
                        f"エクイティカーブデータフレーム: {equity_df is not None}"
                    )
                    if equity_df is not None:
                        logger.info(f"エクイティカーブの長さ: {len(equity_df)}")
                        logger.info(
                            f"エクイティカーブの列: {list(equity_df.columns) if hasattr(equity_df, 'columns') else '列なし'}"
                        )

                        if len(equity_df) > 0:
                            first_equity = None
                            last_equity = None

                            if (
                                hasattr(equity_df, "columns")
                                and "Equity" in equity_df.columns
                            ):
                                first_equity = equity_df.iloc[0]["Equity"]
                                last_equity = equity_df.iloc[-1]["Equity"]
                                logger.info(
                                    f"Equity列を使用: 最初={first_equity}, 最後={last_equity}"
                                )
                            else:
                                try:
                                    if hasattr(equity_df, "iloc"):
                                        first_equity = equity_df.iloc[0]
                                        last_equity = equity_df.iloc[-1]
                                        logger.info(
                                            f"ilocを使用: 最初={first_equity}, 最後={last_equity}"
                                        )
                                except Exception as e:
                                    logger.error(
                                        f"エクイティ値にアクセスできません: {e}"
                                    )

                            if (
                                first_equity is not None
                                and last_equity is not None
                                and first_equity > 0
                            ):
                                computed_return = float(
                                    (last_equity - first_equity)
                                    / float(first_equity)
                                    * 100.0
                                )
                                statistics["total_return"] = computed_return
                                statistics["final_equity"] = float(last_equity)
                                logger.info(
                                    f"計算されたtotal_return: {computed_return}, final_equity: {last_equity}"
                                )

                # If final_equity is still 0, try to get it from equity curve
                current_final_equity = statistics.get("final_equity", 0)
                logger.info(f"現在のfinal_equity: {current_final_equity}")
                if current_final_equity == 0:
                    equity_df = getattr(actual_stats, "_equity_curve", None)
                    if equity_df is not None and len(equity_df) > 0:
                        last_equity = None
                        if (
                            hasattr(equity_df, "columns")
                            and "Equity" in equity_df.columns
                        ):
                            last_equity = equity_df.iloc[-1]["Equity"]
                        else:
                            try:
                                if hasattr(equity_df, "iloc"):
                                    last_equity = equity_df.iloc[-1]
                            except Exception as e:
                                logger.error(
                                    f"最後のエクイティにアクセスできません: {e}"
                                )

                        if last_equity is not None:
                            statistics["final_equity"] = float(last_equity)
                            logger.info(
                                f"エクイティカーブからfinal_equityを更新: {last_equity}"
                            )

                # 最終的な整合性チェック
                final_total_trades = statistics.get("total_trades", 0)
                final_total_return = statistics.get("total_return", 0)

                if final_total_trades == 0 and final_total_return != 0:
                    logger.warning(
                        f"取引数が0なのにリターンが{final_total_return}%です。これは戦略が取引条件を満たしたが、backtesting.pyで取引が拒否された可能性があります。"
                    )
                    # 取引関連指標を明示的に0に設定
                    statistics["win_rate"] = 0.0
                    statistics["profit_factor"] = 0.0
                    statistics["avg_win"] = 0.0
                    statistics["avg_loss"] = 0.0
                    statistics["best_trade"] = 0.0
                    statistics["worst_trade"] = 0.0
                    statistics["avg_trade"] = 0.0

                logger.info(
                    f"フォールバック後の最終統計情報: total_return={statistics.get('total_return')}, total_trades={statistics.get('total_trades')}, final_equity={statistics.get('final_equity')}"
                )

            except Exception as e:
                logger.error(f"再計算フォールバック失敗: {e}")
                logger.error("フォールバックエラーの詳細", exc_info=True)

            return statistics

        except Exception as e:
            logger.error(f"統計情報の抽出中にエラー: {e}")
            logger.error(f"Statsの型: {type(stats)}")
            logger.error(
                f"実際のstatsの型: {type(actual_stats) if 'actual_stats' in locals() else '利用不可'}"
            )
            return {}

    def _convert_trade_history(self, stats: Any) -> List[Dict[str, Any]]:
        """取引履歴を変換"""
        try:
            # statsがpropertyの場合、実際の値を取得
            if hasattr(stats, "__call__"):
                try:
                    actual_stats = stats()
                except Exception:
                    actual_stats = stats
            else:
                actual_stats = stats

            trades_df = getattr(actual_stats, "_trades", None)

            # デバッグログ: 取引データフレームの詳細情報
            logger.info(f"取引データフレーム確認: {trades_df is not None}")
            if trades_df is not None:
                logger.info(f"取引データフレームの型: {type(trades_df)}")
                logger.info(
                    f"取引データフレームの長さ: {len(trades_df) if hasattr(trades_df, '__len__') else 'N/A'}"
                )
                if hasattr(trades_df, "columns"):
                    logger.info(f"取引データフレームの列: {list(trades_df.columns)}")
                if hasattr(trades_df, "empty"):
                    logger.info(f"取引データフレームが空: {trades_df.empty}")

            if trades_df is None or (hasattr(trades_df, "empty") and trades_df.empty):
                logger.warning("取引データフレームが空またはNoneです")
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

            logger.info(f"変換された取引履歴数: {len(trades)}")
            return trades

        except Exception as e:
            logger.warning(f"取引履歴の変換中にエラー: {e}")
            return []

    def _convert_equity_curve(self, stats: Any) -> List[Dict[str, Any]]:
        """エクイティカーブを変換"""
        try:
            # statsがpropertyの場合、実際の値を取得
            if hasattr(stats, "__call__"):
                try:
                    actual_stats = stats()
                except Exception:
                    actual_stats = stats
            else:
                actual_stats = stats

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
