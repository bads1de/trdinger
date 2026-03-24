"""
フィットネス計算モジュール

バックテスト結果からパフォーマンスメトリクスを抽出し、
適応度（Fitness）値を計算します。
"""

import logging
import math
from typing import Any, Dict, Tuple

from ..config.ga import GAConfig
from .evaluation_metrics import calculate_trade_frequency_penalty, calculate_ulcer_index

logger = logging.getLogger(__name__)


class FitnessCalculator:
    """
    フィットネス計算を担当するクラス

    IndividualEvaluator から委譲を受け、パフォーマンスメトリクスの抽出、
    単一目的・多目的フィットネスの計算、ロング・ショートバランス評価を行います。
    """

    def __init__(self) -> None:
        pass

    def extract_performance_metrics(
        self, backtest_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        バックテスト結果からパフォーマンスメトリクスを抽出

        Args:
            backtest_result: バックテスト結果

        Returns:
            抽出されたパフォーマンスメトリクス
        """
        performance_metrics = backtest_result.get("performance_metrics", {})

        metrics = {
            "total_return": performance_metrics.get("total_return", 0.0),
            "sharpe_ratio": performance_metrics.get("sharpe_ratio", 0.0),
            "max_drawdown": performance_metrics.get("max_drawdown", 1.0),
            "win_rate": performance_metrics.get("win_rate", 0.0),
            "profit_factor": performance_metrics.get("profit_factor", 0.0),
            "sortino_ratio": performance_metrics.get("sortino_ratio", 0.0),
            "calmar_ratio": performance_metrics.get("calmar_ratio", 0.0),
            "total_trades": performance_metrics.get("total_trades", 0),
        }

        for key, value in list(metrics.items()):

            def is_invalid_value(val):
                return (
                    val is None
                    or (isinstance(val, float) and not math.isfinite(val))
                    or not isinstance(val, (int, float))
                )

            if is_invalid_value(value):
                if key == "max_drawdown":
                    metrics[key] = 1.0
                elif key == "total_trades":
                    metrics[key] = 0
                else:
                    metrics[key] = 0.0
            elif (
                key == "max_drawdown" and isinstance(value, (int, float)) and value < 0
            ):
                metrics[key] = 0.0

        equity_curve = backtest_result.get("equity_curve", [])
        metrics["ulcer_index"] = calculate_ulcer_index(equity_curve)

        trade_history = backtest_result.get("trade_history", [])
        metrics["trade_frequency_penalty"] = calculate_trade_frequency_penalty(
            total_trades=metrics["total_trades"],
            start_date=backtest_result.get("start_date"),
            end_date=backtest_result.get("end_date"),
            trade_history=trade_history,
        )

        return metrics

    def calculate_fitness(
        self, backtest_result: Dict[str, Any], config: GAConfig, **kwargs
    ) -> float:
        """
        フィットネス計算（ロング・ショートバランス評価を含む）

        Args:
            backtest_result: バックテスト結果
            config: GA設定

        Returns:
            フィットネス値
        """
        try:
            metrics = self.extract_performance_metrics(backtest_result)

            total_return = metrics["total_return"]
            sharpe_ratio = metrics["sharpe_ratio"]
            max_drawdown = metrics["max_drawdown"]
            win_rate = metrics["win_rate"]
            total_trades = metrics["total_trades"]
            ulcer_index = metrics.get("ulcer_index", 0.0)
            trade_penalty = metrics.get("trade_frequency_penalty", 0.0)

            if total_trades == 0:
                logger.warning("取引回数が0のため、低いフィットネス値を設定")
                return config.zero_trades_penalty

            min_trades_req = int(config.fitness_constraints.get("min_trades", 0))
            if total_trades < min_trades_req:
                return config.constraint_violation_penalty

            max_dd_limit = config.fitness_constraints.get("max_drawdown_limit", None)
            if isinstance(max_dd_limit, (float, int)) and max_drawdown > float(
                max_dd_limit
            ):
                return config.constraint_violation_penalty

            if total_return < 0 or sharpe_ratio < config.fitness_constraints.get(
                "min_sharpe_ratio", 0
            ):
                return config.constraint_violation_penalty

            balance_score = self.calculate_long_short_balance(backtest_result)

            fitness_weights = config.fitness_weights.copy()

            fitness = (
                fitness_weights.get("total_return", 0.3) * total_return
                + fitness_weights.get("sharpe_ratio", 0.4) * sharpe_ratio
                + fitness_weights.get("max_drawdown", 0.2) * (1 - max_drawdown)
                + fitness_weights.get("win_rate", 0.1) * win_rate
                + fitness_weights.get("balance_score", 0.1) * balance_score
            )

            ulcer_scale = 1.0
            trade_scale = 1.0
            if getattr(config, "dynamic_objective_reweighting", False):
                dynamic_scalars = getattr(config, "objective_dynamic_scalars", {})
                ulcer_scale = dynamic_scalars.get("ulcer_index", 1.0)
                trade_scale = dynamic_scalars.get("trade_frequency_penalty", 1.0)

            fitness -= (
                fitness_weights.get("ulcer_index_penalty", 0.0)
                * ulcer_scale
                * ulcer_index
            )
            fitness -= (
                fitness_weights.get("trade_frequency_penalty", 0.0)
                * trade_scale
                * trade_penalty
            )

            return max(0.0, fitness)

        except Exception as e:
            logger.error(f"フィットネス計算エラー: {e}")
            return config.constraint_violation_penalty

    def calculate_long_short_balance(
        self, backtest_result: Dict[str, Any]
    ) -> float:
        """
        ロング・ショートバランススコアを計算

        Args:
            backtest_result: バックテスト結果

        Returns:
            バランススコア（0.0-1.0）
        """
        try:
            trade_history = backtest_result.get("trade_history", [])
            if not trade_history:
                return 0.5

            long_trades = []
            short_trades = []
            long_pnl = 0.0
            short_pnl = 0.0

            for trade in trade_history:
                size = trade.get("size", 0.0)
                pnl = trade.get("pnl", 0.0)

                if size > 0:
                    long_trades.append(trade)
                    long_pnl += pnl
                elif size < 0:
                    short_trades.append(trade)
                    short_pnl += pnl

            total_trades = len(long_trades) + len(short_trades)
            if total_trades == 0:
                return 0.5

            long_ratio = len(long_trades) / total_trades
            short_ratio = len(short_trades) / total_trades
            trade_balance = 1.0 - abs(long_ratio - short_ratio)

            total_pnl = long_pnl + short_pnl
            profit_balance = 0.5

            if total_pnl > 0:
                if long_pnl > 0 and short_pnl > 0:
                    profit_balance = 1.0
                elif long_pnl > 0 or short_pnl > 0:
                    profit_balance = 0.7
            elif long_pnl < 0 and short_pnl < 0:
                profit_balance = 0.1
            else:
                profit_balance = 0.3

            balance_score = 0.6 * trade_balance + 0.4 * profit_balance

            return max(0.0, min(1.0, balance_score))

        except Exception as e:
            logger.error(f"ロング・ショートバランス計算エラー: {e}")
            return 0.5

    def calculate_multi_objective_fitness(
        self, backtest_result: Dict[str, Any], config: GAConfig, **kwargs
    ) -> Tuple[float, ...]:
        """
        多目的適応度の計算

        バックテスト結果からパフォーマンスメトリクスを抽出し、
        GA設定で定義された各目的関数に対応する値を算出してタプルとして返します。

        Args:
            backtest_result: バックテスト実行結果
            config: GA設定
            **kwargs: 追加の評価コンテキスト

        Returns:
            各目的関数の評価値を含むタプル
        """
        try:
            metrics = self.extract_performance_metrics(backtest_result)
            total_trades = metrics["total_trades"]

            min_trades_req = int(config.fitness_constraints.get("min_trades", 0))
            if total_trades < min_trades_req:
                penalty_values = []
                for obj in config.objectives:
                    if obj in [
                        "max_drawdown",
                        "ulcer_index",
                        "trade_frequency_penalty",
                    ]:
                        penalty_values.append(1.0)
                    else:
                        penalty_values.append(config.constraint_violation_penalty)
                return tuple(penalty_values)

            fitness_values = []

            for objective in config.objectives:
                if objective == "weighted_score":
                    value = self.calculate_fitness(backtest_result, config, **kwargs)
                elif objective == "total_return":
                    value = metrics["total_return"]
                elif objective == "sharpe_ratio":
                    value = metrics["sharpe_ratio"]
                elif objective == "max_drawdown":
                    value = metrics["max_drawdown"]
                elif objective == "win_rate":
                    value = metrics["win_rate"]
                elif objective == "profit_factor":
                    value = metrics["profit_factor"]
                elif objective == "sortino_ratio":
                    value = metrics["sortino_ratio"]
                elif objective == "calmar_ratio":
                    value = metrics["calmar_ratio"]
                elif objective == "balance_score":
                    value = self.calculate_long_short_balance(backtest_result)
                elif objective == "ulcer_index":
                    value = metrics.get("ulcer_index", 0.0)
                elif objective == "trade_frequency_penalty":
                    value = metrics.get("trade_frequency_penalty", 0.0)
                else:
                    logger.warning(f"未知の目的: {objective}")
                    value = 0.0

                dynamic_scalars = getattr(config, "objective_dynamic_scalars", {})
                scale = dynamic_scalars.get(objective, 1.0)
                fitness_values.append(float(value) * scale)

            return tuple(fitness_values)

        except Exception as e:
            logger.error(f"多目的フィットネス計算エラー: {e}")
            return tuple(0.0 for _ in config.objectives)
