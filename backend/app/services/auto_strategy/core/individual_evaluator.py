"""
個体評価器

遺伝的アルゴリズムの個体評価を担当します。
"""

import logging
from typing import Any, Dict

from ..config import GAConfig
from app.services.backtest.backtest_service import BacktestService

logger = logging.getLogger(__name__)


class IndividualEvaluator:
    """
    個体評価器

    遺伝的アルゴリズムの個体評価を担当します。
    """

    def __init__(self, backtest_service: BacktestService):
        """初期化"""
        self.backtest_service = backtest_service
        self._fixed_backtest_config = None

    def set_backtest_config(self, backtest_config: Dict[str, Any]):
        """バックテスト設定を設定"""
        self._fixed_backtest_config = self._select_timeframe_config(backtest_config)

    def evaluate_individual(self, individual, config: GAConfig):
        """
        個体評価

        Args:
            individual: 評価する個体
            config: GA設定

        Returns:
            フィットネス値のタプル
        """
        try:
            # 遺伝子デコード（リファクタリング改善）
            from ..serializers.gene_serialization import GeneSerializer
            from ..models.strategy_models import StrategyGene

            gene_serializer = GeneSerializer()
            gene = gene_serializer.from_list(individual, StrategyGene)

            # バックテスト実行用の設定を構築
            backtest_config = (
                self._fixed_backtest_config.copy()
                if self._fixed_backtest_config
                else {}
            )

            # 戦略設定を追加（test_strategy_generationと同じ形式）
            from app.services.auto_strategy.serializers.gene_serialization import (
                GeneSerializer,
            )

            serializer = GeneSerializer()
            backtest_config["strategy_config"] = {
                "strategy_type": "GENERATED_GA",
                "parameters": {"strategy_gene": serializer.strategy_gene_to_dict(gene)},
            }

            # strategy_nameフィールドを追加
            backtest_config["strategy_name"] = f"GA_Individual_{gene.id[:8]}"


            result = self.backtest_service.run_backtest(backtest_config)

            # フィットネス計算（単一目的・多目的対応）
            if config.enable_multi_objective:
                fitness_values = self._calculate_multi_objective_fitness(result, config)
                return fitness_values
            else:
                fitness = self._calculate_fitness(result, config)
                return (fitness,)

        except Exception as e:
            logger.error(f"個体評価エラー: {e}")
            if config.enable_multi_objective:
                # 多目的最適化の場合、目的数に応じたデフォルト値を返す
                return tuple(0.0 for _ in config.objectives)
            else:
                return (0.0,)

    def _extract_performance_metrics(self, backtest_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        バックテスト結果からパフォーマンスメトリクスを抽出

        Args:
            backtest_result: バックテスト結果

        Returns:
            抽出されたパフォーマンスメトリクス
        """
        performance_metrics = backtest_result.get("performance_metrics", {})

        # 主要メトリクスを安全に抽出（デフォルト値を設定）
        metrics = {
            "total_return": performance_metrics.get("total_return", 0.0),
            "sharpe_ratio": performance_metrics.get("sharpe_ratio", 0.0),
            "max_drawdown": performance_metrics.get("max_drawdown", 1.0),
            "win_rate": performance_metrics.get("win_rate", 0.0),
            "profit_factor": performance_metrics.get("profit_factor", 0.0),
            "sortino_ratio": performance_metrics.get("sortino_ratio", 0.0),
            "calmar_ratio": performance_metrics.get("calmar_ratio", 0.0),
            "total_trades": performance_metrics.get("total_trades", 0)
        }

        # 無効な値を処理（None, inf, nanなど）
        import math
        for key, value in metrics.items():
            def is_invalid_value(val):
                return (val is None or
                       (isinstance(val, float) and not math.isfinite(val)) or
                       not isinstance(val, (int, float)))

            if is_invalid_value(value):
                if key == "max_drawdown":
                    metrics[key] = 1.0  # 最大ドローダウンは1.0（100%）が上限
                elif key == "total_trades":
                    metrics[key] = 0
                else:
                    metrics[key] = 0.0
            elif key == "max_drawdown" and isinstance(value, (int, float)) and value < 0:
                metrics[key] = 0.0  # 負のドローダウンは0に修正

        return metrics

    def _calculate_fitness(
        self, backtest_result: Dict[str, Any], config: GAConfig
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
            # パフォーマンスメトリクスを抽出
            metrics = self._extract_performance_metrics(backtest_result)

            total_return = metrics["total_return"]
            sharpe_ratio = metrics["sharpe_ratio"]
            max_drawdown = metrics["max_drawdown"]
            win_rate = metrics["win_rate"]
            total_trades = metrics["total_trades"]

            # 取引回数が0の場合は低いフィットネス値を返す
            if total_trades == 0:
                logger.warning("取引回数が0のため、低いフィットネス値を設定")
                return 0.1  # 完全に0ではなく、わずかな値を返す

            # 追加の制約チェック
            min_trades_req = int(config.fitness_constraints.get("min_trades", 0))
            if total_trades < min_trades_req:
                return 0.0

            max_dd_limit = config.fitness_constraints.get("max_drawdown_limit", None)
            if isinstance(max_dd_limit, (float, int)) and max_drawdown > float(
                max_dd_limit
            ):
                return 0.0

            if total_return < 0 or sharpe_ratio < config.fitness_constraints.get(
                "min_sharpe_ratio", 0
            ):
                return 0.0

            # ロング・ショートバランス評価を計算
            balance_score = self._calculate_long_short_balance(backtest_result)

            # 重み付きフィットネス計算（バランススコアを追加）
            fitness_weights = config.fitness_weights
            fitness = (
                fitness_weights.get("total_return", 0.3) * total_return
                + fitness_weights.get("sharpe_ratio", 0.4) * sharpe_ratio
                + fitness_weights.get("max_drawdown", 0.2) * (1 - max_drawdown)
                + fitness_weights.get("win_rate", 0.1) * win_rate
                + fitness_weights.get("balance_score", 0.1) * balance_score
            )

            return max(0.0, fitness)

        except Exception as e:
            logger.error(f"フィットネス計算エラー: {e}")
            return 0.0

    def _calculate_long_short_balance(self, backtest_result: Dict[str, Any]) -> float:
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
                return 0.5  # 取引がない場合は中立スコア

            long_trades = []
            short_trades = []
            long_pnl = 0.0
            short_pnl = 0.0

            # 取引をロング・ショートに分類
            for trade in trade_history:
                size = trade.get("size", 0.0)
                pnl = trade.get("pnl", 0.0)

                if size > 0:  # ロング取引
                    long_trades.append(trade)
                    long_pnl += pnl
                elif size < 0:  # ショート取引
                    short_trades.append(trade)
                    short_pnl += pnl

            total_trades = len(long_trades) + len(short_trades)
            if total_trades == 0:
                return 0.5

            # 取引回数バランス（理想は50:50）
            long_ratio = len(long_trades) / total_trades
            short_ratio = len(short_trades) / total_trades
            trade_balance = 1.0 - abs(long_ratio - short_ratio)

            # 利益バランス（両方向で利益を出せているか）
            total_pnl = long_pnl + short_pnl
            profit_balance = 0.5  # デフォルト

            if total_pnl > 0:
                # 両方向で利益が出ている場合は高スコア
                if long_pnl > 0 and short_pnl > 0:
                    profit_balance = 1.0
                # 片方向のみで利益の場合は中程度
                elif long_pnl > 0 or short_pnl > 0:
                    profit_balance = 0.7
            # 両方で損失が出ている場合は低いスコア
            elif long_pnl < 0 and short_pnl < 0:
                profit_balance = 0.1
            else:
                profit_balance = 0.3

            # 総合バランススコア（取引回数バランス60%、利益バランス40%）
            balance_score = 0.6 * trade_balance + 0.4 * profit_balance

            return max(0.0, min(1.0, balance_score))

        except Exception as e:
            logger.error(f"ロング・ショートバランス計算エラー: {e}")
            return 0.5  # エラー時は中立スコア

    def _select_timeframe_config(
        self, backtest_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        タイムフレーム設定の選択

        Args:
            backtest_config: バックテスト設定

        Returns:
            選択されたタイムフレーム設定
        """
        if not backtest_config:
            return {}

        # 簡単な実装: 設定をそのまま返す
        return backtest_config.copy()

    def _calculate_multi_objective_fitness(
        self, backtest_result: Dict[str, Any], config: GAConfig
    ) -> tuple:
        """
        多目的最適化用フィットネス計算

        Args:
            backtest_result: バックテスト結果
            config: GA設定

        Returns:
            各目的の評価値のタプル
        """
        try:
            # パフォーマンスメトリクスを抽出
            metrics = self._extract_performance_metrics(backtest_result)

            fitness_values = []

            for objective in config.objectives:
                if objective == "total_return":
                    value = metrics["total_return"]
                elif objective == "sharpe_ratio":
                    value = metrics["sharpe_ratio"]
                elif objective == "max_drawdown":
                    # ドローダウンは最小化したいので、符号を反転させる
                    # DEAP側で-1.0の重みが設定されているため、ここでは正の値のまま
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
                    value = self._calculate_long_short_balance(backtest_result)
                else:
                    logger.warning(f"未知の目的: {objective}")
                    value = 0.0

                fitness_values.append(float(value))

            # 取引回数が0の場合は低い評価値を設定
            total_trades = metrics["total_trades"]
            if total_trades == 0:
                logger.warning("取引回数が0のため、低い評価値を設定")
                fitness_values = [0.1 for _ in fitness_values]

            return tuple(fitness_values)

        except Exception as e:
            logger.error(f"多目的フィットネス計算エラー: {e}")
            # エラー時は目的数に応じたデフォルト値を返す
            return tuple(0.0 for _ in config.objectives)
