"""
個体評価器

遺伝的アルゴリズムの個体評価を担当します。
"""

import logging
from typing import Dict, Any

from ..models.ga_config import GAConfig
from app.core.services.backtest_service import BacktestService

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
            # 遺伝子デコード
            from ..models.gene_encoding import GeneEncoder
            from ..models.gene_strategy import StrategyGene

            gene_encoder = GeneEncoder()
            gene = gene_encoder.decode_list_to_strategy_gene(individual, StrategyGene)

            # バックテスト実行用の設定を構築
            backtest_config = (
                self._fixed_backtest_config.copy()
                if self._fixed_backtest_config
                else {}
            )

            # 戦略設定を追加（test_strategy_generationと同じ形式）
            backtest_config["strategy_config"] = {
                "strategy_type": "GENERATED_GA",
                "parameters": {"strategy_gene": gene.to_dict()},
            }

            # strategy_nameフィールドを追加
            backtest_config["strategy_name"] = f"GA_Individual_{gene.id[:8]}"

            # デバッグログ: 取引量設定を確認
            # risk_management = gene.risk_management
            # position_size = risk_management.get("position_size", 0.1)
            # logger.debug(
            #     f"GA個体評価 - position_size: {position_size}, gene_id: {gene.id}"
            # )

            result = self.backtest_service.run_backtest(backtest_config)

            # フィットネス計算
            fitness = self._calculate_fitness(result, config)

            return (fitness,)

        except Exception as e:
            logger.error(f"個体評価エラー: {e}")
            return (0.0,)

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
            # performance_metricsから基本メトリクスを取得
            performance_metrics = backtest_result.get("performance_metrics", {})

            total_return = performance_metrics.get("total_return", 0.0)
            sharpe_ratio = performance_metrics.get("sharpe_ratio", 0.0)
            max_drawdown = performance_metrics.get("max_drawdown", 1.0)
            win_rate = performance_metrics.get("win_rate", 0.0)
            total_trades = performance_metrics.get("total_trades", 0)

            # 取引回数が0の場合は低いフィットネス値を返す
            if total_trades == 0:
                logger.warning("取引回数が0のため、低いフィットネス値を設定")
                return 0.1  # 完全に0ではなく、わずかな値を返す

            # 制約チェック
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
