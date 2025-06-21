"""
フィットネス計算器

GAにおける個体評価とフィットネス計算を担当するモジュール。
バックテスト結果から戦略の適応度を計算し、制約条件をチェックします。
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

from ..models.strategy_gene import StrategyGene
from ..models.gene_encoding import GeneEncoder
from ..models.ga_config import GAConfig
from ..factories.strategy_factory import StrategyFactory
from app.core.services.backtest_service import BacktestService


logger = logging.getLogger(__name__)


class FitnessCalculator:
    """
    フィットネス計算器

    個体評価、フィットネス計算、制約チェックを担当します。
    """

    def __init__(
        self, backtest_service: BacktestService, strategy_factory: StrategyFactory
    ):
        """
        初期化

        Args:
            backtest_service: バックテストサービス
            strategy_factory: 戦略ファクトリー
        """
        self.backtest_service = backtest_service
        self.strategy_factory = strategy_factory
        self.gene_encoder = GeneEncoder()

    def evaluate_individual(
        self,
        individual: List[float],
        config: GAConfig,
        backtest_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float]:
        """
        個体の評価（フィットネス計算）

        Args:
            individual: 評価する個体（数値リスト）
            config: GA設定
            backtest_config: バックテスト設定

        Returns:
            フィットネス値のタプル
        """
        try:
            # ログレベルに応じた出力制御
            if config.enable_detailed_logging:
                logger.info(f"個体評価開始: 遺伝子長={len(individual)}")
                logger.info(f"バックテスト設定: {backtest_config}")

            # 数値リストから戦略遺伝子にデコード
            gene = self.gene_encoder.decode_list_to_strategy_gene(
                individual, StrategyGene
            )

            # 戦略の妥当性チェック
            is_valid, errors = self.strategy_factory.validate_gene(gene)
            if not is_valid:
                logger.debug(f"無効な戦略: {errors}")
                return (0.0,)  # 無効な戦略には最低スコア

            # 戦略クラスを生成
            self.strategy_factory.create_strategy_class(gene)

            # バックテスト設定を構築
            test_config = self._build_backtest_config(gene, backtest_config)

            # バックテスト実行
            result = self.backtest_service.run_backtest(test_config)

            # フィットネス計算
            fitness = self.calculate_fitness(result, config)

            return (fitness,)

        except Exception as e:
            logger.error(f"個体評価エラー: {e}")
            return (0.0,)  # エラー時は最低スコア

    def _build_backtest_config(
        self, gene: StrategyGene, backtest_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        バックテスト設定を構築

        Args:
            gene: 戦略遺伝子
            backtest_config: ベースとなるバックテスト設定

        Returns:
            構築されたバックテスト設定
        """
        if backtest_config:
            test_config = {
                "strategy_name": f"GA_Generated_{gene.id}",
                "symbol": backtest_config.get("symbol", "BTC/USDT"),
                "timeframe": backtest_config.get("timeframe", "1d"),
                "start_date": backtest_config.get("start_date", "2024-01-01"),
                "end_date": backtest_config.get("end_date", "2024-04-09"),
                "initial_capital": backtest_config.get("initial_capital", 100000),
                "commission_rate": backtest_config.get("commission_rate", 0.001),
                "strategy_config": {
                    "strategy_type": "GENERATED_TEST",
                    "parameters": {"strategy_gene": gene.to_dict()},
                },
            }
        else:
            # フォールバック設定
            test_config = {
                "strategy_name": f"GA_Generated_{gene.id}",
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": "2024-01-01",
                "end_date": "2024-04-09",
                "initial_capital": 100000,
                "commission_rate": 0.001,
                "strategy_config": {
                    "strategy_type": "GENERATED_TEST",
                    "parameters": {"strategy_gene": gene.to_dict()},
                },
            }

        return test_config

    def calculate_fitness(
        self, backtest_result: Dict[str, Any], config: GAConfig
    ) -> float:
        """
        バックテスト結果からフィットネスを計算

        Args:
            backtest_result: バックテスト結果
            config: GA設定

        Returns:
            フィットネス値
        """
        try:
            metrics = backtest_result.get("performance_metrics", {})

            # 制約条件のチェック
            if not self._check_constraints(metrics, config):
                return 0.0

            # フィットネス計算
            fitness = self._calculate_weighted_fitness(metrics, config)

            # ボーナス適用
            fitness = self._apply_performance_bonus(fitness, metrics)

            return fitness

        except Exception as e:
            logger.error(f"フィットネス計算エラー: {e}")
            return 0.0

    def _check_constraints(self, metrics: Dict[str, Any], config: GAConfig) -> bool:
        """
        制約条件をチェック

        Args:
            metrics: パフォーマンス指標
            config: GA設定

        Returns:
            制約条件を満たす場合True
        """
        constraints = config.fitness_constraints

        # 最小取引数チェック（緩和: 1取引以上）
        if metrics.get("total_trades", 0) < constraints.get("min_trades", 1):
            return False

        # 最大ドローダウンチェック（緩和: 50%まで許可）
        max_drawdown = abs(metrics.get("max_drawdown", 1.0))
        if max_drawdown > constraints.get("max_drawdown_limit", 0.5):
            return False

        # 最小シャープレシオチェック（緩和: -1.0まで許可）
        sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
        if sharpe_ratio < constraints.get("min_sharpe_ratio", -1.0):
            return False

        return True

    def _calculate_weighted_fitness(
        self, metrics: Dict[str, Any], config: GAConfig
    ) -> float:
        """
        重み付きフィットネスを計算

        Args:
            metrics: パフォーマンス指標
            config: GA設定

        Returns:
            重み付きフィットネス値
        """
        weights = config.fitness_weights

        # 主要指標の取得
        total_return = metrics.get("total_return", 0.0)
        sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
        max_drawdown = abs(metrics.get("max_drawdown", 1.0))
        win_rate = metrics.get("win_rate", 0.0) / 100.0  # パーセンテージを小数に変換

        # 正規化（より実用的な範囲設定）: 各指標を0から1の範囲にスケーリング
        # 1. リターン正規化: -50%〜+200% の範囲を0〜1にマッピング
        #    (total_return + 50) で負の値をオフセットし、250 (200 - (-50)) で割る
        normalized_return = max(0, min(1, (total_return + 50) / 250))

        # 2. シャープレシオ正規化: -2〜+4 の範囲を0〜1にマッピング (優秀な戦略は2以上)
        #    (sharpe_ratio + 2) で負の値をオフセットし、6 (4 - (-2)) で割る
        normalized_sharpe = max(0, min(1, (sharpe_ratio + 2) / 6))

        # 3. ドローダウン正規化: 0〜50% の範囲を1〜0にマッピング (低いほど良い)
        #    1から (max_drawdown / 0.5) を引くことで、ドローダウンが低いほど値が高くなるようにする
        normalized_drawdown = max(0, min(1, 1 - (max_drawdown / 0.5)))

        # 4. 勝率正規化: 0〜100% の範囲を0〜1にマッピング
        normalized_win_rate = max(0, min(1, win_rate))

        # 重み付き合計
        fitness = (
            weights.get("total_return", 0.35) * normalized_return
            + weights.get("sharpe_ratio", 0.35) * normalized_sharpe
            + weights.get("max_drawdown", 0.25) * normalized_drawdown
            + weights.get("win_rate", 0.05) * normalized_win_rate
        )

        return fitness

    def _apply_performance_bonus(
        self, fitness: float, metrics: Dict[str, Any]
    ) -> float:
        """
        優秀な戦略にボーナスを適用

        Args:
            fitness: ベースフィットネス値
            metrics: パフォーマンス指標

        Returns:
            ボーナス適用後のフィットネス値
        """
        total_return = metrics.get("total_return", 0.0)
        sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
        max_drawdown = abs(metrics.get("max_drawdown", 1.0))

        # ボーナス: 優秀な戦略への追加評価
        # 特定のパフォーマンス基準を満たす戦略に対して、フィットネススコアにボーナスを適用し、
        # その戦略がGAによって優先的に選択されるようにします。
        if total_return > 20 and sharpe_ratio > 1.5 and max_drawdown < 0.15:
            fitness *= 1.2  # 20%ボーナス (リターン20%超、シャープレシオ1.5超、最大ドローダウン15%未満)
        elif total_return > 50 and sharpe_ratio > 2.0 and max_drawdown < 0.10:
            fitness *= 1.5  # 50%ボーナス（非常に優秀）(リターン50%超、シャープレシオ2.0超、最大ドローダウン10%未満)

        return fitness
