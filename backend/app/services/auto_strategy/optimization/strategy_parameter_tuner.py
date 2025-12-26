"""
戦略パラメータチューナー

Optuna を使用して戦略パラメータを最適化します。
"""

import logging
from typing import Any, Dict, Optional

from app.services.ml.optimization.optuna_optimizer import OptunaOptimizer

from ..config.ga import GAConfig
from ..core.individual_evaluator import IndividualEvaluator
from ..genes.strategy import StrategyGene
from .strategy_parameter_space import StrategyParameterSpace

logger = logging.getLogger(__name__)


class StrategyParameterTuner:
    """
    Optuna による戦略パラメータチューニング

    GAで発見された戦略構造に対して、Optunaを使用して
    パラメータ（インジケータ期間、TPSL設定など）を最適化します。
    """

    def __init__(
        self,
        evaluator: IndividualEvaluator,
        config: GAConfig,
        n_trials: int = 30,
        use_wfa: bool = True,
        include_indicators: bool = True,
        include_tpsl: bool = True,
        include_thresholds: bool = False,
    ):
        """
        初期化

        Args:
            evaluator: 評価器（バックテスト実行）
            config: GA設定
            n_trials: Optuna試行回数
            use_wfa: WFA評価を使用するか（過学習防止）
            include_indicators: インジケーターパラメータを最適化するか
            include_tpsl: TPSLパラメータを最適化するか
            include_thresholds: 条件閾値を最適化するか
        """
        self.evaluator = evaluator
        self.config = config
        self.n_trials = n_trials
        self.use_wfa = use_wfa
        self.include_indicators = include_indicators
        self.include_tpsl = include_tpsl
        self.include_thresholds = include_thresholds

        self.parameter_space_builder = StrategyParameterSpace()
        self.optimizer = OptunaOptimizer()

    def tune(self, gene: StrategyGene) -> StrategyGene:
        """
        単一の戦略遺伝子に対してパラメータチューニングを実行

        Optunaを使用して、指標の期間、TP/SL、あるいは取引条件の閾値などの
        連続変数/離散変数を最適化し、より高いフィットネスを持つ遺伝子を返します。

        Args:
            gene: チューニング対象の戦略遺伝子

        Returns:
            最適化されたパラメータを適用した新しいStrategyGene
        """
        logger.info("[Tuning] 戦略パラメータチューニングを開始")

        parameter_space = self.parameter_space_builder.build_parameter_space(
            gene, self.include_indicators, self.include_tpsl, self.include_thresholds
        )

        if not parameter_space:
            logger.warning("最適化可能なパラメータがありません")
            return gene

        # 目的関数
        def objective(params: Dict[str, Any]) -> float:
            tuned = self.parameter_space_builder.apply_params_to_gene(gene, params)
            return self._evaluate_gene(tuned)

        try:
            res = self.optimizer.optimize(objective, parameter_space, self.n_trials)

            # 最適パラメータの適用とメタデータ更新
            best_gene = self.parameter_space_builder.apply_params_to_gene(
                gene, res.best_params
            )
            best_gene.metadata.update(
                {
                    "optuna_tuned": True,
                    "optuna_best_score": res.best_score,
                    "optuna_trials": res.total_evaluations,
                    "optuna_time": res.optimization_time,
                }
            )
            return best_gene

        except Exception as e:
            logger.error(f"チューニングエラー: {e}")
            return gene
        finally:
            self.optimizer.cleanup()

    def _evaluate_gene(self, gene: StrategyGene) -> float:
        """
        遺伝子のフィットネスを評価

        Args:
            gene: 評価対象の遺伝子

        Returns:
            フィットネス値（高いほど良い）
        """
        try:
            # WFA評価が有効な場合
            if self.use_wfa and self.config.enable_walk_forward:
                # WFA設定を一時的に有効化したconfigを使用
                wfa_config = self._create_wfa_config()
                fitness_tuple = self.evaluator.evaluate_individual(gene, wfa_config)
            else:
                fitness_tuple = self.evaluator.evaluate_individual(gene, self.config)

            # フィットネスタプルから主要スコアを抽出
            if isinstance(fitness_tuple, tuple) and len(fitness_tuple) > 0:
                return float(fitness_tuple[0])
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"遺伝子評価中にエラー: {e}")
            return 0.0

    def _create_wfa_config(self) -> GAConfig:
        """WFA用の設定を作成"""
        # 元のconfigをコピーしてWFAを有効化
        import copy

        wfa_config = copy.deepcopy(self.config)
        wfa_config.enable_walk_forward = True

        # WFAフォールド数を減らして高速化（チューニング用）
        if wfa_config.wfa_n_folds > 3:
            wfa_config.wfa_n_folds = 3

        return wfa_config

    def tune_multiple(
        self, genes: list[StrategyGene], top_n: Optional[int] = None
    ) -> list[StrategyGene]:
        """
        複数の遺伝子をチューニング

        Args:
            genes: チューニング対象の遺伝子リスト
            top_n: チューニングする上位N個（Noneの場合は全て）

        Returns:
            チューニングされた遺伝子のリスト
        """
        if top_n is not None:
            genes = genes[:top_n]

        tuned_genes = []
        for idx, gene in enumerate(genes):
            logger.info(f"遺伝子 {idx + 1}/{len(genes)} をチューニング中...")
            tuned_gene = self.tune(gene)
            tuned_genes.append(tuned_gene)

        return tuned_genes
