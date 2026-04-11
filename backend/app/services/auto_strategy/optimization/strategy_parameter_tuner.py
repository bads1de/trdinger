"""
戦略パラメータチューナー

Optuna を使用して戦略パラメータを最適化します。
"""

import logging
import uuid
from typing import Any, Dict, Optional

from app.services.ml.optimization.optuna_optimizer import OptunaOptimizer

from ..config.ga import GAConfig
from ..core.evaluation.individual_evaluator import IndividualEvaluator
from ..core.engine.fitness_utils import extract_primary_fitness_from_result
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

    @classmethod
    def from_ga_config(
        cls,
        evaluator: IndividualEvaluator,
        config: GAConfig,
    ) -> "StrategyParameterTuner":
        """GAConfig のチューニング関連フィールドからチューナーを構築する。"""
        return cls(
            evaluator=evaluator,
            config=config,
            n_trials=config.tuning_config.n_trials,
            use_wfa=config.tuning_config.use_wfa,
            include_indicators=config.tuning_config.include_indicators,
            include_tpsl=config.tuning_config.include_tpsl,
            include_thresholds=config.tuning_config.include_thresholds,
        )

    def tune(self, gene: StrategyGene) -> StrategyGene:
        """
        単一の戦略遺伝子に対して、Optuna を用いたパラメータの微調整（ローカルサーチ）を実行します。

        このメソッドは、遺伝子の「論理構造（どの指標をどう組み合わせるか）」は変えずに、
        「数値パラメータ（期間、閾値、TP/SL幅等）」の最適な組み合わせを探索します。

        手順：
        1. `parameter_space_builder` を使用して、遺伝子から探索可能な変数を抽出。
        2. 目的関数（`objective`）を定義。各試行でパラメータを適用し、バックテストで評価。
        3. 指定された試行回数（`n_trials`）だけ探索を実行。
        4. 発見された最良のパラメータを適用した新しい遺伝子を生成し、メタデータを付与して返却。

        Args:
            gene (StrategyGene): チューニング対象の親個体。

        Returns:
            StrategyGene: パラメータが最適化された新しい戦略遺伝子。
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
            self._refresh_tuned_gene_identity(tuned, gene)
            return self._evaluate_gene(tuned)

        try:
            res = self.optimizer.optimize(objective, parameter_space, self.n_trials)

            # 最適パラメータの適用とメタデータ更新
            best_gene = self.parameter_space_builder.apply_params_to_gene(
                gene, res.best_params
            )
            self._refresh_tuned_gene_identity(best_gene, gene)
            best_gene.metadata.update(
                {
                    "optuna_tuned": True,
                    "optuna_best_score": res.best_score,
                    "optuna_trials": res.total_evaluations,
                    "optuna_time": res.optimization_time,
                    "optuna_source_gene_id": getattr(gene, "id", ""),
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
            if self.use_wfa and self.config.evaluation_config.enable_walk_forward:
                # WFA設定を一時的に有効化したconfigを使用
                wfa_config = self._create_wfa_config()
                fitness_tuple = self.evaluator.evaluate_individual(gene, wfa_config)
            else:
                fitness_tuple = self.evaluator.evaluate_individual(gene, self.config)

            return extract_primary_fitness_from_result(fitness_tuple)

        except Exception as e:
            logger.warning(f"遺伝子評価中にエラー: {e}")
            return 0.0

    def _create_wfa_config(self) -> GAConfig:
        """WFA用の設定を作成"""
        # 元のconfigをコピーしてWFAを有効化
        import copy

        wfa_config = copy.deepcopy(self.config)
        wfa_config.evaluation_config.enable_walk_forward = True

        # WFAフォールド数を減らして高速化（チューニング用）
        if wfa_config.evaluation_config.wfa_n_folds > 3:
            wfa_config.evaluation_config.wfa_n_folds = 3

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

    @staticmethod
    def _refresh_tuned_gene_identity(
        tuned_gene: StrategyGene, source_gene: StrategyGene
    ) -> None:
        """チューニング後遺伝子が元個体と cache key を共有しないよう ID を更新する。"""
        source_id = str(getattr(source_gene, "id", "") or "")
        tuned_id = str(getattr(tuned_gene, "id", "") or "")
        if tuned_id and tuned_id != source_id:
            return
        tuned_gene.id = str(uuid.uuid4())
