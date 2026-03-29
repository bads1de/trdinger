"""
実験永続化サービス
GA実験に関連するデータのデータベースへの保存、更新、取得を管理します。
"""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.services.auto_strategy.serializers.serialization import GeneSerializer
from app.services.auto_strategy.core.evaluation.report_persistence import (
    attach_evaluation_summary,
)
from database.repositories.backtest_result_repository import BacktestResultRepository
from database.repositories.ga_experiment_repository import GAExperimentRepository
from database.repositories.generated_strategy_repository import (
    GeneratedStrategyRepository,
)

from ..config import GAConfig

logger = logging.getLogger(__name__)


class ExperimentPersistenceService:
    """
    実験永続化サービス
    """

    def __init__(self, db_session_factory):
        """
        初期化

        Args:
            db_session_factory: DBセッションファクトリ
        """
        self.db_session_factory = db_session_factory
        self.serializer = GeneSerializer()

    def create_experiment(
        self,
        experiment_id: str,
        experiment_name: str,
        ga_config: GAConfig,
        backtest_config: Dict[str, Any],
    ) -> str:
        """
        実験を作成
        """
        with self.db_session_factory() as db:
            ga_experiment_repo = GAExperimentRepository(db)
            config_data = {
                "ga_config": ga_config.to_dict(),
                "backtest_config": backtest_config,
                "experiment_id": experiment_id,
            }
            ga_experiment_repo.create_experiment(
                experiment_id=experiment_id,
                name=experiment_name,
                config=config_data,
                total_generations=ga_config.generations,
                status="running",
            )
            logger.info(
                f"実験を作成しました: {experiment_name} (UUID: {experiment_id})"
            )
            return experiment_id

    def save_experiment_result(
        self,
        experiment_id: str,
        result: Dict[str, Any],
        ga_config: GAConfig,
        backtest_config: Dict[str, Any],
        experiment_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """実験結果をデータベースに保存"""
        if not isinstance(experiment_info, dict):
            experiment_info = self.get_experiment_info(experiment_id)

        if not experiment_info:
            logger.error(f"実験情報が見つかりません: {experiment_id}")
            return

        logger.info(f"実験結果保存開始: {experiment_id}")

        # backtest_config は互換性のために受け取るが、この保存処理では使わない。

        with self.db_session_factory() as db:
            self._save_best_strategy(
                db, experiment_id, experiment_info, result, ga_config
            )
            self._save_other_strategies(db, experiment_info, result, ga_config)

            if ga_config.enable_multi_objective and "pareto_front" in result:
                self._save_pareto_front(db, experiment_info, result, ga_config)

        logger.info(f"実験結果保存完了: {experiment_id}")

    def save_backtest_result(self, result_data: Dict[str, Any]) -> None:
        """詳細バックテスト結果をデータベースに保存"""
        if not result_data:
            logger.warning("保存対象のバックテスト結果がありません")
            return

        with self.db_session_factory() as db:
            backtest_result_repo = BacktestResultRepository(db)
            backtest_result_repo.save_backtest_result(result_data)
            logger.info("最良戦略のバックテスト結果を保存しました。")

    def _save_best_strategy(
        self,
        db: Session,
        experiment_id: str,
        experiment_info: Dict[str, Any],
        result: Dict[str, Any],
        ga_config: GAConfig,
    ):
        """最良戦略を保存する"""
        generated_strategy_repo = GeneratedStrategyRepository(db)

        db_experiment_id = experiment_info["db_id"]
        best_strategy = result["best_strategy"]
        best_fitness = result["best_fitness"]
        evaluation_summary = result.get("best_evaluation_summary")

        # フィットネス値の整理
        if ga_config.enable_multi_objective and isinstance(best_fitness, (list, tuple)):
            fitness_values = list(best_fitness)
            fitness_score = best_fitness[0] if best_fitness else 0.0
        else:
            fitness_values = None
            fitness_score = (
                best_fitness if isinstance(best_fitness, (int, float)) else 0.0
            )

        gene_data = self.serializer.strategy_gene_to_dict(best_strategy)
        gene_data = attach_evaluation_summary(gene_data, evaluation_summary)
        best_strategy_record = generated_strategy_repo.save_strategy(
            experiment_id=db_experiment_id,
            gene_data=gene_data,
            generation=ga_config.generations,
            fitness_score=fitness_score,
            fitness_values=fitness_values,
        )

        logger.info(
            f"最良戦略を保存しました (ID: {best_strategy_record.id})"
        )

    def _save_other_strategies(
        self,
        db: Session,
        experiment_info: Dict[str, Any],
        result: Dict[str, Any],
        ga_config: GAConfig,
    ):
        """最良戦略以外の戦略をバッチ保存する"""
        all_strategies = result.get("all_strategies", [])
        if not all_strategies or len(all_strategies) <= 1:
            return

        best_strategy = result["best_strategy"]
        db_experiment_id = experiment_info["db_id"]
        generated_strategy_repo = GeneratedStrategyRepository(db)
        evaluation_summaries = result.get("evaluation_summaries", {})

        strategies_data = []
        fitness_scores = result.get("fitness_scores", [])
        for i, strategy in enumerate(all_strategies[:100]):  # 上位100件に制限
            if strategy.id != best_strategy.id:
                # fitness_scoresの範囲内にあるかチェック
                fitness_score = fitness_scores[i] if i < len(fitness_scores) else 0.0
                strategy_key = self._get_strategy_result_key(strategy)
                gene_data = self.serializer.strategy_gene_to_dict(strategy)
                gene_data = attach_evaluation_summary(
                    gene_data,
                    evaluation_summaries.get(strategy_key),
                )
                strategies_data.append(
                    {
                        "experiment_id": db_experiment_id,
                        "gene_data": gene_data,
                        "generation": ga_config.generations,
                        "fitness_score": fitness_score,
                    }
                )

        if strategies_data:
            saved_count = generated_strategy_repo.save_strategies_batch(strategies_data)
            logger.info(f"追加戦略を一括保存しました: {saved_count} 件")

    def _save_pareto_front(
        self,
        db: Session,
        experiment_info: Dict[str, Any],
        result: Dict[str, Any],
        ga_config: GAConfig,
    ):
        """パレート最適解を保存する"""
        pareto_front = result.get("pareto_front", [])
        if not pareto_front:
            return

        db_experiment_id = experiment_info["db_id"]
        generated_strategy_repo = GeneratedStrategyRepository(db)
        evaluation_summaries = result.get("evaluation_summaries", {})

        strategies_data = []
        for solution in pareto_front:
            strategy = solution.get("strategy")
            fitness_values = solution.get("fitness_values")

            if strategy and fitness_values:
                strategy_key = self._get_strategy_result_key(strategy)
                gene_data = self.serializer.strategy_gene_to_dict(strategy)
                gene_data = attach_evaluation_summary(
                    gene_data,
                    evaluation_summaries.get(strategy_key),
                )
                strategies_data.append(
                    {
                        "experiment_id": db_experiment_id,
                        "gene_data": gene_data,
                        "generation": ga_config.generations,
                        "fitness_score": (fitness_values[0] if fitness_values else 0.0),
                        "fitness_values": fitness_values,
                    }
                )

        if strategies_data:
            saved_count = generated_strategy_repo.save_strategies_batch(strategies_data)
            logger.info(f"パレート最適解を一括保存しました: {saved_count} 件")

    def complete_experiment(self, experiment_id: str):
        """実験を完了状態にする"""
        experiment_info = self.get_experiment_info(experiment_id)
        if experiment_info:
            with self.db_session_factory() as db:
                repo = GAExperimentRepository(db)
                repo.update_experiment_status(experiment_info["db_id"], "completed")

    def fail_experiment(self, experiment_id: str):
        """実験を失敗状態にする"""
        experiment_info = self.get_experiment_info(experiment_id)
        if experiment_info:
            with self.db_session_factory() as db:
                repo = GAExperimentRepository(db)
                repo.update_experiment_status(experiment_info["db_id"], "failed")

    def stop_experiment(self, experiment_id: str):
        """実験を停止状態にする"""
        experiment_info = self.get_experiment_info(experiment_id)
        if experiment_info:
            with self.db_session_factory() as db:
                repo = GAExperimentRepository(db)
                repo.update_experiment_status(experiment_info["db_id"], "stopped")

    def list_experiments(self) -> List[Dict[str, Any]]:
        """実験一覧を取得"""
        with self.db_session_factory() as db:
            repo = GAExperimentRepository(db)
            experiments = repo.get_recent_experiments(limit=100)
            return [
                {
                    "id": exp.id,
                    "experiment_name": exp.name,
                    "status": exp.status,
                    "created_at": (
                        exp.created_at.isoformat() if exp.created_at else None
                    ),
                    "completed_at": (
                        exp.completed_at.isoformat() if exp.completed_at else None
                    ),
                }
                for exp in experiments
            ]

    def get_experiment_info(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """実験情報を取得（experiment_idカラムによる高速検索）。"""
        with self.db_session_factory() as db:
            repo = GAExperimentRepository(db)

            # 1. experiment_idカラムによる直接検索（最速）
            exp = repo.get_by_experiment_id(experiment_id)
            if exp:
                return self._map_experiment_info(exp)

            # 2. 後方互換: 名前またはDB IDで照合
            experiments = repo.get_recent_experiments(limit=100)
            for exp in experiments:
                if str(exp.name) == experiment_id or str(exp.id) == experiment_id:
                    return self._map_experiment_info(exp)

            logger.warning(f"実験が見つかりません: {experiment_id}")
            return None

    def _map_experiment_info(self, exp: Any) -> Dict[str, Any]:
        """DBモデルから辞書形式に変換"""
        return {
            "db_id": exp.id,
            "name": exp.name,
            "status": exp.status,
            "config": exp.config,
            "created_at": exp.created_at,
            "completed_at": exp.completed_at,
        }

    @staticmethod
    def _get_strategy_result_key(strategy: Any) -> str:
        """result 内部の summary 対応付けキーを返す。"""
        strategy_id = getattr(strategy, "id", None)
        if strategy_id not in (None, ""):
            return str(strategy_id)
        return str(id(strategy))
