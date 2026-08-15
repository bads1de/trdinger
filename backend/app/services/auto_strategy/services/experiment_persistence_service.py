"""
実験永続化サービス
GA実験に関連するデータのデータベースへの保存、更新、取得を管理します。
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any, cast

from sqlalchemy.orm import Session

from app.services.auto_strategy.core.evaluation.report_persistence import (
    attach_evaluation_summary,
    attach_validation_summary,
)
from app.services.auto_strategy.genes.genetic_utils import GeneticUtils
from app.services.auto_strategy.serializers.serialization import GeneSerializer
from app.utils.datetime_utils import isoformat_or_none
from database.repositories.backtest_result_repository import (
    BacktestResultRepository,
)
from database.repositories.ga_experiment_repository import (
    GAExperimentRepository,
)
from database.repositories.generated_strategy_repository import (
    GeneratedStrategyRepository,
)

from ..config import GAConfig

logger = logging.getLogger(__name__)


class ExperimentPersistenceService:
    """
    実験永続化サービス
    """

    def __init__(self, db_session_factory: Any) -> None:
        """
        初期化

        Args:
            db_session_factory: DBセッションファクトリ
        """
        self.db_session_factory = db_session_factory
        self.serializer = GeneSerializer()
        # GA実行中にセッションを再利用するためのキャッシュ
        self._active_session: Session | None = None

    @contextmanager
    def _get_db_session(self) -> Generator[Session, None, None]:
        """
        DBセッションを取得するコンテキストマネージャー。

        GA実験実行中は既存のセッションを再利用し、
        実験完了後に新しいセッションを作成することで、
        多数のDB接続/切断を回避します。

        Yields:
            Session: SQLAlchemyセッション
        """
        if self._active_session is not None and not getattr(
            self._active_session, "closed", True
        ):
            # 既存のセッションを再利用
            yield self._active_session
        else:
            # 新しいセッションを作成
            with self.db_session_factory() as session:
                self._active_session = session
                try:
                    yield session
                except Exception:
                    self._active_session = None
                    raise

    def _close_active_session(self) -> None:
        """アクティブなセッションを閉じる"""
        if self._active_session is not None:
            try:
                if not getattr(self._active_session, "closed", True):
                    self._active_session.close()
            except Exception:
                pass
            finally:
                self._active_session = None

    def create_experiment(
        self,
        experiment_id: str,
        experiment_name: str,
        ga_config: GAConfig,
        backtest_config: dict[str, Any],
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
        result: dict[str, Any],
        ga_config: GAConfig,
        backtest_config: dict[str, Any],
        experiment_info: dict[str, Any] | None = None,
        backtest_result_id: int | None = None,
    ) -> None:
        """実験結果をデータベースに保存"""
        if not isinstance(experiment_info, dict):
            experiment_info = self.get_experiment_info(experiment_id)

        if not experiment_info:
            logger.error(f"実験情報が見つかりません: {experiment_id}")
            return

        logger.info(f"実験結果保存開始: {experiment_id}")

        # 自動検証結果（戦略キー → 検証辞書）
        validation_results = result.get("validation_results", {})

        with self.db_session_factory() as db:
            self._save_best_strategy(
                db,
                experiment_id,
                experiment_info,
                result,
                ga_config,
                validation_results=validation_results,
                backtest_result_id=backtest_result_id,
            )
            self._save_other_strategies(
                db,
                experiment_info,
                result,
                ga_config,
                validation_results=validation_results,
            )

            if "pareto_front" in result:
                self._save_pareto_front(
                    db,
                    experiment_info,
                    result,
                    ga_config,
                    validation_results=validation_results,
                )

        logger.info(f"実験結果保存完了: {experiment_id}")

    def save_backtest_result(self, result_data: dict[str, Any]) -> dict[str, Any] | None:
        """詳細バックテスト結果をデータベースに保存

        Returns:
            保存されたバックテスト結果の辞書（id を含む）。
            保存対象が無い場合は None。
        """
        if not result_data:
            logger.warning("保存対象のバックテスト結果がありません")
            return None

        with self.db_session_factory() as db:
            backtest_result_repo = BacktestResultRepository(db)
            saved = backtest_result_repo.save_backtest_result(result_data)
            logger.info("最良戦略のバックテスト結果を保存しました。")
            return saved

    def _save_best_strategy(
        self,
        db: Session,
        experiment_id: str,
        experiment_info: dict[str, Any],
        result: dict[str, Any],
        ga_config: GAConfig,
        *,
        validation_results: dict[str, Any] | None = None,
        backtest_result_id: int | None = None,
    ) -> None:
        """最良戦略を保存する"""
        generated_strategy_repo = GeneratedStrategyRepository(db)

        best_strategy = result.get("best_strategy")
        if best_strategy is None:
            logger.warning("最良戦略が無いため保存をスキップします")
            return

        db_experiment_id = experiment_info["db_id"]
        best_fitness = result["best_fitness"]
        evaluation_summary = result.get("best_evaluation_summary")

        # フィットネス値の整理
        if isinstance(best_fitness, (list, tuple)):
            fitness_values = list(best_fitness)
            fitness_score = best_fitness[0] if best_fitness else 0.0
        else:
            fitness_values = None
            fitness_score = (
                best_fitness if isinstance(best_fitness, (int, float)) else 0.0
            )

        gene_data = self.serializer.strategy_gene_to_dict(best_strategy)
        gene_data = attach_evaluation_summary(gene_data, evaluation_summary)
        if validation_results:
            best_key = self._get_strategy_result_key(best_strategy)
            gene_data = attach_validation_summary(
                gene_data, validation_results.get(best_key)
            )
        best_strategy_record = generated_strategy_repo.save_strategy(
            experiment_id=db_experiment_id,
            gene_data=gene_data,
            generation=ga_config.generations,
            fitness_score=fitness_score,
            fitness_values=fitness_values,
            backtest_result_id=backtest_result_id,
        )

        logger.info(f"最良戦略を保存しました (ID: {best_strategy_record.id})")

    def _save_other_strategies(
        self,
        db: Session,
        experiment_info: dict[str, Any],
        result: dict[str, Any],
        ga_config: GAConfig,
        *,
        validation_results: dict[str, Any] | None = None,
    ) -> None:
        """最良戦略以外の戦略をバッチ保存する"""
        all_strategies = result.get("all_strategies", [])
        if not all_strategies or len(all_strategies) <= 1:
            return

        best_strategy = result.get("best_strategy")
        if best_strategy is None:
            return

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
                if validation_results:
                    gene_data = attach_validation_summary(
                        gene_data, validation_results.get(strategy_key)
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
        experiment_info: dict[str, Any],
        result: dict[str, Any],
        ga_config: GAConfig,
        *,
        validation_results: dict[str, Any] | None = None,
    ) -> None:
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
                if validation_results:
                    gene_data = attach_validation_summary(
                        gene_data, validation_results.get(strategy_key)
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

    def complete_experiment(self, experiment_id: str) -> None:
        """実験を完了状態にする"""
        self._update_experiment_status(experiment_id, "completed")

    def fail_experiment(
        self, experiment_id: str, error_message: str | None = None
    ) -> None:
        """実験を失敗状態にする"""
        self._update_experiment_status(
            experiment_id, "failed", error_message=error_message
        )

    def stop_experiment(self, experiment_id: str) -> None:
        """実験を停止状態にする"""
        self._update_experiment_status(experiment_id, "stopped")

    def _update_experiment_status(
        self,
        experiment_id: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """実験ステータス更新の共通処理。"""
        experiment_info = self.get_experiment_info(experiment_id)
        if experiment_info:
            with self._get_db_session() as db:
                repo = GAExperimentRepository(db)
                repo.update_experiment_status(
                    experiment_info["db_id"], status, error_message=error_message
                )
            self._close_active_session()

    def update_experiment_progress(
        self,
        experiment_id: str,
        current_generation: int,
        total_generations: int,
        best_fitness: float | None = None,
    ) -> bool:
        """実験の進捗状況を更新する"""
        experiment_info = self.get_experiment_info(experiment_id)
        if not experiment_info:
            return False
        with self._get_db_session() as db:
            repo = GAExperimentRepository(db)
            return repo.update_progress(
                experiment_info["db_id"],
                current_generation,
                total_generations,
                best_fitness,
            )

    def list_experiments(self) -> list[dict[str, Any]]:
        """実験一覧を取得"""
        with self.db_session_factory() as db:
            repo = GAExperimentRepository(db)
            experiments = repo.get_recent_experiments(limit=100)
            return [
                {
                    "id": exp.id,
                    "experiment_name": exp.name,
                    "status": exp.status,
                    "progress": exp.progress,
                    "current_generation": exp.current_generation,
                    "total_generations": exp.total_generations,
                    "best_fitness": exp.best_fitness,
                    "error_message": exp.error_message,
                    "created_at": isoformat_or_none(
                        cast(datetime | None, exp.created_at)
                    ),
                    "completed_at": isoformat_or_none(
                        cast(datetime | None, exp.completed_at)
                    ),
                }
                for exp in experiments
            ]

    def count_running_experiments(self) -> int:
        """実行中の実験数を取得"""
        with self.db_session_factory() as db:
            repo = GAExperimentRepository(db)
            return repo.count_by_status("running")

    def reconcile_stale_running_experiments(self) -> int:
        """サーバー再起動後の孤児実験（running のまま）を停止状態へ巻き戻す"""
        with self.db_session_factory() as db:
            repo = GAExperimentRepository(db)
            return repo.reconcile_stale_running_experiments()

    def delete_experiment(self, experiment_id: str) -> tuple[int, int] | None:
        """
        実験を削除する

        Args:
            experiment_id: フロントエンド由来の実験UUID

        Returns:
            (削除した実験数, 削除した戦略数) のタプル。
            実験が見つからない場合は None。

        Raises:
            ValueError: 実行中の実験は削除不可
        """
        experiment_info = self.get_experiment_info(experiment_id)
        if experiment_info is None:
            return None
        with self.db_session_factory() as db:
            repo = GAExperimentRepository(db)
            return repo.delete_experiment(experiment_info["db_id"])

    def get_experiment_detail(self, experiment_id: str) -> dict[str, Any] | None:
        """実験詳細を取得（進捗情報を含む）"""
        with self.db_session_factory() as db:
            repo = GAExperimentRepository(db)
            exp = repo.get_by_experiment_id(experiment_id)
            if not exp:
                return None
            return {
                "id": exp.id,
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "status": exp.status,
                "progress": exp.progress,
                "current_generation": exp.current_generation,
                "total_generations": exp.total_generations,
                "best_fitness": exp.best_fitness,
                "error_message": exp.error_message,
                "created_at": isoformat_or_none(cast(datetime | None, exp.created_at)),
                "completed_at": isoformat_or_none(
                    cast(datetime | None, exp.completed_at)
                ),
            }

    def get_experiment_info(self, experiment_id: str) -> dict[str, Any] | None:
        """実験情報を取得（experiment_idカラムによる高速検索）。"""
        with self.db_session_factory() as db:
            repo = GAExperimentRepository(db)

            # 1. experiment_idカラムによる直接検索（最速）
            exp = repo.get_by_experiment_id(experiment_id)
            if exp:
                return self._map_experiment_info(exp)

            logger.warning(f"実験が見つかりません: {experiment_id}")
            return None

    def _map_experiment_info(self, exp: Any) -> dict[str, Any]:
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
    def _get_strategy_result_key(strategy: object) -> str:
        """result 内部の summary 対応付けキーを返す。"""
        return GeneticUtils.get_strategy_result_key(strategy)
