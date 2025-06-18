"""
GA実験リポジトリ

遺伝的アルゴリズム実験の永続化処理を管理します。
"""

from typing import List, Optional, Dict, Any, cast
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc
import logging

from .base_repository import BaseRepository
from database.models import GAExperiment
from app.core.utils.database_utils import DatabaseQueryHelper

logger = logging.getLogger(__name__)


class GAExperimentRepository(BaseRepository):
    """GA実験のリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, GAExperiment)

    def create_experiment(
        self,
        name: str,
        config: Dict[str, Any],
        total_generations: int,
        status: str = "running",
    ) -> GAExperiment:
        """
        新しいGA実験を作成

        Args:
            name: 実験名
            config: GA設定（JSON形式）
            total_generations: 総世代数
            status: 初期ステータス

        Returns:
            作成されたGA実験
        """
        try:
            experiment = GAExperiment(
                name=name,
                config=config,
                status=status,
                total_generations=total_generations,
                current_generation=0,
                progress=0.0,
            )

            self.db.add(experiment)
            self.db.commit()
            self.db.refresh(experiment)

            logger.info(f"GA実験を作成しました: {experiment.id} ({name})")
            return experiment

        except Exception as e:
            self.db.rollback()
            logger.error(f"GA実験作成エラー: {e}")
            raise

    def update_experiment_progress(
        self,
        experiment_id: int,
        current_generation: int,
        progress: float,
        best_fitness: Optional[float] = None,
    ) -> bool:
        """
        実験の進捗を更新

        Args:
            experiment_id: 実験ID
            current_generation: 現在の世代数
            progress: 進捗率（0.0-1.0）
            best_fitness: 最高フィットネス

        Returns:
            更新成功フラグ
        """
        try:
            experiment = (
                self.db.query(GAExperiment)
                .filter(GAExperiment.id == experiment_id)
                .first()
            )

            if not experiment:
                logger.warning(f"実験が見つかりません: {experiment_id}")
                return False

            experiment.current_generation = current_generation  # type: ignore
            experiment.progress = progress  # type: ignore

            if best_fitness is not None:
                experiment.best_fitness = best_fitness  # type: ignore

            self.db.commit()
            logger.debug(
                f"実験進捗を更新: {experiment_id} (世代: {current_generation}, 進捗: {progress:.2%})"
            )
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"実験進捗更新エラー: {e}")
            return False

    def update_experiment_status(
        self, experiment_id: int, status: str, completed_at: Optional[datetime] = None
    ) -> bool:
        """
        実験のステータスを更新

        Args:
            experiment_id: 実験ID
            status: 新しいステータス
            completed_at: 完了時刻

        Returns:
            更新成功フラグ
        """
        try:
            experiment = (
                self.db.query(GAExperiment)
                .filter(GAExperiment.id == experiment_id)
                .first()
            )

            if not experiment:
                logger.warning(f"実験が見つかりません: {experiment_id}")
                return False

            experiment.status = status  # type: ignore

            if completed_at:
                experiment.completed_at = cast(datetime, completed_at)  # type: ignore
            elif status in ["completed", "error", "cancelled"]:
                experiment.completed_at = cast(datetime, datetime.utcnow())  # type: ignore

            self.db.commit()
            logger.info(f"実験ステータスを更新: {experiment_id} -> {status}")
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"実験ステータス更新エラー: {e}")
            return False

    def get_experiment_by_id(self, experiment_id: int) -> Optional[GAExperiment]:
        """
        IDで実験を取得

        Args:
            experiment_id: 実験ID

        Returns:
            GA実験（存在しない場合はNone）
        """
        try:
            return (
                self.db.query(GAExperiment)
                .filter(GAExperiment.id == experiment_id)
                .first()
            )

        except Exception as e:
            logger.error(f"実験取得エラー: {e}")
            return None

    def get_experiments_by_status(
        self, status: str, limit: Optional[int] = None
    ) -> List[GAExperiment]:
        """
        ステータス別で実験を取得

        Args:
            status: ステータス
            limit: 取得件数制限

        Returns:
            GA実験のリスト
        """
        try:
            filters = {"status": status}
            return DatabaseQueryHelper.get_filtered_records(
                db=self.db,
                model_class=GAExperiment,
                filters=filters,
                order_by_column="created_at",
                order_asc=False,
                limit=limit,
            )

        except Exception as e:
            logger.error(f"実験取得エラー: {e}")
            return []

    def get_recent_experiments(self, limit: int = 10) -> List[GAExperiment]:
        """
        最近の実験を取得

        Args:
            limit: 取得件数制限

        Returns:
            GA実験のリスト
        """
        try:
            return DatabaseQueryHelper.get_filtered_records(
                db=self.db,
                model_class=GAExperiment,
                order_by_column="created_at",
                order_asc=False,
                limit=limit,
            )

        except Exception as e:
            logger.error(f"最近の実験取得エラー: {e}")
            return []

    def complete_experiment(
        self, experiment_id: int, best_fitness: float, final_generation: int
    ) -> bool:
        """
        実験を完了状態にする

        Args:
            experiment_id: 実験ID
            best_fitness: 最終的な最高フィットネス
            final_generation: 最終世代数

        Returns:
            完了処理成功フラグ
        """
        try:
            experiment = (
                self.db.query(GAExperiment)
                .filter(GAExperiment.id == experiment_id)
                .first()
            )

            if not experiment:
                logger.warning(f"実験が見つかりません: {experiment_id}")
                return False

            experiment.status = "completed"  # type: ignore
            experiment.best_fitness = best_fitness  # type: ignore
            experiment.current_generation = final_generation  # type: ignore
            experiment.progress = 1.0  # type: ignore
            experiment.completed_at = cast(datetime, datetime.utcnow())  # type: ignore

            self.db.commit()
            logger.info(
                f"実験を完了しました: {experiment_id} (フィットネス: {best_fitness:.4f})"
            )
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"実験完了処理エラー: {e}")
            return False

    def get_experiment_statistics(self) -> Dict[str, Any]:
        """
        実験の統計情報を取得

        Returns:
            統計情報の辞書
        """
        try:
            total_experiments = self.db.query(GAExperiment).count()

            running_experiments = (
                self.db.query(GAExperiment)
                .filter(GAExperiment.status == "running")
                .count()
            )

            completed_experiments = (
                self.db.query(GAExperiment)
                .filter(GAExperiment.status == "completed")
                .count()
            )

            error_experiments = (
                self.db.query(GAExperiment)
                .filter(GAExperiment.status == "error")
                .count()
            )

            # 最高フィットネスの実験
            best_experiment = (
                self.db.query(GAExperiment)
                .filter(GAExperiment.best_fitness.isnot(None))
                .order_by(desc(GAExperiment.best_fitness))
                .first()
            )

            return {
                "total_experiments": total_experiments,
                "running_experiments": running_experiments,
                "completed_experiments": completed_experiments,
                "error_experiments": error_experiments,
                "best_fitness": (
                    best_experiment.best_fitness if best_experiment else None
                ),
                "best_experiment_id": best_experiment.id if best_experiment else None,
            }

        except Exception as e:
            logger.error(f"実験統計取得エラー: {e}")
            return {}
