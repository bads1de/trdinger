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
            logger.error(f"GA実験の作成中にエラーが発生しました: {e}")
            raise



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
            # BaseRepositoryの汎用メソッドを使用して実験を取得
            experiments = self.get_filtered_data(
                filters={"id": experiment_id},
                limit=1,
            )

            if not experiments:
                logger.warning(f"指定されたIDの実験が見つかりません: {experiment_id}")
                return False

            experiment = experiments[0]
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
            logger.error(f"実験ステータスの更新中にエラーが発生しました: {e}")
            return False



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
            # BaseRepositoryの汎用メソッドを使用
            return self.get_filtered_data(
                filters={"status": status},
                order_by_column="created_at",
                order_asc=False,
                limit=limit,
            )

        except Exception as e:
            logger.error(f"ステータスによる実験の取得中にエラーが発生しました: {e}")
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
            # BaseRepositoryの汎用メソッドを使用
            return self.get_latest_records(
                timestamp_column="created_at",
                limit=limit,
            )

        except Exception as e:
            logger.error(f"最近の実験の取得中にエラーが発生しました: {e}")
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
                logger.warning(f"指定されたIDの実験が見つかりません: {experiment_id}")
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
            logger.error(f"実験完了処理中にエラーが発生しました: {e}")
            return False



    def delete_all_experiments(self) -> int:
        """
        すべてのGA実験を削除

        Returns:
            削除された件数
        """
        try:
            deleted_count = self.db.query(GAExperiment).delete()
            self.db.commit()
            logger.info(f"すべてのGA実験を削除しました: {deleted_count} 件")
            return deleted_count
        except Exception as e:
            self.db.rollback()
            logger.error(f"すべてのGA実験の削除中にエラーが発生しました: {e}")
            raise
