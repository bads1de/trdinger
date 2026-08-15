"""
GA実験リポジトリ

遺伝的アルゴリズム実験の永続化処理を管理します。
"""

import logging
from datetime import datetime, timezone
from typing import Any, cast

from sqlalchemy.orm import Session

from database.models import GAExperiment

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class GAExperimentRepository(BaseRepository):
    """GA実験のリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, GAExperiment)

    def get_by_experiment_id(self, experiment_id: str) -> GAExperiment | None:
        """
        フロントエンド由来のexperiment_idで実験を取得

        Args:
            experiment_id: フロントエンドで生成されたUUID

        Returns:
            GA実験、または見つからない場合はNone
        """
        return (
            self.db.query(GAExperiment)
            .filter(GAExperiment.experiment_id == experiment_id)
            .first()
        )

    def create_experiment(
        self,
        name: str,
        config: dict[str, Any],
        total_generations: int,
        status: str = "running",
        experiment_id: str | None = None,
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
        from app.utils.error_handler import safe_operation

        @safe_operation(context="GA実験作成", is_api_call=False)
        def _create_experiment() -> GAExperiment:
            experiment = GAExperiment(
                experiment_id=experiment_id,
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

        return _create_experiment()

    def update_experiment_status(
        self,
        experiment_id: int,
        status: str,
        completed_at: datetime | None = None,
        error_message: str | None = None,
    ) -> bool:
        """
        実験のステータスを更新

        Args:
            experiment_id: 実験ID
            status: 新しいステータス
            completed_at: 完了時刻
            error_message: 失敗理由（status=failed 時に保存）

        Returns:
            更新成功フラグ
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="実験ステータス更新", is_api_call=False, default_return=False
        )
        def _update_experiment_status() -> bool:
            # BaseRepositoryの汎用メソッドを使用して実験を取得
            experiments = self.get_filtered_data(
                filters={"id": experiment_id},
                limit=1,
            )

            if not experiments:
                logger.warning(f"指定されたIDの実験が見つかりません: {experiment_id}")
                return False

            experiment = experiments[0]
            experiment.status = status

            if error_message is not None:
                experiment.error_message = error_message

            if completed_at:
                experiment.completed_at = cast(datetime, completed_at)  # type: ignore
            elif status in ["completed", "error", "cancelled"]:
                experiment.completed_at = cast(datetime, datetime.now(timezone.utc))  # type: ignore

            self.db.commit()
            logger.info(f"実験ステータスを更新: {experiment_id} -> {status}")
            return True

        return _update_experiment_status()

    def get_experiments_by_status(
        self, status: str, limit: int | None = None
    ) -> list[GAExperiment]:
        """
        ステータス別で実験を取得

        Args:
            status: ステータス
            limit: 取得件数制限

        Returns:
            GA実験のリスト
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="ステータス別実験取得", is_api_call=False, default_return=[]
        )
        def _get_experiments_by_status() -> list[GAExperiment]:
            # BaseRepositoryの汎用メソッドを使用
            return self.get_filtered_data(
                filters={"status": status},
                order_by_column="created_at",
                order_asc=False,
                limit=limit,
            )

        return _get_experiments_by_status()

    def count_by_status(self, status: str) -> int:
        """
        ステータス別の実験数を取得

        Args:
            status: ステータス

        Returns:
            該当ステータスの実験数
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="ステータス別実験数取得", is_api_call=False, default_return=0
        )
        def _count_by_status() -> int:
            return (
                self.db.query(GAExperiment)
                .filter(GAExperiment.status == status)
                .count()
            )

        return _count_by_status()

    def get_recent_experiments(self, limit: int = 10) -> list[GAExperiment]:
        """
        最近の実験を取得

        Args:
            limit: 取得件数制限

        Returns:
            GA実験のリスト
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context="最近実験取得", is_api_call=False, default_return=[])
        def _get_recent_experiments() -> list[GAExperiment]:
            # BaseRepositoryの汎用メソッドを使用
            return self.get_latest_records(
                timestamp_column="created_at",
                limit=limit,
            )

        return _get_recent_experiments()

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
        from app.utils.error_handler import safe_operation

        @safe_operation(context="実験完了処理", is_api_call=False, default_return=False)
        def _complete_experiment() -> bool:
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
            experiment.completed_at = cast(datetime, datetime.now(timezone.utc))  # type: ignore

            self.db.commit()
            logger.info(
                f"実験を完了しました: {experiment_id} (フィットネス: {best_fitness:.4f})"
            )
            return True

        return _complete_experiment()

    def update_progress(
        self,
        experiment_id: int,
        current_generation: int,
        total_generations: int,
        best_fitness: float | None = None,
        status: str | None = None,
    ) -> bool:
        """
        実験の進捗状況を更新する

        Args:
            experiment_id: 実験ID
            current_generation: 現在の世代
            total_generations: 総世代数
            best_fitness: 現在の最高フィットネス（オプション）
            status: ステータス（オプション）

        Returns:
            更新処理成功フラグ
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context="実験進捗更新", is_api_call=False, default_return=False)
        def _update_progress() -> bool:
            experiment = (
                self.db.query(GAExperiment)
                .filter(GAExperiment.id == experiment_id)
                .first()
            )

            if not experiment:
                logger.warning(f"指定されたIDの実験が見つかりません: {experiment_id}")
                return False

            experiment.current_generation = current_generation  # type: ignore
            experiment.progress = (current_generation + 1) / total_generations  # type: ignore

            if best_fitness is not None:
                experiment.best_fitness = best_fitness  # type: ignore
            if status is not None:
                experiment.status = status  # type: ignore

            self.db.commit()
            return True

        return _update_progress()

    def reconcile_stale_running_experiments(self) -> int:
        """
        サーバー再起動後に残った running 状態の実験を停止状態へ巻き戻す

        Returns:
            巻き戻した実験数
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="孤児実験リコンサイル", is_api_call=False, default_return=0
        )
        def _reconcile() -> int:
            stale = (
                self.db.query(GAExperiment)
                .filter(GAExperiment.status == "running")
                .all()
            )
            for experiment in stale:
                experiment.status = "stopped"
                experiment.error_message = (
                    "サーバー再起動により実行コンテキストが失われたため停止しました"
                )
                experiment.completed_at = cast(
                    datetime, datetime.now(timezone.utc)
                )  # type: ignore
            if stale:
                self.db.commit()
                logger.info(
                    f"孤児実験を停止状態へ巻き戻しました: {len(stale)} 件"
                )
            return len(stale)

        return _reconcile()

    def delete_all_experiments(self) -> int:
        """
        すべてのGA実験を削除

        Returns:
            削除された件数
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context="全GA実験削除", is_api_call=False)
        def _delete_all_experiments() -> int:
            deleted_count = self.db.query(GAExperiment).delete()
            self.db.commit()
            return deleted_count

        return _delete_all_experiments()

    def delete_experiment(self, experiment_db_id: int) -> tuple[int, int] | None:
        """
        実験と関連データ（戦略・BT結果）を削除

        FK制約のため、先に戦略（と戦略が参照するBT結果）を削除してから
        実験本体を削除します。

        Args:
            experiment_db_id: 実験のDB ID（integer PK）

        Returns:
            (削除した実験数, 削除した戦略数) のタプル。
            実験が見つからない場合は None。
        """
        from app.utils.error_handler import safe_operation
        from database.models import BacktestResult, GeneratedStrategy

        @safe_operation(context="GA実験削除", is_api_call=False)
        def _delete_experiment() -> tuple[int, int] | None:
            experiment = (
                self.db.query(GAExperiment)
                .filter(GAExperiment.id == experiment_db_id)
                .first()
            )
            if not experiment:
                return None

            if experiment.status == "running":
                raise ValueError("実行中の実験は削除できません。先に停止してください。")

            # 1. 実験に紐づく戦略を取得
            strategies = (
                self.db.query(GeneratedStrategy)
                .filter(GeneratedStrategy.experiment_id == experiment_db_id)
                .all()
            )

            # 2. 戦略が参照するBT結果IDを収集（他戦略から参照されていないもののみ）
            bt_result_ids = {
                s.backtest_result_id
                for s in strategies
                if s.backtest_result_id is not None
            }
            strategies_count = len(strategies)

            # 3. 戦略を削除
            if strategies:
                for strategy in strategies:
                    self.db.delete(strategy)
                self.db.flush()

            # 4. どの戦略からも参照されなくなったBT結果を削除
            if bt_result_ids:
                for bt_id in bt_result_ids:
                    referenced = (
                        self.db.query(GeneratedStrategy)
                        .filter(GeneratedStrategy.backtest_result_id == bt_id)
                        .first()
                        is not None
                    )
                    if not referenced:
                        self.db.query(BacktestResult).filter(
                            BacktestResult.id == bt_id
                        ).delete()

            # 5. 実験本体を削除
            self.db.delete(experiment)
            self.db.commit()

            logger.info(
                f"実験を削除しました: DB ID={experiment_db_id} "
                f"(戦略 {strategies_count} 件も削除)"
            )
            return (1, strategies_count)

        return _delete_experiment()
