"""
進捗追跡器

GA実験の進捗追跡・コールバック管理を担当するモジュール。
"""

import logging
from typing import Dict, Optional, Callable

from ..models.ga_config import GAProgress

from database.repositories.ga_experiment_repository import GAExperimentRepository
from database.connection import SessionLocal


logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    進捗追跡器

    GA実験の進捗追跡・コールバック管理を担当します。
    """

    def __init__(self):
        """初期化"""
        self.progress_data: Dict[str, GAProgress] = {}
        self.progress_callbacks: Dict[str, Callable[[GAProgress], None]] = {}

    def set_progress_callback(
        self,
        experiment_id: str,
        callback: Callable[[GAProgress], None],
        db_experiment_id: Optional[int] = None,
    ):
        """
        進捗コールバックを設定

        Args:
            experiment_id: 実験ID
            callback: 進捗コールバック関数
            db_experiment_id: データベース実験ID
        """

        def combined_callback(progress: GAProgress):
            """進捗更新とデータベース保存を組み合わせたコールバック"""
            try:
                # メモリ上の進捗データを更新
                self.progress_data[experiment_id] = progress

                # データベースに進捗を保存
                if db_experiment_id:
                    self._save_progress_to_db(progress, db_experiment_id)

                # 外部コールバックを呼び出し
                if callback:
                    callback(progress)

            except Exception as e:
                logger.error(f"進捗コールバックエラー: {e}")

        self.progress_callbacks[experiment_id] = combined_callback

    def get_progress_callback(
        self, experiment_id: str
    ) -> Optional[Callable[[GAProgress], None]]:
        """
        進捗コールバックを取得

        Args:
            experiment_id: 実験ID

        Returns:
            進捗コールバック関数（存在しない場合はNone）
        """
        return self.progress_callbacks.get(experiment_id)

    def update_progress(self, experiment_id: str, progress: GAProgress):
        """
        進捗を更新

        Args:
            experiment_id: 実験ID
            progress: 進捗情報
        """
        try:
            self.progress_data[experiment_id] = progress

            # コールバックが設定されている場合は呼び出し
            callback = self.progress_callbacks.get(experiment_id)
            if callback:
                callback(progress)

        except Exception as e:
            logger.error(f"進捗更新エラー: {e}")

    def get_progress(self, experiment_id: str) -> Optional[GAProgress]:
        """
        進捗を取得

        Args:
            experiment_id: 実験ID

        Returns:
            進捗情報（存在しない場合はNone）
        """
        return self.progress_data.get(experiment_id)

    def create_final_progress(
        self,
        experiment_id: str,
        result: Dict,
        ga_config,
        status: str = "completed",
    ) -> GAProgress:
        """
        最終進捗情報を作成

        Args:
            experiment_id: 実験ID
            result: GA実行結果
            ga_config: GA設定
            status: ステータス

        Returns:
            最終進捗情報
        """
        try:
            final_progress = GAProgress(
                experiment_id=experiment_id,
                current_generation=ga_config.generations,
                total_generations=ga_config.generations,
                best_fitness=result.get("best_fitness", 0.0),
                average_fitness=result.get("best_fitness", 0.0),  # 簡略化
                execution_time=result.get("execution_time", 0.0),
                estimated_remaining_time=0.0,
                status=status,
            )

            self.progress_data[experiment_id] = final_progress
            return final_progress

        except Exception as e:
            logger.error(f"最終進捗作成エラー: {e}")
            # エラー時のフォールバック
            return GAProgress(
                experiment_id=experiment_id,
                current_generation=0,
                total_generations=ga_config.generations,
                best_fitness=0.0,
                average_fitness=0.0,
                execution_time=0.0,
                estimated_remaining_time=0.0,
                status="error",
            )

    def create_error_progress(
        self,
        experiment_id: str,
        ga_config,
        error_message: str = "",
    ) -> GAProgress:
        """
        エラー進捗情報を作成

        Args:
            experiment_id: 実験ID
            ga_config: GA設定
            error_message: エラーメッセージ

        Returns:
            エラー進捗情報
        """
        try:
            error_progress = GAProgress(
                experiment_id=experiment_id,
                current_generation=0,
                total_generations=ga_config.generations,
                best_fitness=0.0,
                average_fitness=0.0,
                execution_time=0.0,
                estimated_remaining_time=0.0,
                status="error",
            )

            self.progress_data[experiment_id] = error_progress
            return error_progress

        except Exception as e:
            logger.error(f"エラー進捗作成エラー: {e}")
            # 最小限のエラー進捗を返す
            return GAProgress(
                experiment_id=experiment_id,
                current_generation=0,
                total_generations=0,
                best_fitness=0.0,
                average_fitness=0.0,
                execution_time=0.0,
                estimated_remaining_time=0.0,
                status="error",
            )

    def cleanup_progress(self, experiment_id: str):
        """
        進捗データをクリーンアップ

        Args:
            experiment_id: 実験ID
        """
        try:
            if experiment_id in self.progress_data:
                del self.progress_data[experiment_id]

            if experiment_id in self.progress_callbacks:
                del self.progress_callbacks[experiment_id]

        except Exception as e:
            logger.error(f"進捗クリーンアップエラー: {e}")

    def _save_progress_to_db(self, progress: GAProgress, db_experiment_id: int):
        """
        進捗をデータベースに保存

        Args:
            progress: 進捗情報
            db_experiment_id: データベース実験ID
        """
        try:
            db = SessionLocal()
            try:
                ga_experiment_repo = GAExperimentRepository(db)
                ga_experiment_repo.update_experiment_progress(
                    experiment_id=db_experiment_id,
                    current_generation=progress.current_generation,
                    progress=progress.progress_percentage / 100.0,
                    best_fitness=progress.best_fitness,
                )
            finally:
                db.close()

        except Exception as e:
            logger.error(f"進捗データベース保存エラー: {e}")

    def get_all_progress(self) -> Dict[str, GAProgress]:
        """
        全ての進捗データを取得

        Returns:
            全進捗データの辞書
        """
        return self.progress_data.copy()

    def has_progress(self, experiment_id: str) -> bool:
        """
        進捗データが存在するかチェック

        Args:
            experiment_id: 実験ID

        Returns:
            進捗データが存在する場合True
        """
        return experiment_id in self.progress_data

    def get_progress_summary(self) -> Dict[str, Dict]:
        """
        進捗サマリーを取得

        Returns:
            進捗サマリーの辞書
        """
        summary = {}
        for experiment_id, progress in self.progress_data.items():
            summary[experiment_id] = {
                "status": progress.status,
                "current_generation": progress.current_generation,
                "total_generations": progress.total_generations,
                "progress_percentage": progress.progress_percentage,
                "best_fitness": progress.best_fitness,
                "execution_time": progress.execution_time,
            }
        return summary
