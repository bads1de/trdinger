"""
実験管理器

GA実験の管理・スレッド制御を担当するモジュール。
"""

import uuid
import time
import threading
import logging
from typing import Callable, Dict, Any, List, Optional, cast

from ..models.ga_config import GAConfig

from database.repositories.ga_experiment_repository import GAExperimentRepository
from database.connection import SessionLocal

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    実験管理器

    GA実験の管理・スレッド制御を担当します。
    """

    def __init__(self):
        """初期化"""
        self.running_experiments: Dict[str, Dict[str, Any]] = {}
        self.experiment_threads: Dict[str, threading.Thread] = {}

    def create_experiment(
        self,
        experiment_name: str,
        ga_config: GAConfig,
        backtest_config: Dict[str, Any],
    ) -> str:
        """
        新しい実験を作成

        Args:
            experiment_name: 実験名
            ga_config: GA設定
            backtest_config: バックテスト設定

        Returns:
            実験ID
        """
        try:
            # 実験IDを生成
            experiment_id = str(uuid.uuid4())
            logger.info(f"新しい実験を作成中: {experiment_id} ({experiment_name})")

            # データベースに実験を保存
            db_experiment_id = self._save_experiment_to_db(experiment_name, ga_config)

            # 実験情報をメモリに記録
            experiment_info = {
                "id": experiment_id,
                "db_id": db_experiment_id,
                "name": experiment_name,
                "ga_config": ga_config,
                "backtest_config": backtest_config,
                "status": "created",
                "start_time": time.time(),
                "thread": None,
                "result": None,
                "error": None,
            }

            self.running_experiments[experiment_id] = experiment_info
            logger.info(f"実験作成完了: {experiment_id}")

            return experiment_id

        except Exception as e:
            logger.error(f"実験作成エラー: {e}")
            raise

    def start_experiment(
        self,
        experiment_id: str,
        experiment_runner: Callable[[str, GAConfig, Dict[str, Any]], None],
    ) -> bool:
        """
        実験を開始

        Args:
            experiment_id: 実験ID
            experiment_runner: 実験実行関数

        Returns:
            開始成功の場合True
        """
        try:
            experiment_info = self.running_experiments.get(experiment_id)
            if not experiment_info:
                logger.error(f"実験が見つかりません: {experiment_id}")
                return False

            if experiment_info["status"] != "created":
                logger.error(f"実験は既に開始されています: {experiment_id}")
                return False

            # バックグラウンドスレッドで実験を実行
            thread = threading.Thread(
                target=experiment_runner,
                args=(
                    experiment_id,
                    experiment_info["ga_config"],
                    experiment_info["backtest_config"],
                ),
                daemon=True,
            )

            experiment_info["thread"] = thread
            experiment_info["status"] = "running"
            self.experiment_threads[experiment_id] = thread

            thread.start()
            logger.info(f"実験開始: {experiment_id}")

            return True

        except Exception as e:
            logger.error(f"実験開始エラー: {e}")
            return False

    def stop_experiment(self, experiment_id: str) -> bool:
        """
        実験を停止

        Args:
            experiment_id: 実験ID

        Returns:
            停止成功の場合True
        """
        try:
            experiment_info = self.running_experiments.get(experiment_id)
            if not experiment_info:
                logger.error(f"実験が見つかりません: {experiment_id}")
                return False

            if experiment_info["status"] != "running":
                logger.warning(f"実験は実行中ではありません: {experiment_id}")
                return False

            # 実験状態を更新
            experiment_info["status"] = "stopped"
            experiment_info["end_time"] = time.time()

            # スレッドの参照を削除
            if experiment_id in self.experiment_threads:
                del self.experiment_threads[experiment_id]

            logger.info(f"実験停止: {experiment_id}")
            return True

        except Exception as e:
            logger.error(f"実験停止エラー: {e}")
            return False

    def complete_experiment(
        self,
        experiment_id: str,
        result: Dict[str, Any],
    ):
        """
        実験を完了状態にする

        Args:
            experiment_id: 実験ID
            result: 実験結果
        """
        try:
            experiment_info = self.running_experiments.get(experiment_id)
            if not experiment_info:
                logger.error(f"実験が見つかりません: {experiment_id}")
                return

            experiment_info["status"] = "completed"
            experiment_info["end_time"] = time.time()
            experiment_info["result"] = result

            # スレッドの参照を削除
            if experiment_id in self.experiment_threads:
                del self.experiment_threads[experiment_id]

            logger.info(f"実験完了: {experiment_id}")

        except Exception as e:
            logger.error(f"実験完了処理エラー: {e}")

    def fail_experiment(
        self,
        experiment_id: str,
        error: str,
    ):
        """
        実験を失敗状態にする

        Args:
            experiment_id: 実験ID
            error: エラーメッセージ
        """
        try:
            experiment_info = self.running_experiments.get(experiment_id)
            if not experiment_info:
                logger.error(f"実験が見つかりません: {experiment_id}")
                return

            experiment_info["status"] = "error"
            experiment_info["end_time"] = time.time()
            experiment_info["error"] = error

            # スレッドの参照を削除
            if experiment_id in self.experiment_threads:
                del self.experiment_threads[experiment_id]

            logger.error(f"実験失敗: {experiment_id} - {error}")

        except Exception as e:
            logger.error(f"実験失敗処理エラー: {e}")

    def get_experiment_info(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        実験情報を取得

        Args:
            experiment_id: 実験ID

        Returns:
            実験情報（存在しない場合はNone）
        """
        return self.running_experiments.get(experiment_id)

    def get_experiment_result(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        実験結果を取得

        Args:
            experiment_id: 実験ID

        Returns:
            実験結果（存在しない場合はNone）
        """
        experiment_info = self.running_experiments.get(experiment_id)
        if experiment_info and experiment_info["status"] == "completed":
            return experiment_info.get("result")
        return None

    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        実験一覧を取得

        Returns:
            実験情報のリスト
        """
        experiments = []
        for experiment_id, info in self.running_experiments.items():
            experiment_summary = {
                "id": experiment_id,
                "name": info["name"],
                "status": info["status"],
                "start_time": info["start_time"],
                "end_time": info.get("end_time"),
                "error": info.get("error"),
            }
            experiments.append(experiment_summary)

        return experiments

    def cleanup_completed_experiments(self, max_age_hours: int = 24):
        """
        完了した実験をクリーンアップ

        Args:
            max_age_hours: 保持する最大時間（時間）
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600

            experiments_to_remove = []
            for experiment_id, info in self.running_experiments.items():
                if info["status"] in ["completed", "error", "stopped"]:
                    end_time = info.get("end_time", info["start_time"])
                    if current_time - end_time > max_age_seconds:
                        experiments_to_remove.append(experiment_id)

            for experiment_id in experiments_to_remove:
                del self.running_experiments[experiment_id]
                if experiment_id in self.experiment_threads:
                    del self.experiment_threads[experiment_id]

            if experiments_to_remove:
                logger.info(
                    f"クリーンアップ完了: {len(experiments_to_remove)}件の実験を削除"
                )

        except Exception as e:
            logger.error(f"実験クリーンアップエラー: {e}")

    def _save_experiment_to_db(
        self,
        experiment_name: str,
        ga_config: GAConfig,
    ) -> int:
        """
        実験をデータベースに保存

        Args:
            experiment_name: 実験名
            ga_config: GA設定

        Returns:
            データベース実験ID
        """
        try:
            db = SessionLocal()
            try:
                ga_experiment_repo = GAExperimentRepository(db)
                db_experiment = ga_experiment_repo.create_experiment(
                    name=experiment_name,
                    config=ga_config.to_dict(),
                    total_generations=ga_config.generations,
                    status="starting",
                )
                db_experiment_id = cast(int, db_experiment.id)
                logger.info(f"データベース実験作成完了: DB ID = {db_experiment_id}")
                return db_experiment_id

            finally:
                db.close()

        except Exception as e:
            logger.error(f"データベース保存エラー: {e}")
            raise

    def get_running_experiment_count(self) -> int:
        """実行中の実験数を取得"""
        return sum(
            1
            for info in self.running_experiments.values()
            if info["status"] == "running"
        )

    def is_experiment_running(self, experiment_id: str) -> bool:
        """実験が実行中かチェック"""
        experiment_info = self.running_experiments.get(experiment_id)
        return experiment_info is not None and experiment_info["status"] == "running"
