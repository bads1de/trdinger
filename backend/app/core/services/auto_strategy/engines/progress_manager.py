"""
進捗管理器

GAの進捗情報を作成・管理するモジュール。
"""

import time
import logging
from typing import List, Any, Optional, Callable

from ..models.ga_config import GAConfig, GAProgress

logger = logging.getLogger(__name__)


class ProgressManager:
    """
    進捗管理器
    
    GAの進捗情報を作成・管理します。
    """

    def __init__(self):
        """初期化"""
        self.progress_callback: Optional[Callable] = None
        self.start_time = 0
        self.current_generation = 0

    def set_progress_callback(self, callback: Callable[[GAProgress], None]):
        """進捗コールバックを設定"""
        self.progress_callback = callback

    def set_start_time(self, start_time: float):
        """開始時刻を設定"""
        self.start_time = start_time

    def set_current_generation(self, generation: int):
        """現在の世代を設定"""
        self.current_generation = generation

    def create_progress_info(
        self, config: GAConfig, population: List[Any], experiment_id: str
    ) -> GAProgress:
        """
        進捗情報を作成

        GAの現在の世代、最高適応度、平均適応度、経過時間、推定残り時間などの
        情報を集約し、GAProgress オブジェクトとして返します。
        これにより、外部システム (例: フロントエンド) がGAの進行状況を
        リアルタイムで把握できるようになります。

        Args:
            config: GA設定
            population: 現在の個体群
            experiment_id: 実験ID

        Returns:
            進捗情報
        """
        try:
            # 現在の個体群から有効な適応度を持つ個体の適応度リストを抽出
            fitnesses = [ind.fitness.values[0] for ind in population if ind.fitness.valid]

            best_fitness = max(fitnesses) if fitnesses else 0.0
            avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0

            execution_time = time.time() - self.start_time
            estimated_remaining = (execution_time / max(1, self.current_generation)) * (
                config.generations - self.current_generation
            )

            return GAProgress(
                experiment_id=experiment_id,
                current_generation=self.current_generation,
                total_generations=config.generations,
                best_fitness=best_fitness,
                average_fitness=avg_fitness,
                execution_time=execution_time,
                estimated_remaining_time=estimated_remaining,
            )

        except Exception as e:
            logger.error(f"進捗情報作成エラー: {e}")
            # エラー時のフォールバック
            return GAProgress(
                experiment_id=experiment_id,
                current_generation=self.current_generation,
                total_generations=config.generations,
                best_fitness=0.0,
                average_fitness=0.0,
                execution_time=0.0,
                estimated_remaining_time=0.0,
            )

    def notify_progress(
        self, config: GAConfig, population: List[Any], experiment_id: str
    ):
        """
        進捗をコールバックに通知
        
        Args:
            config: GA設定
            population: 現在の個体群
            experiment_id: 実験ID
        """
        if self.progress_callback:
            try:
                progress = self.create_progress_info(config, population, experiment_id)
                self.progress_callback(progress)
            except Exception as e:
                logger.error(f"進捗通知エラー: {e}")

    def get_execution_time(self) -> float:
        """実行時間を取得"""
        return time.time() - self.start_time

    def get_estimated_remaining_time(self, config: GAConfig) -> float:
        """推定残り時間を取得"""
        if self.current_generation == 0:
            return 0.0
        
        execution_time = self.get_execution_time()
        return (execution_time / self.current_generation) * (
            config.generations - self.current_generation
        )
