"""
実行中 GA エンジンのレジストリ。
"""


from app.utils.registry import Registry

from ..core.engine.ga_engine import GeneticAlgorithmEngine


class ExperimentEngineRegistry(Registry[str, GeneticAlgorithmEngine]):
    """実行中の GA エンジンをスレッドセーフに保持する。"""

    def register(self, experiment_id: str, engine: GeneticAlgorithmEngine) -> None:
        self.set(experiment_id, engine)

    def release(
        self,
        experiment_id: str,
        engine: GeneticAlgorithmEngine | None = None,
    ) -> None:
        self.remove(experiment_id, expected=engine)
