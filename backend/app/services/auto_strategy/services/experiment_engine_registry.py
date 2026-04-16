"""
実行中 GA エンジンのレジストリ。
"""

import threading
from typing import Dict, Optional

from ..core.engine.ga_engine import GeneticAlgorithmEngine


class ExperimentEngineRegistry:
    """実行中の GA エンジンをスレッドセーフに保持する。"""

    def __init__(self) -> None:
        self._active_engines: Dict[str, "GeneticAlgorithmEngine"] = {}
        self._lock = threading.RLock()

    @property
    def active_engines(self) -> Dict[str, "GeneticAlgorithmEngine"]:
        return self._active_engines

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def register(self, experiment_id: str, engine: "GeneticAlgorithmEngine") -> None:
        with self._lock:
            self._active_engines[experiment_id] = engine

    def get(self, experiment_id: str) -> Optional["GeneticAlgorithmEngine"]:
        with self._lock:
            return self._active_engines.get(experiment_id)

    def release(
        self,
        experiment_id: str,
        engine: Optional["GeneticAlgorithmEngine"] = None,
    ) -> None:
        with self._lock:
            current = self._active_engines.get(experiment_id)
            if current is None:
                return
            if engine is None or current is engine:
                self._active_engines.pop(experiment_id, None)

    def clear(self) -> None:
        with self._lock:
            self._active_engines.clear()
