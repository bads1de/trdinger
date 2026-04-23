"""
段階的特徴量選択戦略（推奨）

複数の手法を順番に適用し、段階的に絞り込む。
Filter -> Wrapper -> Embedded の順序が推奨。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import FeatureSelectionConfig, SelectionMethod
from ..strategy_registry import build_staged_strategy_map
from .base import BaseSelectionStrategy

logger = logging.getLogger(__name__)


class StagedStrategy(BaseSelectionStrategy):
    """
    段階的特徴量選択（推奨）

    複数の手法を順番に適用し、段階的に絞り込む。
    Filter -> Wrapper -> Embedded の順序が推奨。
    """

    def __init__(
        self,
        strategies: Optional[
            Dict[SelectionMethod, BaseSelectionStrategy]
        ] = None,
    ):
        self.strategy_map = strategies or self._default_strategies()

    def _default_strategies(
        self,
    ) -> Dict[SelectionMethod, BaseSelectionStrategy]:
        return build_staged_strategy_map()

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        config: FeatureSelectionConfig,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        current_mask = np.ones(X.shape[1], dtype=bool)
        current_X = X
        current_names = feature_names.copy()
        stage_results = []

        for method in config.staged_methods:
            if method not in self.strategy_map:
                logger.warning(f"Unknown method in staged selection: {method}")
                continue

            strategy = self.strategy_map[method]

            try:
                stage_mask, stage_details = strategy.select(
                    current_X, y, current_names, config
                )

                global_indices = np.where(current_mask)[0]
                for i, selected in enumerate(stage_mask):
                    if not selected:
                        current_mask[global_indices[i]] = False

                current_X = current_X[:, stage_mask]
                current_names = [
                    current_names[i] for i, s in enumerate(stage_mask) if s
                ]

                stage_results.append(
                    {
                        "method": method.value,
                        "selected_count": int(stage_mask.sum()),
                        "details": stage_details,
                    }
                )

                logger.info(
                    f"Stage [{method.value}]: {len(feature_names)} -> {current_mask.sum()} features"
                )

            except Exception as e:
                logger.warning(f"Stage [{method.value}] failed: {e}")
                stage_results.append({"method": method.value, "error": str(e)})

        return current_mask, {
            "method": "staged",
            "stages": stage_results,
            "final_count": int(current_mask.sum()),
        }
