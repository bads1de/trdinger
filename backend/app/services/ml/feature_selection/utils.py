"""
特徴量選択のユーティリティモジュール
"""

import logging

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

# LightGBM をデフォルトモデルとして使用（高速・高精度）
try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def get_default_estimator(
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = 1,
) -> BaseEstimator:
    """
    デフォルトのestimatorを取得

    LightGBMが利用可能ならLightGBM、そうでなければRandomForestを返す。
    """
    if LIGHTGBM_AVAILABLE:
        return LGBMClassifier(
            n_estimators=n_estimators,
            importance_type="gain",
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=-1,  # 警告抑制
            force_col_wise=True,  # 警告抑制
        )
    else:
        logger.warning("LightGBM not available, falling back to RandomForest")
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )
