"""
特徴量選択のユーティリティモジュール
"""

import logging

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logger = logging.getLogger(__name__)

# LightGBM をデフォルトモデルとして使用（高速・高精度）
try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def is_regression_target(y: np.ndarray) -> bool:
    """ターゲットが回帰タスクかどうかを判定"""
    unique_values = np.unique(y)
    return len(unique_values) > 20 or not np.all(unique_values == unique_values.astype(int))


def get_default_estimator(
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = 1,
) -> BaseEstimator:
    """
    デフォルトの分類estimatorを取得

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


def get_default_regressor(
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = 1,
) -> BaseEstimator:
    """
    デフォルトの回帰estimatorを取得

    LightGBMが利用可能ならLGBMRegressor、そうでなければRandomForestRegressorを返す。
    """
    if LIGHTGBM_AVAILABLE:
        return LGBMRegressor(
            n_estimators=n_estimators,
            importance_type="gain",
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=-1,
            force_col_wise=True,
        )
    else:
        logger.warning("LightGBM not available, falling back to RandomForestRegressor")
        return RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )


def get_task_appropriate_estimator(
    y: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = 1,
) -> BaseEstimator:
    """タスクタイプに応じたestimatorを返す"""
    if is_regression_target(y):
        return get_default_regressor(n_estimators, random_state, n_jobs)
    return get_default_estimator(n_estimators, random_state, n_jobs)
