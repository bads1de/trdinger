"""
Feature Selection パッケージ

特徴量選択のためのユーティリティと選択手法を提供します。
"""

from .config import FeatureSelectionConfig, SelectionMethod
from .feature_selector import FeatureSelector
from .strategies import (
    BaseSelectionStrategy,
    LassoStrategy,
    PermutationStrategy,
    RFECVStrategy,
    ShadowFeatureStrategy,
    StagedStrategy,
    TreeBasedStrategy,
    UnivariateStrategy,
    VarianceStrategy,
)
from .utils import get_default_estimator

__all__ = [
    # メインセレクター
    "FeatureSelector",
    # 設定
    "SelectionMethod",
    "FeatureSelectionConfig",
    # ユーティリティ
    "get_default_estimator",
    # 戦略
    "BaseSelectionStrategy",
    "VarianceStrategy",
    "UnivariateStrategy",
    "RFECVStrategy",
    "LassoStrategy",
    "TreeBasedStrategy",
    "PermutationStrategy",
    "ShadowFeatureStrategy",
    "StagedStrategy",
]
