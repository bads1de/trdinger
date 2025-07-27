"""
AutoML特徴量エンジニアリングモジュール

このモジュールは、AutoMLライブラリを使用した自動特徴量生成・選択機能を提供します。

主要コンポーネント:
- TSFreshFeatureCalculator: TSFreshライブラリによる時系列特徴量生成
- FeaturetoolsCalculator: Featuretoolsライブラリによる深層特徴量合成
- AutoFeatSelector: AutoFeatライブラリによる遺伝的アルゴリズム特徴量選択
- AutoMLConfig: AutoML設定管理
"""

from .tsfresh_calculator import TSFreshFeatureCalculator
from .featuretools_calculator import FeaturetoolsCalculator
from .autofeat_calculator import AutoFeatCalculator

__all__ = [
    "TSFreshFeatureCalculator",
    "FeaturetoolsCalculator",
    "AutoFeatCalculator",
]
