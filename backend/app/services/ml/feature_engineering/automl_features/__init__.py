"""
AutoML特徴量エンジニアリングモジュール

このモジュールは、AutoMLライブラリを使用した自動特徴量生成・選択機能を提供します。

主要コンポーネント:
- TSFreshFeatureCalculator: TSFreshライブラリによる時系列特徴量生成
- AutoFeatSelector: AutoFeatライブラリによる遺伝的アルゴリズム特徴量選択
- AutoMLConfig: AutoML設定管理

注意: FeaturetoolsCalculatorはメモリ問題のため削除されました。
"""

from .tsfresh_calculator import TSFreshFeatureCalculator
from .autofeat_calculator import AutoFeatCalculator

__all__ = [
    "TSFreshFeatureCalculator",
    "AutoFeatCalculator",
]
