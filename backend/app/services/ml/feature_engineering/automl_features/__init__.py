"""
AutoML特徴量エンジニアリングモジュール

このモジュールは、AutoMLライブラリを使用した自動特徴量生成・選択機能を提供します。

主要コンポーネント:
- AutoFeatCalculator: AutoFeatライブラリによる遺伝的アルゴリズム特徴量生成
- AutoMLConfig: AutoML設定管理
"""

from .autofeat_calculator import AutoFeatCalculator

__all__ = [
    "AutoFeatCalculator",
]
