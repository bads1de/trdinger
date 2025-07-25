"""
アンサンブル学習モジュール

複数のMLモデルを組み合わせて予測精度と頑健性を向上させるアンサンブル学習機能を提供します。

サポートされているアンサンブル手法:
- バギング (Bagging): 同じアルゴリズムのモデルを異なるデータサブセットで学習
- スタッキング (Stacking): 複数の異なるアルゴリズムの予測をメタモデルで統合
"""

from .base_ensemble import BaseEnsemble
from .bagging import BaggingEnsemble
from .stacking import StackingEnsemble
from .ensemble_trainer import EnsembleTrainer

__all__ = [
    "BaseEnsemble",
    "BaggingEnsemble", 
    "StackingEnsemble",
    "EnsembleTrainer"
]
