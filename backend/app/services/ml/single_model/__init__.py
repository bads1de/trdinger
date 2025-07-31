"""
単一モデルトレーニングモジュール

BaseMLTrainerを継承した単一モデル用のトレーナークラスを提供します。
アンサンブル学習を使用せず、指定された単一のモデルで学習を行います。
"""

from .single_model_trainer import SingleModelTrainer

__all__ = [
    "SingleModelTrainer",
]
