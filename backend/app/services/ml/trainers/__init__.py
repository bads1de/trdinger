"""
MLトレーナーモジュール

機械学習モデルのトレーニングを担当するトレーナークラスを提供します。
各トレーナーはモデル固有の訓練ロジックをカプセル化し、
統一されたインターフェースで結果を返します。

主なコンポーネント:
- base_ml_trainer.py: トレーナー基底クラス（共通ロジック）
- volatility_regression_trainer.py: ボラティリティ回帰専用トレーナー
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_ml_trainer import BaseMLTrainer
    from .volatility_regression_trainer import VolatilityRegressionTrainer

_ATTRIBUTE_EXPORTS = {
    "BaseMLTrainer": ".base_ml_trainer",
    "VolatilityRegressionTrainer": ".volatility_regression_trainer",
}

__all__ = ["BaseMLTrainer", "VolatilityRegressionTrainer"]

from app.utils.lazy_import import setup_lazy_import  # noqa: E402

setup_lazy_import(globals(), _ATTRIBUTE_EXPORTS)
