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

from importlib import import_module
from typing import Any

_ATTRIBUTE_EXPORTS = {
    "BaseMLTrainer": ".base_ml_trainer",
    "VolatilityRegressionTrainer": ".volatility_regression_trainer",
}


def __getattr__(name: str) -> Any:
    module_path = _ATTRIBUTE_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *_ATTRIBUTE_EXPORTS})

__all__ = ["BaseMLTrainer", "VolatilityRegressionTrainer"]
