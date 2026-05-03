"""
バックエンド共通型定義

`Any` の使用を排除し、型安全なコードベースを維持するための型エイリアス・Protocolを定義します。
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from catboost import CatBoost as CatBoostModel
    from lightgbm import Booster as LGBMBooster
    from xgboost import Booster as XGBBooster

# ─────────────────────────────────────────────
# シリアライズ可能なプリミティブ型
# ─────────────────────────────────────────────
SerializablePrimitive = Union[str, int, float, bool, None]
SerializableValue = Union[
    SerializablePrimitive,
    list["SerializableValue"],
    dict[str, "SerializableValue"],
]

# ─────────────────────────────────────────────
# MLモデル関連
# ─────────────────────────────────────────────
class MLModelProtocol(Protocol):
    """MLモデルの共通インターフェースを定義するProtocol"""

    is_trained: bool
    feature_columns: list[str] | None

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


# 学習済みモデル（具体的な型）
# MLModelProtocol を実装するモデルまたは生のBoosterモデル
TrainedModel = Union[MLModelProtocol, "LGBMBooster", "XGBBooster", "CatBoostModel"]

# ─────────────────────────────────────────────
# バックテスト・戦略関連
# ─────────────────────────────────────────────
BacktestResultDict = dict[str, SerializableValue]
StrategyGeneDict = dict[str, SerializableValue]

# ─────────────────────────────────────────────
# 日時関連
# ─────────────────────────────────────────────
DatetimeLike = Union[datetime, str, float, int]


