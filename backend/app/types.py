"""
バックエンド共通型定義

`Any` の使用を排除し、型安全なコードベースを維持するための型エイリアス・Protocolを定義します。
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol, TypeVar, Union

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
# 勾配ブースティングモデルの共通インターフェース
ModelType = TypeVar(
    "ModelType",
    "LGBMBooster",
    "XGBBooster",
    "CatBoostModel",
    covariant=True,
)

# 学習済みモデル（具体的な型）
TrainedModel = Union["LGBMBooster", "XGBBooster", "CatBoostModel"]

# 予測メタデータ
PredictionMetadata = dict[str, Union[float, int, str, list[float]]]


class MLModelProtocol(Protocol):
    """MLモデルの共通インターフェースを定義するProtocol"""

    @property
    def is_trained(self) -> bool: ...

    @property
    def feature_columns(self) -> list[str] | None: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


# ─────────────────────────────────────────────
# バックテスト・戦略関連
# ─────────────────────────────────────────────
BacktestResultDict = dict[str, SerializableValue]
StrategyGeneDict = dict[str, SerializableValue]

# ─────────────────────────────────────────────
# 日時関連
# ─────────────────────────────────────────────
DatetimeLike = Union[datetime, str, float, int]

# ─────────────────────────────────────────────
# 汎用ジェネリクス
# ─────────────────────────────────────────────
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
