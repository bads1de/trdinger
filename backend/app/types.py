"""
バックエンド共通型定義

`Any` の使用を排除し、型安全なコードベースを維持するための型エイリアス・Protocolを定義します。
"""

from __future__ import annotations

from datetime import datetime
from typing import Union

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
# バックテスト・戦略関連
# ─────────────────────────────────────────────
BacktestResultDict = dict[str, SerializableValue]
StrategyGeneDict = dict[str, SerializableValue]

# ─────────────────────────────────────────────
# 日時関連
# ─────────────────────────────────────────────
DatetimeLike = Union[datetime, str, float, int]
