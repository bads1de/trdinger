"""
データベース層共通型定義
"""

from __future__ import annotations

from typing import Union

# シリアライズ可能なプリミティブ型
SerializablePrimitive = Union[str, int, float, bool, None]
SerializableValue = Union[
    SerializablePrimitive,
    list["SerializableValue"],
    dict[str, "SerializableValue"],
]
