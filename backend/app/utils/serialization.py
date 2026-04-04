"""
オブジェクト・辞書間の汎用シリアライズユーティリティ

dataclass、Enum、datetime 等の Python オブジェクトと
JSON 辞書との相互変換を一箇所に集約し、各所の重複実装を排除します。
"""

import json
import logging
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _convert_value(value: Any) -> Any:
    """単一値を JSON シリアライズ可能な型に再帰変換する。"""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return dataclass_to_dict(value)
    if isinstance(value, dict):
        return {k: _convert_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_convert_value(v) for v in value]
    if hasattr(value, "__dict__"):
        # dataclass 以外の複雑オブジェクト
        return str(value)
    return value


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """
    dataclass インスタンスを JSON シリアライズ可能な辞書に変換する。

    Enum          → .value
    datetime      → .isoformat()
    nested dataclass → 再帰変換
    list / dict   → 要素を再帰変換
    その他複雑オブジェクト → str()

    Args:
        obj: dataclass インスタンス（``__dataclass_fields__`` を持つこと）

    Returns:
        変換済み辞書。変換に失敗した場合は空辞書を返す。
    """
    try:
        result: Dict[str, Any] = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = _convert_value(value)
        return result
    except Exception as e:
        logger.error(f"辞書変換エラー ({type(obj).__name__}): {e}", exc_info=True)
        return {}


def dataclass_to_json(obj: Any, *, indent: int = 2) -> str:
    """dataclass インスタンスを JSON 文字列に変換する。"""
    try:
        return json.dumps(dataclass_to_dict(obj), ensure_ascii=False, indent=indent)
    except Exception as e:
        logger.error(f"JSON変換エラー ({type(obj).__name__}): {e}", exc_info=True)
        return "{}"


def dataclass_from_json(cls: Type[T], json_str: str) -> T:
    """
    JSON 文字列から dataclass インスタンスを復元する。

    ``cls.from_dict(data)`` が定義されていればそれを使用し、
    なければフィールドを ``setattr`` で設定する。

    Args:
        cls: 復元先の dataclass 型
        json_str: JSON 文字列

    Returns:
        復元されたインスタンス

    Raises:
        ValueError: JSON のパースまたは復元に失敗した場合
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON復元エラー: {e}", exc_info=True)
        raise ValueError(f"JSON のパースに失敗しました: {e}")

    return dataclass_from_dict(cls, data)


def dataclass_from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """
    辞書から dataclass インスタンスを復元する。

    ``cls.from_dict(data)`` が定義されていればそれを優先的に使用する。

    Args:
        cls: 復元先の dataclass 型
        data: 辞書

    Returns:
        復元されたインスタンス
    """
    try:
        # クラス側に from_dict があればそちらを優先
        if hasattr(cls, "from_dict") and callable(getattr(cls, "from_dict")):
            return cls.from_dict(data)  # type: ignore[attr-defined, return-value]

        # フォールバック: デフォルトインスタンスを作って setattr
        instance = cls()
        for key, value in data.items():
            if hasattr(instance, key):
                try:
                    setattr(instance, key, value)
                except Exception as e:
                    logger.warning(f"フィールド設定エラー: {key} = {value}, {e}")
        return instance
    except Exception as e:
        logger.error(f"辞書復元エラー ({cls.__name__}): {e}", exc_info=True)
        raise ValueError(f"辞書からの復元に失敗しました: {e}")
