"""
Serialization ユーティリティのユニットテスト

dataclass、Enum、datetime 等の Python オブジェクトと
JSON 辞書との相互変換をテストします。
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest

from app.utils.serialization import (
    _convert_value,
    dataclass_to_dict,
    dataclass_to_json,
    dataclass_from_json,
    dataclass_from_dict,
)


# テスト用のEnum
class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Status(Enum):
    ACTIVE = 1
    INACTIVE = 0


# テスト用のdataclass
@dataclass
class SimpleDataclass:
    name: str
    value: int
    price: float = 100.0


@dataclass
class DefaultDataclass:
    """デフォルト値を持つdataclass（from_dictフォールバックテスト用）"""
    name: str = ""
    value: int = 0
    price: float = 100.0


@dataclass
class NestedDataclass:
    id: int
    inner: SimpleDataclass
    tags: List[str] = field(default_factory=list)


@dataclass
class ComplexDataclass:
    color: Color
    status: Status
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataclassWithFromDict:
    name: str
    value: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataclassWithFromDict":
        return cls(name=data["name"], value=data["value"] * 2)


class TestConvertValue:
    """_convert_value関数のテスト"""

    def test_enum_value(self):
        """Enum値が変換されること"""
        assert _convert_value(Color.RED) == "red"
        assert _convert_value(Status.ACTIVE) == 1

    def test_datetime_value(self):
        """datetime値がISO形式に変換されること"""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        assert _convert_value(dt) == "2024-01-15T10:30:45"

    def test_dataclass_value(self):
        """dataclass値が辞書に変換されること"""
        obj = SimpleDataclass(name="test", value=42)
        result = _convert_value(obj)
        assert result == {"name": "test", "value": 42, "price": 100.0}

    def test_dict_value(self):
        """辞書値が再帰変換されること"""
        data = {"color": Color.RED, "count": 5}
        result = _convert_value(data)
        assert result == {"color": "red", "count": 5}

    def test_list_value(self):
        """リスト値が再帰変換されること"""
        data = [Color.RED, Color.BLUE, 42]
        result = _convert_value(data)
        assert result == ["red", "blue", 42]

    def test_tuple_value(self):
        """タプル値がリストに変換されること"""
        data = (1, 2, 3)
        result = _convert_value(data)
        assert result == [1, 2, 3]

    def test_complex_object_value(self):
        """複雑オブジェクトが文字列に変換されること"""
        class CustomObj:
            def __str__(self):
                return "CustomObject"
        
        obj = CustomObj()
        result = _convert_value(obj)
        assert isinstance(result, str)

    def test_primitive_values(self):
        """プリミティブ値がそのまま返されること"""
        assert _convert_value(42) == 42
        assert _convert_value(3.14) == 3.14
        assert _convert_value("hello") == "hello"
        assert _convert_value(None) is None
        assert _convert_value(True) is True


class TestDataclassToDict:
    """dataclass_to_dict関数のテスト"""

    def test_simple_dataclass(self):
        """シンプルなdataclassが変換されること"""
        obj = SimpleDataclass(name="test", value=42)
        result = dataclass_to_dict(obj)
        assert result == {"name": "test", "value": 42, "price": 100.0}

    def test_nested_dataclass(self):
        """ネストされたdataclassが再帰変換されること"""
        obj = NestedDataclass(
            id=1,
            inner=SimpleDataclass(name="inner", value=10),
            tags=["a", "b"]
        )
        result = dataclass_to_dict(obj)
        assert result["id"] == 1
        assert result["inner"] == {"name": "inner", "value": 10, "price": 100.0}
        assert result["tags"] == ["a", "b"]

    def test_complex_dataclass(self):
        """Enumとdatetimeを含むdataclassが変換されること"""
        obj = ComplexDataclass(
            color=Color.GREEN,
            status=Status.ACTIVE,
            created_at=datetime(2024, 1, 1),
            metadata={"key": "value"}
        )
        result = dataclass_to_dict(obj)
        assert result["color"] == "green"
        assert result["status"] == 1
        assert result["created_at"] == "2024-01-01T00:00:00"
        assert result["metadata"] == {"key": "value"}

    def test_error_returns_empty_dict(self):
        """エラー時に空辞書が返されること"""
        result = dataclass_to_dict("not a dataclass")
        assert result == {}


class TestDataclassToJson:
    """dataclass_to_json関数のテスト"""

    def test_simple_to_json(self):
        """シンプルなdataclassがJSONに変換されること"""
        obj = SimpleDataclass(name="test", value=42)
        result = dataclass_to_json(obj)
        parsed = json.loads(result)
        assert parsed["name"] == "test"
        assert parsed["value"] == 42

    def test_custom_indent(self):
        """カスタムインデントでJSONが生成されること"""
        obj = SimpleDataclass(name="test", value=42)
        result = dataclass_to_json(obj, indent=4)
        assert "    " in result

    def test_error_returns_empty_json(self):
        """エラー時に空のJSONが返されること"""
        result = dataclass_to_json("not a dataclass")
        assert result == "{}"


class TestDataclassFromJson:
    """dataclass_from_json関数のテスト"""

    def test_from_json_with_from_dict(self):
        """from_dictメソッドがあるクラスが復元されること"""
        json_str = '{"name": "test", "value": 10}'
        result = dataclass_from_json(DataclassWithFromDict, json_str)
        assert result.name == "test"
        assert result.value == 20  # from_dictで2倍される

    def test_invalid_json_raises_error(self):
        """不正なJSONでValueErrorが発生すること"""
        with pytest.raises(ValueError, match="JSON のパースに失敗しました"):
            dataclass_from_json(SimpleDataclass, "invalid json")


class TestDataclassFromDict:
    """dataclass_from_dict関数のテスト"""

    def test_from_dict_with_class_method(self):
        """from_dictメソッドがあるクラスが優先されること"""
        data = {"name": "test", "value": 10}
        result = dataclass_from_dict(DataclassWithFromDict, data)
        assert result.name == "test"
        assert result.value == 20

    def test_from_dict_fallback_setattr(self):
        """from_dictがない場合、setattrで復元されること"""
        data = {"name": "test", "value": 42, "price": 200.0}
        result = dataclass_from_dict(DefaultDataclass, data)
        assert result.name == "test"
        assert result.value == 42
        assert result.price == 200.0

    def test_from_dict_ignores_unknown_keys(self):
        """未知のキーが無視されること"""
        data = {"name": "test", "value": 42, "unknown": "ignored"}
        result = dataclass_from_dict(DefaultDataclass, data)
        assert result.name == "test"
        assert not hasattr(result, "unknown")

    def test_from_dict_error_raises_value_error(self):
        """エラー時にValueErrorが発生すること"""
        # from_dictメソッドを持たないクラスでエラーを発生させる
        class NoFromDict:
            def __init__(self, required_arg):
                pass
        
        with pytest.raises(ValueError, match="辞書からの復元に失敗しました"):
            dataclass_from_dict(NoFromDict, {"key": "value"})
