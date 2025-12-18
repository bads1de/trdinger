"""
BaseGeneクラスのテスト
"""

from datetime import datetime
from enum import Enum


from app.services.auto_strategy.genes.base_gene import BaseGene


class MockEnum(Enum):
    VALUE1 = "value1"
    VALUE2 = "value2"


class MockGene(BaseGene):
    """テスト用の遺伝子クラス"""

    def __init__(
        self,
        normal_field: str = None,
        enum_field: MockEnum = None,
        datetime_field: datetime = None,
        optional_field: str = "default",
    ):
        self.normal_field = normal_field
        self.enum_field = enum_field
        self.datetime_field = datetime_field
        self.optional_field = optional_field

    def _validate_parameters(self, errors):
        pass

    # アノテーションを追加（テスト用）
    __annotations__ = {
        "normal_field": str,
        "enum_field": MockEnum,
        "datetime_field": datetime,
        "optional_field": str,
    }


class MockGeneWithoutAnnotations(BaseGene):
    """アノテーションなしのテストクラス"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def _validate_parameters(self, errors):
        pass


# アノテーションなしにするため、クラス定義後に削除
if hasattr(MockGeneWithoutAnnotations, "__annotations__"):
    del MockGeneWithoutAnnotations.__annotations__


class TestGeneUtils:
    """BaseGene.from_dict のテスト"""

    def test_from_dict_with_valid_data(self):
        """有効なデータでの復元テスト"""
        data = {
            "normal_field": "test_value",
            "enum_field": "value1",
            "datetime_field": "2023-01-01T00:00:00",
            "optional_field": "custom_value",
        }

        gene = MockGene.from_dict(data)

        assert gene.normal_field == "test_value"
        assert gene.enum_field == MockEnum.VALUE1
        assert gene.datetime_field == datetime(2023, 1, 1, 0, 0, 0)
        assert gene.optional_field == "custom_value"

    def test_from_dict_with_invalid_enum_value(self):
        """無効なEnum値での復元テスト（デフォルト値が設定される）"""
        data = {
            "normal_field": "test_value",
            "enum_field": "invalid_value",
            "datetime_field": "2023-01-01T00:00:00",
        }

        gene = MockGene.from_dict(data)

        assert gene.normal_field == "test_value"
        # 無効なEnum値の場合は最初の値をデフォルトとして設定
        assert gene.enum_field == MockEnum.VALUE1
        assert gene.datetime_field == datetime(2023, 1, 1, 0, 0, 0)

    def test_from_dict_with_invalid_datetime_value(self):
        """無効なdatetime値での復元テスト（デフォルト値が設定される）"""
        data = {
            "normal_field": "test_value",
            "enum_field": "value2",
            "datetime_field": "invalid_datetime",
        }

        gene = MockGene.from_dict(data)

        assert gene.normal_field == "test_value"
        assert gene.enum_field == MockEnum.VALUE2
        # 無効なdatetime値の場合は現在時刻が設定される（正確な値はテストしにくいので型チェック）
        assert isinstance(gene.datetime_field, datetime)

    def test_from_dict_with_enum_object(self):
        """Enumオブジェクトが直接渡された場合のテスト"""
        data = {
            "normal_field": "test_value",
            "enum_field": MockEnum.VALUE2,
            "datetime_field": "2023-01-01T00:00:00",
        }

        gene = MockGene.from_dict(data)

        assert gene.normal_field == "test_value"
        assert gene.enum_field == MockEnum.VALUE2
        assert gene.datetime_field == datetime(2023, 1, 1, 0, 0, 0)

    def test_from_dict_with_datetime_object(self):
        """datetimeオブジェクトが直接渡された場合のテスト"""
        test_datetime = datetime(2023, 6, 15, 12, 30, 45)
        data = {
            "normal_field": "test_value",
            "enum_field": "value1",
            "datetime_field": test_datetime,
        }

        gene = MockGene.from_dict(data)

        assert gene.normal_field == "test_value"
        assert gene.enum_field == MockEnum.VALUE1
        assert gene.datetime_field == test_datetime

    def test_from_dict_without_annotations(self):
        """アノテーションなしクラスのテスト"""
        data = {"field1": "value1", "field2": 42, "field3": {"nested": "data"}}

        # クラスの状態を確認
        print(f"DEBUG: Annotations: {getattr(MockGeneWithoutAnnotations, '__annotations__', 'N/A')}")
        
        gene = MockGeneWithoutAnnotations.from_dict(data)

        # 生成されたオブジェクトの状態を確認
        print(f"DEBUG: Gene __dict__: {gene.__dict__}")

        assert gene.field1 == "value1"
        assert gene.field2 == 42
        assert gene.field3 == {"nested": "data"}

    def test_from_dict_with_missing_fields(self):
        """一部のフィールドが欠けている場合のテスト"""
        data = {
            "normal_field": "test_value",
            "enum_field": "value1",
            # datetime_field が欠けている
        }

        gene = MockGene.from_dict(data)

        assert gene.normal_field == "test_value"
        assert gene.enum_field == MockEnum.VALUE1
        # datetime_field は設定されないのでNoneになる
        assert gene.datetime_field is None

    def test_convert_value_enum(self):
        """Enum変換のテスト"""
        # 文字列からの変換
        result = BaseGene._convert_value("value1", MockEnum)
        assert result == MockEnum.VALUE1

        # 無効な文字列
        result = BaseGene._convert_value("invalid", MockEnum)
        assert result == MockEnum.VALUE1  # デフォルト値

        # すでにEnumオブジェクト
        result = BaseGene._convert_value(MockEnum.VALUE2, MockEnum)
        assert result == MockEnum.VALUE2

    def test_convert_value_datetime(self):
        """datetime変換のテスト"""
        # 文字列からの変換
        result = BaseGene._convert_value("2023-01-01T00:00:00", datetime)
        assert result == datetime(2023, 1, 1, 0, 0, 0)

        # 無効な文字列
        result = BaseGene._convert_value("invalid", datetime)
        assert isinstance(result, datetime)  # 現在時刻が設定される

        # すでにdatetimeオブジェクト
        test_dt = datetime(2023, 6, 15, 12, 30, 45)
        result = BaseGene._convert_value(test_dt, datetime)
        assert result == test_dt

    def test_convert_value_other_types(self):
        """その他の型のテスト"""
        # 文字列
        result = BaseGene._convert_value("test", str)
        assert result == "test"

        # 数値
        result = BaseGene._convert_value(42, int)
        assert result == 42

        # 辞書
        result = BaseGene._convert_value({"key": "value"}, dict)
        assert result == {"key": "value"}

    def test_is_enum_type(self):
        """Enum型チェックのテスト"""
        assert BaseGene._is_enum_type(MockEnum) is True
        assert BaseGene._is_enum_type(str) is False
        assert BaseGene._is_enum_type(datetime) is False

    def test_is_datetime_type(self):
        """datetime型チェックのテスト"""
        assert BaseGene._is_datetime_type(datetime) is True
        assert BaseGene._is_datetime_type(str) is False
        assert BaseGene._is_datetime_type(MockEnum) is False




