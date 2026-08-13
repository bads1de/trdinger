from app.services.auto_strategy.utils.normalization import (
    NormalizationUtils,
    normalize_enum_name,
)


class TestNormalizationUtils:
    def test_normalize_enum_name_enum(self):
        from enum import Enum

        class Method(Enum):
            FIXED = "fixed_percentage"

        assert normalize_enum_name(Method.FIXED) == "fixed_percentage"

    def test_normalize_enum_name_string(self):
        assert normalize_enum_name("volatility_based") == "volatility_based"

    def test_normalize_enum_name_none(self):
        assert normalize_enum_name(None) == ""
        assert normalize_enum_name(None, default="fallback") == "fallback"

    def test_class_method_access(self):
        # NormalizationUtilsクラス経由でのアクセスも確認
        assert NormalizationUtils.normalize_enum_name("x") == "x"
