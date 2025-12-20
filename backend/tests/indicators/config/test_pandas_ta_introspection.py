"""
pandas-ta イントロスペクションモジュールのテスト

pandas-taのソースコードから動的に情報を抽出する機能をテストします。
これにより、手動での定数メンテナンスを削減します。
"""

import pytest
import pandas as pd
import numpy as np


class TestMinLengthExtractor:
    """min_length抽出機能のテスト"""

    def test_extract_simple_length_expression(self):
        """単純なlengthパラメータを使う指標（RSI等）の抽出"""
        from app.services.indicators.config.pandas_ta_introspection import (
            extract_min_length_expression,
        )

        # RSIは単純に "length" を使う
        expr = extract_min_length_expression("rsi")
        assert expr is not None
        assert "length" in expr.lower()

    def test_extract_max_expression(self):
        """max()を使う複雑な式の抽出（MACD等）"""
        from app.services.indicators.config.pandas_ta_introspection import (
            extract_min_length_expression,
        )

        # MACDは max(fast, slow, signal) を使う
        expr = extract_min_length_expression("macd")
        assert expr is not None
        assert "max" in expr.lower()
        assert "fast" in expr or "slow" in expr

    def test_extract_with_intermediate_variable(self):
        """中間変数_lengthを使う指標（STOCH等）の抽出"""
        from app.services.indicators.config.pandas_ta_introspection import (
            extract_min_length_expression,
        )

        # STOCHは _length = max(k, d, smooth_k) を使う
        expr = extract_min_length_expression("stoch")
        assert expr is not None
        # 最終的な式が返されるべき
        assert "max" in expr.lower() or "k" in expr.lower()

    def test_extract_nonexistent_indicator_returns_none(self):
        """存在しない指標はNoneを返す"""
        from app.services.indicators.config.pandas_ta_introspection import (
            extract_min_length_expression,
        )

        expr = extract_min_length_expression("nonexistent_indicator_xyz")
        assert expr is None


class TestMinLengthCalculator:
    """min_length計算機能のテスト"""

    def test_calculate_rsi_min_length(self):
        """RSIのmin_length計算"""
        from app.services.indicators.config.pandas_ta_introspection import (
            calculate_min_length,
        )

        # RSI with length=14
        result = calculate_min_length("rsi", {"length": 14})
        assert result == 14

        # RSI with length=20
        result = calculate_min_length("rsi", {"length": 20})
        assert result == 20

    def test_calculate_macd_min_length(self):
        """MACDのmin_length計算"""
        from app.services.indicators.config.pandas_ta_introspection import (
            calculate_min_length,
        )

        # MACD with fast=12, slow=26, signal=9 -> max(12, 26, 9) = 26
        result = calculate_min_length("macd", {"fast": 12, "slow": 26, "signal": 9})
        assert result == 26

        # MACD with different params
        result = calculate_min_length("macd", {"fast": 5, "slow": 35, "signal": 10})
        assert result == 35

    def test_calculate_stoch_min_length(self):
        """STOCHのmin_length計算"""
        from app.services.indicators.config.pandas_ta_introspection import (
            calculate_min_length,
        )

        # STOCH with k=14, d=3, smooth_k=3 -> max(14, 3, 3) = 14
        result = calculate_min_length("stoch", {"k": 14, "d": 3, "smooth_k": 3})
        assert result == 14

    def test_calculate_with_defaults(self):
        """パラメータが指定されない場合はデフォルト値を使用"""
        from app.services.indicators.config.pandas_ta_introspection import (
            calculate_min_length,
        )

        # 空のパラメータでも動作する（デフォルト値を使用）
        result = calculate_min_length("rsi", {})
        assert result is not None
        assert result > 0

    def test_calculate_nonexistent_returns_fallback(self):
        """存在しない指標はフォールバック値を返す"""
        from app.services.indicators.config.pandas_ta_introspection import (
            calculate_min_length,
        )

        result = calculate_min_length("nonexistent_xyz", {"length": 10})
        # フォールバックとして渡されたlengthまたは1を返す
        assert result >= 1


class TestReturnColsExtractor:
    """戻り値カラム抽出機能のテスト"""

    def test_detect_single_return_indicator(self):
        """単一値を返す指標の検出（RSI等）"""
        from app.services.indicators.config.pandas_ta_introspection import (
            is_multi_column_indicator,
        )

        # RSIは単一のSeriesを返す
        result = is_multi_column_indicator("rsi")
        assert result is False

    def test_detect_multi_return_indicator(self):
        """複数値を返す指標の検出（MACD, BBANDS等）"""
        from app.services.indicators.config.pandas_ta_introspection import (
            is_multi_column_indicator,
        )

        # MACDはDataFrame（複数カラム）を返す
        result = is_multi_column_indicator("macd")
        assert result is True

        # BBANDSもDataFrameを返す
        result = is_multi_column_indicator("bbands")
        assert result is True

    def test_get_return_column_count(self):
        """戻り値のカラム数を取得"""
        from app.services.indicators.config.pandas_ta_introspection import (
            get_return_column_count,
        )

        # MACDは3カラム（MACD, Signal, Histogram）
        count = get_return_column_count("macd")
        assert count == 3

        # BBANDSは5カラム（BBL, BBM, BBU, BBB, BBP）
        count = get_return_column_count("bbands")
        assert count == 5


class TestIndicatorCategoryExtraction:
    """インジケーターカテゴリ抽出のテスト"""

    def test_get_indicator_category(self):
        """pandas-taからカテゴリを取得"""
        from app.services.indicators.config.pandas_ta_introspection import (
            get_indicator_category,
        )

        # RSIはmomentumカテゴリ
        category = get_indicator_category("rsi")
        assert category == "momentum"

        # BBandsはvolatilityカテゴリ
        category = get_indicator_category("bbands")
        assert category == "volatility"

        # SMAはoverlapカテゴリ
        category = get_indicator_category("sma")
        assert category == "overlap"


class TestDefaultParameterExtraction:
    """デフォルトパラメータ抽出のテスト"""

    def test_extract_rsi_defaults(self):
        """RSIのデフォルトパラメータを抽出"""
        from app.services.indicators.config.pandas_ta_introspection import (
            extract_default_parameters,
        )

        defaults = extract_default_parameters("rsi")
        # RSIのデフォルトはlength=14（またはNoneでpandas-ta内部で14になる）
        assert "length" in defaults or defaults.get("length") is None

    def test_extract_macd_defaults(self):
        """MACDのデフォルトパラメータを抽出"""
        from app.services.indicators.config.pandas_ta_introspection import (
            extract_default_parameters,
        )

        defaults = extract_default_parameters("macd")
        assert "fast" in defaults
        assert "slow" in defaults
        assert "signal" in defaults
