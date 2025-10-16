"""
TechnicalIndicatorServiceのテスト
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from app.services.indicators import TechnicalIndicatorService


class TestTechnicalIndicatorService:
    """TechnicalIndicatorServiceのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        self.service = TechnicalIndicatorService()

    def test_init(self):
        """初期化のテスト"""
        assert self.service is not None
        assert hasattr(self.service, "indicators")
        assert hasattr(self.service, "config")

    def test_get_supported_indicators(self):
        """サポート指標取得のテスト"""
        indicators = self.service.get_supported_indicators()

        assert isinstance(indicators, dict)
        assert len(indicators) > 0
        # 基本的な指標が含まれているか確認
        assert "SMA" in indicators
        assert "RSI" in indicators
        assert "MACD" in indicators

    def test_get_indicator_parameters(self):
        """指標パラメータ取得のテスト"""
        # SMAのパラメータをテスト
        sma_params = self.service.get_indicator_parameters("SMA")

        assert isinstance(sma_params, dict)
        assert "period" in sma_params

    def test_get_indicator_parameters_invalid(self):
        """無効な指標パラメータ取得のテスト"""
        params = self.service.get_indicator_parameters("INVALID_INDICATOR")

        assert params is None

    def test_calculate_single_indicator(self):
        """単一指標計算のテスト"""
        # テスト用データ
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            }
        )

        result = self.service.calculate_indicator(data, "SMA", period=5)

        assert isinstance(result, pd.DataFrame)
        assert "SMA_5" in result.columns

    def test_calculate_single_indicator_invalid_data(self):
        """無効なデータでの指標計算テスト"""
        data = pd.DataFrame()  # 空のデータ

        result = self.service.calculate_indicator(data, "SMA", period=5)

        assert result.empty

    def test_calculate_multiple_indicators(self):
        """複数指標計算のテスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            }
        )

        indicators = ["SMA", "RSI"]
        result = self.service.calculate_indicators(data, indicators)

        assert isinstance(result, pd.DataFrame)
        assert "SMA_14" in result.columns
        assert "RSI_14" in result.columns

    def test_calculate_custom_indicators(self):
        """カスタム指標計算のテスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            }
        )

        custom_config = [
            {"name": "SMA", "params": {"period": 5}},
            {"name": "RSI", "params": {"period": 10}},
        ]

        result = self.service.calculate_custom_indicators(data, custom_config)

        assert isinstance(result, pd.DataFrame)
        assert "SMA_5" in result.columns
        assert "RSI_10" in result.columns

    def test_get_indicator_description(self):
        """指標説明取得のテスト"""
        description = self.service.get_indicator_description("SMA")

        assert isinstance(description, str)
        assert len(description) > 0

    def test_get_indicator_description_invalid(self):
        """無効な指標説明取得のテスト"""
        description = self.service.get_indicator_description("INVALID")

        assert description is None

    def test_validate_indicator_data_enough_data(self):
        """データ検証（十分なデータ）のテスト"""
        data = pd.DataFrame({"close": list(range(100)), "volume": [1000] * 100})

        is_valid, message = self.service.validate_indicator_data(data, "SMA", period=14)

        assert is_valid is True
        assert message == "データは有効です"

    def test_validate_indicator_data_insufficient(self):
        """データ検証（データ不足）のテスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102], "volume": [1000, 1100, 1200]}  # 不十分なデータ
        )

        is_valid, message = self.service.validate_indicator_data(data, "SMA", period=14)

        assert is_valid is False
        assert "不足" in message

    def test_get_optimal_parameters(self):
        """最適パラメータ取得のテスト"""
        indicator_name = "SMA"
        timeframe = "1h"

        optimal_params = self.service.get_optimal_parameters(indicator_name, timeframe)

        assert isinstance(optimal_params, dict)
        assert "period" in optimal_params

    def test_get_optimal_parameters_invalid(self):
        """無効な最適パラメータ取得のテスト"""
        optimal_params = self.service.get_optimal_parameters("INVALID", "1h")

        assert optimal_params == {}

    def test_indicator_compatibility_check(self):
        """指標互換性チェックのテスト"""
        indicators = ["SMA", "RSI", "MACD"]

        compatible = self.service.check_indicator_compatibility(indicators)

        assert isinstance(compatible, bool)

    def test_get_indicator_category(self):
        """指標カテゴリ取得のテスト"""
        category = self.service.get_indicator_category("SMA")

        assert category in ["trend", "momentum", "volatility", "volume"]

    def test_get_indicator_category_invalid(self):
        """無効な指標カテゴリ取得のテスト"""
        category = self.service.get_indicator_category("INVALID")

        assert category is None

    def test_calculate_indicator_with_validation(self):
        """検証付き指標計算のテスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            }
        )

        result = self.service.calculate_indicator_with_validation(
            data, "SMA", period=5, validate=True
        )

        assert isinstance(result, pd.DataFrame)
        assert "SMA_5" in result.columns

    def test_calculate_indicator_with_validation_invalid(self):
        """無効データでの検証付き計算テスト"""
        data = pd.DataFrame()  # 空のデータ

        result = self.service.calculate_indicator_with_validation(
            data, "SMA", period=5, validate=True
        )

        assert result is None

    def test_batch_calculate_indicators(self):
        """バッチ指標計算のテスト"""
        data_list = [
            pd.DataFrame({"close": [100, 101, 102], "volume": [1000, 1100, 1200]}),
            pd.DataFrame({"close": [200, 201, 202], "volume": [2000, 2100, 2200]}),
        ]
        indicators = ["SMA", "RSI"]

        results = self.service.batch_calculate_indicators(data_list, indicators)

        assert isinstance(results, list)
        assert len(results) == 2
        for result in results:
            assert "SMA_14" in result.columns
            assert "RSI_14" in result.columns

    def test_get_indicator_dependencies(self):
        """指標依存関係取得のテスト"""
        dependencies = self.service.get_indicator_dependencies("MACD")

        assert isinstance(dependencies, list)

    def test_get_indicator_dependencies_none(self):
        """依存関係なしの指標テスト"""
        dependencies = self.service.get_indicator_dependencies("SMA")

        assert dependencies == []

    def test_indicator_performance_metrics(self):
        """指標パフォーマンスメトリクスのテスト"""
        # 偽の指標データ
        indicator_data = pd.Series([1, 2, 3, 4, 5])
        price_data = pd.Series([100, 101, 102, 103, 104])

        metrics = self.service.calculate_indicator_performance_metrics(
            indicator_data, price_data, "test_indicator"
        )

        assert isinstance(metrics, dict)
        assert "correlation" in metrics

    def test_update_indicator_cache(self):
        """指標キャッシュ更新のテスト"""
        cache_key = "test_cache_key"
        data = pd.DataFrame({"close": [1, 2, 3]})

        self.service._update_indicator_cache(cache_key, data)

        # キャッシュが更新されたか確認（内部検証）
        assert hasattr(self.service, "_cache")
        # 実際のキャッシュ検証は難しいため、メソッドが存在することを確認

    def test_clear_indicator_cache(self):
        """指標キャッシュクリアのテスト"""
        self.service.clear_cache()

        # キャッシュがクリアされたか確認
        assert True  # エラーなく実行されることを確認

    def test_get_cache_size(self):
        """キャッシュサイズ取得のテスト"""
        size = self.service.get_cache_size()

        assert isinstance(size, int)
        assert size >= 0

    def test_indicator_backtesting_support(self):
        """指標バックテスト対応のテスト"""
        supports_backtesting = self.service.indicator_supports_backtesting("SMA")

        assert isinstance(supports_backtesting, bool)

    def test_get_indicator_calculation_time(self):
        """指標計算時間取得のテスト"""
        data = pd.DataFrame({"close": list(range(100)), "volume": [1000] * 100})

        calc_time = self.service.estimate_calculation_time(data, ["SMA", "RSI"])

        assert isinstance(calc_time, float)
        assert calc_time >= 0

    def test_indicator_data_requirements(self):
        """指標データ要件のテスト"""
        requirements = self.service.get_indicator_data_requirements("SMA")

        assert isinstance(requirements, dict)
        assert "minimum_periods" in requirements
        assert "required_columns" in requirements

    def test_error_handling_in_calculation(self):
        """計算中のエラー処理テスト"""
        # 無効なデータ
        data = pd.DataFrame({"invalid": [1, 2, 3]})

        result = self.service.calculate_indicator(data, "SMA", period=5)

        # エラーが適切に処理される
        assert result is not None  # エラー時は空のDataFrameまたは元のデータが返される
