"""
包括的インジケーターテストスイート

100個以上のインジケーターの初期化・動作テストを実行
"""

import pytest
import numpy as np
import pandas as pd
import talib
from unittest.mock import patch

# 全ての指標クラスをインポート
from app.core.services.indicators import (
    TrendIndicators,
    MomentumIndicators,
    VolatilityIndicators,
    VolumeIndicators,
    PriceTransformIndicators,
    CycleIndicators,
    StatisticsIndicators,
    MathTransformIndicators,
    MathOperatorsIndicators,
    PatternRecognitionIndicators,
    TALibError,
)


class TestComprehensiveIndicators:
    """包括的インジケーターテスト"""

    @pytest.fixture
    def comprehensive_sample_data(self):
        """包括的テスト用のサンプルデータ"""
        np.random.seed(42)
        size = 200  # より多くのデータポイント
        
        # より現実的な価格データを生成
        base_price = 100
        price_changes = np.random.normal(0, 0.02, size)  # 2%の標準偏差
        close_prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = close_prices[-1] * (1 + change)
            close_prices.append(max(new_price, 1))  # 価格が1以下にならないように
        
        close_prices = np.array(close_prices)
        
        # OHLC データを生成
        high_prices = close_prices * np.random.uniform(1.0, 1.05, size)
        low_prices = close_prices * np.random.uniform(0.95, 1.0, size)
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        return {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": np.random.uniform(1000, 10000, size),
        }

    def test_all_indicators_can_be_imported(self):
        """全てのインジケータークラスがインポート可能であることを確認"""
        indicator_classes = [
            TrendIndicators,
            MomentumIndicators,
            VolatilityIndicators,
            VolumeIndicators,
            PriceTransformIndicators,
            CycleIndicators,
            StatisticsIndicators,
            MathTransformIndicators,
            MathOperatorsIndicators,
            PatternRecognitionIndicators,
        ]
        
        for indicator_class in indicator_classes:
            assert indicator_class is not None
            # クラスが適切なメソッドを持っていることを確認
            assert hasattr(indicator_class, '__name__')

    def test_trend_indicators_comprehensive(self, comprehensive_sample_data):
        """トレンド系インジケーターの包括的テスト"""
        data = comprehensive_sample_data
        
        # 基本的なトレンド系インジケーター
        indicators_to_test = [
            ("sma", {"period": 20}),
            ("ema", {"period": 20}),
            ("wma", {"period": 20}),
            ("trima", {"period": 20}),
            ("kama", {"period": 30}),
        ]
        
        for indicator_name, params in indicators_to_test:
            try:
                method = getattr(TrendIndicators, indicator_name)
                if indicator_name in ["sar"]:
                    result = method(data["high"], data["low"], **params)
                else:
                    result = method(data["close"], **params)
                
                assert isinstance(result, np.ndarray)
                assert len(result) == len(data["close"])
                assert not np.all(np.isnan(result))  # 全てがNaNではないことを確認
                
            except Exception as e:
                pytest.fail(f"{indicator_name} failed: {str(e)}")

    def test_momentum_indicators_comprehensive(self, comprehensive_sample_data):
        """モメンタム系インジケーターの包括的テスト"""
        data = comprehensive_sample_data
        
        # 基本的なモメンタム系インジケーター
        indicators_to_test = [
            ("rsi", {"period": 14}),
            ("adx", {"period": 14}),
            ("cci", {"period": 14}),
            ("willr", {"period": 14}),
            ("adxr", {"period": 14}),
            ("roc", {"period": 10}),
            ("rocp", {"period": 10}),
            ("rocr", {"period": 10}),
            ("rocr100", {"period": 10}),
            ("trix", {"period": 30}),
        ]
        
        for indicator_name, params in indicators_to_test:
            try:
                method = getattr(MomentumIndicators, indicator_name)
                if indicator_name in ["adx", "adxr", "cci", "willr"]:
                    result = method(data["high"], data["low"], data["close"], **params)
                elif indicator_name == "mfi":
                    result = method(data["high"], data["low"], data["close"], data["volume"], **params)
                else:
                    result = method(data["close"], **params)
                
                assert isinstance(result, np.ndarray)
                assert len(result) == len(data["close"])
                
            except Exception as e:
                pytest.fail(f"{indicator_name} failed: {str(e)}")

    def test_complex_momentum_indicators(self, comprehensive_sample_data):
        """複雑な戻り値を持つモメンタム系インジケーターのテスト"""
        data = comprehensive_sample_data

        # MACDEXT (3つの戻り値)
        try:
            macd, signal, histogram = MomentumIndicators.macdext(data["close"])
            assert isinstance(macd, np.ndarray)
            assert isinstance(signal, np.ndarray)
            assert isinstance(histogram, np.ndarray)
            assert len(macd) == len(data["close"])
        except Exception as e:
            pytest.fail(f"MACDEXT failed: {str(e)}")

        # MACDFIX (3つの戻り値)
        try:
            macd, signal, histogram = MomentumIndicators.macdfix(data["close"])
            assert isinstance(macd, np.ndarray)
            assert isinstance(signal, np.ndarray)
            assert isinstance(histogram, np.ndarray)
            assert len(macd) == len(data["close"])
        except Exception as e:
            pytest.fail(f"MACDFIX failed: {str(e)}")

        # STOCHRSI (2つの戻り値)
        try:
            fastk, fastd = MomentumIndicators.stochrsi(data["close"])
            assert isinstance(fastk, np.ndarray)
            assert isinstance(fastd, np.ndarray)
            assert len(fastk) == len(data["close"])
        except Exception as e:
            pytest.fail(f"STOCHRSI failed: {str(e)}")

    def test_volume_indicators_comprehensive(self, comprehensive_sample_data):
        """出来高系インジケーターの包括的テスト"""
        data = comprehensive_sample_data
        
        indicators_to_test = [
            ("obv", {}),
            ("ad", {}),
        ]
        
        for indicator_name, params in indicators_to_test:
            try:
                method = getattr(VolumeIndicators, indicator_name)
                if indicator_name == "obv":
                    result = method(data["close"], data["volume"], **params)
                else:  # ad
                    result = method(data["high"], data["low"], data["close"], data["volume"], **params)
                
                assert isinstance(result, np.ndarray)
                assert len(result) == len(data["close"])
                
            except Exception as e:
                pytest.fail(f"{indicator_name} failed: {str(e)}")

    def test_statistics_indicators_comprehensive(self, comprehensive_sample_data):
        """統計系インジケーターの包括的テスト"""
        data = comprehensive_sample_data
        
        indicators_to_test = [
            ("linearreg", {"period": 14}),
            ("stddev", {"period": 5}),
            ("tsf", {"period": 14}),
            ("var", {"period": 5}),
        ]
        
        for indicator_name, params in indicators_to_test:
            try:
                method = getattr(StatisticsIndicators, indicator_name)
                if indicator_name in ["beta", "correl"]:
                    result = method(data["high"], data["low"], **params)
                else:
                    result = method(data["close"], **params)
                
                assert isinstance(result, np.ndarray)
                assert len(result) == len(data["close"])
                
            except Exception as e:
                pytest.fail(f"{indicator_name} failed: {str(e)}")

    def test_math_transform_indicators_comprehensive(self, comprehensive_sample_data):
        """数学変換系インジケーターの包括的テスト"""
        data = comprehensive_sample_data
        
        # 正の値のみを使用（対数関数などのため）
        positive_data = np.abs(data["close"]) + 1
        
        indicators_to_test = [
            "sqrt", "ln", "log10", "exp", "ceil", "floor",
            "sin", "cos", "tan", "asin", "acos", "atan"
        ]
        
        for indicator_name in indicators_to_test:
            try:
                method = getattr(MathTransformIndicators, indicator_name)
                
                # 三角関数の逆関数は値域を制限
                if indicator_name in ["asin", "acos"]:
                    test_data = np.clip(positive_data / np.max(positive_data), -1, 1)
                elif indicator_name in ["atan"]:
                    test_data = positive_data
                else:
                    test_data = positive_data
                
                result = method(test_data)
                
                assert isinstance(result, np.ndarray)
                assert len(result) == len(test_data)
                
            except Exception as e:
                pytest.fail(f"{indicator_name} failed: {str(e)}")

    def test_math_operators_indicators_comprehensive(self, comprehensive_sample_data):
        """数学演算子系インジケーターの包括的テスト"""
        data = comprehensive_sample_data
        
        # 単項演算子（期間ベース）
        period_based_indicators = [
            ("max", {"period": 30}),
            ("min", {"period": 30}),
            ("sum", {"period": 30}),
        ]
        
        for indicator_name, params in period_based_indicators:
            try:
                method = getattr(MathOperatorsIndicators, indicator_name)
                result = method(data["close"], **params)
                
                assert isinstance(result, np.ndarray)
                assert len(result) == len(data["close"])
                
            except Exception as e:
                pytest.fail(f"{indicator_name} failed: {str(e)}")
        
        # 二項演算子
        binary_indicators = ["add", "sub", "mult", "div"]
        
        for indicator_name in binary_indicators:
            try:
                method = getattr(MathOperatorsIndicators, indicator_name)
                result = method(data["high"], data["low"])
                
                assert isinstance(result, np.ndarray)
                assert len(result) == len(data["close"])
                
            except Exception as e:
                pytest.fail(f"{indicator_name} failed: {str(e)}")

    def test_pattern_recognition_indicators_comprehensive(self, comprehensive_sample_data):
        """パターン認識系インジケーターの包括的テスト"""
        data = comprehensive_sample_data
        
        indicators_to_test = [
            "cdl_doji", "cdl_hammer", "cdl_hanging_man", "cdl_shooting_star",
            "cdl_engulfing", "cdl_harami", "cdl_piercing"
        ]
        
        for indicator_name in indicators_to_test:
            try:
                method = getattr(PatternRecognitionIndicators, indicator_name)
                result = method(data["open"], data["high"], data["low"], data["close"])
                
                assert isinstance(result, np.ndarray)
                assert len(result) == len(data["close"])
                # パターン認識の結果は整数値（通常-100, 0, 100だが、他の値もあり得る）
                unique_values = np.unique(result[~np.isnan(result)])
                # 結果が整数であることを確認
                assert all(isinstance(val, (int, np.integer)) or val == int(val) for val in unique_values)
                
            except Exception as e:
                pytest.fail(f"{indicator_name} failed: {str(e)}")

    def test_indicator_registry_comprehensive_initialization(self):
        """インジケーターレジストリの包括的初期化テスト"""
        from app.core.services.indicators.config.indicator_definitions import initialize_all_indicators
        from app.core.services.indicators.config.indicator_config import indicator_registry
        
        # 初期化前のレジストリをクリア
        indicator_registry._configs.clear()
        
        # 初期化実行
        initialize_all_indicators()
        
        # 登録されたインジケーターの確認
        registered_indicators = list(indicator_registry._configs.keys())
        
        # 90個以上のインジケーターが登録されていることを確認（段階的拡張）
        assert len(registered_indicators) >= 90, f"登録されたインジケーター数: {len(registered_indicators)}"
        
        # カテゴリ別の確認
        categories = {}
        for indicator_name in registered_indicators:
            config = indicator_registry.get_indicator_config(indicator_name)
            if config:
                category = config.category
                if category not in categories:
                    categories[category] = []
                categories[category].append(indicator_name)
        
        # 各カテゴリに適切な数のインジケーターが含まれていることを確認
        expected_categories = [
            "trend", "momentum", "volatility", "volume", 
            "price_transform", "cycle", "statistics", 
            "math_transform", "math_operators", "pattern_recognition"
        ]
        
        for category in expected_categories:
            assert category in categories, f"カテゴリ {category} が見つかりません"
            assert len(categories[category]) > 0, f"カテゴリ {category} にインジケーターがありません"
        
        # 登録されたインジケーターの一覧を出力（デバッグ用）
        print(f"\n=== 登録されたインジケーター総数: {len(registered_indicators)} ===")
        for category, indicators in categories.items():
            print(f"{category}: {len(indicators)}個 - {sorted(indicators)}")

    def test_error_handling_comprehensive(self):
        """エラーハンドリングの包括的テスト"""
        
        # 不正なデータでのテスト
        invalid_data = np.array([])
        
        with pytest.raises((TALibError, ValueError)):
            TrendIndicators.sma(invalid_data, period=20)
        
        # 期間が長すぎる場合のテスト
        short_data = np.array([1, 2, 3])
        
        with pytest.raises((TALibError, ValueError)):
            TrendIndicators.sma(short_data, period=20)
        
        # データ長が一致しない場合のテスト
        data1 = np.array([1, 2, 3, 4, 5])
        data2 = np.array([1, 2, 3])
        
        with pytest.raises((TALibError, ValueError)):
            MathOperatorsIndicators.add(data1, data2)
