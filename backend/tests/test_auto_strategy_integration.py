"""
オートストラテジーとの統合テスト

型変換なし実装でのオートストラテジー機能の動作を確認するテストです。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from app.services.auto_strategy.calculators.indicator_calculator import IndicatorCalculator
from app.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
from app.services.auto_strategy.models.gene_validation import GeneValidator


class TestAutoStrategyIntegration:
    """オートストラテジー統合テストクラス"""

    @pytest.fixture
    def market_data(self):
        """市場データのモック"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='1H')
        
        # より現実的な価格データ
        base_price = 50000  # BTC価格想定
        returns = np.random.normal(0, 0.02, 200)
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 1000))  # 最低価格設定
        
        closes = np.array(prices)
        highs = closes * (1 + np.abs(np.random.normal(0, 0.005, 200)))
        lows = closes * (1 - np.abs(np.random.normal(0, 0.005, 200)))
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        volumes = np.random.randint(100, 1000, 200)
        
        return pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes
        }, index=dates)

    @pytest.fixture
    def mock_backtest_data(self, market_data):
        """backtesting.py用のモックデータ"""
        class MockData:
            def __init__(self, df):
                self.df = df
        return MockData(market_data)

    def test_indicator_calculator_basic_functionality(self, mock_backtest_data):
        """IndicatorCalculatorの基本機能テスト"""
        calculator = IndicatorCalculator()
        
        # 基本的なテクニカル指標
        basic_indicators = [
            ('SMA', {'period': 20}),
            ('EMA', {'period': 20}),
            ('RSI', {'period': 14}),
            ('MACD', {'fast': 12, 'slow': 26, 'signal': 9}),
            ('BB', {'period': 20, 'std': 2}),
            ('ATR', {'period': 14}),
        ]
        
        for indicator_name, params in basic_indicators:
            try:
                result = calculator.calculate_indicator(
                    mock_backtest_data, indicator_name, params
                )
                
                if isinstance(result, tuple):
                    # 複数出力の場合
                    assert len(result) >= 2, f"{indicator_name} should have multiple outputs"
                    for output in result:
                        assert isinstance(output, np.ndarray), f"{indicator_name} output should be numpy array"
                        assert len(output) == len(mock_backtest_data.df), f"{indicator_name} length mismatch"
                else:
                    # 単一出力の場合
                    assert isinstance(result, np.ndarray), f"{indicator_name} should return numpy array"
                    assert len(result) == len(mock_backtest_data.df), f"{indicator_name} length mismatch"
                    
            except Exception as e:
                pytest.fail(f"{indicator_name} calculation failed: {e}")

    def test_indicator_calculator_math_operators(self, mock_backtest_data):
        """IndicatorCalculatorの数学演算子テスト"""
        calculator = IndicatorCalculator()
        
        # 数学演算子系指標
        math_indicators = [
            ('ADD', {'data0': mock_backtest_data.df['Close'], 'data1': mock_backtest_data.df['Volume']/1000}),
            ('SUB', {'data0': mock_backtest_data.df['High'], 'data1': mock_backtest_data.df['Low']}),
            ('MULT', {'data0': mock_backtest_data.df['Close'], 'data1': 1.01}),
            ('DIV', {'data0': mock_backtest_data.df['Close'], 'data1': mock_backtest_data.df['Open']}),
            ('MAX', {'period': 20}),
            ('MIN', {'period': 20}),
            ('SUM', {'period': 20}),
        ]
        
        for indicator_name, params in math_indicators:
            try:
                result = calculator.calculate_indicator(
                    mock_backtest_data, indicator_name, params
                )
                assert isinstance(result, np.ndarray), f"{indicator_name} should return numpy array"
                assert len(result) == len(mock_backtest_data.df), f"{indicator_name} length mismatch"
                
            except Exception as e:
                pytest.fail(f"{indicator_name} calculation failed: {e}")

    def test_indicator_calculator_error_handling(self, mock_backtest_data):
        """IndicatorCalculatorのエラーハンドリングテスト"""
        calculator = IndicatorCalculator()
        
        # 存在しない指標
        with pytest.raises(Exception):
            calculator.calculate_indicator(mock_backtest_data, 'NONEXISTENT', {})
        
        # 無効なパラメータ
        with pytest.raises(Exception):
            calculator.calculate_indicator(mock_backtest_data, 'SMA', {'period': 0})
        
        # 期間が長すぎる
        with pytest.raises(Exception):
            calculator.calculate_indicator(mock_backtest_data, 'SMA', {'period': 1000})

    def test_gene_validator_indicator_validation(self):
        """GeneValidatorの指標検証テスト"""
        validator = GeneValidator()
        
        # 有効な指標名
        valid_indicators = ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR', 'STOCH']
        for indicator in valid_indicators:
            assert validator.is_valid_indicator_name(indicator), f"{indicator} should be valid"
        
        # パラメータ付き指標名
        parameterized_indicators = ['SMA_20', 'RSI_14', 'MACD_12_26_9']
        for indicator in parameterized_indicators:
            assert validator.is_valid_indicator_name(indicator), f"{indicator} should be valid"
        
        # 無効な指標名
        invalid_indicators = ['', '   ', 'INVALID', 'HT_DCPERIOD', 'HT_SINE']
        for indicator in invalid_indicators:
            assert not validator.is_valid_indicator_name(indicator), f"{indicator} should be invalid"

    def test_smart_condition_generator_without_cycle(self):
        """SmartConditionGeneratorでcycle系指標が除外されていることを確認"""
        generator = SmartConditionGenerator()
        
        # cycle系指標が含まれていないことを確認
        from app.services.auto_strategy.generators.smart_condition_generator import INDICATOR_CHARACTERISTICS
        
        cycle_indicators = ['HT_DCPERIOD', 'HT_DCPHASE', 'HT_SINE', 'HT_PHASOR', 'HT_TRENDMODE']
        for indicator in cycle_indicators:
            assert indicator not in INDICATOR_CHARACTERISTICS, f"{indicator} should not be in characteristics"

    def test_ml_indicator_integration(self, mock_backtest_data):
        """ML指標との統合テスト"""
        calculator = IndicatorCalculator()
        
        # ML指標のモック
        with patch.object(calculator.ml_orchestrator, 'calculate_single_ml_indicator') as mock_ml:
            mock_ml.return_value = np.random.rand(len(mock_backtest_data.df))
            
            # ML指標の計算
            ml_indicators = ['ML_UP_PROB', 'ML_DOWN_PROB', 'ML_RANGE_PROB']
            for indicator in ml_indicators:
                try:
                    result = calculator.calculate_indicator(
                        mock_backtest_data, indicator, {}
                    )
                    assert isinstance(result, np.ndarray), f"{indicator} should return numpy array"
                    assert len(result) == len(mock_backtest_data.df), f"{indicator} length mismatch"
                    
                except Exception as e:
                    pytest.fail(f"{indicator} calculation failed: {e}")

    def test_performance_with_large_dataset(self):
        """大規模データセットでのパフォーマンステスト"""
        # 大規模データセット（1年分の1分足データ）
        large_data = pd.DataFrame({
            'Open': np.random.rand(525600) * 100 + 50000,
            'High': np.random.rand(525600) * 100 + 50100,
            'Low': np.random.rand(525600) * 100 + 49900,
            'Close': np.random.rand(525600) * 100 + 50000,
            'Volume': np.random.randint(100, 1000, 525600)
        })
        
        class MockLargeData:
            def __init__(self, df):
                self.df = df
        
        mock_large_data = MockLargeData(large_data)
        calculator = IndicatorCalculator()
        
        import time
        
        # パフォーマンステスト
        start_time = time.time()
        
        # 複数の指標を計算
        indicators_to_test = [
            ('SMA', {'period': 20}),
            ('RSI', {'period': 14}),
            ('MACD', {'fast': 12, 'slow': 26, 'signal': 9}),
        ]
        
        for indicator_name, params in indicators_to_test:
            result = calculator.calculate_indicator(mock_large_data, indicator_name, params)
            assert len(result) == len(large_data) or isinstance(result, tuple)
        
        elapsed_time = time.time() - start_time
        
        # 処理時間が合理的であることを確認（30秒以内）
        assert elapsed_time < 30, f"Processing took too long: {elapsed_time:.2f} seconds"

    def test_memory_usage_optimization(self, mock_backtest_data):
        """メモリ使用量最適化のテスト"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        calculator = IndicatorCalculator()
        
        # 大量の計算を実行
        for i in range(50):
            calculator.calculate_indicator(mock_backtest_data, 'SMA', {'period': 20})
            calculator.calculate_indicator(mock_backtest_data, 'RSI', {'period': 14})
            calculator.calculate_indicator(mock_backtest_data, 'MACD', {'fast': 12, 'slow': 26, 'signal': 9})
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # メモリ増加が50MB以下であることを確認
        assert memory_increase < 50 * 1024 * 1024, f"Memory increase too large: {memory_increase / 1024 / 1024:.2f} MB"

    def test_data_type_consistency_across_components(self, mock_backtest_data):
        """コンポーネント間でのデータ型一貫性テスト"""
        calculator = IndicatorCalculator()
        
        # IndicatorCalculatorでの計算
        calc_result = calculator.calculate_indicator(mock_backtest_data, 'SMA', {'period': 20})
        
        # TechnicalIndicatorServiceでの計算
        from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
        service = TechnicalIndicatorService()
        service_result = service.calculate_indicator(mock_backtest_data.df, 'SMA', {'period': 20})
        
        # 結果が同じであることを確認
        np.testing.assert_array_almost_equal(calc_result, service_result, decimal=10)

    def test_edge_cases_handling(self, mock_backtest_data):
        """エッジケースの処理テスト"""
        calculator = IndicatorCalculator()
        
        # 最小データセット
        min_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [95, 96],
            'Close': [102, 103],
            'Volume': [1000, 1100]
        })
        
        class MockMinData:
            def __init__(self, df):
                self.df = df
        
        mock_min_data = MockMinData(min_data)
        
        # 短期間指標での計算
        try:
            result = calculator.calculate_indicator(mock_min_data, 'SMA', {'period': 2})
            assert isinstance(result, np.ndarray)
            assert len(result) == 2
        except Exception as e:
            pytest.fail(f"Minimum dataset calculation failed: {e}")
        
        # 期間が長すぎる場合
        with pytest.raises(Exception):
            calculator.calculate_indicator(mock_min_data, 'SMA', {'period': 10})
