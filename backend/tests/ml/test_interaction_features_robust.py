"""
InteractionFeatureCalculator の堅牢性テスト

数値計算の精度、ゼロ除算、オーバーフロー、エッジケースなどの
堅牢性に関する包括的なテストを実施します。
"""

import pytest
import pandas as pd
import numpy as np
import psutil
import gc
from unittest.mock import patch
import warnings

from app.core.services.ml.feature_engineering.interaction_features import InteractionFeatureCalculator


class TestInteractionFeatureCalculatorRobust:
    """InteractionFeatureCalculator の堅牢性テストクラス"""
    
    @pytest.fixture
    def calculator(self):
        """InteractionFeatureCalculator インスタンスを生成"""
        return InteractionFeatureCalculator()
    
    @pytest.fixture
    def extreme_values_data(self):
        """極端な値を含むテストデータ"""
        dates = pd.date_range('2024-01-01', periods=100, freq='h', tz='UTC')
        
        return pd.DataFrame({
            'Close': np.random.uniform(40000, 50000, 100),
            'Volume': np.random.uniform(1000, 10000, 100),
            
            # 極端な値を含む特徴量
            'Price_Change_5': np.concatenate([
                [0.0, np.inf, -np.inf, np.nan],  # 特殊値
                np.random.uniform(-0.1, 0.1, 96)  # 通常値
            ]),
            'Price_Momentum_14': np.concatenate([
                [1e-10, 1e10, -1e10, 0.0],  # 極端な値
                np.random.uniform(-0.1, 0.1, 96)  # 通常値
            ]),
            'ATR_20': np.concatenate([
                [0.0, 1e6, 1e-6, np.nan],  # 極端な値
                np.random.uniform(100, 1000, 96)  # 通常値
            ]),
            'Volatility_Spike': np.random.choice([True, False], 100),
            'Volume_Ratio': np.concatenate([
                [0.0, np.inf, -1.0, np.nan],  # 極端な値
                np.random.uniform(0.5, 2.0, 96)  # 通常値
            ]),
            'Volume_Spike': np.random.choice([True, False], 100),
            'RSI': np.concatenate([
                [0.0, 100.0, -10.0, 110.0],  # 境界値外
                np.random.uniform(20, 80, 96)  # 通常値
            ]),
            'Trend_Strength': np.concatenate([
                [-2.0, 2.0, np.inf, -np.inf],  # 極端な値
                np.random.uniform(-1, 1, 96)  # 通常値
            ]),
            'Breakout_Strength': np.concatenate([
                [-1.0, 2.0, np.nan, 0.0],  # 境界値外
                np.random.uniform(0, 1, 96)  # 通常値
            ]),
            'FR_Normalized': np.concatenate([
                [0.0, 1.0, -1.0, np.nan],  # 極端な値
                np.random.uniform(-0.01, 0.01, 96)  # 通常値
            ]),
            'FR_Extreme_High': np.random.choice([True, False], 100),
            'FR_Extreme_Low': np.random.choice([True, False], 100),
            'OI_Change_Rate': np.concatenate([
                [0.0, np.inf, -np.inf, np.nan],  # 特殊値
                np.random.uniform(-0.1, 0.1, 96)  # 通常値
            ]),
            'OI_Trend': np.concatenate([
                [-10.0, 10.0, 0.0, np.nan],  # 極端な値
                np.random.uniform(-1, 1, 96)  # 通常値
            ])
        }, index=dates)
    
    def test_extreme_numerical_values(self, calculator, extreme_values_data):
        """極端な数値での計算テスト"""
        result = calculator.calculate_interaction_features(extreme_values_data)
        
        # 結果が生成されることを確認
        assert len(result) == len(extreme_values_data)
        
        # 相互作用特徴量が追加されていることを確認
        interaction_features = calculator.get_feature_names()
        for feature in interaction_features:
            assert feature in result.columns
        
        # 無限大値やNaN値が適切に処理されていることを確認
        for feature in interaction_features:
            if feature in result.columns:
                # 無限大値がないことを確認
                infinite_count = np.isinf(result[feature]).sum()
                assert infinite_count == 0, f"Feature {feature} contains {infinite_count} infinite values"
                
                # NaN値の数が合理的であることを確認（元データのNaNに起因するもの以外）
                nan_count = result[feature].isna().sum()
                assert nan_count <= 10, f"Feature {feature} contains too many NaN values: {nan_count}"
    
    def test_zero_division_protection(self, calculator):
        """ゼロ除算保護のテスト"""
        dates = pd.date_range('2024-01-01', periods=10, freq='h', tz='UTC')
        
        # ゼロ値を多く含むデータ
        zero_data = pd.DataFrame({
            'Close': [45000] * 10,
            'Volume': [5000] * 10,
            'Price_Change_5': [0.0] * 10,  # 全てゼロ
            'Price_Momentum_14': [0.0] * 10,  # 全てゼロ
            'ATR_20': [0.0] * 10,  # 全てゼロ
            'Volatility_Spike': [False] * 10,
            'Volume_Ratio': [0.0] * 10,  # 全てゼロ
            'Volume_Spike': [False] * 10,
            'RSI': [50.0] * 10,
            'Trend_Strength': [0.0] * 10,  # 全てゼロ
            'Breakout_Strength': [0.0] * 10,  # 全てゼロ
            'FR_Normalized': [0.0] * 10,  # 全てゼロ
            'FR_Extreme_High': [False] * 10,
            'FR_Extreme_Low': [False] * 10,
            'OI_Change_Rate': [0.0] * 10,  # 全てゼロ
            'OI_Trend': [0.0] * 10  # 全てゼロ
        }, index=dates)
        
        # ゼロ除算エラーが発生しないことを確認
        result = calculator.calculate_interaction_features(zero_data)
        
        # 結果が生成されることを確認
        assert len(result) == 10
        
        # 相互作用特徴量が全てゼロまたは有限値であることを確認
        interaction_features = calculator.get_feature_names()
        for feature in interaction_features:
            if feature in result.columns:
                assert np.isfinite(result[feature]).all() or (result[feature] == 0).all(), \
                    f"Feature {feature} contains non-finite values with zero inputs"
    
    def test_numerical_precision(self, calculator):
        """数値精度のテスト"""
        dates = pd.date_range('2024-01-01', periods=5, freq='h', tz='UTC')
        
        # 高精度が必要な小さな値
        precision_data = pd.DataFrame({
            'Close': [45000.123456789] * 5,
            'Volume': [5000.987654321] * 5,
            'Price_Change_5': [1e-10, -1e-10, 1e-15, -1e-15, 0.0],
            'Price_Momentum_14': [1e-8, -1e-8, 1e-12, -1e-12, 0.0],
            'ATR_20': [1e-6, 1e-6, 1e-6, 1e-6, 1e-6],
            'Volatility_Spike': [True, False, True, False, True],
            'Volume_Ratio': [1.000000001, 0.999999999, 1.0, 1.0, 1.0],
            'Volume_Spike': [True, False, True, False, True],
            'RSI': [50.000000001, 49.999999999, 50.0, 50.0, 50.0],
            'Trend_Strength': [1e-10, -1e-10, 0.0, 0.0, 0.0],
            'Breakout_Strength': [1e-10, 1e-10, 0.0, 0.0, 0.0],
            'FR_Normalized': [1e-12, -1e-12, 0.0, 0.0, 0.0],
            'FR_Extreme_High': [True, False, True, False, True],
            'FR_Extreme_Low': [False, True, False, True, False],
            'OI_Change_Rate': [1e-15, -1e-15, 0.0, 0.0, 0.0],
            'OI_Trend': [1e-10, -1e-10, 0.0, 0.0, 0.0]
        }, index=dates)
        
        result = calculator.calculate_interaction_features(precision_data)
        
        # 数値精度が保たれていることを確認
        for feature in calculator.get_feature_names():
            if feature in result.columns:
                # 極端に小さな値が適切に処理されていることを確認
                values = result[feature].values
                assert np.all(np.isfinite(values)), f"Feature {feature} contains non-finite values"
                
                # 精度が保たれていることを確認（相対誤差が小さい）
                if not np.all(values == 0):
                    relative_error = np.abs(values - np.round(values, 15)) / (np.abs(values) + 1e-15)
                    assert np.all(relative_error < 1e-10), f"Feature {feature} has precision issues"
    
    def test_data_type_consistency(self, calculator):
        """データ型の一貫性テスト"""
        dates = pd.date_range('2024-01-01', periods=10, freq='h', tz='UTC')
        
        # 異なるデータ型を含むデータ
        mixed_type_data = pd.DataFrame({
            'Close': np.array([45000] * 10, dtype=np.float32),  # float32
            'Volume': np.array([5000] * 10, dtype=np.int64),    # int64
            'Price_Change_5': np.array([0.01] * 10, dtype=np.float64),  # float64
            'Price_Momentum_14': np.array([0.02] * 10, dtype=np.float32),  # float32
            'ATR_20': np.array([100] * 10, dtype=np.int32),     # int32
            'Volatility_Spike': np.array([True] * 10, dtype=bool),  # bool
            'Volume_Ratio': np.array([1.5] * 10, dtype=np.float64),  # float64
            'Volume_Spike': np.array([False] * 10, dtype=bool),  # bool
            'RSI': np.array([50] * 10, dtype=np.int16),         # int16
            'Trend_Strength': np.array([0.5] * 10, dtype=np.float32),  # float32
            'Breakout_Strength': np.array([0.3] * 10, dtype=np.float64),  # float64
            'FR_Normalized': np.array([0.001] * 10, dtype=np.float32),  # float32
            'FR_Extreme_High': np.array([True] * 10, dtype=bool),  # bool
            'FR_Extreme_Low': np.array([False] * 10, dtype=bool),  # bool
            'OI_Change_Rate': np.array([0.05] * 10, dtype=np.float64),  # float64
            'OI_Trend': np.array([0.1] * 10, dtype=np.float32)  # float32
        }, index=dates)
        
        result = calculator.calculate_interaction_features(mixed_type_data)
        
        # 結果のデータ型が適切であることを確認
        interaction_features = calculator.get_feature_names()
        for feature in interaction_features:
            if feature in result.columns:
                # 数値型であることを確認
                assert pd.api.types.is_numeric_dtype(result[feature]), \
                    f"Feature {feature} should be numeric, got {result[feature].dtype}"
                
                # float型であることを確認（計算結果は通常float）
                assert result[feature].dtype in [np.float32, np.float64], \
                    f"Feature {feature} should be float type, got {result[feature].dtype}"
    
    def test_memory_efficiency(self, calculator):
        """メモリ効率性のテスト"""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 大量データでのメモリ使用量テスト
        large_size = 50000  # 5万行
        dates = pd.date_range('2024-01-01', periods=large_size, freq='min', tz='UTC')
        
        large_data = pd.DataFrame({
            'Close': np.random.uniform(40000, 50000, large_size),
            'Volume': np.random.uniform(1000, 10000, large_size),
            'Price_Change_5': np.random.uniform(-0.05, 0.05, large_size),
            'Price_Momentum_14': np.random.uniform(-0.1, 0.1, large_size),
            'ATR_20': np.random.uniform(100, 1000, large_size),
            'Volatility_Spike': np.random.choice([True, False], large_size),
            'Volume_Ratio': np.random.uniform(0.5, 2.0, large_size),
            'Volume_Spike': np.random.choice([True, False], large_size),
            'RSI': np.random.uniform(20, 80, large_size),
            'Trend_Strength': np.random.uniform(-1, 1, large_size),
            'Breakout_Strength': np.random.uniform(0, 1, large_size),
            'FR_Normalized': np.random.uniform(-0.01, 0.01, large_size),
            'FR_Extreme_High': np.random.choice([True, False], large_size),
            'FR_Extreme_Low': np.random.choice([True, False], large_size),
            'OI_Change_Rate': np.random.uniform(-0.1, 0.1, large_size),
            'OI_Trend': np.random.uniform(-1, 1, large_size)
        }, index=dates)
        
        # 処理実行
        result = calculator.calculate_interaction_features(large_data)
        
        # メモリ使用量チェック
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # メモリ効率性の確認（200MB以下）
        assert memory_increase < 200, f"Memory usage too high: {memory_increase:.2f}MB for {large_size} rows"
        
        # 結果の整合性確認
        assert len(result) == large_size
        
        # メモリクリーンアップ
        del large_data, result
        gc.collect()
    
    def test_error_propagation(self, calculator):
        """エラー伝播のテスト"""
        dates = pd.date_range('2024-01-01', periods=5, freq='h', tz='UTC')
        
        # 部分的に不正なデータ
        partial_invalid_data = pd.DataFrame({
            'Close': [45000, 45100, np.nan, 45300, 45400],
            'Volume': [5000, 5100, 5200, np.inf, 5400],
            'Price_Change_5': [0.01, 0.02, 0.03, 0.04, 0.05],
            'Price_Momentum_14': [0.1, 0.2, 0.3, 0.4, 0.5],
            'ATR_20': [100, 200, 300, 400, 500],
            'Volatility_Spike': [True, False, True, False, True],
            'Volume_Ratio': [1.0, 1.1, 1.2, 1.3, 1.4],
            'Volume_Spike': [False, True, False, True, False],
            'RSI': [50, 60, 70, 80, 90],
            'Trend_Strength': [0.1, 0.2, 0.3, 0.4, 0.5],
            'Breakout_Strength': [0.1, 0.2, 0.3, 0.4, 0.5],
            'FR_Normalized': [0.001, 0.002, 0.003, 0.004, 0.005],
            'FR_Extreme_High': [True, False, True, False, True],
            'FR_Extreme_Low': [False, True, False, True, False],
            'OI_Change_Rate': [0.01, 0.02, 0.03, 0.04, 0.05],
            'OI_Trend': [0.1, 0.2, 0.3, 0.4, 0.5]
        }, index=dates)
        
        # 部分的に不正なデータでも処理が完了することを確認
        result = calculator.calculate_interaction_features(partial_invalid_data)
        
        # 結果が生成されることを確認
        assert len(result) == 5
        
        # 有効な行では正しい計算が行われていることを確認
        valid_rows = [0, 1, 4]  # インデックス2と3は不正データ
        for row_idx in valid_rows:
            for feature in calculator.get_feature_names():
                if feature in result.columns:
                    value = result.iloc[row_idx][feature]
                    assert np.isfinite(value) or pd.isna(value), \
                        f"Feature {feature} at row {row_idx} should be finite or NaN, got {value}"


if __name__ == "__main__":
    pytest.main([__file__])
