"""
相互作用特徴量計算のテスト

InteractionFeatureCalculatorクラスの機能をテストします。
ボラティリティ×モメンタム、出来高×トレンド、FR×RSI、OI×価格変動の相互作用特徴量計算をテストします。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

# テスト対象のインポート
from app.core.services.ml.feature_engineering.interaction_features import InteractionFeatureCalculator


class TestInteractionFeatureCalculator:
    """InteractionFeatureCalculator のテストクラス"""
    
    @pytest.fixture
    def sample_features_data(self):
        """テスト用の特徴量データを生成"""
        dates = pd.date_range(
            start='2024-01-01 00:00:00',
            periods=100,
            freq='h',
            tz='UTC'
        )
        
        # 基本的なOHLCVデータ
        data = {
            'Open': np.random.uniform(40000, 50000, len(dates)),
            'High': np.random.uniform(40000, 50000, len(dates)),
            'Low': np.random.uniform(40000, 50000, len(dates)),
            'Close': np.random.uniform(40000, 50000, len(dates)),
            'Volume': np.random.uniform(1000, 10000, len(dates)),
            
            # 価格特徴量
            'Price_Change_5': np.random.uniform(-0.05, 0.05, len(dates)),
            'Price_Momentum_14': np.random.uniform(-0.1, 0.1, len(dates)),
            'ATR': np.random.uniform(100, 1000, len(dates)),
            'Volatility_Spike': np.random.choice([True, False], len(dates)),
            
            # 出来高特徴量
            'Volume_Ratio': np.random.uniform(0.5, 2.0, len(dates)),
            'Volume_Spike': np.random.choice([True, False], len(dates)),
            
            # テクニカル特徴量
            'RSI': np.random.uniform(20, 80, len(dates)),
            'Trend_Strength': np.random.uniform(-1, 1, len(dates)),
            'Breakout_Strength': np.random.uniform(0, 1, len(dates)),
            
            # 市場データ特徴量（実際の特徴量名に合わせて修正）
            'FR_Normalized': np.random.uniform(-0.01, 0.01, len(dates)),
            'FR_Extreme_High': np.random.choice([True, False], len(dates)),
            'FR_Extreme_Low': np.random.choice([True, False], len(dates)),
            'OI_Change_Rate': np.random.uniform(-0.1, 0.1, len(dates)),
            'OI_Trend': np.random.uniform(-1, 1, len(dates))
        }
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    @pytest.fixture
    def calculator(self):
        """InteractionFeatureCalculator インスタンスを生成"""
        return InteractionFeatureCalculator()
    
    def test_calculate_interaction_features_basic(self, calculator, sample_features_data):
        """基本的な相互作用特徴量計算のテスト"""
        result = calculator.calculate_interaction_features(sample_features_data)
        
        # 期待される相互作用特徴量カラムが存在することを確認
        expected_columns = [
            'Volatility_Momentum_Interaction',
            'Volatility_Spike_Momentum',
            'Volume_Trend_Interaction',
            'Volume_Breakout',
            'FR_RSI_Extreme',
            'FR_Overbought',
            'FR_Oversold',
            'OI_Price_Divergence',
            'OI_Momentum_Alignment'
        ]
        
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"
        
        # 元の特徴量が保持されていることを確認
        for col in sample_features_data.columns:
            assert col in result.columns, f"Original column {col} should be preserved"
    
    def test_volatility_momentum_interactions(self, calculator, sample_features_data):
        """ボラティリティ×モメンタム相互作用のテスト"""
        result = calculator.calculate_interaction_features(sample_features_data)
        
        # Volatility_Momentum_Interaction の計算確認
        expected_interaction = sample_features_data['ATR'] * sample_features_data['Price_Momentum_14']
        pd.testing.assert_series_equal(
            result['Volatility_Momentum_Interaction'], 
            expected_interaction, 
            check_names=False
        )
        
        # Volatility_Spike_Momentum の計算確認
        expected_spike_momentum = (
            sample_features_data['Volatility_Spike'].astype(float) * 
            sample_features_data['Price_Change_5']
        )
        pd.testing.assert_series_equal(
            result['Volatility_Spike_Momentum'], 
            expected_spike_momentum, 
            check_names=False
        )
    
    def test_volume_trend_interactions(self, calculator, sample_features_data):
        """出来高×トレンド相互作用のテスト"""
        result = calculator.calculate_interaction_features(sample_features_data)
        
        # Volume_Trend_Interaction の計算確認
        expected_volume_trend = sample_features_data['Volume_Ratio'] * sample_features_data['Trend_Strength']
        pd.testing.assert_series_equal(
            result['Volume_Trend_Interaction'], 
            expected_volume_trend, 
            check_names=False
        )
        
        # Volume_Breakout の計算確認
        expected_volume_breakout = (
            sample_features_data['Volume_Spike'].astype(float) * 
            sample_features_data['Breakout_Strength']
        )
        pd.testing.assert_series_equal(
            result['Volume_Breakout'], 
            expected_volume_breakout, 
            check_names=False
        )
    
    def test_fr_rsi_interactions(self, calculator, sample_features_data):
        """FR×RSI相互作用のテスト"""
        result = calculator.calculate_interaction_features(sample_features_data)
        
        # FR_RSI_Extreme の計算確認
        expected_fr_rsi = sample_features_data['FR_Normalized'] * (sample_features_data['RSI'] - 50)
        pd.testing.assert_series_equal(
            result['FR_RSI_Extreme'], 
            expected_fr_rsi, 
            check_names=False
        )
        
        # FR_Overbought の計算確認
        expected_overbought = (
            sample_features_data['FR_High_Flag'].astype(float) * 
            (sample_features_data['RSI'] > 70).astype(float)
        )
        pd.testing.assert_series_equal(
            result['FR_Overbought'], 
            expected_overbought, 
            check_names=False
        )
        
        # FR_Oversold の計算確認
        expected_oversold = (
            sample_features_data['FR_Low_Flag'].astype(float) * 
            (sample_features_data['RSI'] < 30).astype(float)
        )
        pd.testing.assert_series_equal(
            result['FR_Oversold'], 
            expected_oversold, 
            check_names=False
        )
    
    def test_oi_price_interactions(self, calculator, sample_features_data):
        """OI×価格変動相互作用のテスト"""
        result = calculator.calculate_interaction_features(sample_features_data)
        
        # OI_Price_Divergence の計算確認
        expected_oi_price = sample_features_data['OI_Change_Rate'] * sample_features_data['Price_Change_5']
        pd.testing.assert_series_equal(
            result['OI_Price_Divergence'], 
            expected_oi_price, 
            check_names=False
        )
        
        # OI_Momentum_Alignment の計算確認
        expected_oi_momentum = sample_features_data['OI_Trend'] * sample_features_data['Price_Momentum_14']
        pd.testing.assert_series_equal(
            result['OI_Momentum_Alignment'], 
            expected_oi_momentum, 
            check_names=False
        )
    
    def test_missing_columns_handling(self, calculator):
        """必要なカラムが不足している場合のテスト"""
        # 不完全なデータ
        incomplete_data = pd.DataFrame({
            'Close': [100, 200, 300],
            'Volume': [1000, 2000, 3000],
            'ATR': [10, 20, 30]  # 一部の特徴量のみ
        })
        
        # 警告が出力されることを確認
        with patch('app.core.services.ml.feature_engineering.interaction_features.logger') as mock_logger:
            result = calculator.calculate_interaction_features(incomplete_data)
            
            # 警告ログが呼ばれることを確認
            mock_logger.warning.assert_called()
            
            # 元のデータが返されることを確認
            pd.testing.assert_frame_equal(result, incomplete_data)
    
    def test_empty_dataframe_handling(self, calculator):
        """空のDataFrameの処理テスト"""
        empty_df = pd.DataFrame()
        result = calculator.calculate_interaction_features(empty_df)
        assert result.empty
    
    def test_zero_division_handling(self, calculator):
        """ゼロ除算の処理テスト"""
        # ゼロ値を含むデータ
        data_with_zeros = pd.DataFrame({
            'ATR': [0, 100, 200],
            'Price_Momentum_14': [0.1, 0.2, 0.3],
            'Volume_Ratio': [0, 1.5, 2.0],
            'Trend_Strength': [0.5, 0, 1.0],
            'FR_Normalized': [0, 0.01, -0.01],
            'RSI': [50, 70, 30],
            'OI_Change_Rate': [0, 0.05, -0.05],
            'Price_Change_5': [0.01, 0, 0.02]
        })
        
        result = calculator.calculate_interaction_features(data_with_zeros)
        
        # NaN値や無限大値がないことを確認
        for col in result.select_dtypes(include=[np.number]).columns:
            assert not result[col].isna().any(), f"Column {col} contains NaN values"
            assert not np.isinf(result[col]).any(), f"Column {col} contains infinite values"
    
    def test_feature_names_method(self, calculator):
        """特徴量名取得メソッドのテスト"""
        feature_names = calculator.get_feature_names()
        
        # 期待される特徴量名が含まれることを確認
        expected_names = [
            'Volatility_Momentum_Interaction',
            'Volatility_Spike_Momentum',
            'Volume_Trend_Interaction',
            'Volume_Breakout',
            'FR_RSI_Extreme',
            'FR_Overbought',
            'FR_Oversold',
            'OI_Price_Divergence',
            'OI_Momentum_Alignment'
        ]
        
        for name in expected_names:
            assert name in feature_names
    
    def test_data_validation_integration(self, calculator, sample_features_data):
        """データバリデーションとの統合テスト"""
        result = calculator.calculate_interaction_features(sample_features_data)
        
        # NaN値や無限大値がないことを確認
        interaction_columns = calculator.get_feature_names()
        for col in interaction_columns:
            if col in result.columns:
                assert not result[col].isna().any(), f"Column {col} contains NaN values"
                assert not np.isinf(result[col]).any(), f"Column {col} contains infinite values"


if __name__ == "__main__":
    pytest.main([__file__])
