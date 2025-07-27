"""
強化されたFeatureEngineeringServiceの統合テスト

新しい時間的特徴量と相互作用特徴量が含まれた
完全なFeatureEngineeringServiceをテストします。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService


class TestEnhancedFeatureEngineering:
    """強化されたFeatureEngineeringServiceのテストクラス"""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """テスト用のOHLCVデータを生成"""
        dates = pd.date_range(
            start='2024-01-01 00:00:00',
            end='2024-01-03 23:00:00',
            freq='h',
            tz='UTC'
        )
        
        data = {
            'Open': np.random.uniform(40000, 50000, len(dates)),
            'High': np.random.uniform(40000, 50000, len(dates)),
            'Low': np.random.uniform(40000, 50000, len(dates)),
            'Close': np.random.uniform(40000, 50000, len(dates)),
            'Volume': np.random.uniform(1000, 10000, len(dates))
        }
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    @pytest.fixture
    def sample_funding_rate_data(self):
        """テスト用のファンディングレートデータを生成"""
        dates = pd.date_range(
            start='2024-01-01 00:00:00',
            end='2024-01-03 23:00:00',
            freq='h',
            tz='UTC'
        )
        
        data = {
            'funding_rate': np.random.uniform(-0.01, 0.01, len(dates))
        }
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    @pytest.fixture
    def sample_open_interest_data(self):
        """テスト用の建玉残高データを生成"""
        dates = pd.date_range(
            start='2024-01-01 00:00:00',
            end='2024-01-03 23:00:00',
            freq='h',
            tz='UTC'
        )
        
        data = {
            'open_interest': np.random.uniform(1000000, 2000000, len(dates))
        }
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    @pytest.fixture
    def enhanced_service(self):
        """強化されたFeatureEngineeringService インスタンスを生成"""
        return FeatureEngineeringService()
    
    def test_enhanced_feature_generation(self, enhanced_service, sample_ohlcv_data, sample_funding_rate_data, sample_open_interest_data):
        """強化された特徴量生成のテスト"""
        result = enhanced_service.calculate_advanced_features(
            sample_ohlcv_data, 
            sample_funding_rate_data, 
            sample_open_interest_data
        )
        
        # 元のOHLCVカラムが保持されていることを確認
        original_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in original_columns:
            assert col in result.columns
        
        # 時間的特徴量が含まれていることを確認
        temporal_features = [
            'Hour_of_Day', 'Day_of_Week', 'Is_Weekend', 'Is_Monday', 'Is_Friday',
            'Asia_Session', 'Europe_Session', 'US_Session',
            'Session_Overlap_Asia_Europe', 'Session_Overlap_Europe_US',
            'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
        ]
        for feature in temporal_features:
            assert feature in result.columns, f"Temporal feature {feature} should be present"
        
        # 相互作用特徴量が含まれていることを確認
        interaction_features = [
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
        for feature in interaction_features:
            assert feature in result.columns, f"Interaction feature {feature} should be present"
        
        # データの整合性確認
        assert len(result) == len(sample_ohlcv_data)
        assert result.index.equals(sample_ohlcv_data.index)
    
    def test_enhanced_feature_names_method(self, enhanced_service):
        """強化された特徴量名取得メソッドのテスト"""
        feature_names = enhanced_service.get_feature_names()
        
        # 時間的特徴量が含まれていることを確認
        temporal_features = [
            'Hour_of_Day', 'Day_of_Week', 'Is_Weekend', 'Is_Monday', 'Is_Friday',
            'Asia_Session', 'Europe_Session', 'US_Session',
            'Session_Overlap_Asia_Europe', 'Session_Overlap_Europe_US',
            'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
        ]
        for feature in temporal_features:
            assert feature in feature_names, f"Temporal feature {feature} should be in feature names"
        
        # 相互作用特徴量が含まれていることを確認
        interaction_features = [
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
        for feature in interaction_features:
            assert feature in feature_names, f"Interaction feature {feature} should be in feature names"
    
    def test_enhanced_feature_data_quality(self, enhanced_service, sample_ohlcv_data, sample_funding_rate_data, sample_open_interest_data):
        """強化された特徴量のデータ品質テスト"""
        result = enhanced_service.calculate_advanced_features(
            sample_ohlcv_data, 
            sample_funding_rate_data, 
            sample_open_interest_data
        )
        
        # 時間的特徴量の品質チェック
        assert (result['Hour_of_Day'] >= 0).all() and (result['Hour_of_Day'] <= 23).all()
        assert (result['Day_of_Week'] >= 0).all() and (result['Day_of_Week'] <= 6).all()
        assert (result['Hour_Sin'] >= -1).all() and (result['Hour_Sin'] <= 1).all()
        assert (result['Hour_Cos'] >= -1).all() and (result['Hour_Cos'] <= 1).all()
        
        # ブール特徴量の型チェック
        boolean_features = ['Is_Weekend', 'Is_Monday', 'Is_Friday', 
                           'Asia_Session', 'Europe_Session', 'US_Session',
                           'Session_Overlap_Asia_Europe', 'Session_Overlap_Europe_US']
        for feature in boolean_features:
            assert result[feature].dtype == bool, f"Feature {feature} should be boolean"
        
        # 相互作用特徴量のNaN/無限大値チェック
        interaction_features = [
            'Volatility_Momentum_Interaction',
            'Volume_Trend_Interaction',
            'FR_RSI_Extreme',
            'OI_Price_Divergence',
            'OI_Momentum_Alignment'
        ]
        for feature in interaction_features:
            if feature in result.columns:
                assert not result[feature].isna().any(), f"Feature {feature} contains NaN values"
                assert not np.isinf(result[feature]).any(), f"Feature {feature} contains infinite values"
    
    def test_enhanced_feature_count(self, enhanced_service, sample_ohlcv_data, sample_funding_rate_data, sample_open_interest_data):
        """強化された特徴量数のテスト"""
        result = enhanced_service.calculate_advanced_features(
            sample_ohlcv_data, 
            sample_funding_rate_data, 
            sample_open_interest_data
        )
        
        # 元のOHLCV（5列）+ 既存特徴量 + 時間的特徴量（14列）+ 相互作用特徴量（9列）
        # 最低でも28列以上は期待される
        assert len(result.columns) >= 28, f"Expected at least 28 features, got {len(result.columns)}"
        
        # 実際の特徴量数をログ出力
        print(f"Total features generated: {len(result.columns)}")
        print(f"Feature names: {sorted(result.columns.tolist())}")
    

    def test_enhanced_feature_performance(self, enhanced_service):
        """強化された特徴量生成のパフォーマンステスト"""
        # 1週間分のデータ
        dates = pd.date_range(
            start='2024-01-01 00:00:00',
            end='2024-01-07 23:00:00',
            freq='h',
            tz='UTC'
        )
        
        large_ohlcv = pd.DataFrame({
            'Open': np.random.uniform(40000, 50000, len(dates)),
            'High': np.random.uniform(40000, 50000, len(dates)),
            'Low': np.random.uniform(40000, 50000, len(dates)),
            'Close': np.random.uniform(40000, 50000, len(dates)),
            'Volume': np.random.uniform(1000, 10000, len(dates))
        }, index=dates)
        
        large_fr = pd.DataFrame({
            'funding_rate': np.random.uniform(-0.01, 0.01, len(dates))
        }, index=dates)
        
        large_oi = pd.DataFrame({
            'open_interest': np.random.uniform(1000000, 2000000, len(dates))
        }, index=dates)
        
        # パフォーマンス測定
        import time
        start_time = time.time()
        result = enhanced_service.calculate_advanced_features(large_ohlcv, large_fr, large_oi)
        end_time = time.time()
        
        # 処理時間が合理的であることを確認（15秒以内）
        processing_time = end_time - start_time
        assert processing_time < 15.0, f"Processing took too long: {processing_time:.2f} seconds"
        
        # 結果の整合性確認
        assert len(result) == len(large_ohlcv)
        print(f"Performance test: {processing_time:.2f} seconds for {len(large_ohlcv)} rows")


if __name__ == "__main__":
    pytest.main([__file__])
