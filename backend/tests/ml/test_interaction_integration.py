"""
InteractionFeatureCalculator の統合テスト

FeatureEngineeringService との統合をテストします。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from app.core.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
from app.core.services.ml.feature_engineering.interaction_features import InteractionFeatureCalculator


class TestInteractionFeatureIntegration:
    """InteractionFeatureCalculator の統合テストクラス"""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """テスト用のOHLCVデータを生成"""
        dates = pd.date_range(
            start='2024-01-01 00:00:00',
            end='2024-01-02 23:00:00',
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
            end='2024-01-02 23:00:00',
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
            end='2024-01-02 23:00:00',
            freq='h',
            tz='UTC'
        )
        
        data = {
            'open_interest': np.random.uniform(1000000, 2000000, len(dates))
        }
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    @pytest.fixture
    def feature_service(self):
        """FeatureEngineeringService インスタンスを生成"""
        return FeatureEngineeringService()
    
    @pytest.fixture
    def interaction_calculator(self):
        """InteractionFeatureCalculator インスタンスを生成"""
        return InteractionFeatureCalculator()
    
    def test_interaction_calculator_standalone(self, interaction_calculator, feature_service, sample_ohlcv_data, sample_funding_rate_data, sample_open_interest_data):
        """InteractionFeatureCalculator 単体での動作テスト"""
        # まず既存の特徴量を生成
        features_df = feature_service.calculate_advanced_features(
            sample_ohlcv_data, 
            sample_funding_rate_data, 
            sample_open_interest_data
        )
        
        # 相互作用特徴量を計算
        result = interaction_calculator.calculate_interaction_features(features_df)
        
        # 元の特徴量が保持されていることを確認
        for col in features_df.columns:
            assert col in result.columns, f"Original column {col} should be preserved"
        
        # 相互作用特徴量が追加されていることを確認
        interaction_columns = interaction_calculator.get_feature_names()
        for col in interaction_columns:
            assert col in result.columns, f"Interaction feature {col} should be present"
        
        # データの整合性確認
        assert len(result) == len(features_df)
        assert result.index.equals(features_df.index)
    
    def test_feature_service_without_interaction(self, feature_service, sample_ohlcv_data, sample_funding_rate_data, sample_open_interest_data):
        """FeatureEngineeringService の既存機能テスト（相互作用特徴量なし）"""
        result = feature_service.calculate_advanced_features(
            sample_ohlcv_data, 
            sample_funding_rate_data, 
            sample_open_interest_data
        )
        
        # 既存の特徴量が生成されていることを確認
        assert len(result.columns) > len(sample_ohlcv_data.columns)
        
        # 相互作用特徴量がまだ含まれていないことを確認
        interaction_columns = [
            'Volatility_Momentum_Interaction',
            'Volume_Trend_Interaction',
            'FR_RSI_Extreme',
            'OI_Price_Divergence'
        ]
        for col in interaction_columns:
            assert col not in result.columns, f"Interaction feature {col} should not be present yet"
    
    def test_manual_interaction_integration(self, feature_service, interaction_calculator, sample_ohlcv_data, sample_funding_rate_data, sample_open_interest_data):
        """手動でのInteractionFeatureCalculator統合テスト"""
        # 既存の特徴量を計算
        features_df = feature_service.calculate_advanced_features(
            sample_ohlcv_data, 
            sample_funding_rate_data, 
            sample_open_interest_data
        )
        
        # 相互作用特徴量を追加
        final_df = interaction_calculator.calculate_interaction_features(features_df)
        
        # 全ての特徴量が含まれていることを確認
        original_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in original_columns:
            assert col in final_df.columns
        
        # 既存の特徴量が保持されていることを確認
        existing_feature_count = len(features_df.columns)
        interaction_feature_count = len(interaction_calculator.get_feature_names())
        
        # 相互作用特徴量が追加されていることを確認
        expected_total = existing_feature_count + interaction_feature_count
        assert len(final_df.columns) == expected_total
        
        # 相互作用特徴量の値が正しく計算されていることを確認
        interaction_columns = interaction_calculator.get_feature_names()
        for col in interaction_columns:
            assert col in final_df.columns
            assert not final_df[col].isna().any(), f"Column {col} contains NaN values"
    
    def test_interaction_features_data_quality(self, interaction_calculator, feature_service, sample_ohlcv_data, sample_funding_rate_data, sample_open_interest_data):
        """相互作用特徴量のデータ品質テスト"""
        # 既存の特徴量を生成
        features_df = feature_service.calculate_advanced_features(
            sample_ohlcv_data, 
            sample_funding_rate_data, 
            sample_open_interest_data
        )
        
        # 相互作用特徴量を計算
        result = interaction_calculator.calculate_interaction_features(features_df)
        
        # 相互作用特徴量の数値品質チェック
        interaction_columns = interaction_calculator.get_feature_names()
        for col in interaction_columns:
            if col in result.columns:
                series = result[col]
                
                # NaN値や無限大値がないことを確認
                assert not series.isna().any(), f"Column {col} contains NaN values"
                assert not np.isinf(series).any(), f"Column {col} contains infinite values"
                
                # 数値型であることを確認
                assert pd.api.types.is_numeric_dtype(series), f"Column {col} should be numeric"
    
    def test_interaction_features_mathematical_consistency(self, interaction_calculator, feature_service, sample_ohlcv_data, sample_funding_rate_data, sample_open_interest_data):
        """相互作用特徴量の数学的一貫性テスト"""
        # 既存の特徴量を生成
        features_df = feature_service.calculate_advanced_features(
            sample_ohlcv_data, 
            sample_funding_rate_data, 
            sample_open_interest_data
        )
        
        # 相互作用特徴量を計算
        result = interaction_calculator.calculate_interaction_features(features_df)
        
        # 数学的一貫性の確認
        if 'ATR' in features_df.columns and 'Price_Momentum_14' in features_df.columns:
            expected_interaction = features_df['ATR'] * features_df['Price_Momentum_14']
            if 'Volatility_Momentum_Interaction' in result.columns:
                pd.testing.assert_series_equal(
                    result['Volatility_Momentum_Interaction'],
                    expected_interaction,
                    check_names=False
                )
        
        # FR×RSI相互作用の確認
        if 'FR_Normalized' in features_df.columns and 'RSI' in features_df.columns:
            expected_fr_rsi = features_df['FR_Normalized'] * (features_df['RSI'] - 50)
            if 'FR_RSI_Extreme' in result.columns:
                pd.testing.assert_series_equal(
                    result['FR_RSI_Extreme'],
                    expected_fr_rsi,
                    check_names=False
                )
    
    def test_performance_with_large_dataset(self, interaction_calculator, feature_service):
        """大きなデータセットでのパフォーマンステスト"""
        # 1週間分の時間データ
        dates = pd.date_range(
            start='2024-01-01 00:00:00',
            end='2024-01-07 23:00:00',
            freq='h',
            tz='UTC'
        )
        
        large_ohlcv = {
            'Open': np.random.uniform(40000, 50000, len(dates)),
            'High': np.random.uniform(40000, 50000, len(dates)),
            'Low': np.random.uniform(40000, 50000, len(dates)),
            'Close': np.random.uniform(40000, 50000, len(dates)),
            'Volume': np.random.uniform(1000, 10000, len(dates))
        }
        
        large_fr = {
            'funding_rate': np.random.uniform(-0.01, 0.01, len(dates))
        }
        
        large_oi = {
            'open_interest': np.random.uniform(1000000, 2000000, len(dates))
        }
        
        large_ohlcv_df = pd.DataFrame(large_ohlcv, index=dates)
        large_fr_df = pd.DataFrame(large_fr, index=dates)
        large_oi_df = pd.DataFrame(large_oi, index=dates)
        
        # 既存の特徴量を生成
        features_df = feature_service.calculate_advanced_features(
            large_ohlcv_df, large_fr_df, large_oi_df
        )
        
        # パフォーマンス測定
        import time
        start_time = time.time()
        result = interaction_calculator.calculate_interaction_features(features_df)
        end_time = time.time()
        
        # 処理時間が合理的であることを確認（5秒以内）
        processing_time = end_time - start_time
        assert processing_time < 5.0, f"Processing took too long: {processing_time:.2f} seconds"
        
        # 結果の整合性確認
        assert len(result) == len(features_df)
        expected_columns = len(features_df.columns) + len(interaction_calculator.get_feature_names())
        assert len(result.columns) == expected_columns


if __name__ == "__main__":
    pytest.main([__file__])
