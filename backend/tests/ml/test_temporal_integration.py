"""
TemporalFeatureCalculator の統合テスト

FeatureEngineeringService との統合をテストします。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from app.core.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
from app.core.services.ml.feature_engineering.temporal_features import TemporalFeatureCalculator


class TestTemporalFeatureIntegration:
    """TemporalFeatureCalculator の統合テストクラス"""
    
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
    def feature_service(self):
        """FeatureEngineeringService インスタンスを生成"""
        return FeatureEngineeringService()
    
    @pytest.fixture
    def temporal_calculator(self):
        """TemporalFeatureCalculator インスタンスを生成"""
        return TemporalFeatureCalculator()
    
    def test_temporal_calculator_standalone(self, temporal_calculator, sample_ohlcv_data):
        """TemporalFeatureCalculator 単体での動作テスト"""
        result = temporal_calculator.calculate_temporal_features(sample_ohlcv_data)
        
        # 元のOHLCVカラムが保持されていることを確認
        original_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in original_columns:
            assert col in result.columns
        
        # 時間的特徴量が追加されていることを確認
        temporal_columns = temporal_calculator.get_feature_names()
        for col in temporal_columns:
            assert col in result.columns
        
        # データの整合性確認
        assert len(result) == len(sample_ohlcv_data)
        assert result.index.equals(sample_ohlcv_data.index)
    
    def test_feature_service_with_temporal_integrated(self, feature_service, sample_ohlcv_data):
        """FeatureEngineeringService の統合後機能テスト（時間的特徴量含む）"""
        result = feature_service.calculate_advanced_features(sample_ohlcv_data)

        # 既存の特徴量が生成されていることを確認
        assert len(result.columns) > len(sample_ohlcv_data.columns)

        # 時間的特徴量が含まれていることを確認（統合後の期待動作）
        temporal_columns = [
            'Hour_of_Day', 'Day_of_Week', 'Is_Weekend', 'Is_Monday', 'Is_Friday',
            'Asia_Session', 'Europe_Session', 'US_Session'
        ]
        for col in temporal_columns:
            assert col in result.columns, f"Temporal feature {col} should be present after integration"
    
    def test_manual_temporal_integration(self, feature_service, temporal_calculator, sample_ohlcv_data):
        """手動でのTemporalFeatureCalculator統合テスト"""
        # 統合前の状態をシミュレートするため、時間的特徴量を除外したサービスを作成
        from app.core.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
        from app.core.services.ml.feature_engineering.price_features import PriceFeatureCalculator
        from app.core.services.ml.feature_engineering.technical_features import TechnicalFeatureCalculator

        # 時間的特徴量なしのサービスを手動構築
        basic_service = FeatureEngineeringService()
        # 時間的特徴量計算を一時的に無効化
        original_temporal_calc = basic_service.temporal_calculator
        basic_service.temporal_calculator = None

        try:
            # 基本特徴量のみを計算
            features_df = basic_service.calculate_advanced_features(sample_ohlcv_data)

            # 時間的特徴量を手動で追加
            final_df = temporal_calculator.calculate_temporal_features(features_df)

            # 全ての特徴量が含まれていることを確認
            original_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in original_columns:
                assert col in final_df.columns

            # 時間的特徴量が追加されていることを確認
            temporal_features = temporal_calculator.get_feature_names()
            for feature in temporal_features:
                assert feature in final_df.columns, f"Temporal feature {feature} should be added"

        finally:
            # 元の状態に復元
            basic_service.temporal_calculator = original_temporal_calc
        
        # 時間的特徴量の値が正しく計算されていることを確認
        temporal_columns = temporal_calculator.get_feature_names()
        for col in temporal_columns:
            assert col in final_df.columns
            assert not final_df[col].isna().any(), f"Column {col} contains NaN values"
    
    def test_temporal_features_data_quality(self, temporal_calculator, sample_ohlcv_data):
        """時間的特徴量のデータ品質テスト"""
        result = temporal_calculator.calculate_temporal_features(sample_ohlcv_data)
        
        # 数値特徴量の範囲チェック
        assert (result['Hour_of_Day'] >= 0).all() and (result['Hour_of_Day'] <= 23).all()
        assert (result['Day_of_Week'] >= 0).all() and (result['Day_of_Week'] <= 6).all()
        
        # 周期的エンコーディングの範囲チェック
        assert (result['Hour_Sin'] >= -1).all() and (result['Hour_Sin'] <= 1).all()
        assert (result['Hour_Cos'] >= -1).all() and (result['Hour_Cos'] <= 1).all()
        assert (result['Day_Sin'] >= -1).all() and (result['Day_Sin'] <= 1).all()
        assert (result['Day_Cos'] >= -1).all() and (result['Day_Cos'] <= 1).all()
        
        # ブール特徴量の型チェック
        boolean_columns = ['Is_Weekend', 'Is_Monday', 'Is_Friday', 
                          'Asia_Session', 'Europe_Session', 'US_Session',
                          'Session_Overlap_Asia_Europe', 'Session_Overlap_Europe_US']
        for col in boolean_columns:
            assert result[col].dtype == bool
    
    def test_temporal_features_consistency(self, temporal_calculator, sample_ohlcv_data):
        """時間的特徴量の一貫性テスト"""
        result = temporal_calculator.calculate_temporal_features(sample_ohlcv_data)
        
        # 同じ時間帯の特徴量が一貫していることを確認
        for hour in range(24):
            hour_mask = result['Hour_of_Day'] == hour
            if hour_mask.any():
                # 同じ時間の Sin/Cos 値が一致することを確認
                hour_sin_values = result.loc[hour_mask, 'Hour_Sin'].unique()
                hour_cos_values = result.loc[hour_mask, 'Hour_Cos'].unique()
                assert len(hour_sin_values) == 1, f"Hour {hour} should have consistent sin value"
                assert len(hour_cos_values) == 1, f"Hour {hour} should have consistent cos value"
        
        # セッション重複の論理的一貫性を確認
        # Asia-Europe overlap は Asia と Europe の両方がTrueの時のみTrueであるべき
        overlap_ae = result['Session_Overlap_Asia_Europe']
        asia_session = result['Asia_Session']
        europe_session = result['Europe_Session']
        
        # overlap が True の時は両方のセッションが True であるべき
        overlap_true_mask = overlap_ae == True
        if overlap_true_mask.any():
            assert asia_session.loc[overlap_true_mask].all(), "Asia session should be True when overlap is True"
            assert europe_session.loc[overlap_true_mask].all(), "Europe session should be True when overlap is True"
    
    def test_performance_with_large_dataset(self, temporal_calculator):
        """大きなデータセットでのパフォーマンステスト"""
        # 1年分の時間データ（8760時間）
        dates = pd.date_range(
            start='2024-01-01 00:00:00',
            end='2024-12-31 23:00:00',
            freq='h',
            tz='UTC'
        )
        
        large_data = {
            'Open': np.random.uniform(40000, 50000, len(dates)),
            'High': np.random.uniform(40000, 50000, len(dates)),
            'Low': np.random.uniform(40000, 50000, len(dates)),
            'Close': np.random.uniform(40000, 50000, len(dates)),
            'Volume': np.random.uniform(1000, 10000, len(dates))
        }
        
        large_df = pd.DataFrame(large_data, index=dates)
        
        # パフォーマンス測定
        import time
        start_time = time.time()
        result = temporal_calculator.calculate_temporal_features(large_df)
        end_time = time.time()
        
        # 処理時間が合理的であることを確認（10秒以内）
        processing_time = end_time - start_time
        assert processing_time < 10.0, f"Processing took too long: {processing_time:.2f} seconds"
        
        # 結果の整合性確認
        assert len(result) == len(large_df)
        assert len(result.columns) == len(large_df.columns) + len(temporal_calculator.get_feature_names())


if __name__ == "__main__":
    pytest.main([__file__])
