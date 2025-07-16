"""
時間的特徴量計算のテスト

TemporalFeatureCalculatorクラスの機能をテストします。
取引セッション、曜日効果、時間帯、週末効果の特徴量計算をテストします。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import patch

# テスト対象のインポート
from app.core.services.ml.feature_engineering.temporal_features import TemporalFeatureCalculator


class TestTemporalFeatureCalculator:
    """TemporalFeatureCalculator のテストクラス"""
    
    @pytest.fixture
    def sample_datetime_data(self):
        """テスト用の日時データを生成"""
        # 複数の時間帯と曜日をカバーするデータ
        dates = pd.date_range(
            start='2024-01-01 00:00:00',
            end='2024-01-07 23:00:00',
            freq='h',
            tz='UTC'
        )
        
        # OHLCV データを作成
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
    def calculator(self):
        """TemporalFeatureCalculator インスタンスを生成"""
        return TemporalFeatureCalculator()
    
    def test_calculate_temporal_features_basic(self, calculator, sample_datetime_data):
        """基本的な時間的特徴量計算のテスト"""
        result = calculator.calculate_temporal_features(sample_datetime_data)

        # 期待される特徴量カラムが存在することを確認
        expected_columns = [
            'Hour_of_Day', 'Day_of_Week', 'Is_Weekend', 'Is_Monday', 'Is_Friday',
            'Asia_Session', 'Europe_Session', 'US_Session',
            'Session_Overlap_Asia_Europe', 'Session_Overlap_Europe_US',
            'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
        ]

        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

        # データ型の確認
        assert result['Hour_of_Day'].dtype in [np.int64, np.int32]
        assert result['Day_of_Week'].dtype in [np.int64, np.int32]
        assert result['Is_Weekend'].dtype == bool
        assert result['Asia_Session'].dtype == bool
    
    def test_trading_session_detection(self, calculator, sample_datetime_data):
        """取引セッション判定のテスト"""
        result = calculator.calculate_temporal_features(sample_datetime_data)

        # 特定の時間帯でのセッション判定をテスト
        # UTC 2:00 (Asia session)
        asia_time = sample_datetime_data.index[sample_datetime_data.index.hour == 2][0]
        asia_idx = sample_datetime_data.index.get_loc(asia_time)
        assert result.iloc[asia_idx]['Asia_Session'] == True

        # UTC 10:00 (Europe session)
        europe_time = sample_datetime_data.index[sample_datetime_data.index.hour == 10][0]
        europe_idx = sample_datetime_data.index.get_loc(europe_time)
        assert result.iloc[europe_idx]['Europe_Session'] == True

        # UTC 16:00 (US session)
        us_time = sample_datetime_data.index[sample_datetime_data.index.hour == 16][0]
        us_idx = sample_datetime_data.index.get_loc(us_time)
        assert result.iloc[us_idx]['US_Session'] == True
    
    def test_weekend_detection(self, calculator, sample_datetime_data):
        """週末判定のテスト"""
        result = calculator.calculate_temporal_features(sample_datetime_data)

        # 土曜日と日曜日が週末として判定されることを確認
        weekend_mask = sample_datetime_data.index.dayofweek.isin([5, 6])  # 土日
        assert (result['Is_Weekend'] == weekend_mask).all()
    
    def test_cyclical_encoding(self, calculator, sample_datetime_data):
        """周期的エンコーディングのテスト"""
        result = calculator.calculate_temporal_features(sample_datetime_data)

        # Sin/Cos エンコーディングの値域チェック
        assert (result['Hour_Sin'] >= -1).all() and (result['Hour_Sin'] <= 1).all()
        assert (result['Hour_Cos'] >= -1).all() and (result['Hour_Cos'] <= 1).all()
        assert (result['Day_Sin'] >= -1).all() and (result['Day_Sin'] <= 1).all()
        assert (result['Day_Cos'] >= -1).all() and (result['Day_Cos'] <= 1).all()

        # 周期性の確認（24時間周期）
        for hour in range(24):
            hour_mask = sample_datetime_data.index.hour == hour
            if hour_mask.any():
                hour_sin_values = result.loc[hour_mask, 'Hour_Sin'].unique()
                hour_cos_values = result.loc[hour_mask, 'Hour_Cos'].unique()
                assert len(hour_sin_values) == 1, f"Hour {hour} should have consistent sin value"
                assert len(hour_cos_values) == 1, f"Hour {hour} should have consistent cos value"
    
    def test_session_overlap_detection(self, calculator, sample_datetime_data):
        """セッション重複時間の判定テスト"""
        result = calculator.calculate_temporal_features(sample_datetime_data)

        # Asia-Europe overlap (UTC 7:00-9:00)
        overlap_ae_mask = sample_datetime_data.index.hour.isin([7, 8])
        assert (result.loc[overlap_ae_mask, 'Session_Overlap_Asia_Europe']).all()

        # Europe-US overlap (UTC 13:00-16:00)
        overlap_eu_mask = sample_datetime_data.index.hour.isin([13, 14, 15])
        assert (result.loc[overlap_eu_mask, 'Session_Overlap_Europe_US']).all()
    
    def test_empty_dataframe_handling(self, calculator):
        """空のDataFrameの処理テスト"""
        empty_df = pd.DataFrame()
        result = calculator.calculate_temporal_features(empty_df)
        assert result.empty
    
    def test_invalid_index_handling(self, calculator):
        """不正なインデックスの処理テスト"""
        # 日時以外のインデックスを持つDataFrame
        df = pd.DataFrame({
            'Close': [100, 200, 300],
            'Volume': [1000, 2000, 3000]
        }, index=[0, 1, 2])  # 数値インデックス

        # エラーハンドリングのテスト
        with pytest.raises(ValueError, match="DatetimeIndex"):
            calculator.calculate_temporal_features(df)
    
    def test_timezone_handling(self, calculator):
        """タイムゾーン処理のテスト"""
        # 異なるタイムゾーンのデータ
        dates_jst = pd.date_range(
            start='2024-01-01 09:00:00',
            periods=24,
            freq='h',
            tz='Asia/Tokyo'
        )

        df_jst = pd.DataFrame({
            'Close': np.random.uniform(40000, 50000, len(dates_jst)),
            'Volume': np.random.uniform(1000, 10000, len(dates_jst))
        }, index=dates_jst)

        result = calculator.calculate_temporal_features(df_jst)

        # UTCに変換されて処理されることを確認
        assert result.index.tz == timezone.utc or result.index.tz.zone == 'UTC'
    
    def test_feature_names_method(self, calculator):
        """特徴量名取得メソッドのテスト"""
        feature_names = calculator.get_feature_names()

        # 期待される特徴量名が含まれることを確認
        expected_names = [
            'Hour_of_Day', 'Day_of_Week', 'Is_Weekend', 'Is_Monday', 'Is_Friday',
            'Asia_Session', 'Europe_Session', 'US_Session',
            'Session_Overlap_Asia_Europe', 'Session_Overlap_Europe_US',
            'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
        ]

        for name in expected_names:
            assert name in feature_names
    
    def test_data_validation_integration(self, calculator, sample_datetime_data):
        """データバリデーションとの統合テスト"""
        result = calculator.calculate_temporal_features(sample_datetime_data)

        # NaN値や無限大値がないことを確認
        for col in result.select_dtypes(include=[np.number]).columns:
            assert not result[col].isna().any(), f"Column {col} contains NaN values"
            assert not np.isinf(result[col]).any(), f"Column {col} contains infinite values"


if __name__ == "__main__":
    pytest.main([__file__])
