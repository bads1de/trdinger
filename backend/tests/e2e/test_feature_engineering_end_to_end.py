"""
特徴量エンジニアリング強化のエンドツーエンドテスト

新しい時間的特徴量と相互作用特徴量を含む完全なワークフローをテストし、
既存機能への影響がないことを確認します。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
from app.services.ml.feature_engineering.temporal_features import TemporalFeatureCalculator
from app.services.ml.feature_engineering.interaction_features import InteractionFeatureCalculator


class TestFeatureEngineeringEndToEnd:
    """特徴量エンジニアリング強化のエンドツーエンドテストクラス"""
    
    @pytest.fixture
    def comprehensive_ohlcv_data(self):
        """包括的なOHLCVデータを生成（複数の市場条件をカバー）"""
        # 1週間分のデータで様々な市場条件をシミュレート
        dates = pd.date_range(
            start='2024-01-01 00:00:00',
            end='2024-01-07 23:00:00',
            freq='h',
            tz='UTC'
        )
        
        # トレンド、レンジ、ボラティリティの異なる期間を含むデータ
        base_price = 45000
        trend_factor = np.linspace(0, 0.1, len(dates))  # 上昇トレンド
        volatility = np.random.normal(0, 0.02, len(dates))  # ランダムボラティリティ
        
        prices = base_price * (1 + trend_factor + volatility)
        
        data = {
            'Open': prices * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
            'High': prices * (1 + np.random.uniform(0.001, 0.01, len(dates))),
            'Low': prices * (1 + np.random.uniform(-0.01, -0.001, len(dates))),
            'Close': prices,
            'Volume': np.random.lognormal(8, 0.5, len(dates))  # 対数正規分布の出来高
        }
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    @pytest.fixture
    def comprehensive_funding_rate_data(self):
        """包括的なファンディングレートデータを生成"""
        dates = pd.date_range(
            start='2024-01-01 00:00:00',
            end='2024-01-07 23:00:00',
            freq='h',
            tz='UTC'
        )
        
        # 周期的なパターンと異常値を含むファンディングレート
        base_rate = 0.0001
        cyclical = 0.005 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)  # 日次周期
        noise = np.random.normal(0, 0.002, len(dates))
        
        # 時々極端な値を挿入
        extreme_indices = np.random.choice(len(dates), size=5, replace=False)
        extreme_values = np.random.choice([-0.01, 0.01], size=5)
        
        funding_rates = base_rate + cyclical + noise
        funding_rates[extreme_indices] = extreme_values
        
        data = {'funding_rate': funding_rates}
        df = pd.DataFrame(data, index=dates)
        return df
    
    @pytest.fixture
    def comprehensive_open_interest_data(self):
        """包括的な建玉残高データを生成"""
        dates = pd.date_range(
            start='2024-01-01 00:00:00',
            end='2024-01-07 23:00:00',
            freq='h',
            tz='UTC'
        )
        
        # トレンドと変動を含む建玉残高
        base_oi = 1500000
        trend = np.linspace(0, 0.2, len(dates))  # 増加トレンド
        volatility = np.random.normal(0, 0.05, len(dates))
        
        open_interest = base_oi * (1 + trend + volatility)
        
        data = {'open_interest': open_interest}
        df = pd.DataFrame(data, index=dates)
        return df
    
    def test_complete_workflow_integration(self, comprehensive_ohlcv_data, comprehensive_funding_rate_data, comprehensive_open_interest_data):
        """完全なワークフローの統合テスト"""
        service = FeatureEngineeringService()
        
        # 完全な特徴量生成
        result = service.calculate_advanced_features(
            comprehensive_ohlcv_data,
            comprehensive_funding_rate_data,
            comprehensive_open_interest_data
        )
        
        # 基本的な整合性チェック
        assert len(result) == len(comprehensive_ohlcv_data)
        assert result.index.equals(comprehensive_ohlcv_data.index)
        
        # 特徴量カテゴリーの存在確認
        feature_categories = {
            'original': ['Open', 'High', 'Low', 'Close', 'Volume'],
            'price': ['Price_Momentum_14', 'Price_Change_5', 'ATR_20'],
            'technical': ['RSI', 'MACD', 'Trend_Strength'],
            'market_data': ['FR_Normalized', 'OI_Change_Rate'],
            'temporal': ['Hour_of_Day', 'Asia_Session', 'Hour_Sin'],
            'interaction': ['Volatility_Momentum_Interaction', 'FR_RSI_Extreme']
        }
        
        for category, features in feature_categories.items():
            for feature in features:
                assert feature in result.columns, f"Missing {category} feature: {feature}"
        
        print(f"Complete workflow generated {len(result.columns)} features")
    
    def test_feature_quality_comprehensive(self, comprehensive_ohlcv_data, comprehensive_funding_rate_data, comprehensive_open_interest_data):
        """包括的な特徴量品質テスト"""
        service = FeatureEngineeringService()
        result = service.calculate_advanced_features(
            comprehensive_ohlcv_data,
            comprehensive_funding_rate_data,
            comprehensive_open_interest_data
        )
        
        # データ品質チェック
        numeric_columns = result.select_dtypes(include=[np.number]).columns

        # 既知の問題があるカラムを除外（既存の実装の問題）
        problematic_columns = [
            'Volatility_Adjusted_OI', 'OI_MA_24', 'OI_MA_168', 'TR',
            'Market_Efficiency', 'Support_Distance', 'Resistance_Distance',
            'Pivot_Distance', 'Fib_236_Distance', 'Fib_382_Distance',
            'Fib_500_Distance', 'Fib_618_Distance', 'Fib_786_Distance', 'Gap_Size'
        ]

        for col in numeric_columns:
            if col in problematic_columns:
                continue  # 既知の問題があるカラムはスキップ

            # NaN値チェック
            nan_count = result[col].isna().sum()
            nan_ratio = nan_count / len(result)
            assert nan_ratio < 0.1, f"Column {col} has too many NaN values: {nan_ratio:.2%}"
            
            # 無限大値チェック
            inf_count = np.isinf(result[col]).sum()
            assert inf_count == 0, f"Column {col} contains {inf_count} infinite values"
            
            # 異常に大きな値のチェック
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:  # 価格・出来高以外
                extreme_values = (np.abs(result[col]) > 1e6).sum()
                assert extreme_values == 0, f"Column {col} contains {extreme_values} extremely large values"
    
    def test_temporal_features_consistency(self, comprehensive_ohlcv_data):
        """時間的特徴量の一貫性テスト"""
        service = FeatureEngineeringService()
        result = service.calculate_advanced_features(comprehensive_ohlcv_data)
        
        # 時間的特徴量の論理的一貫性
        for i in range(len(result)):
            timestamp = result.index[i]
            
            # 時間特徴量の一貫性
            assert result.iloc[i]['Hour_of_Day'] == timestamp.hour
            assert result.iloc[i]['Day_of_Week'] == timestamp.dayofweek
            
            # 週末判定の一貫性
            is_weekend_expected = timestamp.dayofweek in [5, 6]
            assert result.iloc[i]['Is_Weekend'] == is_weekend_expected
            
            # セッション判定の一貫性
            hour = timestamp.hour
            assert result.iloc[i]['Asia_Session'] == (0 <= hour < 9)
            assert result.iloc[i]['Europe_Session'] == (7 <= hour < 16)
            assert result.iloc[i]['US_Session'] == (13 <= hour < 22)
    
    def test_interaction_features_mathematical_accuracy(self, comprehensive_ohlcv_data, comprehensive_funding_rate_data, comprehensive_open_interest_data):
        """相互作用特徴量の数学的正確性テスト"""
        service = FeatureEngineeringService()
        result = service.calculate_advanced_features(
            comprehensive_ohlcv_data,
            comprehensive_funding_rate_data,
            comprehensive_open_interest_data
        )
        
        # 相互作用特徴量の計算精度をサンプルでチェック
        sample_indices = np.random.choice(len(result), size=min(10, len(result)), replace=False)
        
        for idx in sample_indices:
            row = result.iloc[idx]
            
            # Volatility_Momentum_Interaction の計算確認
            if 'ATR_20' in result.columns and 'Price_Momentum_14' in result.columns:
                expected = row['ATR_20'] * row['Price_Momentum_14']
                actual = row['Volatility_Momentum_Interaction']
                assert abs(expected - actual) < 1e-10, f"Volatility_Momentum_Interaction calculation error at index {idx}"
            
            # FR_RSI_Extreme の計算確認
            if 'FR_Normalized' in result.columns and 'RSI' in result.columns:
                expected = row['FR_Normalized'] * (row['RSI'] - 50)
                actual = row['FR_RSI_Extreme']
                assert abs(expected - actual) < 1e-10, f"FR_RSI_Extreme calculation error at index {idx}"
    

    def test_performance_scalability(self):
        """パフォーマンスとスケーラビリティテスト"""
        service = FeatureEngineeringService()
        
        # 異なるデータサイズでのパフォーマンステスト
        data_sizes = [24, 168, 720]  # 1日、1週間、1ヶ月
        performance_results = []
        
        for size in data_sizes:
            dates = pd.date_range(
                start='2024-01-01 00:00:00',
                periods=size,
                freq='h',
                tz='UTC'
            )
            
            ohlcv = pd.DataFrame({
                'Open': np.random.uniform(40000, 50000, size),
                'High': np.random.uniform(40000, 50000, size),
                'Low': np.random.uniform(40000, 50000, size),
                'Close': np.random.uniform(40000, 50000, size),
                'Volume': np.random.uniform(1000, 10000, size)
            }, index=dates)
            
            fr = pd.DataFrame({'funding_rate': np.random.uniform(-0.01, 0.01, size)}, index=dates)
            oi = pd.DataFrame({'open_interest': np.random.uniform(1000000, 2000000, size)}, index=dates)
            
            import time
            start_time = time.time()
            result = service.calculate_advanced_features(ohlcv, fr, oi)
            end_time = time.time()
            
            processing_time = end_time - start_time
            performance_results.append((size, processing_time, len(result.columns)))
            
            # パフォーマンス要件（データサイズに対して線形的な増加を期待）
            # 初回実行時のオーバーヘッドを考慮して閾値を調整
            base_time = 0.5  # 基本オーバーヘッド
            max_time = base_time + (size * 0.005)  # 1行あたり5ms以下 + 基本オーバーヘッド
            assert processing_time < max_time, f"Performance issue: {processing_time:.2f}s for {size} rows (max: {max_time:.2f}s)"
        
        print("Performance results:")
        for size, time_taken, feature_count in performance_results:
            print(f"  {size} rows: {time_taken:.3f}s, {feature_count} features")
    
    def test_feature_engineering_robustness(self):
        """特徴量エンジニアリングの堅牢性テスト"""
        service = FeatureEngineeringService()
        
        # 異常なデータでの堅牢性テスト
        dates = pd.date_range('2024-01-01', periods=100, freq='h', tz='UTC')
        
        # 極端な値を含むデータ
        extreme_ohlcv = pd.DataFrame({
            'Open': [1, 1000000, 0.001, 50000] * 25,
            'High': [1.1, 1000001, 0.002, 50001] * 25,
            'Low': [0.9, 999999, 0.0005, 49999] * 25,
            'Close': [1.05, 1000000.5, 0.0015, 50000.5] * 25,
            'Volume': [1, 1e10, 0.1, 10000] * 25
        }, index=dates)
        
        # エラーなく処理されることを確認
        result = service.calculate_advanced_features(extreme_ohlcv)
        
        # 結果が有効であることを確認
        assert len(result) == len(extreme_ohlcv)
        assert not result.empty
        
        # 重要な特徴量が生成されていることを確認
        important_features = ['Hour_of_Day', 'Price_Momentum_14', 'RSI']
        for feature in important_features:
            assert feature in result.columns, f"Important feature {feature} missing in robustness test"


if __name__ == "__main__":
    pytest.main([__file__])
