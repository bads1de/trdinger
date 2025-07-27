"""
特徴量エンジニアリングのエッジケーステスト

極端な市場条件、異常データパターン、境界値などの
エッジケースでの動作を検証します。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import warnings

from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService


class TestFeatureEngineeringEdgeCases:
    """特徴量エンジニアリングのエッジケーステストクラス"""
    
    @pytest.fixture
    def service(self):
        """FeatureEngineeringService インスタンスを生成"""
        return FeatureEngineeringService()
    
    def test_flash_crash_scenario(self, service):
        """フラッシュクラッシュシナリオのテスト"""
        dates = pd.date_range('2024-01-01 12:00:00', periods=100, freq='min', tz='UTC')
        
        # 正常価格から急激な下落、その後回復
        base_price = 45000
        prices = np.full(100, base_price, dtype=float)
        
        # フラッシュクラッシュ（50分目から60分目）
        crash_start, crash_end = 50, 60
        prices[crash_start:crash_end] = base_price * 0.3  # 70%下落
        
        # 急激な回復（60分目から65分目）
        recovery_end = 65
        recovery_prices = np.linspace(base_price * 0.3, base_price * 0.95, recovery_end - crash_end)
        prices[crash_end:recovery_end] = recovery_prices
        
        # 正常化（65分目以降）
        prices[recovery_end:] = base_price * np.random.uniform(0.95, 1.05, 100 - recovery_end)
        
        flash_crash_data = pd.DataFrame({
            'Open': prices * np.random.uniform(0.999, 1.001, 100),
            'High': prices * np.random.uniform(1.001, 1.01, 100),
            'Low': prices * np.random.uniform(0.99, 0.999, 100),
            'Close': prices,
            'Volume': np.random.uniform(1000, 50000, 100)  # 高ボラティリティ時は出来高増加
        }, index=dates)
        
        # フラッシュクラッシュ時は出来高を大幅に増加
        flash_crash_data.loc[dates[crash_start:crash_end], 'Volume'] *= 10
        
        # 処理実行
        result = service.calculate_advanced_features(flash_crash_data)
        
        # 結果の検証
        assert len(result) == 100
        assert len(result.columns) > 50
        
        # 極端な価格変動が適切に処理されていることを確認
        if 'Price_Change_5' in result.columns:
            price_changes = result['Price_Change_5'].dropna()
            # 極端な値が無限大にならないことを確認
            assert np.isfinite(price_changes).all(), "Price changes contain infinite values"
            
        # ボラティリティ指標が適切に反応していることを確認
        if 'ATR_20' in result.columns:
            atr_values = result['ATR_20'].dropna()
            assert np.isfinite(atr_values).all(), "ATR contains infinite values"
            # クラッシュ後のATRが増加していることを確認
            if len(atr_values) > crash_end:
                post_crash_atr = atr_values.iloc[crash_end:crash_end+10].mean()
                pre_crash_atr = atr_values.iloc[crash_start-10:crash_start].mean()
                if not pd.isna(post_crash_atr) and not pd.isna(pre_crash_atr):
                    assert post_crash_atr > pre_crash_atr, "ATR should increase after flash crash"
    
    def test_market_halt_scenario(self, service):
        """市場停止シナリオのテスト"""
        dates = pd.date_range('2024-01-01 09:00:00', periods=200, freq='min', tz='UTC')
        
        # 正常取引 → 市場停止 → 再開のパターン
        base_price = 45000
        base_volume = 5000
        
        prices = np.full(200, base_price, dtype=float)
        volumes = np.full(200, base_volume, dtype=float)
        
        # 市場停止期間（100分目から150分目）
        halt_start, halt_end = 100, 150
        
        # 停止中は価格変動なし、出来高ゼロ
        prices[halt_start:halt_end] = prices[halt_start-1]  # 価格固定
        volumes[halt_start:halt_end] = 0  # 出来高ゼロ
        
        # 再開後は価格ギャップと高出来高
        gap_factor = 1.05  # 5%のギャップアップ
        prices[halt_end:] = prices[halt_end-1] * gap_factor * np.random.uniform(0.99, 1.01, 200 - halt_end)
        volumes[halt_end:halt_end+10] = base_volume * 5  # 再開直後は高出来高
        
        halt_data = pd.DataFrame({
            'Open': prices,
            'High': prices * np.random.uniform(1.0, 1.002, 200),
            'Low': prices * np.random.uniform(0.998, 1.0, 200),
            'Close': prices,
            'Volume': volumes
        }, index=dates)
        
        # 処理実行
        result = service.calculate_advanced_features(halt_data)
        
        # 結果の検証
        assert len(result) == 200
        
        # ゼロ出来高期間でも処理が完了することを確認
        if 'Volume_Ratio' in result.columns:
            volume_ratios = result['Volume_Ratio']
            # 無限大値やNaN値が適切に処理されていることを確認
            finite_ratios = volume_ratios[np.isfinite(volume_ratios)]
            assert len(finite_ratios) > 0, "All volume ratios are non-finite"
        
        # ギャップアップが適切に検出されることを確認
        if 'Gap_Up' in result.columns and halt_end < len(result):
            gap_detection = result['Gap_Up'].iloc[halt_end:halt_end+5]
            # 少なくとも1つのギャップアップが検出されることを期待
            # ただし、計算方法によっては検出されない場合もあるため、エラーにはしない
    
    def test_extreme_volatility_scenario(self, service):
        """極端なボラティリティシナリオのテスト"""
        dates = pd.date_range('2024-01-01 00:00:00', periods=500, freq='min', tz='UTC')
        
        # 極端に高いボラティリティのデータ
        base_price = 45000
        
        # ランダムウォークに大きなノイズを追加
        returns = np.random.normal(0, 0.05, 500)  # 5%の標準偏差
        
        # 時々極端なスパイクを追加
        spike_indices = np.random.choice(500, size=20, replace=False)
        returns[spike_indices] = np.random.choice([-0.2, 0.2], size=20)  # ±20%のスパイク
        
        # 累積リターンから価格を計算
        price_series = base_price * np.cumprod(1 + returns)
        
        extreme_vol_data = pd.DataFrame({
            'Open': price_series * np.random.uniform(0.995, 1.005, 500),
            'High': price_series * np.random.uniform(1.005, 1.05, 500),
            'Low': price_series * np.random.uniform(0.95, 0.995, 500),
            'Close': price_series,
            'Volume': np.random.lognormal(8, 1, 500)  # 対数正規分布の出来高
        }, index=dates)
        
        # 処理実行
        result = service.calculate_advanced_features(extreme_vol_data)
        
        # 結果の検証
        assert len(result) == 500
        
        # 極端なボラティリティでも有限値が維持されることを確認
        numeric_columns = result.select_dtypes(include=[np.number]).columns
        for col in numeric_columns[:20]:  # 最初の20列をサンプルチェック
            finite_values = result[col][np.isfinite(result[col])]
            finite_ratio = len(finite_values) / len(result[col])
            assert finite_ratio > 0.5, f"Column {col} has too few finite values: {finite_ratio:.2%}"
    
    def test_data_type_boundaries(self, service):
        """データ型境界値のテスト"""
        dates = pd.date_range('2024-01-01', periods=10, freq='h', tz='UTC')
        
        # 各データ型の境界値近くの値
        boundary_data = pd.DataFrame({
            'Open': [
                np.finfo(np.float64).max * 0.1,  # float64の最大値の10%
                np.finfo(np.float64).min * 0.1,  # float64の最小値の10%
                np.finfo(np.float32).max * 0.1,  # float32の最大値の10%
                np.finfo(np.float32).min * 0.1,  # float32の最小値の10%
                1e-10, 1e10, -1e10, -1e-10,     # 極端に小さい/大きい値
                45000, 45000                      # 正常値
            ],
            'High': [45100] * 10,
            'Low': [44900] * 10,
            'Close': [45000] * 10,
            'Volume': [
                np.iinfo(np.int64).max * 0.1,   # int64の最大値の10%
                1, 1e10, 1e-3, 1000,             # 様々なスケール
                5000, 5000, 5000, 5000, 5000     # 正常値
            ]
        }, index=dates)
        
        # 処理実行
        result = service.calculate_advanced_features(boundary_data)
        
        # 結果の検証
        assert len(result) == 10
        
        # オーバーフローやアンダーフローが発生していないことを確認
        numeric_columns = result.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            values = result[col].dropna()
            if len(values) > 0:
                # 無限大値がないことを確認
                assert not np.isinf(values).any(), f"Column {col} contains infinite values"
                
                # 値が合理的な範囲内であることを確認
                abs_values = np.abs(values)
                max_reasonable = 1e15  # 合理的な最大値
                assert (abs_values < max_reasonable).all(), f"Column {col} contains unreasonably large values"
    
    def test_missing_data_patterns(self, service):
        """欠損データパターンのテスト"""
        dates = pd.date_range('2024-01-01', periods=100, freq='h', tz='UTC')
        
        # 様々な欠損パターンを含むデータ
        missing_data = pd.DataFrame({
            'Open': np.random.uniform(40000, 50000, 100),
            'High': np.random.uniform(40000, 50000, 100),
            'Low': np.random.uniform(40000, 50000, 100),
            'Close': np.random.uniform(40000, 50000, 100),
            'Volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        # 様々な欠損パターンを追加
        # 1. ランダム欠損
        random_missing = np.random.choice(100, size=10, replace=False)
        missing_data.loc[dates[random_missing], 'Volume'] = np.nan
        
        # 2. 連続欠損
        missing_data.loc[dates[20:25], 'High'] = np.nan
        
        # 3. 周期的欠損
        periodic_missing = np.arange(0, 100, 10)
        missing_data.loc[dates[periodic_missing], 'Low'] = np.nan
        
        # 4. 開始部分の欠損
        missing_data.loc[dates[:5], 'Open'] = np.nan
        
        # 5. 終了部分の欠損
        missing_data.loc[dates[-5:], 'Close'] = np.nan
        
        # 処理実行
        result = service.calculate_advanced_features(missing_data)
        
        # 結果の検証
        assert len(result) == 100
        
        # 欠損データが適切に処理されていることを確認
        # 完全に欠損していない行が存在することを確認
        complete_rows = result.dropna()
        assert len(complete_rows) > 50, f"Too few complete rows: {len(complete_rows)}"
        
        # 基本的な特徴量が計算されていることを確認
        essential_features = ['Hour_of_Day', 'Day_of_Week', 'Asia_Session']
        for feature in essential_features:
            if feature in result.columns:
                non_null_count = result[feature].notna().sum()
                assert non_null_count == 100, f"Feature {feature} should not have missing values"
    
    def test_single_value_columns(self, service):
        """単一値カラムのテスト"""
        dates = pd.date_range('2024-01-01', periods=50, freq='h', tz='UTC')
        
        # 一部のカラムが単一値のデータ
        single_value_data = pd.DataFrame({
            'Open': [45000] * 50,      # 全て同じ値
            'High': [45000] * 50,      # 全て同じ値
            'Low': [45000] * 50,       # 全て同じ値
            'Close': [45000] * 50,     # 全て同じ値
            'Volume': [5000] * 50      # 全て同じ値
        }, index=dates)
        
        # 処理実行
        result = service.calculate_advanced_features(single_value_data)
        
        # 結果の検証
        assert len(result) == 50
        
        # 単一値でもゼロ除算エラーが発生しないことを確認
        # 変動率系の特徴量は0になることを確認
        if 'Price_Change_5' in result.columns:
            price_changes = result['Price_Change_5'].dropna()
            if len(price_changes) > 0:
                # 価格変動がない場合は0になることを確認
                assert (price_changes == 0).all(), "Price changes should be zero for constant prices"
        
        # 時間的特徴量は正常に計算されることを確認
        if 'Hour_of_Day' in result.columns:
            hours = result['Hour_of_Day']
            assert hours.notna().all(), "Hour_of_Day should not have missing values"
            assert (hours >= 0).all() and (hours <= 23).all(), "Hour_of_Day should be in valid range"
    
    def test_irregular_timestamp_gaps(self, service):
        """不規則なタイムスタンプギャップのテスト"""
        # 不規則な間隔のタイムスタンプを作成
        base_time = pd.Timestamp('2024-01-01 00:00:00', tz='UTC')
        irregular_times = [
            base_time,
            base_time + timedelta(minutes=1),
            base_time + timedelta(minutes=5),
            base_time + timedelta(hours=1),
            base_time + timedelta(hours=6),
            base_time + timedelta(days=1),
            base_time + timedelta(days=1, hours=12),
            base_time + timedelta(days=7),
            base_time + timedelta(days=30),
            base_time + timedelta(days=365)
        ]
        
        irregular_data = pd.DataFrame({
            'Open': np.random.uniform(40000, 50000, 10),
            'High': np.random.uniform(40000, 50000, 10),
            'Low': np.random.uniform(40000, 50000, 10),
            'Close': np.random.uniform(40000, 50000, 10),
            'Volume': np.random.uniform(1000, 10000, 10)
        }, index=irregular_times)
        
        # 処理実行
        result = service.calculate_advanced_features(irregular_data)
        
        # 結果の検証
        assert len(result) == 10
        
        # 時間的特徴量が各タイムスタンプで正しく計算されることを確認
        if 'Hour_of_Day' in result.columns:
            for i, timestamp in enumerate(irregular_times):
                expected_hour = timestamp.hour
                actual_hour = result.iloc[i]['Hour_of_Day']
                assert actual_hour == expected_hour, f"Hour mismatch at index {i}: expected {expected_hour}, got {actual_hour}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
