"""
TemporalFeatureCalculator の堅牢性テスト

エッジケース、異常データ、パフォーマンス、メモリ使用量などの
堅牢性に関する包括的なテストを実施します。
"""

import pytest
import pandas as pd
import numpy as np
import psutil
import gc
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
import warnings

from app.core.services.ml.feature_engineering.temporal_features import TemporalFeatureCalculator


class TestTemporalFeatureCalculatorRobust:
    """TemporalFeatureCalculator の堅牢性テストクラス"""
    
    @pytest.fixture
    def calculator(self):
        """TemporalFeatureCalculator インスタンスを生成"""
        return TemporalFeatureCalculator()
    
    def test_extreme_timestamp_values(self, calculator):
        """極端なタイムスタンプ値のテスト"""
        # 非常に古い日付
        old_dates = pd.date_range('1900-01-01', periods=24, freq='h', tz='UTC')
        old_data = pd.DataFrame({
            'Close': np.random.uniform(1, 100, 24),
            'Volume': np.random.uniform(1, 1000, 24)
        }, index=old_dates)
        
        result = calculator.calculate_temporal_features(old_data)
        assert len(result) == 24
        assert 'Hour_of_Day' in result.columns
        
        # 未来の日付
        future_dates = pd.date_range('2100-01-01', periods=24, freq='h', tz='UTC')
        future_data = pd.DataFrame({
            'Close': np.random.uniform(1, 100, 24),
            'Volume': np.random.uniform(1, 1000, 24)
        }, index=future_dates)
        
        result = calculator.calculate_temporal_features(future_data)
        assert len(result) == 24
        assert 'Hour_of_Day' in result.columns
    
    def test_timezone_edge_cases(self, calculator):
        """タイムゾーンのエッジケースのテスト"""
        # 夏時間切り替え時期のテスト
        dst_dates = pd.date_range(
            '2024-03-10 01:00:00', 
            '2024-03-10 04:00:00', 
            freq='h', 
            tz='US/Eastern'
        )
        
        dst_data = pd.DataFrame({
            'Close': np.random.uniform(40000, 50000, len(dst_dates)),
            'Volume': np.random.uniform(1000, 10000, len(dst_dates))
        }, index=dst_dates)
        
        result = calculator.calculate_temporal_features(dst_data)
        assert len(result) == len(dst_dates)
        assert result.index.tz == timezone.utc
        
        # 異なるタイムゾーンの混在（エラーケース）
        mixed_tz_dates = [
            pd.Timestamp('2024-01-01 12:00:00', tz='UTC'),
            pd.Timestamp('2024-01-01 13:00:00', tz='Asia/Tokyo'),
            pd.Timestamp('2024-01-01 14:00:00', tz='US/Eastern')
        ]
        
        # 混在タイムゾーンは正規化されることを確認
        for tz_date in mixed_tz_dates:
            single_data = pd.DataFrame({
                'Close': [45000],
                'Volume': [5000]
            }, index=[tz_date])
            
            result = calculator.calculate_temporal_features(single_data)
            assert result.index.tz == timezone.utc
    
    def test_irregular_time_intervals(self, calculator):
        """不規則な時間間隔のテスト"""
        # 不規則な間隔のタイムスタンプ
        irregular_dates = [
            pd.Timestamp('2024-01-01 00:00:00', tz='UTC'),
            pd.Timestamp('2024-01-01 00:17:00', tz='UTC'),  # 17分後
            pd.Timestamp('2024-01-01 02:33:00', tz='UTC'),  # 2時間16分後
            pd.Timestamp('2024-01-01 23:59:00', tz='UTC'),  # 21時間26分後
        ]
        
        irregular_data = pd.DataFrame({
            'Close': np.random.uniform(40000, 50000, len(irregular_dates)),
            'Volume': np.random.uniform(1000, 10000, len(irregular_dates))
        }, index=irregular_dates)
        
        result = calculator.calculate_temporal_features(irregular_data)
        assert len(result) == len(irregular_dates)
        
        # 各行の時間特徴量が正しく計算されていることを確認
        for i, timestamp in enumerate(irregular_dates):
            assert result.iloc[i]['Hour_of_Day'] == timestamp.hour
            assert result.iloc[i]['Day_of_Week'] == timestamp.dayofweek
    
    def test_single_row_data(self, calculator):
        """単一行データのテスト"""
        single_date = pd.date_range('2024-01-01 12:00:00', periods=1, tz='UTC')
        single_data = pd.DataFrame({
            'Close': [45000],
            'Volume': [5000]
        }, index=single_date)
        
        result = calculator.calculate_temporal_features(single_data)
        assert len(result) == 1
        assert result.iloc[0]['Hour_of_Day'] == 12
        assert result.iloc[0]['Europe_Session'] == True
    
    def test_duplicate_timestamps(self, calculator):
        """重複タイムスタンプのテスト"""
        duplicate_dates = [
            pd.Timestamp('2024-01-01 12:00:00', tz='UTC'),
            pd.Timestamp('2024-01-01 12:00:00', tz='UTC'),  # 重複
            pd.Timestamp('2024-01-01 13:00:00', tz='UTC')
        ]
        
        duplicate_data = pd.DataFrame({
            'Close': [45000, 45100, 45200],
            'Volume': [5000, 5100, 5200]
        }, index=duplicate_dates)
        
        # 重複インデックスでも処理できることを確認
        result = calculator.calculate_temporal_features(duplicate_data)
        assert len(result) == 3
        assert result.iloc[0]['Hour_of_Day'] == result.iloc[1]['Hour_of_Day'] == 12
    
    def test_memory_usage_large_dataset(self, calculator):
        """大量データでのメモリ使用量テスト"""
        # メモリ使用量を測定
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 1年分のデータ（8760時間）
        large_dates = pd.date_range(
            '2024-01-01 00:00:00', 
            periods=8760, 
            freq='h', 
            tz='UTC'
        )
        
        large_data = pd.DataFrame({
            'Close': np.random.uniform(40000, 50000, len(large_dates)),
            'Volume': np.random.uniform(1000, 10000, len(large_dates))
        }, index=large_dates)
        
        # 特徴量計算実行
        result = calculator.calculate_temporal_features(large_data)
        
        # メモリ使用量チェック
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # メモリ増加が合理的な範囲内であることを確認（100MB以下）
        assert memory_increase < 100, f"Memory usage increased by {memory_increase:.2f}MB"
        
        # 結果の整合性確認
        assert len(result) == len(large_data)
        assert len(result.columns) == len(large_data.columns) + len(calculator.get_feature_names())
        
        # メモリクリーンアップ
        del large_data, result
        gc.collect()
    
    def test_performance_scalability(self, calculator):
        """パフォーマンススケーラビリティテスト"""
        import time
        
        data_sizes = [100, 1000, 10000]
        processing_times = []
        
        for size in data_sizes:
            dates = pd.date_range('2024-01-01', periods=size, freq='h', tz='UTC')
            data = pd.DataFrame({
                'Close': np.random.uniform(40000, 50000, size),
                'Volume': np.random.uniform(1000, 10000, size)
            }, index=dates)
            
            start_time = time.time()
            result = calculator.calculate_temporal_features(data)
            end_time = time.time()
            
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            
            # 結果の整合性確認
            assert len(result) == size
            
            # パフォーマンス要件（1000行/秒以上）
            rows_per_second = size / processing_time
            assert rows_per_second > 1000, f"Performance too slow: {rows_per_second:.0f} rows/sec for {size} rows"
        
        # スケーラビリティの確認（処理時間がデータサイズに対して線形的に増加）
        # 10倍のデータで処理時間が20倍以下であることを確認
        if len(processing_times) >= 2:
            ratio = processing_times[-1] / processing_times[0]
            size_ratio = data_sizes[-1] / data_sizes[0]
            assert ratio < size_ratio * 2, f"Poor scalability: time ratio {ratio:.2f} vs size ratio {size_ratio}"
    
    def test_concurrent_access(self, calculator):
        """並行アクセステスト"""
        import threading
        import queue
        
        def worker(calc, data, result_queue):
            try:
                result = calc.calculate_temporal_features(data)
                result_queue.put(('success', len(result)))
            except Exception as e:
                result_queue.put(('error', str(e)))
        
        # 複数のスレッドで同時実行
        dates = pd.date_range('2024-01-01', periods=100, freq='h', tz='UTC')
        test_data = pd.DataFrame({
            'Close': np.random.uniform(40000, 50000, 100),
            'Volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        result_queue = queue.Queue()
        threads = []
        
        # 5つのスレッドで同時実行
        for i in range(5):
            thread = threading.Thread(target=worker, args=(calculator, test_data, result_queue))
            threads.append(thread)
            thread.start()
        
        # 全スレッドの完了を待機
        for thread in threads:
            thread.join()
        
        # 結果の確認
        success_count = 0
        while not result_queue.empty():
            status, result = result_queue.get()
            if status == 'success':
                success_count += 1
                assert result == 100  # 全ての行が処理されたことを確認
            else:
                pytest.fail(f"Thread failed with error: {result}")
        
        assert success_count == 5, f"Expected 5 successful threads, got {success_count}"
    
    def test_error_recovery(self, calculator):
        """エラー回復テスト"""
        # 正常なデータ
        normal_dates = pd.date_range('2024-01-01', periods=10, freq='h', tz='UTC')
        normal_data = pd.DataFrame({
            'Close': np.random.uniform(40000, 50000, 10),
            'Volume': np.random.uniform(1000, 10000, 10)
        }, index=normal_dates)
        
        # 正常処理の確認
        result1 = calculator.calculate_temporal_features(normal_data)
        assert len(result1) == 10
        
        # 異常なデータでエラーを発生させる
        invalid_data = pd.DataFrame({
            'Close': [45000],
            'Volume': [5000]
        }, index=[0])  # 数値インデックス（DatetimeIndexではない）
        
        # エラーが適切に処理されることを確認
        with pytest.raises(ValueError):
            calculator.calculate_temporal_features(invalid_data)
        
        # エラー後も正常なデータを処理できることを確認
        result2 = calculator.calculate_temporal_features(normal_data)
        assert len(result2) == 10
        
        # 結果が一貫していることを確認
        pd.testing.assert_frame_equal(result1, result2)


if __name__ == "__main__":
    pytest.main([__file__])
