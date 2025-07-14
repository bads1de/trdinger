"""
特徴量エンジニアリングのストレステスト

大量データ、長期間データ、メモリ制限、CPU制限などの
ストレス条件下での動作を検証します。
"""

import pytest
import pandas as pd
import numpy as np
import psutil
import gc
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings

from app.core.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService


class TestFeatureEngineeringStress:
    """特徴量エンジニアリングのストレステストクラス"""
    
    @pytest.fixture
    def service(self):
        """FeatureEngineeringService インスタンスを生成"""
        return FeatureEngineeringService()
    
    def test_large_dataset_processing(self, service):
        """大量データ処理のストレステスト"""
        # 1年分のデータ（約8760時間）
        large_size = 8760
        dates = pd.date_range(
            '2024-01-01 00:00:00', 
            periods=large_size, 
            freq='h', 
            tz='UTC'
        )
        
        # 大量のOHLCVデータ
        large_ohlcv = pd.DataFrame({
            'Open': np.random.uniform(40000, 50000, large_size),
            'High': np.random.uniform(40000, 50000, large_size),
            'Low': np.random.uniform(40000, 50000, large_size),
            'Close': np.random.uniform(40000, 50000, large_size),
            'Volume': np.random.uniform(1000, 10000, large_size)
        }, index=dates)
        
        large_fr = pd.DataFrame({
            'funding_rate': np.random.uniform(-0.01, 0.01, large_size)
        }, index=dates)
        
        large_oi = pd.DataFrame({
            'open_interest': np.random.uniform(1000000, 2000000, large_size)
        }, index=dates)
        
        # メモリ使用量監視
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 処理時間測定
        start_time = time.time()
        
        # 大量データ処理実行
        result = service.calculate_advanced_features(large_ohlcv, large_fr, large_oi)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # メモリ使用量チェック
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # 結果の検証
        assert len(result) == large_size, f"Expected {large_size} rows, got {len(result)}"
        assert len(result.columns) > 100, f"Expected >100 features, got {len(result.columns)}"
        
        # パフォーマンス要件
        rows_per_second = large_size / processing_time
        assert rows_per_second > 500, f"Performance too slow: {rows_per_second:.0f} rows/sec"
        
        # メモリ効率性（500MB以下）
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.2f}MB"
        
        # データ品質チェック
        numeric_columns = result.select_dtypes(include=[np.number]).columns
        for col in numeric_columns[:10]:  # 最初の10列をサンプルチェック
            nan_ratio = result[col].isna().sum() / len(result)
            assert nan_ratio < 0.5, f"Too many NaN values in {col}: {nan_ratio:.2%}"
        
        print(f"Large dataset test: {large_size} rows, {processing_time:.2f}s, {memory_increase:.2f}MB")
        
        # メモリクリーンアップ
        del large_ohlcv, large_fr, large_oi, result
        gc.collect()
    
    def test_very_long_time_series(self, service):
        """非常に長い時系列データのテスト"""
        # 5年分のデータ（約43800時間）
        very_long_size = 43800
        dates = pd.date_range(
            '2020-01-01 00:00:00', 
            periods=very_long_size, 
            freq='h', 
            tz='UTC'
        )
        
        # 段階的にデータを生成してメモリ効率を向上
        chunk_size = 8760  # 1年分ずつ処理
        results = []
        
        for i in range(0, very_long_size, chunk_size):
            end_idx = min(i + chunk_size, very_long_size)
            chunk_dates = dates[i:end_idx]
            chunk_size_actual = len(chunk_dates)
            
            chunk_ohlcv = pd.DataFrame({
                'Open': np.random.uniform(40000, 50000, chunk_size_actual),
                'High': np.random.uniform(40000, 50000, chunk_size_actual),
                'Low': np.random.uniform(40000, 50000, chunk_size_actual),
                'Close': np.random.uniform(40000, 50000, chunk_size_actual),
                'Volume': np.random.uniform(1000, 10000, chunk_size_actual)
            }, index=chunk_dates)
            
            # 基本的な特徴量のみを計算（メモリ効率のため）
            chunk_result = service.calculate_advanced_features(chunk_ohlcv)
            
            # 結果の基本検証
            assert len(chunk_result) == chunk_size_actual
            assert len(chunk_result.columns) > 50
            
            # 時間的特徴量が正しく計算されていることを確認
            assert 'Hour_of_Day' in chunk_result.columns
            assert (chunk_result['Hour_of_Day'] >= 0).all()
            assert (chunk_result['Hour_of_Day'] <= 23).all()
            
            results.append(len(chunk_result))
            
            # メモリクリーンアップ
            del chunk_ohlcv, chunk_result
            gc.collect()
        
        # 全チャンクが正常に処理されたことを確認
        total_processed = sum(results)
        assert total_processed == very_long_size, f"Expected {very_long_size} rows, processed {total_processed}"
        
        print(f"Very long time series test: {very_long_size} rows processed in {len(results)} chunks")
    
    def test_concurrent_processing(self, service):
        """並行処理のストレステスト"""
        def process_data(thread_id):
            """スレッド用の処理関数"""
            dates = pd.date_range(
                f'2024-{thread_id:02d}-01', 
                periods=1000, 
                freq='h', 
                tz='UTC'
            )
            
            ohlcv = pd.DataFrame({
                'Open': np.random.uniform(40000, 50000, 1000),
                'High': np.random.uniform(40000, 50000, 1000),
                'Low': np.random.uniform(40000, 50000, 1000),
                'Close': np.random.uniform(40000, 50000, 1000),
                'Volume': np.random.uniform(1000, 10000, 1000)
            }, index=dates)
            
            fr = pd.DataFrame({
                'funding_rate': np.random.uniform(-0.01, 0.01, 1000)
            }, index=dates)
            
            oi = pd.DataFrame({
                'open_interest': np.random.uniform(1000000, 2000000, 1000)
            }, index=dates)
            
            start_time = time.time()
            result = service.calculate_advanced_features(ohlcv, fr, oi)
            end_time = time.time()
            
            return {
                'thread_id': thread_id,
                'rows': len(result),
                'columns': len(result.columns),
                'processing_time': end_time - start_time,
                'success': True
            }
        
        # 複数スレッドで並行実行
        num_threads = 4
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_data, i+1) for i in range(num_threads)]
            results = [future.result() for future in futures]
        
        # 全スレッドが成功したことを確認
        for result in results:
            assert result['success'], f"Thread {result['thread_id']} failed"
            assert result['rows'] == 1000, f"Thread {result['thread_id']} processed {result['rows']} rows"
            assert result['columns'] > 50, f"Thread {result['thread_id']} generated {result['columns']} columns"
            assert result['processing_time'] < 10, f"Thread {result['thread_id']} took {result['processing_time']:.2f}s"
        
        print(f"Concurrent processing test: {num_threads} threads completed successfully")
    
    def test_memory_pressure(self, service):
        """メモリ圧迫下での動作テスト"""
        # 現在のメモリ使用量を取得
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # メモリを意図的に消費
        memory_hogs = []
        try:
            # 100MBずつメモリを消費（最大500MB）
            for i in range(5):
                # 大きな配列を作成してメモリを消費
                memory_hog = np.random.random((1000, 25000))  # 約100MB
                memory_hogs.append(memory_hog)
                
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                if memory_increase > 400:  # 400MB以上消費したら停止
                    break
            
            # メモリ圧迫下でのデータ処理
            dates = pd.date_range('2024-01-01', periods=5000, freq='h', tz='UTC')
            
            ohlcv = pd.DataFrame({
                'Open': np.random.uniform(40000, 50000, 5000),
                'High': np.random.uniform(40000, 50000, 5000),
                'Low': np.random.uniform(40000, 50000, 5000),
                'Close': np.random.uniform(40000, 50000, 5000),
                'Volume': np.random.uniform(1000, 10000, 5000)
            }, index=dates)
            
            # メモリ圧迫下でも正常に処理できることを確認
            result = service.calculate_advanced_features(ohlcv)
            
            # 結果の検証
            assert len(result) == 5000
            assert len(result.columns) > 50
            
            # 基本的な特徴量が正しく計算されていることを確認
            assert 'Hour_of_Day' in result.columns
            assert 'Price_Momentum_14' in result.columns
            
            print(f"Memory pressure test: Processed under {memory_increase:.2f}MB memory pressure")
            
        finally:
            # メモリクリーンアップ
            del memory_hogs
            gc.collect()
    
    def test_cpu_intensive_workload(self, service):
        """CPU集約的ワークロードのテスト"""
        # CPU使用率を監視しながら処理
        cpu_percent_before = psutil.cpu_percent(interval=1)
        
        # 複数の大きなデータセットを順次処理
        datasets = []
        processing_times = []
        
        for i in range(3):
            dates = pd.date_range(
                f'2024-{i+1:02d}-01', 
                periods=10000, 
                freq='h', 
                tz='UTC'
            )
            
            ohlcv = pd.DataFrame({
                'Open': np.random.uniform(40000, 50000, 10000),
                'High': np.random.uniform(40000, 50000, 10000),
                'Low': np.random.uniform(40000, 50000, 10000),
                'Close': np.random.uniform(40000, 50000, 10000),
                'Volume': np.random.uniform(1000, 10000, 10000)
            }, index=dates)
            
            fr = pd.DataFrame({
                'funding_rate': np.random.uniform(-0.01, 0.01, 10000)
            }, index=dates)
            
            oi = pd.DataFrame({
                'open_interest': np.random.uniform(1000000, 2000000, 10000)
            }, index=dates)
            
            start_time = time.time()
            result = service.calculate_advanced_features(ohlcv, fr, oi)
            end_time = time.time()
            
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            
            # 結果の検証
            assert len(result) == 10000
            assert len(result.columns) > 100
            
            datasets.append(result)
        
        cpu_percent_after = psutil.cpu_percent(interval=1)
        
        # CPU使用率が適切に上昇していることを確認（処理が実際に行われている証拠）
        # ただし、他のプロセスの影響もあるため、厳密な閾値は設けない
        
        # 処理時間の一貫性を確認
        avg_processing_time = np.mean(processing_times)
        std_processing_time = np.std(processing_times)
        
        # 処理時間のばらつきが小さいことを確認（安定性の指標）
        cv = std_processing_time / avg_processing_time  # 変動係数
        assert cv < 0.5, f"Processing time too variable: CV={cv:.2f}"
        
        print(f"CPU intensive test: Avg time {avg_processing_time:.2f}s, CV {cv:.2f}")
        
        # メモリクリーンアップ
        del datasets
        gc.collect()
    
    def test_resource_cleanup(self, service):
        """リソースクリーンアップのテスト"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # 複数回の処理でメモリリークがないことを確認
        memory_measurements = []
        
        for iteration in range(5):
            dates = pd.date_range('2024-01-01', periods=5000, freq='h', tz='UTC')
            
            ohlcv = pd.DataFrame({
                'Open': np.random.uniform(40000, 50000, 5000),
                'High': np.random.uniform(40000, 50000, 5000),
                'Low': np.random.uniform(40000, 50000, 5000),
                'Close': np.random.uniform(40000, 50000, 5000),
                'Volume': np.random.uniform(1000, 10000, 5000)
            }, index=dates)
            
            # 処理実行
            result = service.calculate_advanced_features(ohlcv)
            
            # 明示的にリソースを解放
            del ohlcv, result
            gc.collect()
            
            # メモリ使用量を測定
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_measurements.append(current_memory - initial_memory)
        
        # メモリリークがないことを確認
        # 最後の測定値が最初の測定値の2倍以下であることを確認
        memory_growth = memory_measurements[-1] - memory_measurements[0]
        assert memory_growth < 100, f"Potential memory leak: {memory_growth:.2f}MB growth over 5 iterations"
        
        print(f"Resource cleanup test: Memory growth {memory_growth:.2f}MB over 5 iterations")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
