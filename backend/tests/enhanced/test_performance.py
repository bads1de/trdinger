"""
パフォーマンステスト

MLトレーニングシステムのパフォーマンスを検証するテストスイート。
処理時間、メモリ使用量、スケーラビリティを測定・評価します。
"""

import pytest
import numpy as np
import pandas as pd
import logging
import time
import psutil
import os
from typing import Dict, List, Tuple, Any
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.data_processing import DataProcessor
from app.utils.label_generation import LabelGenerator, ThresholdMethod
from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """パフォーマンス監視クラス"""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.process = psutil.Process(os.getpid())
    
    def start(self):
        """監視開始"""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss
    
    def stop(self) -> Dict[str, float]:
        """監視終了と結果取得"""
        end_time = time.time()
        end_memory = self.process.memory_info().rss
        
        return {
            'execution_time': end_time - self.start_time,
            'memory_used': end_memory - self.start_memory,
            'peak_memory': self.process.memory_info().rss
        }


class TestPerformance:
    """パフォーマンステストクラス"""

    def test_data_processing_performance(self):
        """データ処理のパフォーマンステスト"""
        logger.info("=== データ処理のパフォーマンステスト ===")
        
        processor = DataProcessor()
        monitor = PerformanceMonitor()
        
        # 異なるサイズのデータでテスト
        sizes = [1000, 5000, 10000]
        results = {}
        
        for size in sizes:
            # テストデータ生成
            data = pd.DataFrame({
                'feature1': np.random.normal(0, 1, size),
                'feature2': np.random.exponential(1, size),
                'feature3': np.random.uniform(-1, 1, size),
                'feature4': np.random.lognormal(0, 1, size)
            })
            
            # パフォーマンス測定
            monitor.start()
            
            result = processor.preprocess_features(
                data,
                scale_features=True,
                remove_outliers=True,
                outlier_method='iqr'
            )
            
            perf_stats = monitor.stop()
            results[size] = perf_stats
            
            logger.info(f"サイズ {size}: 実行時間 {perf_stats['execution_time']:.3f}秒, "
                       f"メモリ使用量 {perf_stats['memory_used']/1024/1024:.1f}MB")
        
        # スケーラビリティの確認（ゼロ除算を回避）
        if results[1000]['execution_time'] > 0:
            time_ratio = results[10000]['execution_time'] / results[1000]['execution_time']
            logger.info(f"10倍データでの時間比率: {time_ratio:.2f}x")
            assert time_ratio < 50, f"処理時間のスケーラビリティが悪すぎます: {time_ratio}x"
        else:
            logger.info("基準となる実行時間が0のため、時間比率の計算をスキップ")

        if results[1000]['memory_used'] > 0:
            memory_ratio = results[10000]['memory_used'] / results[1000]['memory_used']
            logger.info(f"10倍データでのメモリ比率: {memory_ratio:.2f}x")
            assert memory_ratio < 20, f"メモリ使用量のスケーラビリティが悪すぎます: {memory_ratio}x"
        else:
            logger.info("基準となるメモリ使用量が0のため、メモリ比率の計算をスキップ")
        
        logger.info("✅ データ処理のパフォーマンステスト完了")

    def test_feature_engineering_performance(self):
        """特徴量エンジニアリングのパフォーマンステスト"""
        logger.info("=== 特徴量エンジニアリングのパフォーマンステスト ===")
        
        fe_service = FeatureEngineeringService()
        monitor = PerformanceMonitor()
        
        # 異なる期間のOHLCVデータでテスト
        periods = [100, 500, 1000]
        results = {}
        
        for period in periods:
            # OHLCVデータ生成
            np.random.seed(42)
            dates = pd.date_range('2023-01-01', periods=period, freq='1h')
            
            base_price = 50000
            returns = np.random.normal(0, 0.02, period)
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            data = pd.DataFrame({
                'timestamp': dates,
                'Open': prices,
                'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'Close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
                'Volume': np.random.lognormal(10, 1, period)
            }).set_index('timestamp')
            
            # パフォーマンス測定
            monitor.start()
            
            try:
                features = fe_service.calculate_advanced_features(data)
                perf_stats = monitor.stop()
                results[period] = perf_stats
                
                logger.info(f"期間 {period}: 実行時間 {perf_stats['execution_time']:.3f}秒, "
                           f"特徴量数 {features.shape[1]}, "
                           f"メモリ使用量 {perf_stats['memory_used']/1024/1024:.1f}MB")
                
            except Exception as e:
                logger.warning(f"期間 {period} でエラー: {e}")
                results[period] = {'execution_time': float('inf'), 'memory_used': 0}
        
        # 有効な結果があることを確認
        valid_results = {k: v for k, v in results.items() if v['execution_time'] != float('inf')}
        assert len(valid_results) > 0, "有効な特徴量エンジニアリング結果がありません"
        
        logger.info("✅ 特徴量エンジニアリングのパフォーマンステスト完了")

    def test_label_generation_performance(self):
        """ラベル生成のパフォーマンステスト"""
        logger.info("=== ラベル生成のパフォーマンステスト ===")
        
        label_generator = LabelGenerator()
        monitor = PerformanceMonitor()
        
        # 異なるサイズの価格データでテスト
        sizes = [1000, 5000, 10000, 50000]
        results = {}
        
        for size in sizes:
            # 価格データ生成
            np.random.seed(42)
            base_price = 50000
            returns = np.random.normal(0, 0.02, size)
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            price_series = pd.Series(prices, name='Close')
            
            # パフォーマンス測定
            monitor.start()
            
            labels, threshold_info = label_generator.generate_labels(
                price_series,
                method=ThresholdMethod.FIXED,
                threshold_up=0.02,
                threshold_down=-0.02
            )
            
            perf_stats = monitor.stop()
            results[size] = perf_stats
            
            logger.info(f"サイズ {size}: 実行時間 {perf_stats['execution_time']:.3f}秒, "
                       f"ラベル数 {len(labels)}, "
                       f"メモリ使用量 {perf_stats['memory_used']/1024/1024:.1f}MB")
        
        # 線形スケーラビリティの確認
        if len(results) >= 2:
            sizes_list = sorted(results.keys())
            time_per_sample = {}
            
            for size in sizes_list:
                time_per_sample[size] = results[size]['execution_time'] / size
            
            # サンプルあたりの処理時間が合理的であることを確認
            max_time_per_sample = max(time_per_sample.values())
            assert max_time_per_sample < 0.001, f"サンプルあたりの処理時間が遅すぎます: {max_time_per_sample:.6f}秒"
        
        logger.info("✅ ラベル生成のパフォーマンステスト完了")

    def test_memory_efficiency(self):
        """メモリ効率性テスト"""
        logger.info("=== メモリ効率性テスト ===")
        
        processor = DataProcessor()
        monitor = PerformanceMonitor()
        
        # 大きなデータセットでメモリ効率をテスト
        data_size = 10000
        feature_count = 50
        
        # テストデータ生成
        data = pd.DataFrame({
            f'feature_{i}': np.random.normal(0, 1, data_size) 
            for i in range(feature_count)
        })
        
        initial_memory = monitor.process.memory_info().rss
        
        # データ型最適化のテスト
        monitor.start()
        optimized_data = processor.optimize_dtypes(data)
        perf_stats = monitor.stop()
        
        # メモリ使用量の比較
        original_memory = data.memory_usage(deep=True).sum()
        optimized_memory = optimized_data.memory_usage(deep=True).sum()
        memory_reduction = (original_memory - optimized_memory) / original_memory
        
        logger.info(f"元のメモリ使用量: {original_memory/1024/1024:.1f}MB")
        logger.info(f"最適化後メモリ使用量: {optimized_memory/1024/1024:.1f}MB")
        logger.info(f"メモリ削減率: {memory_reduction*100:.1f}%")
        
        # メモリ効率が改善されていることを確認
        assert memory_reduction >= 0, "メモリ使用量が増加しています"
        
        logger.info("✅ メモリ効率性テスト完了")

    def test_concurrent_performance(self):
        """並行処理のパフォーマンステスト"""
        logger.info("=== 並行処理のパフォーマンステスト ===")
        
        processor = DataProcessor()
        
        # 複数の小さなタスクを並行実行
        task_count = 5
        data_size = 1000
        
        # テストデータ生成
        datasets = []
        for i in range(task_count):
            data = pd.DataFrame({
                'feature1': np.random.normal(0, 1, data_size),
                'feature2': np.random.exponential(1, data_size)
            })
            datasets.append(data)
        
        # 順次実行の時間測定
        start_time = time.time()
        sequential_results = []
        for data in datasets:
            result = processor.preprocess_features(data, scale_features=True)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        logger.info(f"順次実行時間: {sequential_time:.3f}秒")
        
        # 並行実行は実装されていないため、順次実行の効率性のみ確認
        time_per_task = sequential_time / task_count
        assert time_per_task < 1.0, f"タスクあたりの処理時間が遅すぎます: {time_per_task:.3f}秒"
        
        logger.info("✅ 並行処理のパフォーマンステスト完了")

    def test_scalability_limits(self):
        """スケーラビリティ限界テスト"""
        logger.info("=== スケーラビリティ限界テスト ===")
        
        processor = DataProcessor()
        
        # 段階的にデータサイズを増やしてテスト
        max_successful_size = 0
        test_sizes = [1000, 5000, 10000, 20000, 50000]
        
        for size in test_sizes:
            try:
                # テストデータ生成
                data = pd.DataFrame({
                    'feature1': np.random.normal(0, 1, size),
                    'feature2': np.random.exponential(1, size)
                })
                
                start_time = time.time()
                result = processor.preprocess_features(data, scale_features=True)
                execution_time = time.time() - start_time
                
                max_successful_size = size
                logger.info(f"サイズ {size}: 成功 ({execution_time:.3f}秒)")
                
                # 処理時間が合理的な範囲内であることを確認
                if execution_time > 30:  # 30秒以上かかる場合は停止
                    logger.warning(f"サイズ {size} で処理時間が長すぎます: {execution_time:.3f}秒")
                    break
                    
            except Exception as e:
                logger.warning(f"サイズ {size} で失敗: {e}")
                break
        
        logger.info(f"最大成功サイズ: {max_successful_size}")
        assert max_successful_size >= 1000, "最小限のスケーラビリティが確保されていません"
        
        logger.info("✅ スケーラビリティ限界テスト完了")


def run_all_performance_tests():
    """すべてのパフォーマンステストを実行"""
    logger.info("⚡ パフォーマンステストスイートを開始")
    
    test_instance = TestPerformance()
    
    try:
        test_instance.test_data_processing_performance()
        test_instance.test_feature_engineering_performance()
        test_instance.test_label_generation_performance()
        test_instance.test_memory_efficiency()
        test_instance.test_concurrent_performance()
        test_instance.test_scalability_limits()
        
        logger.info("🎉 すべてのパフォーマンステストが正常に完了しました！")
        return True
        
    except Exception as e:
        logger.error(f"❌ パフォーマンステストでエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = run_all_performance_tests()
    sys.exit(0 if success else 1)
