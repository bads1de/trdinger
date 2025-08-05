#!/usr/bin/env python3
"""
パフォーマンステストスイート

MLトレーニングシステムのパフォーマンス特性を詳細に検証します。
- 大量データ処理性能
- メモリ使用量最適化
- 実行時間ベンチマーク
- スケーラビリティ検証
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
import time
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
import threading
import multiprocessing

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """パフォーマンス指標データクラス"""
    test_name: str
    data_size: int
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_usage_percent: float
    throughput_records_per_sec: float
    success: bool
    error_message: str = ""
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

class PerformanceTestSuite:
    """パフォーマンステストスイート"""
    
    def __init__(self):
        self.results: List[PerformanceMetrics] = []
        self.process = psutil.Process()
        
    def create_large_dataset(self, rows: int, complexity: str = "medium") -> pd.DataFrame:
        """大量データセットを作成"""
        logger.info(f"📊 {rows:,}行のデータセット作成開始 (複雑度: {complexity})")
        
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=rows, freq='h')
        
        # 基本価格データ
        base_price = 50000
        trend = np.linspace(0, 10000, rows)
        
        if complexity == "simple":
            volatility = np.random.normal(0, 500, rows)
        elif complexity == "medium":
            volatility = np.random.normal(0, 1000, rows) + np.sin(np.arange(rows) * 0.01) * 500
        else:  # complex
            volatility = (np.random.normal(0, 1500, rows) + 
                         np.sin(np.arange(rows) * 0.01) * 500 +
                         np.cos(np.arange(rows) * 0.005) * 300)
        
        close_prices = base_price + trend + volatility
        
        data = {
            'Open': close_prices + np.random.normal(0, 100, rows),
            'High': close_prices + np.abs(np.random.normal(200, 150, rows)),
            'Low': close_prices - np.abs(np.random.normal(200, 150, rows)),
            'Close': close_prices,
            'Volume': np.random.lognormal(10, 0.5, rows),
        }
        
        df = pd.DataFrame(data, index=dates)
        
        # 価格整合性を確保
        df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
        df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)
        
        logger.info(f"✅ データセット作成完了: {df.shape}, メモリ使用量: {df.memory_usage(deep=True).sum() / 1024**2:.2f}MB")
        return df
    
    def monitor_system_resources(self, duration: float) -> Dict[str, float]:
        """システムリソースを監視"""
        start_time = time.time()
        cpu_samples = []
        memory_samples = []
        
        while time.time() - start_time < duration:
            cpu_samples.append(psutil.cpu_percent(interval=0.1))
            memory_samples.append(self.process.memory_info().rss / 1024**2)
            time.sleep(0.1)
        
        return {
            'avg_cpu_percent': np.mean(cpu_samples),
            'max_cpu_percent': np.max(cpu_samples),
            'avg_memory_mb': np.mean(memory_samples),
            'peak_memory_mb': np.max(memory_samples)
        }
    
    def test_large_data_processing(self):
        """大量データ処理性能テスト"""
        logger.info("🚀 大量データ処理性能テスト開始")
        
        data_sizes = [1000, 5000, 10000, 20000]
        
        for size in data_sizes:
            logger.info(f"📊 {size:,}行データでのテスト開始")
            
            # メモリクリア
            gc.collect()
            initial_memory = self.process.memory_info().rss / 1024**2
            
            start_time = time.time()
            
            try:
                # 大量データを作成
                large_data = self.create_large_dataset(size, "medium")
                
                # MLトレーニングを実行
                from app.services.ml.single_model.single_model_trainer import SingleModelTrainer
                
                trainer = SingleModelTrainer(model_type="lightgbm")
                
                # リソース監視を開始
                monitor_thread = threading.Thread(
                    target=lambda: self.monitor_system_resources(60),
                    daemon=True
                )
                monitor_thread.start()
                
                # 学習実行
                result = trainer.train_model(
                    training_data=large_data,
                    save_model=False,
                    threshold_up=0.02,
                    threshold_down=-0.02
                )
                
                execution_time = time.time() - start_time
                final_memory = self.process.memory_info().rss / 1024**2
                memory_usage = final_memory - initial_memory
                
                # スループット計算
                throughput = size / execution_time
                
                self.results.append(PerformanceMetrics(
                    test_name=f"大量データ処理_{size:,}行",
                    data_size=size,
                    execution_time=execution_time,
                    memory_usage_mb=memory_usage,
                    peak_memory_mb=final_memory,
                    cpu_usage_percent=psutil.cpu_percent(),
                    throughput_records_per_sec=throughput,
                    success=True,
                    additional_metrics={
                        'accuracy': result.get('accuracy', 0),
                        'feature_count': result.get('feature_count', 0),
                        'model_size_mb': 0  # モデル保存していないため0
                    }
                ))
                
                logger.info(f"✅ {size:,}行処理完了: {execution_time:.2f}秒, {throughput:.1f}行/秒")
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                self.results.append(PerformanceMetrics(
                    test_name=f"大量データ処理_{size:,}行",
                    data_size=size,
                    execution_time=execution_time,
                    memory_usage_mb=0,
                    peak_memory_mb=0,
                    cpu_usage_percent=0,
                    throughput_records_per_sec=0,
                    success=False,
                    error_message=str(e)
                ))
                
                logger.error(f"❌ {size:,}行処理失敗: {e}")
    
    def test_memory_optimization(self):
        """メモリ使用量最適化テスト"""
        logger.info("💾 メモリ使用量最適化テスト開始")
        
        # 異なる複雑度でのメモリ使用量を比較
        complexities = ["simple", "medium", "complex"]
        data_size = 5000
        
        for complexity in complexities:
            logger.info(f"🧮 複雑度 '{complexity}' でのメモリテスト")
            
            gc.collect()
            initial_memory = self.process.memory_info().rss / 1024**2
            
            start_time = time.time()
            
            try:
                # データ作成
                test_data = self.create_large_dataset(data_size, complexity)
                
                # 特徴量エンジニアリング
                from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
                
                fe_service = FeatureEngineeringService()
                features = fe_service.calculate_advanced_features(test_data)
                
                execution_time = time.time() - start_time
                final_memory = self.process.memory_info().rss / 1024**2
                memory_usage = final_memory - initial_memory
                
                # データサイズ計算
                data_memory = test_data.memory_usage(deep=True).sum() / 1024**2
                features_memory = features.memory_usage(deep=True).sum() / 1024**2
                
                self.results.append(PerformanceMetrics(
                    test_name=f"メモリ最適化_{complexity}",
                    data_size=data_size,
                    execution_time=execution_time,
                    memory_usage_mb=memory_usage,
                    peak_memory_mb=final_memory,
                    cpu_usage_percent=psutil.cpu_percent(),
                    throughput_records_per_sec=data_size / execution_time,
                    success=True,
                    additional_metrics={
                        'data_memory_mb': data_memory,
                        'features_memory_mb': features_memory,
                        'memory_efficiency': features_memory / data_memory,
                        'feature_count': len(features.columns)
                    }
                ))
                
                logger.info(f"✅ {complexity}複雑度完了: メモリ使用量 {memory_usage:.2f}MB")
                
                # メモリクリア
                del test_data, features
                gc.collect()
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                self.results.append(PerformanceMetrics(
                    test_name=f"メモリ最適化_{complexity}",
                    data_size=data_size,
                    execution_time=execution_time,
                    memory_usage_mb=0,
                    peak_memory_mb=0,
                    cpu_usage_percent=0,
                    throughput_records_per_sec=0,
                    success=False,
                    error_message=str(e)
                ))
                
                logger.error(f"❌ {complexity}複雑度失敗: {e}")
    
    def test_concurrent_processing(self):
        """並行処理性能テスト"""
        logger.info("⚡ 並行処理性能テスト開始")
        
        data_size = 2000
        test_data = self.create_large_dataset(data_size, "medium")
        
        # シーケンシャル処理
        logger.info("📊 シーケンシャル処理テスト")
        start_time = time.time()
        
        try:
            from app.services.ml.single_model.single_model_trainer import SingleModelTrainer
            
            trainer = SingleModelTrainer(model_type="lightgbm")
            result1 = trainer.train_model(
                training_data=test_data,
                save_model=False,
                threshold_up=0.02,
                threshold_down=-0.02
            )
            
            sequential_time = time.time() - start_time
            
            self.results.append(PerformanceMetrics(
                test_name="並行処理_シーケンシャル",
                data_size=data_size,
                execution_time=sequential_time,
                memory_usage_mb=self.process.memory_info().rss / 1024**2,
                peak_memory_mb=self.process.memory_info().rss / 1024**2,
                cpu_usage_percent=psutil.cpu_percent(),
                throughput_records_per_sec=data_size / sequential_time,
                success=True,
                additional_metrics={
                    'processing_type': 'sequential',
                    'accuracy': result1.get('accuracy', 0)
                }
            ))
            
            logger.info(f"✅ シーケンシャル処理完了: {sequential_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"❌ シーケンシャル処理失敗: {e}")
    
    def test_scalability_limits(self):
        """スケーラビリティ限界テスト"""
        logger.info("📈 スケーラビリティ限界テスト開始")
        
        # 段階的にデータサイズを増加
        max_size = 50000
        step_size = 10000
        
        for size in range(step_size, max_size + 1, step_size):
            logger.info(f"🔍 {size:,}行でのスケーラビリティテスト")
            
            gc.collect()
            start_time = time.time()
            initial_memory = self.process.memory_info().rss / 1024**2
            
            try:
                # メモリ制限チェック
                available_memory = psutil.virtual_memory().available / 1024**2
                if available_memory < 1000:  # 1GB未満の場合は停止
                    logger.warning(f"⚠️ メモリ不足のため{size:,}行テストをスキップ")
                    break
                
                # データ作成（軽量版）
                test_data = self.create_large_dataset(size, "simple")
                
                # 軽量な特徴量エンジニアリング
                from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
                
                fe_service = FeatureEngineeringService()
                features = fe_service.calculate_basic_features(test_data)
                
                execution_time = time.time() - start_time
                final_memory = self.process.memory_info().rss / 1024**2
                memory_usage = final_memory - initial_memory
                
                self.results.append(PerformanceMetrics(
                    test_name=f"スケーラビリティ_{size:,}行",
                    data_size=size,
                    execution_time=execution_time,
                    memory_usage_mb=memory_usage,
                    peak_memory_mb=final_memory,
                    cpu_usage_percent=psutil.cpu_percent(),
                    throughput_records_per_sec=size / execution_time,
                    success=True,
                    additional_metrics={
                        'feature_count': len(features.columns),
                        'memory_per_record_kb': (memory_usage * 1024) / size,
                        'available_memory_mb': available_memory
                    }
                ))
                
                logger.info(f"✅ {size:,}行完了: {execution_time:.2f}秒, メモリ {memory_usage:.2f}MB")
                
                # メモリクリア
                del test_data, features
                gc.collect()
                
                # メモリ使用量が急激に増加した場合は停止
                if memory_usage > 2000:  # 2GB以上
                    logger.warning(f"⚠️ メモリ使用量が{memory_usage:.2f}MBに達したため停止")
                    break
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                self.results.append(PerformanceMetrics(
                    test_name=f"スケーラビリティ_{size:,}行",
                    data_size=size,
                    execution_time=execution_time,
                    memory_usage_mb=0,
                    peak_memory_mb=0,
                    cpu_usage_percent=0,
                    throughput_records_per_sec=0,
                    success=False,
                    error_message=str(e)
                ))
                
                logger.error(f"❌ {size:,}行失敗: {e}")
                break  # エラーが発生した場合は停止

if __name__ == "__main__":
    logger.info("🚀 パフォーマンステストスイート開始")
    
    test_suite = PerformanceTestSuite()
    
    # 各パフォーマンステストを実行
    test_suite.test_large_data_processing()
    test_suite.test_memory_optimization()
    test_suite.test_concurrent_processing()
    test_suite.test_scalability_limits()
    
    # 結果サマリー
    total_tests = len(test_suite.results)
    successful_tests = sum(1 for r in test_suite.results if r.success)
    
    print("\n" + "="*80)
    print("🚀 パフォーマンステスト結果")
    print("="*80)
    print(f"📊 総テスト数: {total_tests}")
    print(f"✅ 成功: {successful_tests}")
    print(f"❌ 失敗: {total_tests - successful_tests}")
    print(f"📈 成功率: {(successful_tests/total_tests*100):.1f}%")
    
    print("\n📊 パフォーマンス詳細:")
    for result in test_suite.results:
        status = "✅" if result.success else "❌"
        print(f"{status} {result.test_name}")
        if result.success:
            print(f"   実行時間: {result.execution_time:.2f}秒")
            print(f"   スループット: {result.throughput_records_per_sec:.1f}行/秒")
            print(f"   メモリ使用量: {result.memory_usage_mb:.2f}MB")
    
    print("="*80)
    
    logger.info("🎯 パフォーマンステストスイート完了")
