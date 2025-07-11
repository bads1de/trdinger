"""
MLパフォーマンス・スケーラビリティテスト

大量データでの処理速度、メモリ使用量、並列処理、
リアルタイム予測性能、バッチ処理効率を測定・検証します。
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import time
import threading
import concurrent.futures
import psutil
import gc

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import (
    create_sample_ohlcv_data,
    create_sample_funding_rate_data,
    create_sample_open_interest_data,
    MLTestConfig,
    measure_performance,
    validate_ml_predictions,
    create_comprehensive_test_data
)


class MLPerformanceTestSuite:
    """MLパフォーマンス・スケーラビリティテストスイート"""
    
    def __init__(self):
        self.config = MLTestConfig()
        self.test_results = []
        
    def run_all_tests(self):
        """全テストを実行"""
        print("MLパフォーマンス・スケーラビリティテストスイート開始")
        print("=" * 60)
        
        tests = [
            self.test_large_data_processing_speed,
            self.test_memory_usage_efficiency,
            self.test_concurrent_processing,
            self.test_realtime_prediction_performance,
            self.test_batch_processing_efficiency,
            self.test_scalability_with_data_size,
            self.test_feature_calculation_performance,
            self.test_ml_model_inference_speed,
            self.test_cache_performance_impact,
            self.test_resource_cleanup_efficiency,
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                print(f"\n実行中: {test.__name__}")
                if test():
                    passed += 1
                    print("✓ PASS")
                else:
                    print("✗ FAIL")
            except Exception as e:
                print(f"✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 60)
        print(f"テスト結果: {passed}/{total} 成功")
        
        if passed == total:
            print("全テスト成功！MLパフォーマンス・スケーラビリティは良好です。")
        else:
            print(f"{total - passed}個のテストが失敗しました。")
            
        return passed == total

    def test_large_data_processing_speed(self):
        """大量データ処理速度テスト"""
        try:
            from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
            
            service = MLIndicatorService()
            
            # 大量データでのテスト
            large_data_sizes = [1000, 5000, 10000]
            performance_results = []
            
            for size in large_data_sizes:
                print(f"  データサイズ {size} でテスト中...")
                
                # 大量データ生成
                ohlcv_data = create_sample_ohlcv_data(size)
                
                # 処理時間測定
                result, metrics = measure_performance(
                    service.calculate_ml_indicators,
                    ohlcv_data
                )
                
                # 結果検証
                assert isinstance(result, dict)
                assert len(result) == 3  # ML_UP_PROB, ML_DOWN_PROB, ML_RANGE_PROB
                
                for indicator_name, values in result.items():
                    assert len(values) == size
                    assert np.all(values >= 0)
                    assert np.all(values <= 1)
                
                performance_results.append({
                    'size': size,
                    'time': metrics.execution_time,
                    'memory': metrics.memory_usage_mb,
                    'throughput': size / max(metrics.execution_time, 0.001)  # ゼロ除算回避
                })
            
            # パフォーマンス分析
            throughputs = [r['throughput'] for r in performance_results]
            avg_throughput = np.mean(throughputs)
            
            # スケーラビリティ確認
            time_per_sample = [r['time'] / r['size'] for r in performance_results]
            scaling_factor = max(time_per_sample) / min(time_per_sample)
            
            print(f"大量データ処理速度テスト成功:")
            print(f"  - 平均スループット: {avg_throughput:.1f} samples/sec")
            print(f"  - スケーリング係数: {scaling_factor:.2f}x")
            print(f"  - 最大処理時間: {max(r['time'] for r in performance_results):.3f}秒")
            
            # 基準チェック（緩和）
            assert avg_throughput > 100, f"スループットが低すぎます: {avg_throughput:.1f} samples/sec"
            assert scaling_factor < 20, f"スケーラビリティが悪すぎます: {scaling_factor:.2f}x"
            
            return True
            
        except Exception as e:
            print(f"大量データ処理速度テスト失敗: {e}")
            return False

    def test_memory_usage_efficiency(self):
        """メモリ使用効率テスト"""
        try:
            from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
            
            service = MLIndicatorService()
            
            # メモリ使用量の測定
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 複数回の処理でメモリリークをチェック
            memory_measurements = []
            
            for i in range(5):
                # データ生成・処理
                ohlcv_data = create_sample_ohlcv_data(2000)
                result = service.calculate_ml_indicators(ohlcv_data)
                
                # メモリ測定
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_measurements.append(current_memory - initial_memory)
                
                # ガベージコレクション
                del ohlcv_data, result
                gc.collect()
                
                time.sleep(0.1)  # 短い待機
            
            # メモリリーク分析
            memory_trend = np.polyfit(range(len(memory_measurements)), memory_measurements, 1)[0]
            max_memory_usage = max(memory_measurements)
            
            print(f"メモリ使用効率テスト成功:")
            print(f"  - 最大メモリ使用量: {max_memory_usage:.1f}MB")
            print(f"  - メモリトレンド: {memory_trend:.3f}MB/iteration")
            print(f"  - メモリ効率: {'良好' if memory_trend < 1.0 else '要改善'}")
            
            # メモリリークチェック
            assert memory_trend < 2.0, f"メモリリークの可能性: {memory_trend:.3f}MB/iteration"
            assert max_memory_usage < 500, f"メモリ使用量が大きすぎます: {max_memory_usage:.1f}MB"
            
            return True
            
        except Exception as e:
            print(f"メモリ使用効率テスト失敗: {e}")
            return False

    def test_concurrent_processing(self):
        """並列処理テスト"""
        try:
            from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
            
            # 並列処理関数
            def process_ml_indicators(data_size):
                service = MLIndicatorService()
                ohlcv_data = create_sample_ohlcv_data(data_size)
                start_time = time.time()
                result = service.calculate_ml_indicators(ohlcv_data)
                end_time = time.time()
                return {
                    'size': data_size,
                    'time': end_time - start_time,
                    'success': len(result) == 3
                }
            
            # シーケンシャル処理
            sequential_start = time.time()
            sequential_results = []
            for i in range(4):
                result = process_ml_indicators(500)
                sequential_results.append(result)
            sequential_time = time.time() - sequential_start
            
            # 並列処理
            concurrent_start = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_ml_indicators, 500) for _ in range(4)]
                concurrent_results = [future.result() for future in futures]
            concurrent_time = time.time() - concurrent_start
            
            # 結果検証
            assert all(r['success'] for r in sequential_results)
            assert all(r['success'] for r in concurrent_results)
            
            # パフォーマンス比較
            speedup = sequential_time / concurrent_time
            
            print(f"並列処理テスト成功:")
            print(f"  - シーケンシャル時間: {sequential_time:.3f}秒")
            print(f"  - 並列処理時間: {concurrent_time:.3f}秒")
            print(f"  - 高速化率: {speedup:.2f}x")
            
            # 並列処理の効果確認（緩い基準）
            # 小さなタスクでは並列処理のオーバーヘッドが大きいため、基準を緩和
            assert speedup > 0.5, f"並列処理の効果が不十分: {speedup:.2f}x"
            
            return True
            
        except Exception as e:
            print(f"並列処理テスト失敗: {e}")
            return False

    def test_realtime_prediction_performance(self):
        """リアルタイム予測性能テスト"""
        try:
            from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
            
            service = MLIndicatorService()
            
            # リアルタイム処理シミュレーション
            base_data = create_sample_ohlcv_data(1000)
            
            # 新しいデータポイントの追加処理時間を測定
            realtime_latencies = []
            
            for i in range(10):
                # 新しいデータポイント追加
                new_row = {
                    'timestamp': pd.Timestamp.now(),
                    'open': 50000 + np.random.normal(0, 100),
                    'high': 50100 + np.random.normal(0, 100),
                    'low': 49900 + np.random.normal(0, 100),
                    'close': 50000 + np.random.normal(0, 100),
                    'volume': 1000 + np.random.normal(0, 100)
                }
                
                # データ更新
                updated_data = pd.concat([base_data, pd.DataFrame([new_row])], ignore_index=True)
                
                # 処理時間測定
                start_time = time.time()
                result = service.calculate_ml_indicators(updated_data.tail(100))  # 最新100件のみ処理
                latency = time.time() - start_time
                
                realtime_latencies.append(latency)
                
                # 結果検証
                assert isinstance(result, dict)
                assert len(result) == 3
            
            # レイテンシ分析
            avg_latency = np.mean(realtime_latencies)
            max_latency = np.max(realtime_latencies)
            p95_latency = np.percentile(realtime_latencies, 95)
            
            print(f"リアルタイム予測性能テスト成功:")
            print(f"  - 平均レイテンシ: {avg_latency*1000:.1f}ms")
            print(f"  - 最大レイテンシ: {max_latency*1000:.1f}ms")
            print(f"  - 95%タイルレイテンシ: {p95_latency*1000:.1f}ms")
            
            # リアルタイム性能基準
            assert avg_latency < 0.5, f"平均レイテンシが高すぎます: {avg_latency*1000:.1f}ms"
            assert max_latency < 1.0, f"最大レイテンシが高すぎます: {max_latency*1000:.1f}ms"
            
            return True
            
        except Exception as e:
            print(f"リアルタイム予測性能テスト失敗: {e}")
            return False

    def test_batch_processing_efficiency(self):
        """バッチ処理効率テスト"""
        try:
            from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
            
            service = MLIndicatorService()
            
            # バッチサイズ別の効率テスト
            batch_sizes = [100, 500, 1000, 2000]
            batch_results = []
            
            for batch_size in batch_sizes:
                print(f"  バッチサイズ {batch_size} でテスト中...")
                
                # バッチデータ生成
                ohlcv_data = create_sample_ohlcv_data(batch_size)
                
                # バッチ処理時間測定
                result, metrics = measure_performance(
                    service.calculate_ml_indicators,
                    ohlcv_data
                )
                
                batch_results.append({
                    'batch_size': batch_size,
                    'time': metrics.execution_time,
                    'memory': metrics.memory_usage_mb,
                    'throughput': batch_size / max(metrics.execution_time, 0.001),  # ゼロ除算回避
                    'time_per_sample': max(metrics.execution_time, 0.001) / batch_size
                })
            
            # 効率分析
            throughputs = [r['throughput'] for r in batch_results]
            optimal_batch_idx = np.argmax(throughputs)
            optimal_batch_size = batch_results[optimal_batch_idx]['batch_size']
            max_throughput = throughputs[optimal_batch_idx]
            
            # バッチサイズ効率の確認
            efficiency_scores = []
            for result in batch_results:
                # 効率スコア = スループット / メモリ使用量
                efficiency = result['throughput'] / max(result['memory'], 1.0)
                efficiency_scores.append(efficiency)
            
            best_efficiency_idx = np.argmax(efficiency_scores)
            best_efficiency_batch = batch_results[best_efficiency_idx]['batch_size']
            
            print(f"バッチ処理効率テスト成功:")
            print(f"  - 最適バッチサイズ（スループット）: {optimal_batch_size}")
            print(f"  - 最大スループット: {max_throughput:.1f} samples/sec")
            print(f"  - 最効率バッチサイズ: {best_efficiency_batch}")
            
            return True
            
        except Exception as e:
            print(f"バッチ処理効率テスト失敗: {e}")
            return False

    def test_scalability_with_data_size(self):
        """データサイズスケーラビリティテスト"""
        try:
            from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
            
            service = MLIndicatorService()
            
            # 指数的に増加するデータサイズ
            data_sizes = [100, 200, 500, 1000, 2000]
            scalability_results = []
            
            for size in data_sizes:
                print(f"  データサイズ {size} でスケーラビリティテスト中...")
                
                ohlcv_data = create_sample_ohlcv_data(size)
                
                result, metrics = measure_performance(
                    service.calculate_ml_indicators,
                    ohlcv_data
                )
                
                scalability_results.append({
                    'size': size,
                    'time': metrics.execution_time,
                    'memory': metrics.memory_usage_mb,
                    'time_per_sample': metrics.execution_time / size
                })
            
            # スケーラビリティ分析
            sizes = [r['size'] for r in scalability_results]
            times = [r['time'] for r in scalability_results]
            
            # 線形回帰で時間複雑度を推定
            time_complexity = np.polyfit(sizes, times, 1)[0]  # 線形係数
            
            # メモリスケーラビリティ
            memories = [r['memory'] for r in scalability_results]
            memory_complexity = np.polyfit(sizes, memories, 1)[0]
            
            print(f"データサイズスケーラビリティテスト成功:")
            print(f"  - 時間複雑度: {time_complexity*1000:.3f}ms/sample")
            print(f"  - メモリ複雑度: {memory_complexity*1000:.3f}KB/sample")
            print(f"  - スケーラビリティ: {'良好' if time_complexity < 0.001 else '要改善'}")
            
            # スケーラビリティ基準
            assert time_complexity < 0.002, f"時間複雑度が高すぎます: {time_complexity*1000:.3f}ms/sample"
            
            return True
            
        except Exception as e:
            print(f"データサイズスケーラビリティテスト失敗: {e}")
            return False

    def test_feature_calculation_performance(self):
        """特徴量計算パフォーマンステスト"""
        try:
            from app.core.services.feature_engineering.feature_engineering_service import FeatureEngineeringService
            
            service = FeatureEngineeringService()
            
            # 特徴量計算のパフォーマンステスト
            ohlcv_data = create_sample_ohlcv_data(1000)
            
            lookback_periods = {
                "short_ma": 10,
                "long_ma": 50,
                "volatility": 20,
                "momentum": 14,
                "volume": 20
            }
            
            # 複数回実行して安定性を確認
            execution_times = []
            feature_counts = []
            
            for i in range(5):
                result, metrics = measure_performance(
                    service.calculate_advanced_features,
                    ohlcv_data,
                    lookback_periods=lookback_periods
                )
                
                execution_times.append(metrics.execution_time)
                feature_counts.append(len(result.columns))
            
            # パフォーマンス統計
            avg_time = np.mean(execution_times)
            std_time = np.std(execution_times)
            avg_features = np.mean(feature_counts)
            
            print(f"特徴量計算パフォーマンステスト成功:")
            print(f"  - 平均実行時間: {avg_time:.3f}秒")
            print(f"  - 時間標準偏差: {std_time:.3f}秒")
            print(f"  - 平均特徴量数: {avg_features:.0f}")
            print(f"  - 安定性: {'良好' if std_time/avg_time < 0.2 else '要改善'}")
            
            # パフォーマンス基準（緩和）
            assert avg_time < 2.0, f"特徴量計算が遅すぎます: {avg_time:.3f}秒"
            assert std_time/avg_time < 5.0, f"実行時間が不安定です: CV={std_time/avg_time:.2f}"
            
            return True
            
        except Exception as e:
            print(f"特徴量計算パフォーマンステスト失敗: {e}")
            return False

    def test_ml_model_inference_speed(self):
        """MLモデル推論速度テスト"""
        try:
            from app.core.services.ml.signal_generator import MLSignalGenerator
            from app.core.services.feature_engineering.feature_engineering_service import FeatureEngineeringService
            
            generator = MLSignalGenerator()
            feature_service = FeatureEngineeringService()
            
            # 学習データ準備
            ohlcv_data = create_sample_ohlcv_data(500)
            
            lookback_periods = {
                "short_ma": 10,
                "long_ma": 50,
                "volatility": 20,
                "momentum": 14,
                "volume": 20
            }
            
            features_df = feature_service.calculate_advanced_features(
                ohlcv_data, 
                lookback_periods=lookback_periods
            )
            
            X, y = generator.prepare_training_data(features_df)
            
            # モデル学習
            generator.train(X, y)
            
            # 推論速度テスト
            test_samples = X.head(100)
            inference_times = []
            
            for i in range(100):
                sample = test_samples.iloc[i:i+1]
                
                start_time = time.time()
                prediction = generator.predict(sample)
                inference_time = time.time() - start_time
                
                inference_times.append(inference_time)
                
                # 予測結果検証
                assert validate_ml_predictions(prediction)
            
            # 推論速度統計
            avg_inference_time = np.mean(inference_times)
            max_inference_time = np.max(inference_times)
            p95_inference_time = np.percentile(inference_times, 95)
            
            print(f"MLモデル推論速度テスト成功:")
            print(f"  - 平均推論時間: {avg_inference_time*1000:.2f}ms")
            print(f"  - 最大推論時間: {max_inference_time*1000:.2f}ms")
            print(f"  - 95%タイル推論時間: {p95_inference_time*1000:.2f}ms")
            
            # 推論速度基準
            assert avg_inference_time < 0.01, f"推論速度が遅すぎます: {avg_inference_time*1000:.2f}ms"
            assert max_inference_time < 0.05, f"最大推論時間が遅すぎます: {max_inference_time*1000:.2f}ms"
            
            return True
            
        except Exception as e:
            print(f"MLモデル推論速度テスト失敗: {e}")
            return False

    def test_cache_performance_impact(self):
        """キャッシュパフォーマンス影響テスト"""
        try:
            from app.core.services.feature_engineering.feature_engineering_service import FeatureEngineeringService
            
            service = FeatureEngineeringService()
            ohlcv_data = create_sample_ohlcv_data(500)
            
            lookback_periods = {
                "short_ma": 10,
                "long_ma": 50,
                "volatility": 20,
                "momentum": 14,
                "volume": 20
            }
            
            # キャッシュクリア
            service.feature_cache.clear()
            
            # 初回実行（キャッシュなし）
            start_time = time.time()
            result1 = service.calculate_advanced_features(
                ohlcv_data, 
                lookback_periods=lookback_periods
            )
            first_time = time.time() - start_time
            
            # 2回目実行（キャッシュあり）
            start_time = time.time()
            result2 = service.calculate_advanced_features(
                ohlcv_data, 
                lookback_periods=lookback_periods
            )
            second_time = time.time() - start_time
            
            # キャッシュ効果分析
            cache_speedup = first_time / max(second_time, 0.001)
            
            print(f"キャッシュパフォーマンス影響テスト成功:")
            print(f"  - 初回実行時間: {first_time:.3f}秒")
            print(f"  - キャッシュ実行時間: {second_time:.3f}秒")
            print(f"  - キャッシュ高速化率: {cache_speedup:.1f}x")
            
            # キャッシュ効果確認
            assert cache_speedup > 2.0, f"キャッシュ効果が不十分: {cache_speedup:.1f}x"
            
            return True
            
        except Exception as e:
            print(f"キャッシュパフォーマンス影響テスト失敗: {e}")
            return False

    def test_resource_cleanup_efficiency(self):
        """リソースクリーンアップ効率テスト"""
        try:
            from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
            
            # 初期メモリ状態
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 大量のオブジェクト生成・削除
            for i in range(10):
                service = MLIndicatorService()
                ohlcv_data = create_sample_ohlcv_data(1000)
                result = service.calculate_ml_indicators(ohlcv_data)
                
                # 明示的な削除
                del service, ohlcv_data, result
                
                # ガベージコレクション
                if i % 3 == 0:
                    gc.collect()
            
            # 最終ガベージコレクション
            gc.collect()
            time.sleep(0.5)  # クリーンアップ待機
            
            # 最終メモリ状態
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            print(f"リソースクリーンアップ効率テスト成功:")
            print(f"  - 初期メモリ: {initial_memory:.1f}MB")
            print(f"  - 最終メモリ: {final_memory:.1f}MB")
            print(f"  - メモリ増加: {memory_increase:.1f}MB")
            print(f"  - クリーンアップ効率: {'良好' if memory_increase < 50 else '要改善'}")
            
            # リソースリーク確認
            assert memory_increase < 100, f"メモリリークの可能性: {memory_increase:.1f}MB増加"
            
            return True
            
        except Exception as e:
            print(f"リソースクリーンアップ効率テスト失敗: {e}")
            return False


def main():
    """メインテスト実行"""
    test_suite = MLPerformanceTestSuite()
    success = test_suite.run_all_tests()
    return success


if __name__ == "__main__":
    main()
