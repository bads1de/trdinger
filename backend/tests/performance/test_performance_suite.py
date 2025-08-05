"""
パフォーマンステストスイート

メモリ使用量、CPU使用率、実行時間、メモリリークなどを監視し、
システムのパフォーマンス問題を発見します。
"""

import gc
import logging
import os
import psutil
import sys
import time
import tracemalloc
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# プロジェクトルートをPythonパスに追加
backend_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# 警告を抑制
warnings.filterwarnings("ignore")

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceTestResult:
    """パフォーマンステスト結果"""

    test_name: str
    component_name: str
    success: bool
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    data_size: int
    throughput_ops_per_sec: float
    memory_leak_detected: bool
    performance_score: float
    error_message: Optional[str] = None
    detailed_metrics: Optional[Dict[str, Any]] = None


class PerformanceTestSuite:
    """パフォーマンステストスイート"""

    def __init__(self):
        self.results: List[PerformanceTestResult] = []
        self.process = psutil.Process()

    def create_performance_test_data(self, size: int) -> pd.DataFrame:
        """パフォーマンステスト用のデータを作成"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=size, freq="1H")

        # リアルな価格データを生成
        base_price = 100
        price_changes = np.random.normal(0, 0.02, size)
        prices = [base_price]

        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))

        prices = np.array(prices)

        return pd.DataFrame(
            {
                "timestamp": dates,
                "Open": prices,
                "High": prices * (1 + np.abs(np.random.normal(0, 0.01, size))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, size))),
                "Close": prices * (1 + np.random.normal(0, 0.005, size)),
                "Volume": np.random.lognormal(10, 1, size),
            }
        )

    def measure_memory_usage(self) -> float:
        """現在のメモリ使用量を測定（MB）"""
        return self.process.memory_info().rss / 1024 / 1024

    def measure_cpu_usage(self, interval: float = 0.1) -> float:
        """CPU使用率を測定"""
        return self.process.cpu_percent(interval=interval)

    def test_feature_engineering_performance(self):
        """特徴量エンジニアリングのパフォーマンステスト"""
        logger.info("🚀 特徴量エンジニアリングパフォーマンステスト開始")

        # メモリトレースを開始
        tracemalloc.start()
        start_memory = self.measure_memory_usage()
        start_time = time.time()

        performance_score = 100.0
        memory_leak_detected = False
        detailed_metrics = {}

        try:
            # 異なるサイズのデータでテスト
            data_sizes = [100, 500, 1000, 2000]
            execution_times = []
            memory_usages = []
            throughputs = []

            for size in data_sizes:
                logger.info(f"📊 データサイズ {size} でテスト中...")

                # データ作成
                test_data = self.create_performance_test_data(size)

                # メモリ使用量測定開始
                memory_before = self.measure_memory_usage()
                size_start_time = time.time()

                # 特徴量エンジニアリング実行
                try:
                    from app.services.ml.feature_engineering.feature_engineering_service import (
                        FeatureEngineeringService,
                    )

                    fe_service = FeatureEngineeringService()
                    result = fe_service.calculate_advanced_features(test_data)

                    # 実行時間とメモリ使用量を記録
                    size_execution_time = time.time() - size_start_time
                    memory_after = self.measure_memory_usage()
                    memory_used = memory_after - memory_before

                    execution_times.append(size_execution_time)
                    memory_usages.append(memory_used)

                    # スループット計算（行/秒）
                    throughput = (
                        size / size_execution_time if size_execution_time > 0 else 0
                    )
                    throughputs.append(throughput)

                    logger.info(
                        f"   サイズ {size}: {size_execution_time:.2f}秒, "
                        f"メモリ {memory_used:.1f}MB, スループット {throughput:.1f}行/秒"
                    )

                    # ガベージコレクション
                    gc.collect()

                except Exception as e:
                    performance_score -= 25.0
                    logger.warning(f"サイズ {size} でエラー: {e}")

            # パフォーマンス分析
            if len(execution_times) >= 2:
                # 実行時間の線形性チェック
                time_ratios = [
                    execution_times[i] / execution_times[i - 1]
                    for i in range(1, len(execution_times))
                ]
                size_ratios = [
                    data_sizes[i] / data_sizes[i - 1] for i in range(1, len(data_sizes))
                ]

                # 理想的には時間比とサイズ比が近い値になるべき
                linearity_score = 0
                for time_ratio, size_ratio in zip(time_ratios, size_ratios):
                    if size_ratio > 0:
                        ratio_diff = abs(time_ratio - size_ratio) / size_ratio
                        if ratio_diff < 0.5:  # 50%以内の差なら良好
                            linearity_score += 1

                linearity_percentage = linearity_score / len(time_ratios) * 100
                detailed_metrics["time_linearity_percentage"] = linearity_percentage

                if linearity_percentage < 50:
                    performance_score -= 20.0
                    logger.warning(
                        f"⚠️ 実行時間の線形性が低い: {linearity_percentage:.1f}%"
                    )

                # メモリ効率チェック
                max_memory_per_row = max(
                    memory_usages[i] / data_sizes[i] for i in range(len(memory_usages))
                )
                detailed_metrics["max_memory_per_row_mb"] = max_memory_per_row

                if max_memory_per_row > 1.0:  # 1MB/行を超える場合
                    performance_score -= 15.0
                    logger.warning(f"⚠️ メモリ効率が低い: {max_memory_per_row:.3f}MB/行")

                # スループット分析
                avg_throughput = sum(throughputs) / len(throughputs)
                detailed_metrics["average_throughput"] = avg_throughput

                if avg_throughput < 10:  # 10行/秒未満
                    performance_score -= 15.0
                    logger.warning(f"⚠️ スループットが低い: {avg_throughput:.1f}行/秒")

            # メモリリーク検出
            current_memory = self.measure_memory_usage()
            memory_increase = current_memory - start_memory
            detailed_metrics["total_memory_increase_mb"] = memory_increase

            if memory_increase > 100:  # 100MB以上の増加
                memory_leak_detected = True
                performance_score -= 25.0
                logger.warning(f"⚠️ メモリリークの可能性: {memory_increase:.1f}MB増加")

            # CPU使用率測定
            cpu_usage = self.measure_cpu_usage(0.5)
            detailed_metrics["cpu_usage_percent"] = cpu_usage

            execution_time = time.time() - start_time
            final_memory = self.measure_memory_usage()

            self.results.append(
                PerformanceTestResult(
                    test_name="特徴量エンジニアリングパフォーマンス",
                    component_name="FeatureEngineeringService",
                    success=performance_score > 70.0,
                    execution_time=execution_time,
                    memory_usage_mb=final_memory - start_memory,
                    cpu_usage_percent=cpu_usage,
                    data_size=sum(data_sizes),
                    throughput_ops_per_sec=(
                        sum(throughputs) / len(throughputs) if throughputs else 0
                    ),
                    memory_leak_detected=memory_leak_detected,
                    performance_score=performance_score,
                    detailed_metrics=detailed_metrics,
                )
            )

            logger.info(
                f"✅ 特徴量エンジニアリングパフォーマンステスト完了: スコア {performance_score:.1f}%"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            final_memory = self.measure_memory_usage()

            self.results.append(
                PerformanceTestResult(
                    test_name="特徴量エンジニアリングパフォーマンス",
                    component_name="FeatureEngineeringService",
                    success=False,
                    execution_time=execution_time,
                    memory_usage_mb=final_memory - start_memory,
                    cpu_usage_percent=0.0,
                    data_size=0,
                    throughput_ops_per_sec=0.0,
                    memory_leak_detected=False,
                    performance_score=0.0,
                    error_message=str(e),
                )
            )

            logger.error(f"❌ 特徴量エンジニアリングパフォーマンステスト失敗: {e}")

        finally:
            tracemalloc.stop()

    def test_data_processing_performance(self):
        """データ処理のパフォーマンステスト"""
        logger.info("🚀 データ処理パフォーマンステスト開始")

        tracemalloc.start()
        start_memory = self.measure_memory_usage()
        start_time = time.time()

        performance_score = 100.0
        memory_leak_detected = False
        detailed_metrics = {}

        try:
            # 大量データでのテスト
            large_data = self.create_performance_test_data(5000)

            # データプロセッサのテスト
            try:
                from app.utils.data_processing import DataProcessor

                processor = DataProcessor()

                # 前処理のパフォーマンステスト
                preprocess_start = time.time()
                memory_before_preprocess = self.measure_memory_usage()

                processed_data = processor.preprocess_features(
                    large_data,
                    imputation_strategy="median",
                    scale_features=True,
                    remove_outliers=True,
                    outlier_threshold=3.0,
                    scaling_method="robust",
                    outlier_method="iqr",
                )

                preprocess_time = time.time() - preprocess_start
                memory_after_preprocess = self.measure_memory_usage()
                preprocess_memory = memory_after_preprocess - memory_before_preprocess

                detailed_metrics["preprocess_time"] = preprocess_time
                detailed_metrics["preprocess_memory_mb"] = preprocess_memory
                detailed_metrics["preprocess_throughput"] = (
                    len(large_data) / preprocess_time
                )

                logger.info(
                    f"前処理: {preprocess_time:.2f}秒, "
                    f"メモリ {preprocess_memory:.1f}MB, "
                    f"スループット {len(large_data) / preprocess_time:.1f}行/秒"
                )

                # パフォーマンス評価
                if preprocess_time > 30:  # 30秒以上
                    performance_score -= 20.0
                    logger.warning(f"⚠️ 前処理が遅い: {preprocess_time:.2f}秒")

                if preprocess_memory > 500:  # 500MB以上
                    performance_score -= 15.0
                    logger.warning(
                        f"⚠️ 前処理のメモリ使用量が多い: {preprocess_memory:.1f}MB"
                    )

            except Exception as e:
                performance_score -= 50.0
                logger.warning(f"データ処理エラー: {e}")

            # メモリリーク検出
            current_memory = self.measure_memory_usage()
            memory_increase = current_memory - start_memory
            detailed_metrics["total_memory_increase_mb"] = memory_increase

            if memory_increase > 200:  # 200MB以上の増加
                memory_leak_detected = True
                performance_score -= 25.0
                logger.warning(f"⚠️ メモリリークの可能性: {memory_increase:.1f}MB増加")

            cpu_usage = self.measure_cpu_usage(0.5)
            detailed_metrics["cpu_usage_percent"] = cpu_usage

            execution_time = time.time() - start_time
            final_memory = self.measure_memory_usage()

            self.results.append(
                PerformanceTestResult(
                    test_name="データ処理パフォーマンス",
                    component_name="DataProcessor",
                    success=performance_score > 70.0,
                    execution_time=execution_time,
                    memory_usage_mb=final_memory - start_memory,
                    cpu_usage_percent=cpu_usage,
                    data_size=len(large_data),
                    throughput_ops_per_sec=len(large_data) / execution_time,
                    memory_leak_detected=memory_leak_detected,
                    performance_score=performance_score,
                    detailed_metrics=detailed_metrics,
                )
            )

            logger.info(
                f"✅ データ処理パフォーマンステスト完了: スコア {performance_score:.1f}%"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            final_memory = self.measure_memory_usage()

            self.results.append(
                PerformanceTestResult(
                    test_name="データ処理パフォーマンス",
                    component_name="DataProcessor",
                    success=False,
                    execution_time=execution_time,
                    memory_usage_mb=final_memory - start_memory,
                    cpu_usage_percent=0.0,
                    data_size=0,
                    throughput_ops_per_sec=0.0,
                    memory_leak_detected=False,
                    performance_score=0.0,
                    error_message=str(e),
                )
            )

            logger.error(f"❌ データ処理パフォーマンステスト失敗: {e}")

        finally:
            tracemalloc.stop()

    def test_model_training_performance(self):
        """モデル学習のパフォーマンステスト"""
        logger.info("🚀 モデル学習パフォーマンステスト開始")

        tracemalloc.start()
        start_memory = self.measure_memory_usage()
        start_time = time.time()

        performance_score = 100.0
        memory_leak_detected = False
        detailed_metrics = {}

        try:
            # 学習用データの準備
            training_data = self.create_performance_test_data(1000)

            # モデル学習のパフォーマンステスト
            try:
                from app.services.ml.base_ml_trainer import BaseMLTrainer

                # シングルモデルトレーナーのテスト
                trainer = BaseMLTrainer(trainer_type="single", model_type="lightgbm")

                # 学習開始
                training_start = time.time()
                memory_before_training = self.measure_memory_usage()

                result = trainer.train_model(training_data)

                training_time = time.time() - training_start
                memory_after_training = self.measure_memory_usage()
                training_memory = memory_after_training - memory_before_training

                detailed_metrics["training_time"] = training_time
                detailed_metrics["training_memory_mb"] = training_memory

                logger.info(
                    f"モデル学習: {training_time:.2f}秒, "
                    f"メモリ {training_memory:.1f}MB"
                )

                # パフォーマンス評価
                if training_time > 60:  # 60秒以上
                    performance_score -= 25.0
                    logger.warning(f"⚠️ 学習時間が長い: {training_time:.2f}秒")

                if training_memory > 1000:  # 1GB以上
                    performance_score -= 20.0
                    logger.warning(
                        f"⚠️ 学習のメモリ使用量が多い: {training_memory:.1f}MB"
                    )

                # 学習結果の検証
                if result and "accuracy" in result:
                    accuracy = result["accuracy"]
                    detailed_metrics["model_accuracy"] = accuracy

                    if accuracy < 0.3:  # 精度が低すぎる場合
                        performance_score -= 15.0
                        logger.warning(f"⚠️ モデル精度が低い: {accuracy:.3f}")

            except Exception as e:
                performance_score -= 50.0
                logger.warning(f"モデル学習エラー: {e}")

            # メモリリーク検出
            current_memory = self.measure_memory_usage()
            memory_increase = current_memory - start_memory
            detailed_metrics["total_memory_increase_mb"] = memory_increase

            if memory_increase > 300:  # 300MB以上の増加
                memory_leak_detected = True
                performance_score -= 25.0
                logger.warning(f"⚠️ メモリリークの可能性: {memory_increase:.1f}MB増加")

            cpu_usage = self.measure_cpu_usage(0.5)
            detailed_metrics["cpu_usage_percent"] = cpu_usage

            execution_time = time.time() - start_time
            final_memory = self.measure_memory_usage()

            self.results.append(
                PerformanceTestResult(
                    test_name="モデル学習パフォーマンス",
                    component_name="BaseMLTrainer",
                    success=performance_score > 70.0,
                    execution_time=execution_time,
                    memory_usage_mb=final_memory - start_memory,
                    cpu_usage_percent=cpu_usage,
                    data_size=len(training_data),
                    throughput_ops_per_sec=len(training_data) / execution_time,
                    memory_leak_detected=memory_leak_detected,
                    performance_score=performance_score,
                    detailed_metrics=detailed_metrics,
                )
            )

            logger.info(
                f"✅ モデル学習パフォーマンステスト完了: スコア {performance_score:.1f}%"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            final_memory = self.measure_memory_usage()

            self.results.append(
                PerformanceTestResult(
                    test_name="モデル学習パフォーマンス",
                    component_name="BaseMLTrainer",
                    success=False,
                    execution_time=execution_time,
                    memory_usage_mb=final_memory - start_memory,
                    cpu_usage_percent=0.0,
                    data_size=0,
                    throughput_ops_per_sec=0.0,
                    memory_leak_detected=False,
                    performance_score=0.0,
                    error_message=str(e),
                )
            )

            logger.error(f"❌ モデル学習パフォーマンステスト失敗: {e}")

        finally:
            tracemalloc.stop()

    def run_all_tests(self):
        """すべてのパフォーマンステストを実行"""
        logger.info("🚀 パフォーマンステストスイート開始")

        # システム情報をログ出力
        logger.info(f"システム情報:")
        logger.info(f"  CPU数: {psutil.cpu_count()}")
        logger.info(
            f"  メモリ総量: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB"
        )
        logger.info(
            f"  利用可能メモリ: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f}GB"
        )

        self.test_feature_engineering_performance()
        self.test_data_processing_performance()
        self.test_model_training_performance()

        # 結果の集計
        total_tests = len(self.results)
        successful_tests = sum(1 for result in self.results if result.success)
        total_execution_time = sum(result.execution_time for result in self.results)
        total_memory_usage = sum(result.memory_usage_mb for result in self.results)
        average_performance = (
            sum(result.performance_score for result in self.results) / total_tests
            if total_tests > 0
            else 0
        )
        memory_leaks_detected = sum(
            1 for result in self.results if result.memory_leak_detected
        )
        average_throughput = (
            sum(result.throughput_ops_per_sec for result in self.results) / total_tests
            if total_tests > 0
            else 0
        )

        logger.info("=" * 80)
        logger.info("🚀 パフォーマンステスト結果")
        logger.info("=" * 80)
        logger.info(f"📊 総テスト数: {total_tests}")
        logger.info(f"✅ 成功: {successful_tests}")
        logger.info(f"❌ 失敗: {total_tests - successful_tests}")
        logger.info(f"📈 成功率: {successful_tests / total_tests * 100:.1f}%")
        logger.info(f"🎯 平均パフォーマンススコア: {average_performance:.1f}%")
        logger.info(f"💾 総メモリ使用量: {total_memory_usage:.1f}MB")
        logger.info(f"⚠️ メモリリーク検出: {memory_leaks_detected}件")
        logger.info(f"⚡ 平均スループット: {average_throughput:.1f}行/秒")
        logger.info(f"⏱️ 総実行時間: {total_execution_time:.2f}秒")

        logger.info("\n🚀 パフォーマンステスト詳細:")
        for result in self.results:
            status = "✅" if result.success else "❌"
            leak_status = "🔴" if result.memory_leak_detected else "🟢"

            logger.info(f"{status} {result.test_name}")
            logger.info(f"   コンポーネント: {result.component_name}")
            logger.info(f"   実行時間: {result.execution_time:.2f}秒")
            logger.info(f"   メモリ使用量: {result.memory_usage_mb:.1f}MB")
            logger.info(f"   CPU使用率: {result.cpu_usage_percent:.1f}%")
            logger.info(f"   スループット: {result.throughput_ops_per_sec:.1f}行/秒")
            logger.info(f"   パフォーマンススコア: {result.performance_score:.1f}%")
            logger.info(f"   メモリリーク: {leak_status}")

            if result.detailed_metrics:
                logger.info("   詳細メトリクス:")
                for key, value in result.detailed_metrics.items():
                    if isinstance(value, float):
                        logger.info(f"     {key}: {value:.3f}")
                    else:
                        logger.info(f"     {key}: {value}")

            if result.error_message:
                logger.info(f"   エラー: {result.error_message[:100]}...")

        logger.info("=" * 80)
        logger.info("🎯 パフォーマンステストスイート完了")

        return self.results


if __name__ == "__main__":
    suite = PerformanceTestSuite()
    results = suite.run_all_tests()
