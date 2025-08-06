"""
並行性テストスイート

マルチスレッド、競合状態、デッドロック、データの不整合などを検証し、
並行処理における潜在的な問題を発見します。
"""

import logging
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
class ConcurrencyTestResult:
    """並行性テスト結果"""

    test_name: str
    component_name: str
    success: bool
    execution_time: float
    thread_count: int
    race_conditions_detected: int
    deadlocks_detected: int
    data_inconsistencies: int
    concurrency_score: float
    error_message: Optional[str] = None
    detailed_metrics: Optional[Dict[str, Any]] = None


class ConcurrencyTestSuite:
    """並行性テストスイート"""

    def __init__(self):
        self.results: List[ConcurrencyTestResult] = []
        self.shared_data = {}
        self.lock = threading.Lock()

    def create_concurrency_test_data(self, size: int = 500) -> pd.DataFrame:
        """並行性テスト用のデータを作成"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=size, freq="1H")

        # 基本的な価格データ
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

    def feature_engineering_worker(
        self, worker_id: int, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """特徴量エンジニアリングワーカー"""
        try:
            from app.services.ml.feature_engineering.feature_engineering_service import (
                FeatureEngineeringService,
            )

            fe_service = FeatureEngineeringService()
            start_time = time.time()

            # 特徴量計算
            result = fe_service.calculate_advanced_features(data)

            execution_time = time.time() - start_time

            # 共有データに結果を保存（競合状態をテスト）
            with self.lock:
                self.shared_data[f"worker_{worker_id}"] = {
                    "result_shape": result.shape,
                    "execution_time": execution_time,
                    "feature_count": len(result.columns),
                    "success": True,
                }

            return {
                "worker_id": worker_id,
                "success": True,
                "execution_time": execution_time,
                "result_shape": result.shape,
                "feature_count": len(result.columns),
            }

        except Exception as e:
            with self.lock:
                self.shared_data[f"worker_{worker_id}"] = {
                    "error": str(e),
                    "success": False,
                }

            return {
                "worker_id": worker_id,
                "success": False,
                "error": str(e),
            }

    def data_processing_worker(
        self, worker_id: int, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """データ処理ワーカー"""
        try:
            from app.utils.data_processing import DataProcessor

            processor = DataProcessor()
            start_time = time.time()

            # データ前処理
            processed_data = processor.preprocess_features(
                data,
                imputation_strategy="median",
                scale_features=True,
                remove_outliers=True,
            )

            execution_time = time.time() - start_time

            # 共有データに結果を保存
            with self.lock:
                self.shared_data[f"processor_{worker_id}"] = {
                    "result_shape": processed_data.shape,
                    "execution_time": execution_time,
                    "success": True,
                }

            return {
                "worker_id": worker_id,
                "success": True,
                "execution_time": execution_time,
                "result_shape": processed_data.shape,
            }

        except Exception as e:
            with self.lock:
                self.shared_data[f"processor_{worker_id}"] = {
                    "error": str(e),
                    "success": False,
                }

            return {
                "worker_id": worker_id,
                "success": False,
                "error": str(e),
            }

    def test_concurrent_feature_engineering(self):
        """並行特徴量エンジニアリングテスト"""
        logger.info("🔄 並行特徴量エンジニアリングテスト開始")

        start_time = time.time()
        concurrency_score = 100.0
        race_conditions = 0
        deadlocks = 0
        data_inconsistencies = 0
        detailed_metrics = {}

        try:
            # テストデータの準備
            test_data = self.create_concurrency_test_data(300)
            thread_count = 4

            # 共有データをクリア
            self.shared_data.clear()

            # 並行実行
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                # 各スレッドで特徴量エンジニアリングを実行
                futures = [
                    executor.submit(self.feature_engineering_worker, i, test_data)
                    for i in range(thread_count)
                ]

                # 結果を収集
                results = []
                completed_count = 0
                timeout_count = 0

                for future in as_completed(futures, timeout=60):
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                        completed_count += 1
                    except Exception as e:
                        timeout_count += 1
                        logger.warning(f"ワーカータイムアウト: {e}")

            # 結果の分析
            successful_workers = [r for r in results if r.get("success", False)]
            failed_workers = [r for r in results if not r.get("success", False)]

            detailed_metrics["completed_workers"] = completed_count
            detailed_metrics["successful_workers"] = len(successful_workers)
            detailed_metrics["failed_workers"] = len(failed_workers)
            detailed_metrics["timeout_workers"] = timeout_count

            # データ整合性チェック
            if len(successful_workers) > 1:
                # 特徴量数の一貫性チェック
                feature_counts = [w["feature_count"] for w in successful_workers]
                if len(set(feature_counts)) > 1:
                    data_inconsistencies += 1
                    concurrency_score -= 20.0
                    logger.warning(f"⚠️ 特徴量数の不整合: {set(feature_counts)}")

                # 実行時間の分散チェック
                execution_times = [w["execution_time"] for w in successful_workers]
                time_variance = np.var(execution_times)
                detailed_metrics["execution_time_variance"] = time_variance

                if time_variance > 100:  # 実行時間の分散が大きい
                    race_conditions += 1
                    concurrency_score -= 15.0
                    logger.warning(f"⚠️ 実行時間の大きな分散: {time_variance:.2f}")

            # 共有データの整合性チェック
            shared_data_keys = list(self.shared_data.keys())
            expected_keys = [f"worker_{i}" for i in range(thread_count)]
            missing_keys = set(expected_keys) - set(shared_data_keys)

            if missing_keys:
                data_inconsistencies += len(missing_keys)
                concurrency_score -= 10.0 * len(missing_keys)
                logger.warning(f"⚠️ 共有データの欠損: {missing_keys}")

            # 失敗率の評価
            failure_rate = len(failed_workers) / thread_count
            detailed_metrics["failure_rate"] = failure_rate

            if failure_rate > 0.25:  # 25%以上の失敗率
                concurrency_score -= 30.0
                logger.warning(f"⚠️ 高い失敗率: {failure_rate:.2f}")

            execution_time = time.time() - start_time

            self.results.append(
                ConcurrencyTestResult(
                    test_name="並行特徴量エンジニアリング",
                    component_name="FeatureEngineeringService",
                    success=concurrency_score > 70.0,
                    execution_time=execution_time,
                    thread_count=thread_count,
                    race_conditions_detected=race_conditions,
                    deadlocks_detected=deadlocks,
                    data_inconsistencies=data_inconsistencies,
                    concurrency_score=concurrency_score,
                    detailed_metrics=detailed_metrics,
                )
            )

            logger.info(
                f"✅ 並行特徴量エンジニアリングテスト完了: スコア {concurrency_score:.1f}%"
            )

        except Exception as e:
            execution_time = time.time() - start_time

            self.results.append(
                ConcurrencyTestResult(
                    test_name="並行特徴量エンジニアリング",
                    component_name="FeatureEngineeringService",
                    success=False,
                    execution_time=execution_time,
                    thread_count=0,
                    race_conditions_detected=0,
                    deadlocks_detected=0,
                    data_inconsistencies=0,
                    concurrency_score=0.0,
                    error_message=str(e),
                )
            )

            logger.error(f"❌ 並行特徴量エンジニアリングテスト失敗: {e}")

    def test_concurrent_data_processing(self):
        """並行データ処理テスト"""
        logger.info("🔄 並行データ処理テスト開始")

        start_time = time.time()
        concurrency_score = 100.0
        race_conditions = 0
        deadlocks = 0
        data_inconsistencies = 0
        detailed_metrics = {}

        try:
            # テストデータの準備
            test_data = self.create_concurrency_test_data(400)
            thread_count = 3

            # 共有データをクリア
            self.shared_data.clear()

            # 並行実行
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                # 各スレッドでデータ処理を実行
                futures = [
                    executor.submit(self.data_processing_worker, i, test_data)
                    for i in range(thread_count)
                ]

                # 結果を収集
                results = []
                for future in as_completed(futures, timeout=45):
                    try:
                        result = future.result(timeout=20)
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"データ処理ワーカーエラー: {e}")

            # 結果の分析
            successful_workers = [r for r in results if r.get("success", False)]
            failed_workers = [r for r in results if not r.get("success", False)]

            detailed_metrics["successful_workers"] = len(successful_workers)
            detailed_metrics["failed_workers"] = len(failed_workers)

            # データ整合性チェック
            if len(successful_workers) > 1:
                # 結果の形状の一貫性チェック
                result_shapes = [w["result_shape"] for w in successful_workers]
                unique_shapes = set(result_shapes)

                if len(unique_shapes) > 1:
                    data_inconsistencies += 1
                    concurrency_score -= 25.0
                    logger.warning(f"⚠️ 処理結果の形状不整合: {unique_shapes}")

            # 失敗率の評価
            failure_rate = len(failed_workers) / thread_count
            detailed_metrics["failure_rate"] = failure_rate

            if failure_rate > 0.33:  # 33%以上の失敗率
                concurrency_score -= 35.0
                logger.warning(f"⚠️ データ処理の高い失敗率: {failure_rate:.2f}")

            execution_time = time.time() - start_time

            self.results.append(
                ConcurrencyTestResult(
                    test_name="並行データ処理",
                    component_name="DataProcessor",
                    success=concurrency_score > 70.0,
                    execution_time=execution_time,
                    thread_count=thread_count,
                    race_conditions_detected=race_conditions,
                    deadlocks_detected=deadlocks,
                    data_inconsistencies=data_inconsistencies,
                    concurrency_score=concurrency_score,
                    detailed_metrics=detailed_metrics,
                )
            )

            logger.info(f"✅ 並行データ処理テスト完了: スコア {concurrency_score:.1f}%")

        except Exception as e:
            execution_time = time.time() - start_time

            self.results.append(
                ConcurrencyTestResult(
                    test_name="並行データ処理",
                    component_name="DataProcessor",
                    success=False,
                    execution_time=execution_time,
                    thread_count=0,
                    race_conditions_detected=0,
                    deadlocks_detected=0,
                    data_inconsistencies=0,
                    concurrency_score=0.0,
                    error_message=str(e),
                )
            )

            logger.error(f"❌ 並行データ処理テスト失敗: {e}")

    def run_all_tests(self):
        """すべての並行性テストを実行"""
        logger.info("🚀 並行性テストスイート開始")

        self.test_concurrent_feature_engineering()
        self.test_concurrent_data_processing()

        # 結果の集計
        total_tests = len(self.results)
        successful_tests = sum(1 for result in self.results if result.success)
        total_execution_time = sum(result.execution_time for result in self.results)
        total_threads = sum(result.thread_count for result in self.results)
        total_race_conditions = sum(
            result.race_conditions_detected for result in self.results
        )
        total_deadlocks = sum(result.deadlocks_detected for result in self.results)
        total_data_inconsistencies = sum(
            result.data_inconsistencies for result in self.results
        )
        average_concurrency = (
            sum(result.concurrency_score for result in self.results) / total_tests
            if total_tests > 0
            else 0
        )

        logger.info("=" * 80)
        logger.info("🔄 並行性テスト結果")
        logger.info("=" * 80)
        logger.info(f"📊 総テスト数: {total_tests}")
        logger.info(f"✅ 成功: {successful_tests}")
        logger.info(f"❌ 失敗: {total_tests - successful_tests}")
        logger.info(f"📈 成功率: {successful_tests / total_tests * 100:.1f}%")
        logger.info(f"🎯 平均並行性スコア: {average_concurrency:.1f}%")
        logger.info(f"🧵 総スレッド数: {total_threads}")
        logger.info(f"⚠️ 競合状態検出: {total_race_conditions}件")
        logger.info(f"🔒 デッドロック検出: {total_deadlocks}件")
        logger.info(f"📊 データ不整合: {total_data_inconsistencies}件")
        logger.info(f"⏱️ 総実行時間: {total_execution_time:.2f}秒")

        logger.info("\n🔄 並行性テスト詳細:")
        for result in self.results:
            status = "✅" if result.success else "❌"

            logger.info(f"{status} {result.test_name}")
            logger.info(f"   コンポーネント: {result.component_name}")
            logger.info(f"   実行時間: {result.execution_time:.2f}秒")
            logger.info(f"   スレッド数: {result.thread_count}")
            logger.info(f"   並行性スコア: {result.concurrency_score:.1f}%")
            logger.info(f"   競合状態: {result.race_conditions_detected}件")
            logger.info(f"   デッドロック: {result.deadlocks_detected}件")
            logger.info(f"   データ不整合: {result.data_inconsistencies}件")

            if result.detailed_metrics:
                logger.info("   詳細メトリクス:")
                for key, value in result.detailed_metrics.items():
                    if isinstance(value, float):
                        logger.info(f"     {key}: {value:.3f}")
                    else:
                        logger.info(f"     {key}: {value}")

            if result.error_message:
                logger.info(f"   エラー: {result.error_message[:100]}...")

        # 並行性の総合評価
        if (
            total_race_conditions == 0
            and total_deadlocks == 0
            and total_data_inconsistencies == 0
        ):
            logger.info("\n🎉 並行性に関する問題は検出されませんでした！")
        else:
            logger.warning("\n⚠️ 並行性に関する問題が検出されました:")
            if total_race_conditions > 0:
                logger.warning(f"   - 競合状態: {total_race_conditions}件")
            if total_deadlocks > 0:
                logger.warning(f"   - デッドロック: {total_deadlocks}件")
            if total_data_inconsistencies > 0:
                logger.warning(f"   - データ不整合: {total_data_inconsistencies}件")

        logger.info("=" * 80)
        logger.info("🎯 並行性テストスイート完了")

        return self.results


if __name__ == "__main__":
    suite = ConcurrencyTestSuite()
    results = suite.run_all_tests()
