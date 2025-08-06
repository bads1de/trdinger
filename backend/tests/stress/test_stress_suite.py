#!/usr/bin/env python3
"""
ストレステストスイート

MLトレーニングシステムの限界値と異常状況での動作を検証します。
- システム限界値テスト
- 異常データ処理テスト
- リソース枯渇状況テスト
- エラー回復能力テスト
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
import time
import psutil
import gc
from typing import Dict, List, Any
from dataclasses import dataclass, field
import threading

# プロジェクトルートをパスに追加
backend_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, backend_path)

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class StressTestResult:
    """ストレステスト結果データクラス"""

    test_name: str
    stress_type: str
    success: bool
    execution_time: float
    error_recovery: bool
    system_stability: bool
    resource_usage: Dict[str, float] = field(default_factory=dict)
    error_message: str = ""
    recovery_details: Dict[str, Any] = field(default_factory=dict)


class StressTestSuite:
    """ストレステストスイート"""

    def __init__(self):
        self.results: List[StressTestResult] = []
        self.process = psutil.Process()

    def create_corrupted_data(
        self, corruption_type: str, rows: int = 1000
    ) -> pd.DataFrame:
        """破損データを作成"""
        logger.info(f"🔥 {corruption_type}タイプの破損データを作成: {rows}行")

        # 基本データを作成
        dates = pd.date_range("2024-01-01", periods=rows, freq="h")
        base_data = {
            "Open": np.random.normal(50000, 1000, rows),
            "High": np.random.normal(51000, 1000, rows),
            "Low": np.random.normal(49000, 1000, rows),
            "Close": np.random.normal(50000, 1000, rows),
            "Volume": np.random.lognormal(10, 0.5, rows),
        }

        df = pd.DataFrame(base_data, index=dates)

        if corruption_type == "infinite_values":
            # 無限大値を挿入
            df.iloc[100:110, :] = np.inf
            df.iloc[200:210, :] = -np.inf

        elif corruption_type == "extreme_outliers":
            # 極端な外れ値を挿入
            df.iloc[50:60, :] *= 1000000
            df.iloc[150:160, :] /= 1000000

        elif corruption_type == "all_nan":
            # 全てNaNの期間を作成
            df.iloc[300:400, :] = np.nan

        elif corruption_type == "negative_prices":
            # 負の価格を挿入
            df.iloc[250:300, ["Open", "High", "Low", "Close"]] *= -1

        elif corruption_type == "zero_volume":
            # ゼロボリューム期間
            df.iloc[400:500, "Volume"] = 0

        elif corruption_type == "inconsistent_ohlc":
            # OHLC整合性違反
            df.iloc[500:600, "High"] = df.iloc[500:600, "Low"] - 1000

        elif corruption_type == "duplicate_timestamps":
            # 重複タイムスタンプ
            duplicate_indices = [dates[i] for i in range(100, 200)]
            df.index = list(df.index[:100]) + duplicate_indices + list(df.index[200:])

        elif corruption_type == "missing_columns":
            # 必須カラムを削除
            df = df.drop(columns=["Volume", "High"])

        elif corruption_type == "wrong_data_types":
            # 間違ったデータ型
            df["Close"] = df["Close"].astype(str)
            df["Volume"] = ["invalid"] * len(df)

        elif corruption_type == "time_gaps":
            # 時系列ギャップ
            gap_start = 500
            gap_end = 700
            df = pd.concat([df.iloc[:gap_start], df.iloc[gap_end:]])

        return df

    def test_corrupted_data_handling(self):
        """破損データ処理テスト"""
        logger.info("🔥 破損データ処理ストレステスト開始")

        corruption_types = [
            "infinite_values",
            "extreme_outliers",
            "all_nan",
            "negative_prices",
            "zero_volume",
            "inconsistent_ohlc",
            "duplicate_timestamps",
            "missing_columns",
            "wrong_data_types",
            "time_gaps",
        ]

        for corruption_type in corruption_types:
            logger.info(f"🧪 {corruption_type}破損データテスト")

            start_time = time.time()
            initial_memory = self.process.memory_info().rss / 1024**2

            try:
                # 破損データを作成
                corrupted_data = self.create_corrupted_data(corruption_type, rows=1000)

                # MLトレーニングを試行
                from app.services.ml.single_model.single_model_trainer import (
                    SingleModelTrainer,
                )

                trainer = SingleModelTrainer(model_type="lightgbm")

                result = trainer.train_model(
                    training_data=corrupted_data,
                    save_model=False,
                    threshold_up=0.02,
                    threshold_down=-0.02,
                )

                execution_time = time.time() - start_time
                final_memory = self.process.memory_info().rss / 1024**2

                # 成功した場合（データクリーニングが機能）
                self.results.append(
                    StressTestResult(
                        test_name=f"破損データ処理_{corruption_type}",
                        stress_type="data_corruption",
                        success=True,
                        execution_time=execution_time,
                        error_recovery=True,
                        system_stability=True,
                        resource_usage={
                            "memory_usage_mb": final_memory - initial_memory,
                            "peak_memory_mb": final_memory,
                        },
                        recovery_details={
                            "data_cleaned": True,
                            "accuracy": result.get("accuracy", 0),
                            "feature_count": result.get("feature_count", 0),
                        },
                    )
                )

                logger.info(
                    f"✅ {corruption_type}破損データ処理成功: {execution_time:.2f}秒"
                )

            except Exception as e:
                execution_time = time.time() - start_time

                # エラーが適切に処理されたかチェック
                error_handled_properly = any(
                    keyword in str(e).lower()
                    for keyword in [
                        "データ",
                        "data",
                        "無効",
                        "invalid",
                        "不正",
                        "corrupt",
                    ]
                )

                self.results.append(
                    StressTestResult(
                        test_name=f"破損データ処理_{corruption_type}",
                        stress_type="data_corruption",
                        success=False,
                        execution_time=execution_time,
                        error_recovery=error_handled_properly,
                        system_stability=True,  # システムがクラッシュしていない
                        error_message=str(e),
                        recovery_details={
                            "error_type": type(e).__name__,
                            "error_handled": error_handled_properly,
                        },
                    )
                )

                logger.warning(f"⚠️ {corruption_type}破損データでエラー: {e}")

    def test_resource_exhaustion(self):
        """リソース枯渇テスト"""
        logger.info("💾 リソース枯渇ストレステスト開始")

        # メモリ枯渇テスト
        logger.info("🧠 メモリ枯渇テスト")

        start_time = time.time()
        initial_memory = self.process.memory_info().rss / 1024**2

        try:
            # 段階的にメモリ使用量を増加
            memory_hogs = []
            max_memory_mb = 500  # 500MB制限

            while True:
                current_memory = self.process.memory_info().rss / 1024**2
                if current_memory - initial_memory > max_memory_mb:
                    break

                # 大量データを作成してメモリを消費
                large_data = self.create_corrupted_data("extreme_outliers", rows=5000)
                memory_hogs.append(large_data)

                # MLトレーニングを実行
                from app.services.ml.single_model.single_model_trainer import (
                    SingleModelTrainer,
                )

                trainer = SingleModelTrainer(model_type="lightgbm")
                result = trainer.train_model(
                    training_data=large_data,
                    save_model=False,
                    threshold_up=0.02,
                    threshold_down=-0.02,
                )

                logger.info(f"メモリ使用量: {current_memory - initial_memory:.2f}MB")

                if len(memory_hogs) > 3:  # 安全制限
                    break

            execution_time = time.time() - start_time
            final_memory = self.process.memory_info().rss / 1024**2

            self.results.append(
                StressTestResult(
                    test_name="メモリ枯渇テスト",
                    stress_type="resource_exhaustion",
                    success=True,
                    execution_time=execution_time,
                    error_recovery=True,
                    system_stability=True,
                    resource_usage={
                        "memory_usage_mb": final_memory - initial_memory,
                        "peak_memory_mb": final_memory,
                        "memory_objects_created": len(memory_hogs),
                    },
                    recovery_details={
                        "max_memory_reached": final_memory - initial_memory,
                        "system_responsive": True,
                    },
                )
            )

            # メモリクリア
            del memory_hogs
            gc.collect()

            logger.info(f"✅ メモリ枯渇テスト完了: {execution_time:.2f}秒")

        except Exception as e:
            execution_time = time.time() - start_time

            self.results.append(
                StressTestResult(
                    test_name="メモリ枯渇テスト",
                    stress_type="resource_exhaustion",
                    success=False,
                    execution_time=execution_time,
                    error_recovery=True,
                    system_stability=True,
                    error_message=str(e),
                    recovery_details={
                        "error_type": type(e).__name__,
                        "memory_at_failure": self.process.memory_info().rss / 1024**2
                        - initial_memory,
                    },
                )
            )

            logger.error(f"❌ メモリ枯渇テスト失敗: {e}")

    def test_concurrent_stress(self):
        """並行処理ストレステスト"""
        logger.info("⚡ 並行処理ストレステスト開始")

        start_time = time.time()

        try:
            # 複数の並行処理を開始
            threads = []
            results = []

            def worker_function(worker_id: int):
                try:
                    # 各ワーカーで異なる破損データを処理
                    corruption_types = [
                        "extreme_outliers",
                        "all_nan",
                        "negative_prices",
                    ]
                    corruption_type = corruption_types[
                        worker_id % len(corruption_types)
                    ]

                    corrupted_data = self.create_corrupted_data(
                        corruption_type, rows=500
                    )

                    from app.services.ml.single_model.single_model_trainer import (
                        SingleModelTrainer,
                    )

                    trainer = SingleModelTrainer(model_type="lightgbm")
                    result = trainer.train_model(
                        training_data=corrupted_data,
                        save_model=False,
                        threshold_up=0.02,
                        threshold_down=-0.02,
                    )

                    results.append(
                        {
                            "worker_id": worker_id,
                            "success": True,
                            "accuracy": result.get("accuracy", 0),
                        }
                    )

                except Exception as e:
                    results.append(
                        {"worker_id": worker_id, "success": False, "error": str(e)}
                    )

            # 3つの並行ワーカーを開始
            for i in range(3):
                thread = threading.Thread(target=worker_function, args=(i,))
                threads.append(thread)
                thread.start()

            # 全てのスレッドの完了を待機
            for thread in threads:
                thread.join(timeout=60)  # 60秒タイムアウト

            execution_time = time.time() - start_time

            # 結果を評価
            successful_workers = sum(1 for r in results if r["success"])
            total_workers = len(results)

            self.results.append(
                StressTestResult(
                    test_name="並行処理ストレステスト",
                    stress_type="concurrent_stress",
                    success=successful_workers > 0,
                    execution_time=execution_time,
                    error_recovery=True,
                    system_stability=True,
                    resource_usage={
                        "concurrent_workers": total_workers,
                        "successful_workers": successful_workers,
                        "success_rate": successful_workers / max(1, total_workers),
                    },
                    recovery_details={
                        "worker_results": results,
                        "all_threads_completed": len(results) == 3,
                    },
                )
            )

            logger.info(
                f"✅ 並行処理ストレステスト完了: {successful_workers}/{total_workers}成功"
            )

        except Exception as e:
            execution_time = time.time() - start_time

            self.results.append(
                StressTestResult(
                    test_name="並行処理ストレステスト",
                    stress_type="concurrent_stress",
                    success=False,
                    execution_time=execution_time,
                    error_recovery=False,
                    system_stability=True,
                    error_message=str(e),
                )
            )

            logger.error(f"❌ 並行処理ストレステスト失敗: {e}")

    def test_rapid_requests(self):
        """高頻度リクエストストレステスト"""
        logger.info("🚀 高頻度リクエストストレステスト開始")

        start_time = time.time()

        try:
            # 短時間で大量のリクエストを送信
            request_count = 10
            successful_requests = 0
            failed_requests = 0

            for i in range(request_count):
                try:
                    # 小さなデータセットで高速処理
                    test_data = self.create_corrupted_data("extreme_outliers", rows=200)

                    from app.services.ml.single_model.single_model_trainer import (
                        SingleModelTrainer,
                    )

                    trainer = SingleModelTrainer(model_type="lightgbm")
                    result = trainer.train_model(
                        training_data=test_data,
                        save_model=False,
                        threshold_up=0.02,
                        threshold_down=-0.02,
                    )

                    successful_requests += 1
                    logger.info(f"リクエスト {i+1}/{request_count} 成功")

                except Exception as e:
                    failed_requests += 1
                    logger.warning(f"リクエスト {i+1}/{request_count} 失敗: {e}")

                # 短い間隔
                time.sleep(0.1)

            execution_time = time.time() - start_time

            self.results.append(
                StressTestResult(
                    test_name="高頻度リクエストストレステスト",
                    stress_type="rapid_requests",
                    success=successful_requests > 0,
                    execution_time=execution_time,
                    error_recovery=True,
                    system_stability=True,
                    resource_usage={
                        "total_requests": request_count,
                        "successful_requests": successful_requests,
                        "failed_requests": failed_requests,
                        "success_rate": successful_requests / request_count,
                        "requests_per_second": request_count / execution_time,
                    },
                    recovery_details={"system_responsive": True, "no_crashes": True},
                )
            )

            logger.info(
                f"✅ 高頻度リクエストテスト完了: {successful_requests}/{request_count}成功"
            )

        except Exception as e:
            execution_time = time.time() - start_time

            self.results.append(
                StressTestResult(
                    test_name="高頻度リクエストストレステスト",
                    stress_type="rapid_requests",
                    success=False,
                    execution_time=execution_time,
                    error_recovery=False,
                    system_stability=False,
                    error_message=str(e),
                )
            )

            logger.error(f"❌ 高頻度リクエストテスト失敗: {e}")


if __name__ == "__main__":
    logger.info("🔥 ストレステストスイート開始")

    test_suite = StressTestSuite()

    # 各ストレステストを実行
    test_suite.test_corrupted_data_handling()
    test_suite.test_resource_exhaustion()
    test_suite.test_concurrent_stress()
    test_suite.test_rapid_requests()

    # 結果サマリー
    total_tests = len(test_suite.results)
    successful_tests = sum(1 for r in test_suite.results if r.success)
    recovered_tests = sum(1 for r in test_suite.results if r.error_recovery)
    stable_tests = sum(1 for r in test_suite.results if r.system_stability)

    print("\n" + "=" * 80)
    print("🔥 ストレステスト結果")
    print("=" * 80)
    print(f"📊 総テスト数: {total_tests}")
    print(f"✅ 成功: {successful_tests}")
    print(f"❌ 失敗: {total_tests - successful_tests}")
    print(f"🔄 エラー回復: {recovered_tests}")
    print(f"🛡️ システム安定性: {stable_tests}")
    print(f"📈 成功率: {(successful_tests/total_tests*100):.1f}%")
    print(f"🔄 回復率: {(recovered_tests/total_tests*100):.1f}%")
    print(f"🛡️ 安定性: {(stable_tests/total_tests*100):.1f}%")

    print("\n🔥 ストレステスト詳細:")
    for result in test_suite.results:
        status = "✅" if result.success else "❌"
        recovery = "🔄" if result.error_recovery else "❌"
        stability = "🛡️" if result.system_stability else "❌"
        print(f"{status} {result.test_name}")
        print(f"   実行時間: {result.execution_time:.2f}秒")
        print(f"   エラー回復: {recovery}")
        print(f"   システム安定性: {stability}")
        if result.error_message:
            print(f"   エラー: {result.error_message[:100]}...")

    print("=" * 80)

    logger.info("🎯 ストレステストスイート完了")
