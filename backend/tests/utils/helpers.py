"""
テストヘルパー関数

テスト全体で使用するヘルパー関数を提供します。
"""

import time
import psutil
import os
from typing import Any, Callable, Dict, List, Tuple
from contextlib import contextmanager


class TestExecutionHelper:
    """テスト実行ヘルパー"""

    @staticmethod
    def run_test_suite(test_methods: List[Callable]) -> Dict[str, Any]:
        """テストスイートを実行し、結果を集計"""
        results = {"passed": 0, "failed": 0, "total": len(test_methods), "details": []}

        for test_method in test_methods:
            try:
                test_method()
                results["passed"] += 1
                results["details"].append(
                    {"test": test_method.__name__, "status": "PASSED"}
                )
            except Exception as e:
                results["failed"] += 1
                results["details"].append(
                    {"test": test_method.__name__, "status": "FAILED", "error": str(e)}
                )

        return results

    @staticmethod
    def print_test_results(results: Dict[str, Any]):
        """テスト結果を出力"""
        print(f"\n{'='*60}")
        print(f"テスト結果: {results['passed']}/{results['total']} 成功")
        print(f"{'='*60}")

        for detail in results["details"]:
            status_symbol = "✓" if detail["status"] == "PASSED" else "✗"
            print(f"{status_symbol} {detail['test']}")
            if detail["status"] == "FAILED":
                print(f"  エラー: {detail['error']}")


@contextmanager
def performance_monitor(operation_name: str):
    """パフォーマンス監視コンテキストマネージャー"""
    process = psutil.Process(os.getpid())

    # 開始時の測定
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB

    try:
        yield
    finally:
        # 終了時の測定
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB

        execution_time = end_time - start_time
        memory_used = end_memory - start_memory

        print(f"\n📊 {operation_name} パフォーマンス:")
        print(f"  実行時間: {execution_time:.2f}秒")
        print(f"  メモリ使用量: {memory_used:.2f}MB")


def assert_financial_precision(actual: float, expected: float, tolerance: float = 1e-8):
    """財務計算の精度をアサート"""
    assert abs(actual - expected) <= tolerance, (
        f"財務計算の精度エラー: 期待値={expected}, 実際値={actual}, "
        f"差={abs(actual - expected)}, 許容誤差={tolerance}"
    )


def assert_dataframe_valid(df, required_columns: List[str], min_rows: int = 1):
    """DataFrameの妥当性をアサート"""
    assert not df.empty, "DataFrameが空です"
    assert len(df) >= min_rows, f"データ行数が不足: {len(df)} < {min_rows}"

    missing_columns = [col for col in required_columns if col not in df.columns]
    assert not missing_columns, f"必要なカラムが不足: {missing_columns}"


def assert_no_memory_leak(func: Callable, max_memory_increase: float = 50.0):
    """メモリリークがないことをアサート"""
    process = psutil.Process(os.getpid())

    # 初期メモリ使用量
    initial_memory = process.memory_info().rss / 1024 / 1024

    # 関数を複数回実行
    for _ in range(10):
        func()

    # 最終メモリ使用量
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = final_memory - initial_memory

    assert memory_increase <= max_memory_increase, (
        f"メモリリークの可能性: {memory_increase:.2f}MB増加 "
        f"(許容値: {max_memory_increase}MB)"
    )


class MockDataHelper:
    """モックデータヘルパー"""

    @staticmethod
    def create_mock_response(data: Any, status_code: int = 200):
        """モックレスポンスを作成"""
        from unittest.mock import Mock

        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.json.return_value = data
        return mock_response

    @staticmethod
    def create_mock_exception(exception_type: type, message: str):
        """モック例外を作成"""
        return exception_type(message)


class ConcurrencyTestHelper:
    """並行処理テストヘルパー"""

    @staticmethod
    def run_concurrent_operations(func: Callable, num_threads: int = 5) -> List[Any]:
        """並行処理でオペレーションを実行"""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []
        errors = []

        def wrapped_func():
            try:
                return func()
            except Exception as e:
                errors.append(e)
                raise

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(wrapped_func) for _ in range(num_threads)]

            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    pass  # エラーは既にerrorsリストに追加済み

        if errors:
            raise Exception(f"並行処理中にエラーが発生: {errors}")

        return results
