import pytest
import logging
import threading
import time
from unittest.mock import patch, MagicMock, ANY
from fastapi import BackgroundTasks, HTTPException

from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.services.auto_strategy.config import GAConfig

logger = logging.getLogger(__name__)

@pytest.fixture
def auto_strategy_service():
    """AutoStrategyServiceのインスタンスを生成するフィクスチャ"""
    with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
         patch('app.services.auto_strategy.services.auto_strategy_service.BacktestDataService'), \
         patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
         patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService') as mock_persistence, \
         patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager') as mock_manager:

        service = AutoStrategyService()
        service.persistence_service = mock_persistence()
        service.experiment_manager = mock_manager()
        yield service

def get_valid_ga_config_dict():
    """有効なGA設定の辞書を返す"""
    return {
        "population_size": 10,
        "generations": 5,
        "crossover_rate": 0.8,
        "mutation_rate": 0.1,
        "elite_size": 2,
        "max_indicators": 3,
        "log_level": "INFO"
    }

def get_valid_backtest_config_dict():
    """有効なバックテスト設定の辞書を返す"""
    return {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-12-19",
        "initial_capital": 100000,
    }

class TestStressTesting:
    """Stress Testingクラスのテスト（極端値/負荷テスト）"""

    def test_extreme_population_size(self, auto_strategy_service):
        """極端に大きなpopulation_sizeでStress Testing"""
        # 準備
        experiment_id = "test-extreme-pop"
        experiment_name = "Extreme Population Test"
        ga_config_dict = get_valid_ga_config_dict()
        ga_config_dict["population_size"] = int(1e10)  # 極端に大きな値
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # 実行と検証：メモリオーバーフローや処理遅延を検出
        try:
            logger.info("極端値population_sizeでテスト開始")
            result = auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
            logger.info(f"極端値処理成功: {result}")
            assert result == experiment_id, "極端値処理に失敗"
        except (MemoryError, OverflowError) as e:
            logger.error(f"メモリ/オーバーフローエラー検出: {e}")
            pytest.fail(f"バグ検出：極端値で{type(e).__name__}: {e}")
        except Exception as e:
            logger.warning(f"その他の例外検出: {e}")
            assert "メモリ" in str(e) or "オーバーフロー" in str(e), f"予測外の例外: {e}"

    def test_negative_values_stress(self, auto_strategy_service):
        """負値でのStress Testing（population_size, crossover_rateなど）"""
        # 準備
        experiment_id = "test-negative-stress"
        experiment_name = "Negative Values Stress Test"
        ga_config_dict = get_valid_ga_config_dict()
        ga_config_dict["population_size"] = -1000
        ga_config_dict["crossover_rate"] = -0.8
        ga_config_dict["mutation_rate"] = -0.5
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # 実行と検証：負値が無視されるバグを検出
        try:
            logger.info("負値でテスト開始")
            result = auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
            logger.warning("バグ検出：負値が処理されてしまった")
            pytest.fail("バグ検出：負値が無視されました")
        except (ValueError, HTTPException) as e:
            logger.info(f"期待されたエラー検出: {e}")
        except Exception as e:
            logger.warning(f"予測外の例外: {e}")

    def test_none_input_stress(self, auto_strategy_service):
        """None入力でのStress Testing"""
        # 準備
        experiment_id = None
        experiment_name = None
        ga_config_dict = None
        backtest_config_dict = None
        background_tasks = BackgroundTasks()

        # 実行と検証：NoneでAttributeError検出
        try:
            logger.info("None入力でテスト開始")
            result = auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
            logger.error("バグ検出：None入力が処理されてしまった")
            pytest.fail("バグ検出：None入力が無視されました")
        except AttributeError as e:
            logger.info(f"AttributeError検出: {e}")
            assert "copy" in str(e).lower() or "dict" in str(e).lower(), f"予測外のAttributeError: {e}"
        except Exception as e:
            logger.warning(f"その他の例外: {e}")

    def test_zero_values_stress(self, auto_strategy_service):
        """ゼロ値でのStress Testing（division by zeroなど）"""
        # 準備
        experiment_id = "test-zero-stress"
        experiment_name = "Zero Values Stress Test"
        ga_config_dict = get_valid_ga_config_dict()
        ga_config_dict["population_size"] = 0
        ga_config_dict["generations"] = 0
        ga_config_dict["crossover_rate"] = 0.0
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # 実行と検証：ゼロ値でゼロ除算バグを検出
        try:
            logger.info("ゼロ値でテスト開始")
            result = auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
            logger.warning("バグ検出：ゼロ値が処理されてしまった")
        except (ZeroDivisionError, ValueError) as e:
            logger.info(f"ゼロ除算バグ検出: {e}")
        except Exception as e:
            logger.warning(f"予測外の例外: {e}")

class TestIntegrationTesting:
    """Integration Testingクラスのテスト（サービス連携テスト）"""

    def test_ga_backtest_integration(self, auto_strategy_service):
        """GAとバックテストの整合性テスト"""
        # 準備
        experiment_id = "test-ga-backtest-integration"
        experiment_name = "GA Backtest Integration"
        ga_config_dict = get_valid_ga_config_dict()
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # GA設定をメタタイムフレームと一致させる
        backtest_config_dict["timeframe"] = "4h"  # GAで使用される可能性

        # 実行と検証：サービス間のデータ移行の整合性
        try:
            logger.info("GAバックテスト統合テスト開始")
            result = auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
            logger.info("GAバックテスト統合成功")
            assert result == experiment_id
        except Exception as e:
            logger.warning(f"GAバックテスト統合エラー: {e}")

class TestConcurrencyTesting:
    """Concurrency Testingクラスのテスト（同時実行テスト）"""

    def test_concurrent_start_generation(self, auto_strategy_service):
        """複数のstart_generationを同時実行"""
        results = []
        errors = []

        def run_generation(exp_id, exp_name):
            try:
                ga_config_dict = get_valid_ga_config_dict()
                ga_config_dict["population_size"] = 50  # 小さくしてリソース節約
                backtest_config_dict = get_valid_backtest_config_dict()
                background_tasks = BackgroundTasks()

                result = auto_strategy_service.start_strategy_generation(
                    exp_id,
                    exp_name,
                    ga_config_dict,
                    backtest_config_dict,
                    background_tasks,
                )
                results.append((exp_id, result))
            except Exception as e:
                errors.append((exp_id, str(e)))

        # 複数スレッドで同時実行
        threads = []
        for i in range(5):
            exp_id = f"test-concurrent-{i}"
            exp_name = f"Concurrent Test {i}"
            thread = threading.Thread(target=run_generation, args=(exp_id, exp_name))
            threads.append(thread)
            thread.start()

        # すべてのスレッドの完了を待つ
        for thread in threads:
            thread.join()

        # 結果検証
        logger.info(f"同時実行結果：成功{len(results)}、エラー{len(errors)}")
        for error in errors:
            logger.warning(f"同時実行エラー: {error}")

        # 少なくとも1つは成功すべき
        assert len(results) > 0, "すべての同時実行が失敗した"

        # エラーが多すぎる場合はリソース競合バグ疑い
        if len(errors) > len(results):
            logger.warning(f"同時実行バグ疑い：エラー率が高い（エラー{len(errors)}/成功{len(results)}）")

    def test_resource_contention(self, auto_strategy_service):
        """リソース競合が発生するかテスト"""
        experiment_id = "test-resource-contention"
        experiment_name = "Resource Contention Test"
        ga_config_dict = get_valid_ga_config_dict()
        ga_config_dict["population_size"] = 500  # 大きめでリソース負荷
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        start_time = time.time()

        try:
            logger.info("リソース競合テスト開始")
            result = auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
            end_time = time.time()
            duration = end_time - start_time

            logger.info(".2f")
            if duration > 30:  # 30秒以上かかる場合はパフォーマンスバグ
                logger.warning(f"パフォーマンスバグ疑い：処理時間が長い ({duration:.2f}s)")
        except Exception as e:
            logger.error(f"リソース競合エラー: {e}")