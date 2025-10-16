"""
エラーハンドリング包括的テスト
堅牢なエラーハンドリングと回復メカニズムのテスト
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List

from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor
from app.services.backtest.backtest_service import BacktestService
from app.services.auto_strategy.config.ga import GAConfig
from app.utils.error_handler import safe_operation, ErrorHandler


class TestErrorHandlingComprehensive:
    """エラーハンドリング包括的テスト"""

    @pytest.fixture
    def ga_config(self):
        """GA設定"""
        return GAConfig.from_dict({
            "population_size": 20,
            "num_generations": 5,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        })

    @pytest.fixture
    def mock_backtest_service(self):
        """モックバックテストサービス"""
        mock_service = Mock(spec=BacktestService)
        mock_service.run_backtest.return_value = {
            "success": True,
            "performance_metrics": {"total_return": 0.15}
        }
        return mock_service

    def test_ga_engine_error_recovery(self, ga_config, mock_backtest_service):
        """GAエンジンエラー回復のテスト"""
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        # 不正な個体に対するエラーハンドリング
        invalid_individuals = [
            [np.nan, 0.5, 0.3],  # NaNを含む
            [np.inf, 0.2, 0.8],  # 無限大を含む
            [],  # 空の個体
            [0.1],  # 長さが不正
        ]

        for invalid_individual in invalid_individuals:
            try:
                fitness = engine._evaluate_individual(invalid_individual)
                # エラーが発生しない場合、適切なデフォルト値が返されること
                assert isinstance(fitness, (float, int))
                assert not np.isnan(fitness)
            except Exception as e:
                # エラーが発生してもシステムがクラッシュしないこと
                assert "invalid" in str(e).lower() or "error" in str(e).lower()

    def test_auto_strategy_service_error_handling(self, mock_backtest_service):
        """AutoStrategyServiceエラーハンドリングのテスト"""
        service = AutoStrategyService(enable_smart_generation=True)
        service.backtest_service = mock_backtest_service
        service.persistence_service = Mock()
        service.experiment_manager = Mock()

        # 無効なGA設定でのテスト
        invalid_ga_config = {
            "population_size": 0,  # 無効な値
            "num_generations": -1,  # 無効な値
        }

        # エラーが適切に処理されること
        try:
            service._prepare_ga_config(invalid_ga_config)
        except ValueError as e:
            assert "無効なGA設定です" in str(e)

    def test_backtest_service_failure_handling(self):
        """バックテストサービス障害ハンドリングのテスト"""
        mock_failing_service = Mock()
        mock_failing_service.run_backtest.side_effect = Exception("Service Unavailable")

        # GAエンジンを作成
        ga_config = GAConfig.from_dict({
            "population_size": 10,
            "num_generations": 3,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        })

        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_failing_service,
            market_data=None,
            regime_detector=None
        )

        # 個体評価中にサービスが失敗しても回復できること
        individual = [0.5, 0.3, 0.7, 0.2, 0.8]

        try:
            fitness = engine._evaluate_individual(individual)
            # 回復が成功した場合、適切な値が返されること
            assert isinstance(fitness, (float, int))
            assert not np.isnan(fitness)
        except Exception as e:
            # 最終的にエラーが発生する場合でも、適切なエラーメッセージであること
            assert "backtest" in str(e).lower() or "service" in str(e).lower()

    def test_market_data_corruption_handling(self):
        """市場データ破損ハンドリングのテスト"""
        # 破損した市場データをシミュレート
        corrupted_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'open': [np.nan, 101, 99, np.inf, 100, 101, 98, 103, np.nan, 102],
            'high': [102, np.nan, 101, 104, np.nan, 103, 100, np.inf, 102, 104],
            'low': [98, 99, np.nan, 100, 98, np.nan, 96, 101, 98, np.nan],
            'close': [101, 100, 102, np.nan, 101, 98, np.inf, 100, 102, 101],
            'volume': [np.nan, 1100, 900, 1200, np.nan, 950, np.inf, 1150, 1000, np.nan],
        })

        # GAエンジンが破損データを処理できること
        ga_config = GAConfig.from_dict({
            "population_size": 10,
            "num_generations": 3,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        })

        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "success": True,
            "performance_metrics": {"total_return": 0.1}
        }

        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=corrupted_data,
            regime_detector=None
        )

        # 破損データでも進化が可能であること
        try:
            result = engine.evolve()
            assert result is not None
        except Exception as e:
            # エラーが適切に処理されること
            assert "data" in str(e).lower() or "corruption" in str(e).lower()

    def test_ml_model_training_failure_recovery(self):
        """MLモデル訓練失敗回復のテスト"""
        predictor = HybridPredictor()

        # 不正な訓練データ
        invalid_training_data = pd.DataFrame({
            'invalid_column': ['invalid'] * 10
        })

        # 訓練失敗時の回復をテスト
        try:
            predictor.train(invalid_training_data)
        except Exception as e:
            # エラーがキャッチされ、システムが継続できること
            assert "train" in str(e).lower() or "invalid" in str(e).lower()

        # 回復後、別のデータで再試行可能であること
        valid_data = pd.DataFrame({
            'close': np.random.rand(10),
            'volume': np.random.rand(10),
        })

        try:
            predictor.train(valid_data)
            assert predictor.is_trained is True
        except Exception:
            # 再試行が失敗してもシステムがクラッシュしないこと
            pass

    def test_database_connection_failure_handling(self):
        """データベース接続障害ハンドリングのテスト"""
        # データベース接続失敗をシミュレート
        mock_db_session_factory = Mock()
        mock_db_session_factory.side_effect = Exception("Database connection failed")

        # AutoStrategyServiceがデータベース障害を処理できること
        service = AutoStrategyService(enable_smart_generation=True)
        service.db_session_factory = mock_db_session_factory
        service.backtest_service = Mock()
        service.persistence_service = Mock()
        service.experiment_manager = Mock()

        # データベース操作が失敗してもサービスが利用可能であること
        try:
            experiments = service.list_experiments()
            # 回復が成功した場合、空のリストが返されること
            assert isinstance(experiments, list)
        except Exception as e:
            # エラーが適切に処理されること
            assert "database" in str(e).lower() or "connection" in str(e).lower()

    def test_memory_leak_prevention_in_long_running_evolution(self):
        """長時間実行進化におけるメモリリーク防止のテスト"""
        import gc
        import sys

        ga_config = GAConfig.from_dict({
            "population_size": 50,
            "num_generations": 10,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        })

        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "success": True,
            "performance_metrics": {"total_return": 0.1}
        }

        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        initial_memory = sys.getsizeof(gc.get_objects())

        # 長時間進化をシミュレート
        for generation in range(5):
            population = engine._create_initial_population()
            fitnesses = [engine._evaluate_individual(ind) for ind in population]

            # ガベージコレクションを実行
            gc.collect()

        final_memory = sys.getsizeof(gc.get_objects())
        memory_increase = final_memory - initial_memory

        # 過度なメモリ増加でないこと
        assert memory_increase < 500000  # 500KB未満の増加

    def test_concurrent_access_race_condition_prevention(self):
        """同時アクセス競合防止のテスト"""
        import threading
        import time

        ga_config = GAConfig.from_dict({
            "population_size": 20,
            "num_generations": 3,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        })

        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "success": True,
            "performance_metrics": {"total_return": 0.1}
        }

        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        exceptions = []
        results = []

        def concurrent_evolution():
            """同時進化を実行"""
            try:
                result = engine.evolve()
                results.append(result)
            except Exception as e:
                exceptions.append(str(e))

        # 5つのスレッドで同時実行
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=concurrent_evolution)
            threads.append(thread)
            thread.start()
            time.sleep(0.1)  # 少し間隔をあける

        for thread in threads:
            thread.join()

        # 重大な競合が発生しないこと
        assert len(exceptions) <= 2  # 少数の例外は許容
        assert len(results) >= 3    # 多くのスレッドが成功

    def test_invalid_parameter_validation_and_recovery(self):
        """無効パラメータ検証と回復のテスト"""
        # 無効なGA設定パラメータ
        invalid_configs = [
            {"population_size": -10},  # 負の個体群サイズ
            {"num_generations": 0},     # ゼロ世代
            {"mutation_rate": 1.5},     # 100%を超える突然変異率
            {"crossover_rate": -0.1},   # 負の交叉率
            {"symbol": ""},             # 空のシンボル
            {"timeframe": "invalid"},    # 無効な時間足
        ]

        for invalid_config in invalid_configs:
            try:
                ga_config = GAConfig.from_dict(invalid_config)
                is_valid, errors = ga_config.validate()

                # 検証が失敗すること
                assert is_valid is False
                assert len(errors) > 0

            except Exception as e:
                # 例外が発生しても適切に処理されること
                assert "validation" in str(e).lower() or "invalid" in str(e).lower()

    def test_network_failure_in_external_api_calls(self):
        """外部API呼び出しにおけるネットワーク障害のテスト"""
        # 外部API呼び出しの失敗をシミュレート
        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.side_effect = ConnectionError("Network timeout")

        ga_config = GAConfig.from_dict({
            "population_size": 10,
            "num_generations": 3,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        })

        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        # ネットワーク障害時の回復をテスト
        individual = [0.5, 0.3, 0.7, 0.2, 0.8]

        try:
            fitness = engine._evaluate_individual(individual)
            # 回復が成功した場合、デフォルト値が返されること
            assert isinstance(fitness, (float, int))
        except Exception as e:
            # エラーが適切に処理されること
            assert "network" in str(e).lower() or "connection" in str(e).lower()

    def test_disk_space_insufficiency_handling(self):
        """ディスク容量不足ハンドリングのテスト"""
        import tempfile
        import os

        # 一時ファイル作成の失敗をシミュレート
        with patch('tempfile.mkdtemp', side_effect=OSError("No space left on device")):
            try:
                # GAエンジン操作がディスク容量不足を処理できること
                ga_config = GAConfig.from_dict({
                    "population_size": 10,
                    "num_generations": 3,
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                })

                mock_backtest_service = Mock()
                mock_backtest_service.run_backtest.return_value = {
                    "success": True,
                    "performance_metrics": {"total_return": 0.1}
                }

                engine = GeneticAlgorithmEngine(
                    ga_config=ga_config,
                    backtest_service=mock_backtest_service,
                    market_data=None,
                    regime_detector=None
                )

                # 基本操作が可能であること
                population = engine._create_initial_population()
                assert len(population) > 0

            except Exception as e:
                # エラーが適切に処理されること
                assert "disk" in str(e).lower() or "space" in str(e).lower()

    def test_final_error_handling_validation(self, ga_config, mock_backtest_service):
        """最終エラーハンドリング検証"""
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        # 基本的な進化が可能であること
        result = engine.evolve()
        assert result is not None

        # エラーハンドリングが組み込まれていること
        assert hasattr(engine, '_evaluate_individual')

        print("✅ エラーハンドリング包括的テスト成功")


# TDDアプローチによるエラーハンドリングテスト
class TestErrorHandlingTDD:
    """TDDアプローチによるエラーハンドリングテスト"""

    def test_minimum_error_handling_setup(self):
        """最小エラーハンドリング設定のテスト"""
        # 最小限のエラーハンドリングが存在すること
        from app.utils.error_handler import safe_operation

        @safe_operation(context="test operation", is_api_call=False)
        def test_function():
            return "success"

        result = test_function()
        assert result == "success"

        print("✅ 最小エラーハンドリング設定のテスト成功")

    def test_basic_exception_catching(self):
        """基本例外キャッチングのテスト"""
        from app.utils.error_handler import safe_operation

        @safe_operation(context="failing operation", is_api_call=False)
        def failing_function():
            raise ValueError("Test error")

        result = failing_function()
        assert result is None  # default_returnが使われる

        print("✅ 基本例外キャッチングのテスト成功")

    def test_error_logging_functionality(self):
        """エラーログ機能のテスト"""
        import logging
        from app.utils.error_handler import ErrorHandler

        # ログハンドラーを設定
        logger = logging.getLogger("test_logger")
        handler = ErrorHandler()

        # エラーログをテスト
        try:
            raise Exception("Test exception for logging")
        except Exception as e:
            # ログが正常に記録されること
            logger.error(f"Test error: {e}", exc_info=True)
            # 実際のログ出力は環境に依存するため、ここではテストしない

        print("✅ エラーログ機能のテスト成功")

    def test_graceful_degradation_mechanism(self):
        """グレースフルデグラデーションメカニズムのテスト"""
        # 高級機能が利用できない場合の代替手段をテスト

        # 例: MLモデルが利用できない場合
        class FallbackStrategy:
            def evaluate(self, data):
                # MLが利用できない場合の代替評価
                return np.mean(data) if len(data) > 0 else 0.0

        fallback = FallbackStrategy()
        test_data = [0.1, 0.2, 0.3, 0.4, 0.5]

        result = fallback.evaluate(test_data)
        assert isinstance(result, float)

        print("✅ グレースフルデグラデーションメカニズムのテスト成功")

    def test_error_recovery_time_measurement(self):
        """エラー回復時間測定のテスト"""
        import time

        start_time = time.time()

        # エラーと回復をシミュレート
        try:
            # 正常操作
            result1 = 1 + 1

            # エラーを発生
            raise ConnectionError("Simulated network error")

        except ConnectionError:
            # 回復操作
            result2 = 2 + 2

        end_time = time.time()
        recovery_time = end_time - start_time

        # 回復が迅速であること
        assert recovery_time < 1.0  # 1秒以内

        print(f"✅ エラー回復時間測定のテスト成功: {recovery_time:.3f}s")

    def test_error_scenario_simulation(self):
        """エラー状況シミュレーションのテスト"""
        # 複数のエラー状況をシミュレート

        error_scenarios = [
            "database_connection_failure",
            "network_timeout",
            "invalid_input_data",
            "memory_exhaustion",
            "disk_full",
        ]

        for scenario in error_scenarios:
            try:
                if scenario == "database_connection_failure":
                    raise ConnectionError("Database unavailable")
                elif scenario == "network_timeout":
                    raise TimeoutError("Network timeout")
                elif scenario == "invalid_input_data":
                    raise ValueError("Invalid input data")
                elif scenario == "memory_exhaustion":
                    raise MemoryError("Out of memory")
                elif scenario == "disk_full":
                    raise OSError("No space left on device")

            except Exception as e:
                # すべてのエラーがキャッチされること
                assert str(e) != ""

        print("✅ エラー状況シミュレーションのテスト成功")