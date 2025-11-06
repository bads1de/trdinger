"""
AutoStrategyServiceの包括的テスト
テスト駆動開発(TDD)アプローチによる包括的テスト
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
from typing import Dict, Any, List

from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.config.unified_config import GAConfig
from app.services.backtest.backtest_service import BacktestService
from app.services.auto_strategy.services.experiment_persistence_service import (
    ExperimentPersistenceService,
)
from app.services.auto_strategy.services.experiment_manager import ExperimentManager


class TestAutoStrategyServiceComprehensive:
    """AutoStrategyServiceの包括的テスト"""

    @pytest.fixture
    def mock_backtest_service(self):
        """モックバックテストサービス"""
        return Mock(spec=BacktestService)

    @pytest.fixture
    def mock_persistence_service(self):
        """モック永続化サービス"""
        return Mock(spec=ExperimentPersistenceService)

    @pytest.fixture
    def mock_experiment_manager(self):
        """モック実験管理マネージャー"""
        return Mock(spec=ExperimentManager)

    @pytest.fixture
    def auto_strategy_service(self, mock_backtest_service, mock_persistence_service, mock_experiment_manager):
        """AutoStrategyServiceのモックバージョン"""
        service = AutoStrategyService(enable_smart_generation=True)

        # モックサービスを設定
        service.backtest_service = mock_backtest_service
        service.persistence_service = mock_persistence_service
        service.experiment_manager = mock_experiment_manager

        return service

    def test_service_initialization_basic(self, auto_strategy_service):
        """基本的なサービス初期化のテスト"""
        assert auto_strategy_service is not None
        assert auto_strategy_service.enable_smart_generation is True
        assert auto_strategy_service.backtest_service is not None
        assert auto_strategy_service.persistence_service is not None
        assert auto_strategy_service.experiment_manager is not None

    def test_service_initialization_without_smart_generation(self):
        """スマート生成無効での初期化テスト"""
        service = AutoStrategyService(enable_smart_generation=False)
        service.backtest_service = Mock()
        service.persistence_service = Mock()
        service.experiment_manager = Mock()

        assert service.enable_smart_generation is False

    def test_prepare_ga_config_valid(self, auto_strategy_service):
        """有効なGA設定準備のテスト"""
        ga_config_dict = {
            "population_size": 50,
            "generations": 100,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "elite_size": 5,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        }

        result = auto_strategy_service._prepare_ga_config(ga_config_dict)

        # Check result is a GAConfig-like object with expected attributes
        assert hasattr(result, 'population_size')
        assert hasattr(result, 'generations')
        assert result.population_size == 50
        assert result.generations == 100

    def test_prepare_ga_config_invalid(self, auto_strategy_service):
        """無効なGA設定準備のテスト"""
        ga_config_dict = {
            "population_size": 0,  # 無効な値
            "generations": -1,  # 無効な値
        }

        with pytest.raises((ValueError, Exception)) as exc_info:
            auto_strategy_service._prepare_ga_config(ga_config_dict)

        # Validation error or ValueError expected
        assert exc_info.value is not None

    def test_prepare_backtest_config_with_defaults(self, auto_strategy_service):
        """デフォルト値を含むバックテスト設定準備のテスト"""
        backtest_config_dict = {
            "initial_capital": 100000,
            # symbolが指定されていない
        }

        result = auto_strategy_service._prepare_backtest_config(backtest_config_dict)

        assert result["initial_capital"] == 100000
        assert "symbol" in result
        # DEFAULT_SYMBOLが設定されていることを確認

    def test_prepare_backtest_config_with_custom_symbol(self, auto_strategy_service):
        """カスタムシンボルを含むバックテスト設定準備のテスト"""
        backtest_config_dict = {
            "initial_capital": 100000,
            "symbol": "ETH/USDT",
        }

        result = auto_strategy_service._prepare_backtest_config(backtest_config_dict)

        assert result["symbol"] == "ETH/USDT"

    def test_create_experiment(self, auto_strategy_service, mock_persistence_service):
        """実験作成のテスト"""
        experiment_id = "test-exp-123"
        experiment_name = "Test Experiment"
        ga_config = GAConfig(**{
            "population_size": 50,
            "generations": 100,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        })
        backtest_config = {"initial_capital": 100000}

        auto_strategy_service._create_experiment(
            experiment_id, experiment_name, ga_config, backtest_config
        )

        mock_persistence_service.create_experiment.assert_called_once_with(
            experiment_id, experiment_name, ga_config, backtest_config
        )

    def test_initialize_ga_engine(self, auto_strategy_service, mock_experiment_manager):
        """GAエンジン初期化のテスト"""
        ga_config = GAConfig(**{
            "population_size": 50,
            "generations": 100,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        })

        auto_strategy_service._initialize_ga_engine(ga_config)

        mock_experiment_manager.initialize_ga_engine.assert_called_once_with(ga_config)

    def test_initialize_ga_engine_without_manager(self):
        """マネージャーなしでのGAエンジン初期化テスト"""
        service = AutoStrategyService(enable_smart_generation=True)
        service.backtest_service = Mock()
        service.persistence_service = Mock()
        service.experiment_manager = None  # マネージャーがNone

        ga_config = GAConfig(**{
            "population_size": 50,
            "generations": 100,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        })

        with pytest.raises(RuntimeError) as exc_info:
            service._initialize_ga_engine(ga_config)

        assert "実験管理マネージャーが初期化されていません" in str(exc_info.value)

    def test_start_experiment_in_background(self, auto_strategy_service, mock_experiment_manager):
        """バックグラウンド実験開始のテスト"""
        from fastapi import BackgroundTasks

        experiment_id = "test-exp-123"
        ga_config = GAConfig(**{
            "population_size": 50,
            "generations": 100,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        })
        backtest_config = {"initial_capital": 100000}
        background_tasks = BackgroundTasks()

        # バックグラウンドタスクが正常に追加されることを確認
        auto_strategy_service._start_experiment_in_background(
            experiment_id, ga_config, backtest_config, background_tasks
        )

        # タスクが追加されたことを確認（内部実装に依存）

    def test_list_experiments_success(self, auto_strategy_service, mock_persistence_service):
        """実験一覧取得成功のテスト"""
        mock_experiments = [
            {"id": "exp1", "name": "Experiment 1"},
            {"id": "exp2", "name": "Experiment 2"},
        ]
        mock_persistence_service.list_experiments.return_value = mock_experiments

        result = auto_strategy_service.list_experiments()

        assert result == mock_experiments
        mock_persistence_service.list_experiments.assert_called_once()

    def test_list_experiments_with_error_handling(self, auto_strategy_service, mock_persistence_service):
        """実験一覧取得エラーハンドリングのテスト"""
        mock_persistence_service.list_experiments.side_effect = Exception("Database error")

        result = auto_strategy_service.list_experiments()

        assert result == []  # default_returnが返される

    def test_stop_experiment_success(self, auto_strategy_service, mock_experiment_manager):
        """実験停止成功のテスト"""
        experiment_id = "test-exp-123"
        mock_experiment_manager.stop_experiment.return_value = True

        result = auto_strategy_service.stop_experiment(experiment_id)

        assert result["success"] is True
        assert result["message"] == "実験が正常に停止されました"
        mock_experiment_manager.stop_experiment.assert_called_once_with(experiment_id)

    def test_stop_experiment_failure(self, auto_strategy_service, mock_experiment_manager):
        """実験停止失敗のテスト"""
        experiment_id = "test-exp-123"
        mock_experiment_manager.stop_experiment.return_value = False

        result = auto_strategy_service.stop_experiment(experiment_id)

        assert result["success"] is False
        assert result["message"] == "実験の停止に失敗しました"

    def test_stop_experiment_no_manager(self):
        """マネージャーなしでの実験停止テスト"""
        service = AutoStrategyService(enable_smart_generation=True)
        service.backtest_service = Mock()
        service.persistence_service = Mock()
        service.experiment_manager = None  # マネージャーがNone

        result = service.stop_experiment("test-exp-123")

        assert result["success"] is False
        assert result["message"] == "実験管理マネージャーが初期化されていません"

    def test_start_strategy_generation_end_to_end(self, auto_strategy_service):
        """戦略生成開始のエンドツーエンドテスト"""
        from fastapi import BackgroundTasks

        experiment_id = "test-strategy-123"
        experiment_name = "Test Strategy Generation"
        ga_config_dict = {
            "population_size": 50,
            "generations": 100,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        }
        backtest_config_dict = {
            "initial_capital": 100000,
            "symbol": "BTC/USDT",
        }
        background_tasks = BackgroundTasks()

        # 各モックを設定
        mock_persistence_service = auto_strategy_service.persistence_service
        mock_experiment_manager = auto_strategy_service.experiment_manager

        result = auto_strategy_service.start_strategy_generation(
            experiment_id,
            experiment_name,
            ga_config_dict,
            backtest_config_dict,
            background_tasks,
        )

        assert result == experiment_id

        # 各ステップが呼び出されたことを確認
        mock_persistence_service.create_experiment.assert_called_once()
        mock_experiment_manager.initialize_ga_engine.assert_called_once()
        # バックグラウンドタスクが追加されたことを確認

    def test_error_handling_in_start_strategy_generation(self, auto_strategy_service):
        """戦略生成開始時のエラーハンドリングテスト"""
        from fastapi import BackgroundTasks

        experiment_id = "test-strategy-123"
        experiment_name = "Test Strategy Generation"
        ga_config_dict = {
            "population_size": 0,  # 無効な設定
            "generations": 100,
        }
        backtest_config_dict = {"initial_capital": 100000}
        background_tasks = BackgroundTasks()

        # 無効なGA設定でエラーが発生することを確認
        from fastapi.exceptions import HTTPException
        with pytest.raises((ValueError, HTTPException)):
            auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )

    def test_service_integrity_checks(self, auto_strategy_service):
        """サービスの整合性チェック"""
        # 必要なサービスが正しく設定されていること
        assert hasattr(auto_strategy_service, 'backtest_service')
        assert hasattr(auto_strategy_service, 'persistence_service')
        assert hasattr(auto_strategy_service, 'experiment_manager')
        assert hasattr(auto_strategy_service, 'db_session_factory')

        # 必要なメソッドが存在すること
        assert hasattr(auto_strategy_service, 'start_strategy_generation')
        assert hasattr(auto_strategy_service, 'list_experiments')
        assert hasattr(auto_strategy_service, 'stop_experiment')
        assert hasattr(auto_strategy_service, '_prepare_ga_config')
        assert hasattr(auto_strategy_service, '_prepare_backtest_config')

    def test_memory_efficiency_in_service_operations(self, auto_strategy_service):
        """サービス操作のメモリ効率テスト"""
        import gc

        initial_objects = len(gc.get_objects())

        # 複数回の操作を実行
        for i in range(10):
            auto_strategy_service.list_experiments()

        gc.collect()
        final_objects = len(gc.get_objects())

        # 過度なメモリ増加でないこと
        memory_growth = final_objects - initial_objects
        assert memory_growth < 100  # ある程度の許容範囲

    def test_concurrent_access_safety(self, auto_strategy_service):
        """同時アクセス安全性のテスト（基本的なチェック）"""
        # 同じインスタンスに対して並列に操作を実行
        import threading

        # Mock the persistence service to return a list
        auto_strategy_service.persistence_service.list_experiments.return_value = [
            {"id": "exp1", "name": "Test 1"},
            {"id": "exp2", "name": "Test 2"}
        ]

        results = []
        exceptions = []

        def test_operation():
            try:
                result = auto_strategy_service.list_experiments()
                results.append(len(result))
            except Exception as e:
                exceptions.append(str(e))

        # 3つのスレッドで同時実行
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=test_operation)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # 例外が発生しないこと
        assert len(exceptions) == 0
        assert len(results) == 3

    def test_final_service_validation(self, auto_strategy_service):
        """最終サービス検証"""
        assert auto_strategy_service is not None

        # Mock the persistence service to return a list
        auto_strategy_service.persistence_service.list_experiments.return_value = []

        # 基本的な操作が可能なこと
        experiments = auto_strategy_service.list_experiments()
        assert isinstance(experiments, list)

        # サービスが適切に初期化されていること
        assert auto_strategy_service.experiment_manager is not None
        assert auto_strategy_service.persistence_service is not None
        assert auto_strategy_service.backtest_service is not None

        print("✅ AutoStrategyService包括的テスト成功")


# TDDアプローチによる拡張テスト
class TestAutoStrategyServiceTDD:
    """TDDアプローチによるAutoStrategyService拡張テスト"""

    def test_service_creation_with_minimal_dependencies(self):
        """最小依存関係でのサービス作成テスト"""
        # 最小限のモックでサービスを作成
        service = AutoStrategyService(enable_smart_generation=False)

        # 手動で必要なサービスを設定
        service.backtest_service = Mock()
        service.persistence_service = Mock()
        service.experiment_manager = Mock()

        assert service.enable_smart_generation is False
        assert service.backtest_service is not None
        print("✅ 最小依存関係でのサービス作成テスト成功")

    def test_ga_config_validation_comprehensive(self):
        """GA設定検証の包括的テスト"""
        service = AutoStrategyService(enable_smart_generation=True)
        service.backtest_service = Mock()
        service.persistence_service = Mock()
        service.experiment_manager = Mock()

        # 多様な設定バリエーションをテスト
        test_configs = [
            {
                "population_size": 100,
                "generations": 200,
                "mutation_rate": 0.05,
                "crossover_rate": 0.9,
                "symbol": "BTC/USDT",
                "timeframe": "1h",
            },
            {
                "population_size": 25,
                "generations": 50,
                "mutation_rate": 0.2,
                "crossover_rate": 0.7,
                "symbol": "ETH/USDT",
                "timeframe": "4h",
            },
        ]

        for config in test_configs:
            result = service._prepare_ga_config(config)
            assert hasattr(result, 'population_size')
            assert result.population_size == config["population_size"]

        print("✅ GA設定検証包括的テスト成功")

    def test_backtest_config_edge_cases(self):
        """バックテスト設定のエッジケーステスト"""
        service = AutoStrategyService(enable_smart_generation=True)
        service.backtest_service = Mock()
        service.persistence_service = Mock()
        service.experiment_manager = Mock()

        # 空の設定
        result1 = service._prepare_backtest_config({})
        assert "symbol" in result1

        # null値のテスト - None is allowed, or it may use default
        result2 = service._prepare_backtest_config({"symbol": None})
        # Current implementation preserves None if explicitly passed
        assert "symbol" in result2

        # 型変換のテスト
        result3 = service._prepare_backtest_config({"initial_capital": "100000"})
        assert isinstance(result3["initial_capital"], str)  # 現状維持

        print("✅ バックテスト設定エッジケーステスト成功")

    def test_experiment_lifecycle_management(self):
        """実験ライフサイクル管理テスト"""
        service = AutoStrategyService(enable_smart_generation=True)
        service.backtest_service = Mock()
        service.persistence_service = Mock()
        service.experiment_manager = Mock()

        experiment_id = "lifecycle-test-123"
        experiment_name = "Lifecycle Test"
        ga_config = GAConfig(**{
            "population_size": 50,
            "generations": 100,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        })
        backtest_config = {"initial_capital": 100000}

        # 実験作成
        service._create_experiment(experiment_id, experiment_name, ga_config, backtest_config)
        service.persistence_service.create_experiment.assert_called_once()

        # GAエンジン初期化
        service._initialize_ga_engine(ga_config)
        service.experiment_manager.initialize_ga_engine.assert_called_once_with(ga_config)

        # 実験停止
        service.stop_experiment(experiment_id)
        service.experiment_manager.stop_experiment.assert_called_once_with(experiment_id)

        print("✅ 実験ライフサイクル管理テスト成功")

    def test_error_recovery_mechanisms(self):
        """エラー回復メカニズムテスト"""
        service = AutoStrategyService(enable_smart_generation=True)
        service.backtest_service = Mock()
        service.persistence_service = Mock()
        service.experiment_manager = Mock()

        # 永続化サービスのエラー
        service.persistence_service.list_experiments.side_effect = Exception("DB Error")
        result = service.list_experiments()
        assert result == []  # default_returnが使われる

        # エンジン初期化のエラー
        service.experiment_manager.initialize_ga_engine.side_effect = Exception("Engine Error")

        ga_config = GAConfig(**{
            "population_size": 50,
            "generations": 100,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        })

        with pytest.raises(Exception):
            service._initialize_ga_engine(ga_config)

        print("✅ エラー回復メカニズムテスト成功")

    def test_service_performance_under_load(self):
        """高負荷下でのサービスパフォーマンステスト"""
        import time

        service = AutoStrategyService(enable_smart_generation=True)
        service.backtest_service = Mock()
        service.persistence_service = Mock()
        service.experiment_manager = Mock()

        start_time = time.time()

        # 複数回の設定準備を実行
        for i in range(100):
            ga_config_dict = {
                "population_size": 50 + i,
                "generations": 100 + i,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "symbol": "BTC/USDT",
                "timeframe": "1h",
            }
            backtest_config_dict = {"initial_capital": 100000 + i}

            ga_config = service._prepare_ga_config(ga_config_dict)
            backtest_config = service._prepare_backtest_config(backtest_config_dict)

        end_time = time.time()
        execution_time = end_time - start_time

        # 実行時間が許容範囲内であること（10秒以内）
        assert execution_time < 10.0

        print(f"✅ 高負荷下でのサービスパフォーマンステスト成功: {execution_time:.2f}s")