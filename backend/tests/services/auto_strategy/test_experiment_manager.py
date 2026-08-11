"""
ExperimentManagerのテスト
"""

from unittest.mock import MagicMock, Mock, patch

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.config.ga.nested_configs import (
    IterativeImprovementConfig,
    ValidationConfig,
)
from app.services.auto_strategy.core.engine.evolution_runner import (
    EvolutionStoppedError,
)
from app.services.auto_strategy.genes import StrategyGene
from app.services.auto_strategy.services import (
    experiment_manager as experiment_manager_module,
)
from app.services.auto_strategy.services.experiment_engine_registry import (
    ExperimentEngineRegistry,
)
from app.services.auto_strategy.services.experiment_manager import ExperimentManager


class TestExperimentManager:
    """ExperimentManagerのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        self.registry = ExperimentEngineRegistry()
        self.mock_backtest_service = Mock()
        self.mock_persistence_service = Mock()
        self.manager = ExperimentManager(
            self.mock_backtest_service,
            self.mock_persistence_service,
            engine_registry=self.registry,
        )

    def teardown_method(self):
        """テスト後の後始末"""
        self.registry.clear()

    def test_init(self):
        """初期化のテスト"""
        assert self.manager.backtest_service == self.mock_backtest_service
        assert self.manager.persistence_service == self.mock_persistence_service
        assert self.manager._engine_registry is self.registry
        assert self.manager._get_active_engine("missing") is None

    def test_initialize_ga_engine(self):
        """GAエンジン初期化のテスト"""
        ga_config = GAConfig()
        ga_config.population_size = 10
        ga_config.generations = 5

        mock_ga_engine = MagicMock()
        with patch(
            "app.services.auto_strategy.core.engine.ga_engine_factory.GeneticAlgorithmEngineFactory.create_engine",
            return_value=mock_ga_engine,
        ):
            returned_engine = self.manager.initialize_ga_engine(
                ga_config, "test_exp_001"
            )

        assert returned_engine is mock_ga_engine
        assert self.manager._get_active_engine("test_exp_001") is mock_ga_engine

    def test_run_experiment_success(self):
        """実験実行成功のテスト"""
        # GA設定の準備
        ga_config = GAConfig()
        ga_config.population_size = 10
        ga_config.generations = 5

        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-12-19",
            "initial_capital": 10000,
        }

        experiment_info = {
            "db_id": 7,
            "name": "AUTO_STRATEGY_GA_2024-01-02_TEST_RUN_",
            "status": "running",
            "config": {"experiment_id": "test_exp_001"},
        }
        self.manager.persistence_service.get_experiment_info.return_value = (
            experiment_info
        )

        strategy = StrategyGene(id="abcdef123456")
        mock_ga_engine = MagicMock()
        mock_ga_engine.run_evolution.return_value = {
            "best_strategy": strategy,
            "best_fitness": 1.5,
            "all_strategies": [strategy],
            "fitness_scores": [1.5],
        }
        mock_ga_engine.is_stop_requested.return_value = False
        with patch(
            "app.services.auto_strategy.core.engine.ga_engine_factory.GeneticAlgorithmEngineFactory.create_engine",
            return_value=mock_ga_engine,
        ):
            self.manager.initialize_ga_engine(ga_config, "test_exp_001")

        self.mock_backtest_service.run_backtest.return_value = {
            "performance_metrics": {"total_return": 0.1},
            "equity_curve": [],
            "trade_history": [],
            "execution_time": 0.25,
        }

        self.manager.run_experiment("test_exp_001", ga_config, backtest_config)

        # 検証
        expected_backtest_config = dict(backtest_config)
        expected_backtest_config["experiment_id"] = "test_exp_001"

        mock_ga_engine.run_evolution.assert_called_once()
        call_args = mock_ga_engine.run_evolution.call_args
        assert call_args[0][0] == ga_config
        assert call_args[0][1] == expected_backtest_config
        assert "progress_callback" in call_args[1]
        self.manager.persistence_service.save_experiment_result.assert_called_once_with(
            "test_exp_001",
            {
                "best_strategy": strategy,
                "best_fitness": 1.5,
                "all_strategies": [strategy],
                "fitness_scores": [1.5],
            },
            ga_config,
            expected_backtest_config,
            experiment_info=experiment_info,
        )
        self.manager.persistence_service.save_backtest_result.assert_called_once()
        saved_backtest_result = (
            self.manager.persistence_service.save_backtest_result.call_args.args[0]
        )
        assert saved_backtest_result["strategy_name"] == "AS_GA_240102_abcdef"
        assert saved_backtest_result["config_json"]["experiment_id"] == "test_exp_001"
        assert saved_backtest_result["config_json"]["db_experiment_id"] == 7
        assert saved_backtest_result["config_json"]["fitness_score"] == 1.5
        self.manager.persistence_service.complete_experiment.assert_called_once_with(
            "test_exp_001"
        )
        self.mock_backtest_service.run_backtest.assert_called_once()
        assert self.manager._get_active_engine("test_exp_001") is None

    def test_run_experiment_exception(self):
        """実験実行中の例外テスト"""
        ga_config = GAConfig()
        backtest_config = {}

        mock_ga_engine = MagicMock()
        mock_ga_engine.run_evolution.side_effect = Exception("Test exception")
        mock_ga_engine.is_stop_requested.return_value = False
        self.manager._register_active_engine("test_exp_001", mock_ga_engine)

        self.manager.run_experiment("test_exp_001", ga_config, backtest_config)

        # エラーハンドリングの検証
        self.manager.persistence_service.fail_experiment.assert_called_once_with(
            "test_exp_001"
        )
        assert self.manager._get_active_engine("test_exp_001") is None

    def test_stop_experiment(self):
        """実験停止のテスト"""
        mock_ga_engine = MagicMock()
        self.manager._register_active_engine("test_exp_001", mock_ga_engine)

        self.manager.stop_experiment("test_exp_001")

        mock_ga_engine.stop_evolution.assert_called_once()
        self.manager.persistence_service.stop_experiment.assert_called_once_with(
            "test_exp_001"
        )

    def test_stop_experiment_across_instances_with_shared_default_registry(self):
        """デフォルト共有レジストリなら別インスタンスから停止できること"""
        ga_config = GAConfig()
        ga_config.population_size = 10
        ga_config.generations = 5
        experiment_id = "test_exp_registry"
        mock_ga_engine = MagicMock()
        other_persistence = Mock()
        experiment_manager_module._DEFAULT_ENGINE_REGISTRY.clear()
        try:
            manager_a = ExperimentManager(
                self.mock_backtest_service, self.mock_persistence_service
            )
            other_manager = ExperimentManager(
                self.mock_backtest_service, other_persistence
            )

            with patch(
                "app.services.auto_strategy.core.engine.ga_engine_factory.GeneticAlgorithmEngineFactory.create_engine",
                return_value=mock_ga_engine,
            ):
                returned_engine = manager_a.initialize_ga_engine(
                    ga_config, experiment_id
                )

            result = other_manager.stop_experiment(experiment_id)

            assert result is True
            assert returned_engine is mock_ga_engine
            mock_ga_engine.stop_evolution.assert_called_once()
            other_persistence.stop_experiment.assert_called_once_with(experiment_id)
        finally:
            experiment_manager_module._DEFAULT_ENGINE_REGISTRY.clear()

    # ------------------------------------------------------------------
    # 自動検証パイプライン統合
    # ------------------------------------------------------------------

    def test_run_experiment_with_validation_pipeline(self):
        """自動検証パイプライン有効時に検証を経て保存されること"""
        ga_config = GAConfig()
        ga_config.population_size = 10
        ga_config.generations = 5
        ga_config.validation_config = ValidationConfig(enabled=True)

        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-12-19",
            "initial_capital": 10000,
        }

        experiment_info = {
            "db_id": 7,
            "name": "AUTO_STRATEGY_GA_2024-01-02_TEST_RUN_",
            "status": "running",
            "config": {"experiment_id": "test_exp_001"},
        }
        self.manager.persistence_service.get_experiment_info.return_value = (
            experiment_info
        )

        strategy = StrategyGene(id="abcdef123456")
        raw_result = {
            "best_strategy": strategy,
            "best_fitness": 1.5,
            "all_strategies": [strategy],
            "fitness_scores": [1.5],
        }
        mock_ga_engine = MagicMock()
        mock_ga_engine.run_evolution.return_value = raw_result
        mock_ga_engine.is_stop_requested.return_value = False
        with patch(
            "app.services.auto_strategy.core.engine.ga_engine_factory.GeneticAlgorithmEngineFactory.create_engine",
            return_value=mock_ga_engine,
        ):
            self.manager.initialize_ga_engine(ga_config, "test_exp_001")

        filtered_result = dict(raw_result)
        filtered_result["validation_results"] = {
            "abcdef123456": {"passed": True, "pass_rate": 0.8},
        }
        mock_validation_service = MagicMock()
        mock_validation_service.validate_and_filter_result.return_value = (
            filtered_result
        )

        self.mock_backtest_service.run_backtest.return_value = {
            "performance_metrics": {"total_return": 0.1},
            "equity_curve": [],
            "trade_history": [],
            "execution_time": 0.25,
        }

        with patch(
            "app.services.auto_strategy.services.experiment_manager.StrategyValidationService",
            return_value=mock_validation_service,
        ):
            self.manager.run_experiment("test_exp_001", ga_config, backtest_config)

        # 検証サービスがGA結果を検証し、検証済み結果が保存される
        mock_validation_service.validate_and_filter_result.assert_called_once_with(
            result=raw_result,
            ga_config=ga_config,
            backtest_config={
                **backtest_config,
                "experiment_id": "test_exp_001",
            },
        )
        saved = self.manager.persistence_service.save_experiment_result.call_args.args[
            1
        ]
        assert saved["validation_results"]["abcdef123456"]["passed"] is True
        self.manager.persistence_service.complete_experiment.assert_called_once_with(
            "test_exp_001"
        )

    def test_run_experiment_skips_save_when_no_passing_strategy(self):
        """合格戦略が無い場合は保存をスキップして完了する"""
        ga_config = GAConfig()
        ga_config.population_size = 10
        ga_config.generations = 5
        ga_config.validation_config = ValidationConfig(enabled=True)

        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
        }

        mock_ga_engine = MagicMock()
        mock_ga_engine.run_evolution.return_value = {
            "best_strategy": StrategyGene(id="abcdef123456"),
            "best_fitness": 1.5,
            "all_strategies": [],
            "fitness_scores": [],
        }
        mock_ga_engine.is_stop_requested.return_value = False
        with patch(
            "app.services.auto_strategy.core.engine.ga_engine_factory.GeneticAlgorithmEngineFactory.create_engine",
            return_value=mock_ga_engine,
        ):
            self.manager.initialize_ga_engine(ga_config, "test_exp_001")

        mock_validation_service = MagicMock()
        mock_validation_service.validate_and_filter_result.return_value = {
            "best_strategy": None,
            "best_fitness": None,
            "all_strategies": [],
            "fitness_scores": [],
            "validation_results": {
                "abcdef123456": {"passed": False, "pass_rate": 0.2},
            },
        }

        with patch(
            "app.services.auto_strategy.services.experiment_manager.StrategyValidationService",
            return_value=mock_validation_service,
        ):
            self.manager.run_experiment("test_exp_001", ga_config, backtest_config)

        # 保存はスキップされ、完了のみ
        self.manager.persistence_service.save_experiment_result.assert_not_called()
        self.manager.persistence_service.save_backtest_result.assert_not_called()
        self.manager.persistence_service.complete_experiment.assert_called_once_with(
            "test_exp_001"
        )

    def test_run_experiment_stop_requested_before_save(self):
        """保存前に停止要求を検知した場合は保存せず停止する"""
        ga_config = GAConfig()
        ga_config.population_size = 10
        ga_config.generations = 5
        backtest_config = {}

        mock_ga_engine = MagicMock()
        mock_ga_engine.run_evolution.return_value = {
            "best_strategy": StrategyGene(id="abcdef123456"),
            "best_fitness": 1.5,
            "all_strategies": [],
            "fitness_scores": [],
        }
        # 1回目の is_stop_requested で True（GA実行直後）
        mock_ga_engine.is_stop_requested.return_value = True
        with patch(
            "app.services.auto_strategy.core.engine.ga_engine_factory.GeneticAlgorithmEngineFactory.create_engine",
            return_value=mock_ga_engine,
        ):
            self.manager.initialize_ga_engine(ga_config, "test_exp_001")

        self.manager.run_experiment("test_exp_001", ga_config, backtest_config)

        self.manager.persistence_service.stop_experiment.assert_called_once_with(
            "test_exp_001"
        )
        self.manager.persistence_service.save_experiment_result.assert_not_called()
        self.manager.persistence_service.complete_experiment.assert_not_called()

    def test_run_experiment_stop_requested_after_save(self):
        """保存後に停止要求を検知した場合は停止状態にする"""
        ga_config = GAConfig()
        ga_config.population_size = 10
        ga_config.generations = 5
        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-12-19",
            "initial_capital": 10000,
        }

        experiment_info = {
            "db_id": 7,
            "name": "AUTO_STRATEGY_GA_2024-01-02_TEST_RUN_",
            "status": "running",
            "config": {"experiment_id": "test_exp_001"},
        }
        self.manager.persistence_service.get_experiment_info.return_value = (
            experiment_info
        )

        mock_ga_engine = MagicMock()
        mock_ga_engine.run_evolution.return_value = {
            "best_strategy": StrategyGene(id="abcdef123456"),
            "best_fitness": 1.5,
            "all_strategies": [StrategyGene(id="abcdef123456")],
            "fitness_scores": [1.5],
        }
        # 1回目: GA実行直後は False、2回目: 保存後は True
        mock_ga_engine.is_stop_requested.side_effect = [False, True]
        with patch(
            "app.services.auto_strategy.core.engine.ga_engine_factory.GeneticAlgorithmEngineFactory.create_engine",
            return_value=mock_ga_engine,
        ):
            self.manager.initialize_ga_engine(ga_config, "test_exp_001")

        self.mock_backtest_service.run_backtest.return_value = {
            "performance_metrics": {"total_return": 0.1},
            "equity_curve": [],
            "trade_history": [],
            "execution_time": 0.25,
        }

        self.manager.run_experiment("test_exp_001", ga_config, backtest_config)

        # 保存は行われ、完了ではなく停止で終わる
        self.manager.persistence_service.save_experiment_result.assert_called_once()
        self.manager.persistence_service.stop_experiment.assert_called_once_with(
            "test_exp_001"
        )
        self.manager.persistence_service.complete_experiment.assert_not_called()

    def test_run_experiment_evolution_stopped_error(self):
        """EvolutionStoppedError 発生時は停止状態にする"""
        ga_config = GAConfig()
        ga_config.population_size = 10
        ga_config.generations = 5
        backtest_config = {}

        mock_ga_engine = MagicMock()
        mock_ga_engine.run_evolution.side_effect = EvolutionStoppedError("stopped")
        mock_ga_engine.is_stop_requested.return_value = False
        self.manager._register_active_engine("test_exp_001", mock_ga_engine)

        self.manager.run_experiment("test_exp_001", ga_config, backtest_config)

        self.manager.persistence_service.stop_experiment.assert_called_once_with(
            "test_exp_001"
        )
        self.manager.persistence_service.fail_experiment.assert_not_called()

    def test_run_experiment_engine_not_initialized(self):
        """エンジン未初期化時は失敗として処理する"""
        ga_config = GAConfig()
        backtest_config = {}

        self.manager.run_experiment("missing_exp", ga_config, backtest_config)

        self.manager.persistence_service.fail_experiment.assert_called_once_with(
            "missing_exp"
        )

    def test_get_validation_service_reuses_engine_evaluator(self):
        """エンジンの評価器を再利用して検証サービスを生成する"""
        mock_engine = MagicMock()
        evaluator = MagicMock()
        mock_engine.individual_evaluator = evaluator

        with patch(
            "app.services.auto_strategy.services.experiment_manager.StrategyValidationService",
        ) as mock_svc_cls:
            service = self.manager._get_validation_service(mock_engine)

        mock_svc_cls.assert_called_once_with(evaluator)
        assert service == mock_svc_cls.return_value

    def test_get_validation_service_fallback_creates_evaluator(self):
        """エンジンに評価器が無い場合は新規評価器で生成する"""
        # spec=[] のMockは未知属性アクセスで AttributeError を起こすため
        # getattr(engine, "individual_evaluator", None) が None になる
        mock_engine = Mock(spec=[])

        with (
            patch(
                "app.services.auto_strategy.core.evaluation.individual_evaluator.IndividualEvaluator",
            ) as mock_evaluator_cls,
            patch(
                "app.services.auto_strategy.services.experiment_manager.StrategyValidationService",
            ) as mock_svc_cls,
        ):
            service = self.manager._get_validation_service(mock_engine)

        mock_evaluator_cls.assert_called_once_with(self.mock_backtest_service)
        mock_svc_cls.assert_called_once_with(mock_evaluator_cls.return_value)
        assert service == mock_svc_cls.return_value

    # ------------------------------------------------------------------
    # 反復改善ループ（PreviousStrategySeedProvider）配線
    # ------------------------------------------------------------------

    def test_initialize_ga_engine_with_iterative_improvement(self):
        """反復改善有効時は PreviousStrategySeedProvider を注入する"""
        ga_config = GAConfig()
        ga_config.population_size = 10
        ga_config.generations = 5
        ga_config.iterative_improvement_config = IterativeImprovementConfig(
            enabled=True, max_seed_strategies=3
        )

        mock_ga_engine = MagicMock()
        mock_provider = MagicMock()
        with (
            patch(
                "app.services.auto_strategy.services.seed_strategy_provider.PreviousStrategySeedProvider",
                return_value=mock_provider,
            ) as mock_provider_cls,
            patch(
                "app.services.auto_strategy.core.engine.ga_engine_factory.GeneticAlgorithmEngineFactory.create_engine",
                return_value=mock_ga_engine,
            ) as mock_create_engine,
        ):
            returned_engine = self.manager.initialize_ga_engine(
                ga_config, "test_exp_001"
            )

        assert returned_engine is mock_ga_engine
        # プロバイダがDBセッションファクトリ付きで生成され、エンジンへ渡される
        mock_provider_cls.assert_called_once_with(
            self.mock_persistence_service.db_session_factory
        )
        create_kwargs = mock_create_engine.call_args.kwargs
        assert create_kwargs["seed_strategy_provider"] is mock_provider

    def test_initialize_ga_engine_without_iterative_improvement(self):
        """反復改善無効時はプロバイダ無しでエンジンを生成する"""
        ga_config = GAConfig()
        ga_config.population_size = 10
        ga_config.generations = 5

        mock_ga_engine = MagicMock()
        with (
            patch(
                "app.services.auto_strategy.services.seed_strategy_provider.PreviousStrategySeedProvider",
            ) as mock_provider_cls,
            patch(
                "app.services.auto_strategy.core.engine.ga_engine_factory.GeneticAlgorithmEngineFactory.create_engine",
                return_value=mock_ga_engine,
            ) as mock_create_engine,
        ):
            returned_engine = self.manager.initialize_ga_engine(
                ga_config, "test_exp_001"
            )

        assert returned_engine is mock_ga_engine
        mock_provider_cls.assert_not_called()
        create_kwargs = mock_create_engine.call_args.kwargs
        assert create_kwargs["seed_strategy_provider"] is None
