"""
GA Engineのテストモジュール
"""

import random
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.auto_strategy.core.engine.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.core.engine.ga_engine_factory import (
    GeneticAlgorithmEngineFactory,
)
from app.services.auto_strategy.core.engine.report_selection import (
    set_two_stage_metadata,
)
from app.services.auto_strategy.core.evaluation.individual_evaluator import (
    IndividualEvaluator,
)
from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.services.backtest.services.backtest_service import BacktestService
from app.services.auto_strategy.config.ga.nested_configs import (
    EvaluationConfig,
    HybridConfig,
)


class TestGeneticAlgorithmEngine:
    """GeneticAlgorithmEngineの初期化とモード切り替えのテスト"""

    @pytest.fixture
    def mock_backtest_service(self):
        """Mock BacktestService"""
        service = Mock(spec=BacktestService)
        service.run_backtest.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.25,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "profit_factor": 1.8,
            "total_trades": 100,
        }
        return service

    @pytest.fixture
    def mock_gene_generator(self):
        """Mock RandomGeneGenerator"""
        generator = Mock(spec=RandomGeneGenerator)
        from app.services.auto_strategy.genes import (
            IndicatorGene,
            PositionSizingGene,
            PositionSizingMethod,
            StrategyGene,
            TPSLGene,
        )

        mock_gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ],
            risk_management={},
            tpsl_gene=TPSLGene(take_profit_pct=0.01, stop_loss_pct=0.005),
            position_sizing_gene=PositionSizingGene(
                method=PositionSizingMethod.FIXED_QUANTITY, fixed_quantity=1000
            ),
            metadata={"generated_by": "Test"},
        )
        generator.generate_random_gene.return_value = mock_gene
        return generator

    def test_standard_mode_initialization(
        self,
        mock_backtest_service,
        mock_gene_generator,
    ):
        """標準GAモードでの初期化を確認"""
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            gene_generator=mock_gene_generator,
            hybrid_mode=False,
        )

        # 標準モードであることを確認
        assert engine.hybrid_mode is False
        assert isinstance(engine.individual_evaluator, IndividualEvaluator)
        assert engine.gene_generator == mock_gene_generator

    def test_hybrid_mode_initialization(
        self,
        mock_backtest_service,
        mock_gene_generator,
    ):
        """ハイブリッドGA+MLモードでの初期化を確認"""
        mock_predictor = Mock()
        mock_feature_adapter = Mock()

        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            gene_generator=mock_gene_generator,
            hybrid_mode=True,
            hybrid_predictor=mock_predictor,
            hybrid_feature_adapter=mock_feature_adapter,
        )

        # ハイブリッドモードであることを確認
        assert engine.hybrid_mode is True
        # 標準名のハイブリッド評価器が使用され、内部最適化は実装側に吸収される
        assert type(engine.individual_evaluator).__name__ == "HybridIndividualEvaluator"

    def test_engine_components_are_set(
        self,
        mock_backtest_service,
        mock_gene_generator,
    ):
        """エンジンのコンポーネントが正しく設定されることを確認"""
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service, gene_generator=mock_gene_generator
        )

        # 必須コンポーネントが設定されていることを確認
        assert engine.backtest_service is not None
        assert engine.gene_generator is not None
        assert engine.individual_evaluator is not None
        assert engine.deap_setup is not None

    @patch(
        "app.services.auto_strategy.generators.seed_strategy_factory.SeedStrategyFactory.get_all_seeds"
    )
    def test_seed_strategies_are_shuffled_deterministically_with_random_state(
        self,
        mock_get_all_seeds,
        mock_backtest_service,
        mock_gene_generator,
    ):
        """random_state があれば seed の注入順が deterministic に並び替わる"""
        from app.services.auto_strategy.genes import StrategyGene
        from app.services.auto_strategy.config.ga import GAConfig

        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            gene_generator=mock_gene_generator,
        )

        seeds = [
            StrategyGene(metadata={"seed_strategy": "a"}),
            StrategyGene(metadata={"seed_strategy": "b"}),
            StrategyGene(metadata={"seed_strategy": "c"}),
            StrategyGene(metadata={"seed_strategy": "d"}),
        ]
        mock_get_all_seeds.return_value = seeds

        config = GAConfig(random_state=42)
        shuffled = engine._get_seed_strategies_for_injection(config)

        expected = list(seeds)
        random.Random(42).shuffle(expected)

        assert [s.metadata["seed_strategy"] for s in shuffled] == [
            s.metadata["seed_strategy"] for s in expected
        ]
        # 元の seed 順は破壊しない
        assert [s.metadata["seed_strategy"] for s in seeds] == ["a", "b", "c", "d"]

    @patch("app.services.auto_strategy.core.engine.ga_engine.random.shuffle")
    @patch(
        "app.services.auto_strategy.generators.seed_strategy_factory.SeedStrategyFactory.get_all_seeds"
    )
    def test_seed_strategies_use_runtime_shuffle_when_random_state_missing(
        self,
        mock_get_all_seeds,
        mock_shuffle,
        mock_backtest_service,
        mock_gene_generator,
    ):
        """random_state がない場合は実行時の shuffle を使う"""
        from app.services.auto_strategy.genes import StrategyGene
        from app.services.auto_strategy.config.ga import GAConfig

        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            gene_generator=mock_gene_generator,
        )

        seeds = [
            StrategyGene(metadata={"seed_strategy": "a"}),
            StrategyGene(metadata={"seed_strategy": "b"}),
            StrategyGene(metadata={"seed_strategy": "c"}),
        ]
        mock_get_all_seeds.return_value = seeds

        def reverse_in_place(values):
            values.reverse()

        mock_shuffle.side_effect = reverse_in_place

        config = GAConfig(random_state=None)
        shuffled = engine._get_seed_strategies_for_injection(config)

        assert [s.metadata["seed_strategy"] for s in shuffled] == ["c", "b", "a"]
        mock_shuffle.assert_called_once()

    @pytest.mark.parametrize("random_state", [np.int32(42), np.int64(42)])
    def test_seed_strategies_accept_numpy_integer_random_state(
        self,
        random_state,
        mock_backtest_service,
        mock_gene_generator,
    ):
        """NumPy スカラーの random_state でも seed の並び替えができる"""
        from app.services.auto_strategy.config.ga import GAConfig
        from app.services.auto_strategy.genes import StrategyGene

        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            gene_generator=mock_gene_generator,
        )

        seeds = [
            StrategyGene(metadata={"seed_strategy": "a"}),
            StrategyGene(metadata={"seed_strategy": "b"}),
            StrategyGene(metadata={"seed_strategy": "c"}),
            StrategyGene(metadata={"seed_strategy": "d"}),
        ]
        with patch(
            "app.services.auto_strategy.generators.seed_strategy_factory.SeedStrategyFactory.get_all_seeds"
        ) as mock_get_all_seeds:
            mock_get_all_seeds.return_value = seeds

            config = GAConfig(random_state=random_state)
            shuffled = engine._get_seed_strategies_for_injection(config)

            expected = list(seeds)
            random.Random(42).shuffle(expected)

            assert [s.metadata["seed_strategy"] for s in shuffled] == [
                s.metadata["seed_strategy"] for s in expected
            ]
            assert [s.metadata["seed_strategy"] for s in seeds] == [
                "a",
                "b",
                "c",
                "d",
            ]

    @patch("app.services.auto_strategy.core.engine.ga_engine.FitnessSharing")
    def test_setup_deap_uses_default_sampling_ratio_when_missing(
        self,
        mock_fitness_sharing_cls,
        mock_backtest_service,
        mock_gene_generator,
    ):
        """fitness_sharing に sampling_ratio がない場合は既定値を使う"""
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            gene_generator=mock_gene_generator,
        )
        engine.deap_setup.setup_deap = Mock()
        engine.deap_setup.get_individual_class = Mock(return_value=None)

        config = Mock()
        config.fitness_sharing = {
            "enable_fitness_sharing": True,
            "sharing_radius": 0.2,
            "sharing_alpha": 1.5,
            "sampling_threshold": 50,
        }
        mock_fitness_sharing_cls.SAMPLING_RATIO = 0.3

        engine.setup_deap(config)

        mock_fitness_sharing_cls.assert_called_once_with(
            sharing_radius=0.2,
            alpha=1.5,
            sampling_threshold=50,
            sampling_ratio=0.3,
        )

    def test_extract_best_individuals_prefers_two_stage_leader(
        self,
        mock_backtest_service,
        mock_gene_generator,
    ):
        """二段階選抜の先頭個体を最終bestとして採用する"""
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            gene_generator=mock_gene_generator,
        )

        from app.services.auto_strategy.genes import StrategyGene

        robust_leader = Mock()
        robust_leader.__class__ = StrategyGene
        robust_leader.fitness.valid = True
        robust_leader.fitness.values = (0.8,)
        set_two_stage_metadata(robust_leader, 0, (1.0, 1.0, 0.8, 0.8))

        raw_best = Mock()
        raw_best.__class__ = StrategyGene
        raw_best.fitness.valid = True
        raw_best.fitness.values = (1.2,)

        config = Mock()

        best_individual, best_gene, best_strategies = (
            engine.result_processor.extract_best_individuals(
                [raw_best, robust_leader],
                config,
                halloffame=[raw_best],
            )
        )

        assert best_individual is robust_leader
        assert best_gene is robust_leader
        assert best_strategies is not None
        assert len(best_strategies) == 1
        assert best_strategies[0]["fitness_values"] == [1.2]

    @patch(
        "app.services.auto_strategy.core.engine.ga_engine_factory.RandomGeneGenerator"
    )
    @patch(
        "app.services.auto_strategy.core.hybrid.hybrid_feature_adapter.HybridFeatureAdapter"
    )
    @patch("app.services.auto_strategy.core.hybrid.hybrid_predictor.HybridPredictor")
    def test_factory_loads_latest_hybrid_model_when_available(
        self,
        mock_predictor_cls,
        mock_adapter_cls,
        mock_gene_generator_cls,
        mock_backtest_service,
    ):
        """hybrid_mode では起動時に最新モデルのロードを試みる"""
        predictor = Mock()
        predictor.load_latest_models.return_value = True
        mock_predictor_cls.return_value = predictor
        mock_adapter_cls.return_value = Mock()
        mock_gene_generator_cls.return_value = Mock()

        config = Mock()
        config.log_level = "INFO"
        config.hybrid_config = HybridConfig(
            mode=True,
            model_types=None,
            model_type="lightgbm",
        )

        engine = GeneticAlgorithmEngineFactory.create_engine(
            mock_backtest_service,
            config,
        )

        predictor.load_latest_models.assert_called_once_with()
        assert engine.hybrid_mode is True

    @patch(
        "app.services.auto_strategy.core.engine.ga_engine_factory.RandomGeneGenerator"
    )
    @patch(
        "app.services.auto_strategy.core.hybrid.hybrid_feature_adapter.HybridFeatureAdapter"
    )
    @patch("app.services.auto_strategy.core.hybrid.hybrid_predictor.HybridPredictor")
    def test_factory_keeps_hybrid_engine_when_no_latest_model_exists(
        self,
        mock_predictor_cls,
        mock_adapter_cls,
        mock_gene_generator_cls,
        mock_backtest_service,
    ):
        """最新モデルがなくても hybrid エンジン初期化は継続する"""
        predictor = Mock()
        predictor.load_latest_models.return_value = False
        mock_predictor_cls.return_value = predictor
        mock_adapter_cls.return_value = Mock()
        mock_gene_generator_cls.return_value = Mock()

        config = Mock()
        config.log_level = "INFO"
        config.hybrid_config = HybridConfig(
            mode=True,
            model_types=None,
            model_type="lightgbm",
        )

        engine = GeneticAlgorithmEngineFactory.create_engine(
            mock_backtest_service,
            config,
        )

        predictor.load_latest_models.assert_called_once_with()
        assert engine.hybrid_mode is True

    def test_tuning_reselection_respects_disabled_two_stage_selection(
        self,
        mock_backtest_service,
        mock_gene_generator,
    ):
        """二段階選抜が無効なら tuning 後の再選抜も raw fitness を使う"""
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            gene_generator=mock_gene_generator,
        )

        current_best = object()
        config = Mock()
        config.two_stage_selection_config = Mock()
        config.two_stage_selection_config.enabled = False

        engine.parameter_tuning_manager.select_tuning_candidates = Mock(
            return_value=["candidate"]
        )
        engine.parameter_tuning_manager.tune_candidate_genes = Mock(
            return_value=["tuned"]
        )
        engine.parameter_tuning_manager.select_best_tuned_candidate = Mock(
            return_value=("wrong", -1.0, None)
        )
        engine.parameter_tuning_manager.select_best_tuned_candidate_by_fitness = Mock(
            return_value=("tuned", 1.23, {"mode": "single"})
        )

        result = engine.parameter_tuning_manager.tune_and_select_best_gene(
            population=[],
            current_best_gene=current_best,
            config=config,
            fallback_fitness=0.5,
            fallback_summary={"mode": "single"},
        )

        assert result == ("tuned", 1.23, {"mode": "single"})
        engine.parameter_tuning_manager.select_best_tuned_candidate.assert_not_called()
        engine.parameter_tuning_manager.select_best_tuned_candidate_by_fitness.assert_called_once_with(
            ["tuned"],
            config,
        )

    def test_tuning_skips_single_objective_reselection_for_multi_objective(
        self,
        mock_backtest_service,
        mock_gene_generator,
    ):
        """多目的では tuning 後に単一スコアで候補比較しない"""
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            gene_generator=mock_gene_generator,
        )

        current_best = object()
        config = Mock()
        config.objectives = ["weighted_score", "max_drawdown"]
        config.two_stage_selection_config = Mock()
        config.two_stage_selection_config.enabled = True

        engine.parameter_tuning_manager.tune_elite_parameters = Mock(
            return_value="tuned-multi"
        )
        engine.parameter_tuning_manager.refresh_best_gene_reporting = Mock(
            return_value=((1.0, -0.2), {"mode": "multi"})
        )
        engine.parameter_tuning_manager.select_tuning_candidates = Mock()
        engine.parameter_tuning_manager.tune_candidate_genes = Mock()
        engine.parameter_tuning_manager.select_best_tuned_candidate = Mock()
        engine.parameter_tuning_manager.select_best_tuned_candidate_by_fitness = Mock()

        result = engine.parameter_tuning_manager.tune_and_select_best_gene(
            population=[],
            current_best_gene=current_best,
            config=config,
            fallback_fitness=(0.5, -0.1),
            fallback_summary={"mode": "multi"},
        )

        assert result == ("tuned-multi", (1.0, -0.2), {"mode": "multi"})
        engine.parameter_tuning_manager.tune_elite_parameters.assert_called_once_with(
            current_best,
            config,
        )
        engine.parameter_tuning_manager.refresh_best_gene_reporting.assert_called_once_with(
            best_gene="tuned-multi",
            config=config,
            fallback_fitness=(0.5, -0.1),
            fallback_summary={"mode": "multi"},
        )
        engine.parameter_tuning_manager.select_tuning_candidates.assert_not_called()
        engine.parameter_tuning_manager.tune_candidate_genes.assert_not_called()
        engine.parameter_tuning_manager.select_best_tuned_candidate.assert_not_called()
        engine.parameter_tuning_manager.select_best_tuned_candidate_by_fitness.assert_not_called()

    @patch("app.services.auto_strategy.core.engine.ga_engine.EvolutionRunner")
    def test_run_evolution_flow(
        self,
        mock_runner_cls,
        mock_backtest_service,
        mock_gene_generator,
    ):
        """GAエンジンの実行フローを確認"""
        # セットアップ
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            gene_generator=mock_gene_generator,
        )

        # Mock DEAP components
        engine.deap_setup = Mock()
        mock_toolbox = Mock()
        # フィットネス属性を持つモック個体（StrategyGeneを模倣）
        mock_ind = Mock()
        # StrategyGeneであることを判定させるため
        from app.services.auto_strategy.genes import StrategyGene

        mock_ind.__class__ = StrategyGene
        mock_ind.fitness.valid = True
        mock_ind.fitness.values = (1.0,)
        mock_toolbox.population.return_value = [mock_ind]
        engine.deap_setup.get_toolbox.return_value = mock_toolbox
        engine.deap_setup.get_individual_class.return_value = Mock()

        # Mock EvolutionRunner instance
        mock_runner_instance = mock_runner_cls.return_value
        # 戻り値は (population, logbook) の 2 要素
        mock_runner_instance.run_evolution.return_value = ([mock_ind], Mock())

        # Config mock - use real GAConfig for complete attribute coverage
        from app.services.auto_strategy.config.ga import GAConfig

        mock_config = GAConfig()
        mock_config.population_size = 10
        mock_config.generations = 5
        mock_config.evaluation_config.enable_parallel = False
        mock_config.fitness_sharing["enable_fitness_sharing"] = False
        mock_config.mutation_rate = 0.1
        mock_config.use_seed_strategies = False
        mock_config.seed_injection_rate = 0.1
        mock_config.fallback_start_date = "2024-01-01"
        mock_config.fallback_end_date = "2024-01-02"
        mock_config.objectives = ["weighted_score"]

        backtest_config = {"symbol": "BTCUSDT", "timeframe": "1h"}

        # 実行
        engine.run_evolution(mock_config, backtest_config)

        # 検証
        engine.deap_setup.setup_deap.assert_called_once()
        mock_toolbox.population.assert_called_with(n=10)
        mock_runner_cls.assert_called_once()
        run_kwargs = mock_runner_instance.run_evolution.call_args.kwargs
        assert "should_stop" in run_kwargs
        assert callable(run_kwargs["should_stop"])
        assert run_kwargs["should_stop"]() is False
        engine.stop_evolution()
        assert run_kwargs["should_stop"]() is True

    @patch("app.services.auto_strategy.core.engine.ga_engine.ParallelEvaluator")
    @patch("app.services.auto_strategy.core.engine.ga_engine.EvolutionRunner")
    def test_run_evolution_with_parallel_config(
        self,
        mock_runner_cls,
        mock_parallel_evaluator_cls,
        mock_backtest_service,
        mock_gene_generator,
    ):
        """並列評価設定時の挙動確認"""
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            gene_generator=mock_gene_generator,
        )

        # Mocks
        engine.deap_setup = Mock()
        mock_toolbox = Mock()
        from app.services.auto_strategy.genes import StrategyGene

        mock_ind = Mock()
        mock_ind.__class__ = StrategyGene
        mock_ind.fitness.valid = True
        mock_ind.fitness.values = (1.0,)
        mock_toolbox.population.return_value = [mock_ind]
        engine.deap_setup.get_toolbox.return_value = mock_toolbox
        engine.deap_setup.get_individual_class.return_value = Mock()

        # Mock EvolutionRunner instance behavior
        mock_runner_instance = mock_runner_cls.return_value
        mock_runner_instance.run_evolution.return_value = ([mock_ind], Mock())

        # Config - use real GAConfig for complete attribute coverage
        from app.services.auto_strategy.config.ga import GAConfig

        mock_config = GAConfig()
        mock_config.evaluation_config.enable_parallel = True
        mock_config.evaluation_config.max_workers = 4
        mock_config.evaluation_config.timeout = 60.0
        mock_config.population_size = 10
        mock_config.fitness_sharing["enable_fitness_sharing"] = False
        mock_config.mutation_rate = 0.1
        mock_config.use_seed_strategies = False
        mock_config.seed_injection_rate = 0.1
        mock_config.fallback_start_date = "2024-01-01"
        mock_config.fallback_end_date = "2024-01-02"
        mock_config.objectives = ["weighted_score"]

        # 並列ワーカー初期化引数の public API を使用
        shared_data = {"main_data": pd.DataFrame({"close": [1, 2]})}
        engine.individual_evaluator.build_parallel_worker_initargs = Mock(
            return_value=(
                {"symbol": "BTC/USDT", "timeframe": "1h"},
                mock_config,
                shared_data,
            )
        )

        # 実行
        engine.run_evolution(mock_config, {"symbol": "BTC/USDT", "timeframe": "1h"})

        # ParallelEvaluatorが初期化されたことを確認
        mock_parallel_evaluator_cls.assert_called_once()
        engine.individual_evaluator.build_parallel_worker_initargs.assert_called_once_with(
            mock_config
        )

        # start()とshutdown()が呼ばれたことを確認
        mock_instance = mock_parallel_evaluator_cls.return_value
        mock_instance.start.assert_called_once()
        mock_instance.shutdown.assert_called_once()

        # RunnerにParallelEvaluatorが渡されたか確認
        # EvolutionRunner(toolbox, stats, fitness_sharing, population, parallel_evaluator)
        runner_pos_args = mock_runner_cls.call_args[0]
        # 5番目の引数 (インデックス4) が parallel_evaluator
        assert runner_pos_args[4] == mock_instance
        run_kwargs = mock_runner_instance.run_evolution.call_args.kwargs
        assert "should_stop" in run_kwargs
        assert callable(run_kwargs["should_stop"])

    @patch("app.services.auto_strategy.core.engine.ga_engine.ParallelEvaluator")
    def test_create_parallel_evaluator_uses_coarse_config_for_multi_fidelity(
        self,
        mock_parallel_evaluator_cls,
        mock_backtest_service,
        mock_gene_generator,
    ):
        """multi-fidelity 有効時は並列ワーカーへ coarse 設定を渡す。"""
        from app.services.auto_strategy.config.ga import GAConfig

        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            gene_generator=mock_gene_generator,
        )
        config = GAConfig(
            evaluation_config=EvaluationConfig(
                enable_parallel=True,
                enable_multi_fidelity_evaluation=True,
                enable_walk_forward=True,
                multi_fidelity_oos_ratio=0.2,
            ),
            enable_purged_kfold=True,
        )
        shared_data = {"main_data": pd.DataFrame({"close": [1, 2]})}
        captured_configs = []

        def _build_initargs(init_config):
            captured_configs.append(init_config)
            return (
                {"symbol": "BTC/USDT", "timeframe": "1h"},
                init_config,
                shared_data,
            )

        engine.individual_evaluator.build_parallel_worker_initargs = Mock(
            side_effect=_build_initargs
        )

        engine._create_parallel_evaluator(config)

        assert len(captured_configs) == 1
        worker_config = captured_configs[0]
        assert worker_config is not config
        assert worker_config.evaluation_config.enable_walk_forward is False
        assert worker_config.enable_purged_kfold is False
        assert worker_config.evaluation_config.oos_split_ratio == 0.2
        assert getattr(worker_config, "_evaluation_fidelity", "full") == "coarse"
