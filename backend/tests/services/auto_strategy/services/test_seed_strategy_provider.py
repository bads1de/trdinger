"""
反復改善ループ（PreviousStrategySeedProvider）のテスト

過去に自動検証へ合格した戦略を DB から取得し、
StrategyGene シードとして復元するロジックをテストします。
"""

from unittest.mock import MagicMock, patch

import pytest

from app.services.auto_strategy.config.ga_config import (
    IterativeImprovementConfig,
)
from app.services.auto_strategy.genes.strategy import StrategyGene
from app.services.auto_strategy.services.seed_strategy_provider import (
    PreviousStrategySeedProvider,
)


@pytest.fixture
def mock_db_session_factory():
    """DBセッションファクトリのモック"""
    factory = MagicMock()
    session = MagicMock()
    factory.return_value.__enter__.return_value = session
    return factory


@pytest.fixture
def iterative_config():
    """反復改善が有効な設定"""
    return IterativeImprovementConfig(
        enabled=True,
        max_seed_strategies=3,
        min_fitness=None,
        validation_passed_only=True,
    )


def _make_record(strategy_id: int, fitness: float, passed: bool = True):
    """テスト用の GeneratedStrategy レコードを構築する。"""
    record = MagicMock()
    record.id = strategy_id
    record.fitness_score = fitness
    gene_data = {
        "id": f"prev-{strategy_id}",
        "indicators": [
            {
                "type": "SMA",
                "parameters": {"period": 20},
                "enabled": True,
                "id": f"ind-{strategy_id}",
            }
        ],
        "long_entry_conditions": [],
        "short_entry_conditions": [],
        "long_exit_conditions": [],
        "short_exit_conditions": [],
        "risk_management": {},
        "metadata": {},
    }
    if passed:
        gene_data["metadata"]["validation"] = {"passed": True, "pass_rate": 0.8}
    record.gene_data = gene_data
    return record


class TestPreviousStrategySeedProvider:
    """PreviousStrategySeedProvider のテスト"""

    def test_disabled_returns_empty(self, mock_db_session_factory):
        """無効な場合は空リストを返す"""
        provider = PreviousStrategySeedProvider(mock_db_session_factory)
        config = MagicMock()
        config.iterative_improvement_config = IterativeImprovementConfig(enabled=False)

        seeds = provider.get_seed_strategies(config)

        assert seeds == []
        mock_db_session_factory.assert_not_called()

    def test_returns_restored_strategy_genes(
        self, mock_db_session_factory, iterative_config
    ):
        """合格戦略が StrategyGene として復元される"""
        records = [
            _make_record(1, 0.9),
            _make_record(2, 0.8),
        ]
        session = mock_db_session_factory.return_value.__enter__.return_value
        # min_fitness=None のため .filter() は1回のみ
        repo = session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all
        repo.return_value = records

        provider = PreviousStrategySeedProvider(mock_db_session_factory)
        config = MagicMock()
        config.iterative_improvement_config = iterative_config

        seeds = provider.get_seed_strategies(config)

        assert len(seeds) == 2
        assert all(isinstance(seed, StrategyGene) for seed in seeds)
        # IDは新しいものが付与される（重複注入回避）
        assert seeds[0].id != "prev-1"
        assert seeds[1].id != "prev-2"

    def test_filters_non_passing_when_validation_passed_only(
        self, mock_db_session_factory, iterative_config
    ):
        """validation_passed_only=True では合格していない戦略は除外される"""
        passing_records = [
            _make_record(1, 0.9, passed=True),
            _make_record(2, 0.8, passed=True),
        ]
        session = mock_db_session_factory.return_value.__enter__.return_value
        repo = session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all
        repo.return_value = passing_records

        provider = PreviousStrategySeedProvider(mock_db_session_factory)
        config = MagicMock()
        config.iterative_improvement_config = iterative_config

        seeds = provider.get_seed_strategies(config)

        assert len(seeds) == 2

    def test_validation_passed_only_false_includes_all(self, mock_db_session_factory):
        """validation_passed_only=False では全戦略が対象になる"""
        records = [
            _make_record(1, 0.9, passed=True),
            _make_record(2, 0.8, passed=False),
        ]
        session = mock_db_session_factory.return_value.__enter__.return_value
        repo = session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all
        repo.return_value = records

        provider = PreviousStrategySeedProvider(mock_db_session_factory)
        config = MagicMock()
        config.iterative_improvement_config = IterativeImprovementConfig(
            enabled=True,
            max_seed_strategies=5,
            validation_passed_only=False,
        )

        seeds = provider.get_seed_strategies(config)

        assert len(seeds) == 2

    def test_db_error_returns_empty(self, mock_db_session_factory):
        """DBエラー時は空リストを返し例外を伝播させない"""
        mock_db_session_factory.side_effect = RuntimeError("db down")

        provider = PreviousStrategySeedProvider(mock_db_session_factory)
        config = MagicMock()
        config.iterative_improvement_config = IterativeImprovementConfig(enabled=True)

        seeds = provider.get_seed_strategies(config)

        assert seeds == []

    def test_restore_error_skips_bad_record(self, mock_db_session_factory):
        """復元に失敗したレコードはスキップされる"""
        good_record = _make_record(1, 0.9, passed=True)
        session = mock_db_session_factory.return_value.__enter__.return_value
        repo = session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all
        repo.return_value = [good_record]

        provider = PreviousStrategySeedProvider(mock_db_session_factory)

        # シリアライザの復元を失敗させる
        serializer = MagicMock()
        serializer.dict_to_strategy_gene.side_effect = ValueError("bad gene")
        provider._serializer = serializer

        config = MagicMock()
        config.iterative_improvement_config = IterativeImprovementConfig(enabled=True)

        seeds = provider.get_seed_strategies(config)

        assert seeds == []

    def test_min_fitness_passed_to_repository(self, mock_db_session_factory):
        """min_fitness 指定時は .filter() が2回呼ばれる"""
        records = [_make_record(1, 0.9, passed=True)]
        session = mock_db_session_factory.return_value.__enter__.return_value
        repo = session.query.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all
        repo.return_value = records

        provider = PreviousStrategySeedProvider(mock_db_session_factory)
        config = MagicMock()
        config.iterative_improvement_config = IterativeImprovementConfig(
            enabled=True,
            min_fitness=0.5,
        )

        seeds = provider.get_seed_strategies(config)

        assert len(seeds) == 1


class TestIterativeImprovementConfig:
    """IterativeImprovementConfig の設定テスト"""

    def test_defaults(self):
        """デフォルト値のテスト"""
        from app.services.auto_strategy.config import GAConfig

        config = GAConfig()
        ic = config.iterative_improvement_config
        assert ic.enabled is False
        assert ic.max_seed_strategies == 5
        assert ic.min_fitness is None
        assert ic.validation_passed_only is True

    def test_from_dict(self):
        """辞書からの復元テスト"""
        from app.services.auto_strategy.config import GAConfig

        config = GAConfig.from_dict(
            {
                "iterative_improvement_config": {
                    "enabled": True,
                    "max_seed_strategies": 10,
                    "min_fitness": 0.5,
                    "validation_passed_only": False,
                }
            }
        )
        ic = config.iterative_improvement_config
        assert ic.enabled is True
        assert ic.max_seed_strategies == 10
        assert ic.min_fitness == 0.5
        assert ic.validation_passed_only is False

    def test_to_dict_round_trip(self):
        """to_dict → from_dict のラウンドトリップテスト"""
        from app.services.auto_strategy.config import GAConfig

        config = GAConfig(
            iterative_improvement_config=IterativeImprovementConfig(
                enabled=True,
                max_seed_strategies=7,
                min_fitness=0.3,
                validation_passed_only=False,
            )
        )
        restored = GAConfig.from_dict(config.to_dict())
        ic = restored.iterative_improvement_config
        assert ic.enabled is True
        assert ic.max_seed_strategies == 7
        assert ic.min_fitness == 0.3
        assert ic.validation_passed_only is False

    def test_exported_from_package(self):
        """パッケージから IterativeImprovementConfig がエクスポートされる"""
        from app.services.auto_strategy.config import IterativeImprovementConfig as I2

        assert I2 is IterativeImprovementConfig


class TestEngineSeedInjection:
    """GAエンジンの反復改善シード注入のテスト（実エンジンを使用）"""

    @pytest.fixture
    def mock_backtest_service(self):
        service = MagicMock()
        return service

    @pytest.fixture
    def mock_gene_generator(self):
        generator = MagicMock()
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

    def _make_engine(self, mock_backtest_service, mock_gene_generator, provider):
        from app.services.auto_strategy.core.engine.ga_engine import (
            GeneticAlgorithmEngine,
        )

        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            gene_generator=mock_gene_generator,
        )
        engine.seed_strategy_provider = provider
        return engine

    def test_previous_seeds_returned_when_enabled(
        self, mock_backtest_service, mock_gene_generator
    ):
        """反復改善が有効なら過去戦略が返される"""
        from app.services.auto_strategy.config import GAConfig

        prev_seed = StrategyGene(metadata={"seed_strategy": "previous"})
        provider = MagicMock()
        provider.get_seed_strategies.return_value = [prev_seed]

        engine = self._make_engine(mock_backtest_service, mock_gene_generator, provider)
        config = GAConfig(
            iterative_improvement_config=IterativeImprovementConfig(enabled=True),
            random_state=None,
        )

        seeds = engine._get_previous_seed_strategies(config)

        assert seeds == [prev_seed]
        provider.get_seed_strategies.assert_called_once_with(config)

    def test_previous_seeds_empty_when_disabled(
        self, mock_backtest_service, mock_gene_generator
    ):
        """反復改善が無効なら空リストを返しプロバイダは呼ばれない"""
        from app.services.auto_strategy.config import GAConfig

        provider = MagicMock()

        engine = self._make_engine(mock_backtest_service, mock_gene_generator, provider)
        config = GAConfig()  # デフォルトでは無効

        seeds = engine._get_previous_seed_strategies(config)

        assert seeds == []
        provider.get_seed_strategies.assert_not_called()

    def test_previous_seeds_empty_on_provider_error(
        self, mock_backtest_service, mock_gene_generator
    ):
        """プロバイダが例外を投げても空リストを返す"""
        from app.services.auto_strategy.config import GAConfig

        provider = MagicMock()
        provider.get_seed_strategies.side_effect = RuntimeError("boom")

        engine = self._make_engine(mock_backtest_service, mock_gene_generator, provider)
        config = GAConfig(
            iterative_improvement_config=IterativeImprovementConfig(enabled=True)
        )

        seeds = engine._get_previous_seed_strategies(config)

        assert seeds == []

    @patch(
        "app.services.auto_strategy.generators.seed_strategy_factory.SeedStrategyFactory.get_all_seeds"
    )
    def test_builtin_seeds_unaffected_by_provider(
        self,
        mock_get_all_seeds,
        mock_backtest_service,
        mock_gene_generator,
    ):
        """組み込みシードのシャッフル動作は従来どおり"""
        from app.services.auto_strategy.config import GAConfig

        seeds = [
            StrategyGene(metadata={"seed_strategy": "a"}),
            StrategyGene(metadata={"seed_strategy": "b"}),
        ]
        mock_get_all_seeds.return_value = seeds

        provider = MagicMock()
        engine = self._make_engine(mock_backtest_service, mock_gene_generator, provider)
        config = GAConfig(random_state=42)

        builtin = engine._get_builtin_seed_strategies(config)

        assert {s.metadata["seed_strategy"] for s in builtin} == {"a", "b"}
        provider.get_seed_strategies.assert_not_called()

    @patch(
        "app.services.auto_strategy.generators.seed_strategy_factory.SeedStrategyFactory.get_all_seeds"
    )
    def test_engine_injects_previous_seeds_first(
        self,
        mock_get_all_seeds,
        mock_backtest_service,
        mock_gene_generator,
    ):
        """反復改善有効時に過去シードが先頭へ注入される"""
        from app.services.auto_strategy.config import GAConfig
        from app.services.auto_strategy.core.engine.ga_engine import (
            GeneticAlgorithmEngine,
        )

        builtin = [
            StrategyGene(metadata={"seed_strategy": "b1"}),
            StrategyGene(metadata={"seed_strategy": "b2"}),
        ]
        mock_get_all_seeds.return_value = builtin

        prev_seed = StrategyGene(metadata={"seed_strategy": "previous"})
        provider = MagicMock()
        provider.get_seed_strategies.return_value = [prev_seed]

        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            gene_generator=mock_gene_generator,
        )
        engine.seed_strategy_provider = provider
        engine.deap_setup = MagicMock()
        mock_toolbox = MagicMock()
        from app.services.auto_strategy.genes import StrategyGene as SG

        def _make_individual(**kwargs):
            """注入されたシードの metadata を保持するモック個体を返す。"""
            ind = MagicMock()
            ind.__class__ = SG
            ind.metadata = kwargs.get("metadata", {})
            return ind

        population = [_make_individual() for _ in range(10)]
        mock_toolbox.population.return_value = population
        engine.deap_setup.get_toolbox.return_value = mock_toolbox
        engine.deap_setup.get_individual_class.return_value = _make_individual

        mock_runner_instance = MagicMock()
        mock_runner_instance.run_evolution.return_value = (population, MagicMock())

        # 結果処理は注入検証の対象外のため最小結果を返すモックに差し替える
        # （実パスの selBest はフィットネス比較に実値を要求する）
        engine._process_results = MagicMock(
            return_value={
                "best_strategy": population[0],
                "best_fitness": 1.0,
                "best_evaluation_summary": None,
                "all_strategies": population,
                "fitness_scores": [1.0] * len(population),
                "pareto_front": [],
                "evaluation_summaries": {},
                "objectives": ["weighted_score"],
                "execution_time": 1.0,
            }
        )

        config = GAConfig()
        config.population_size = 10
        config.generations = 5
        config.evaluation_config.enable_parallel = False
        config.fitness_sharing["enable_fitness_sharing"] = False
        config.mutation_rate = 0.1
        config.use_seed_strategies = True
        config.seed_injection_rate = 0.3
        config.fallback_start_date = "2024-01-01"
        config.fallback_end_date = "2024-01-02"
        config.objectives = ["weighted_score"]
        config.iterative_improvement_config = IterativeImprovementConfig(
            enabled=True, max_seed_strategies=1
        )
        config.random_state = None

        with patch(
            "app.services.auto_strategy.core.engine.ga_engine.EvolutionRunner",
            return_value=mock_runner_instance,
        ):
            engine.run_evolution(config, {"symbol": "BTCUSDT", "timeframe": "1h"})

        # 過去シード1件 + 組み込み（10*0.3=3件中）が注入される
        injected_previous = [
            ind
            for ind in population[:3]
            if ind.metadata.get("seed_strategy") == "previous"
        ]
        assert len(injected_previous) == 1
        # 注入された過去シードが先頭にある（優先注入）
        assert population[0].metadata.get("seed_strategy") == "previous"

    def test_factory_passes_provider_to_engine(self):
        """ファクトリがプロバイダをエンジンへ渡す"""
        from app.services.auto_strategy.config import GAConfig
        from app.services.auto_strategy.core.engine.ga_engine import (
            GeneticAlgorithmEngine,
        )
        from app.services.auto_strategy.core.engine.ga_engine_factory import (
            GeneticAlgorithmEngineFactory,
        )

        config = GAConfig()
        backtest_service = MagicMock()
        provider = MagicMock()

        with (
            patch(
                "app.services.auto_strategy.core.engine.ga_engine_factory.RandomGeneGenerator"
            ),
            patch.object(
                GeneticAlgorithmEngine, "__init__", return_value=None
            ) as mock_init,
        ):
            GeneticAlgorithmEngineFactory.create_engine(
                backtest_service,
                config,
                seed_strategy_provider=provider,
            )

        init_kwargs = mock_init.call_args.kwargs
        assert init_kwargs["seed_strategy_provider"] is provider
