"""
DEAP環境初期化のテストモジュール

DEAPSetupクラスの機能をテストする。
"""

from unittest.mock import Mock, patch, call
from types import SimpleNamespace

import pytest
from deap import base

from app.services.auto_strategy.config.ga import GAConfig
from app.services.auto_strategy.core.engine.deap_setup import DEAPSetup
from app.services.auto_strategy.genes import StrategyGene


class MockCreator(SimpleNamespace):
    """delattrをサポートするCreatorモック"""

    def __init__(self):
        super().__init__()
        # createメソッド自体はMock
        self.create = Mock(side_effect=self._create_side_effect)

    def _create_side_effect(self, name, *args, **kwargs):
        # createが呼ばれたら属性を追加
        setattr(self, name, Mock())


class TestDEAPSetup:
    """DEAPSetupクラスのテスト"""

    @pytest.fixture
    def deap_setup(self):
        """DEAPSetupインスタンス"""
        return DEAPSetup()

    @pytest.fixture
    def mock_config(self):
        """Mock GAConfig"""
        config = Mock(spec=GAConfig)
        config.objectives = ["sharpe_ratio", "max_drawdown"]
        config.objective_weights = [1.0, -1.0]
        config.mutation_rate = 0.2
        return config

    @pytest.fixture
    def mock_functions(self):
        """Mock関数群"""
        return {
            "create_individual": Mock(return_value=[0.1, 0.2, 0.3]),
            "evaluate": Mock(return_value=(1.5, 0.25)),
            "crossover": Mock(return_value=([0.1, 0.2], [0.3, 0.4])),
            "mutate": Mock(return_value=[0.1, 0.2, 0.3]),
        }

    def test_initialization(self, deap_setup):
        """初期化テスト"""
        assert deap_setup.toolbox is None
        assert deap_setup.Individual is None

    @patch(
        "app.services.auto_strategy.core.engine.deap_setup.creator", new_callable=MockCreator
    )
    def test_setup_deap_creates_fitness_class(
        self, mock_creator, deap_setup, mock_config, mock_functions
    ):
        """フィットネスクラスの作成テスト"""
        deap_setup.setup_deap(
            mock_config,
            mock_functions["create_individual"],
            mock_functions["evaluate"],
            mock_functions["crossover"],
            mock_functions["mutate"],
        )

        # FitnessMultiが作成されたことを確認
        assert deap_setup.fitness_class_name is not None
        assert deap_setup.fitness_class_name.startswith("FitnessMulti_")
        mock_creator.create.assert_any_call(
            deap_setup.fitness_class_name, base.Fitness, weights=(1.0, -1.0)
        )

    @patch(
        "app.services.auto_strategy.core.engine.deap_setup.creator", new_callable=MockCreator
    )
    def test_setup_deap_creates_individual_class(
        self, mock_creator, deap_setup, mock_config, mock_functions
    ):
        """個体クラスの作成テスト"""
        deap_setup.setup_deap(
            mock_config,
            mock_functions["create_individual"],
            mock_functions["evaluate"],
            mock_functions["crossover"],
            mock_functions["mutate"],
        )

        # create呼び出しの検証（Individual）
        calls = mock_creator.create.call_args_list
        # 引数リストの中から第1引数が動的なIndividual名であるものを探す
        individual_call = None
        for c in calls:
            if c[0][0] == deap_setup.individual_class_name:
                individual_call = c
                break

        assert individual_call is not None
        assert deap_setup.individual_class_name is not None
        assert deap_setup.individual_class_name.startswith("Individual_")
        assert individual_call[0][1] == StrategyGene
        assert "fitness" in individual_call[1]
        assert individual_call[1]["fitness"] == getattr(
            mock_creator, deap_setup.fitness_class_name
        )

    @patch(
        "app.services.auto_strategy.core.engine.deap_setup.creator", new_callable=MockCreator
    )
    def test_setup_deap_registers_toolbox_functions(
        self, mock_creator, deap_setup, mock_config, mock_functions
    ):
        """ツールボックス関数の登録テスト"""
        # Toolbox自体は本物のdeap.base.Toolboxを使用し、登録結果を検証する
        deap_setup.setup_deap(
            mock_config,
            mock_functions["create_individual"],
            mock_functions["evaluate"],
            mock_functions["crossover"],
            mock_functions["mutate"],
        )

        toolbox = deap_setup.get_toolbox()
        assert toolbox is not None
        assert hasattr(toolbox, "individual")
        assert hasattr(toolbox, "population")
        assert hasattr(toolbox, "evaluate")
        assert hasattr(toolbox, "mate")
        assert hasattr(toolbox, "mutate")
        assert hasattr(toolbox, "select")

    @patch(
        "app.services.auto_strategy.core.engine.deap_setup.creator", new_callable=MockCreator
    )
    def test_setup_deap_registers_nsga2_for_any_objective_count(
        self, mock_creator, deap_setup, mock_config, mock_functions
    ):
        """選択関数として常に NSGA-II を登録すること"""

        # base.Toolboxをモック化してregister呼び出しを検証可能にする
        with patch(
            "app.services.auto_strategy.core.engine.deap_setup.base.Toolbox"
        ) as MockToolbox:
            mock_toolbox_instance = MockToolbox.return_value

            with patch(
                "app.services.auto_strategy.core.engine.deap_setup.tools"
            ) as mock_tools:
                deap_setup.setup_deap(
                    mock_config,
                    mock_functions["create_individual"],
                    mock_functions["evaluate"],
                    mock_functions["crossover"],
                    mock_functions["mutate"],
                )

                # register呼び出しを検証
                args = mock_toolbox_instance.register.call_args_list
                select_call = [call for call in args if call[0][0] == "select"][0]
                assert select_call[0][1] == mock_tools.selNSGA2

    @patch(
        "app.services.auto_strategy.core.engine.deap_setup.creator", new_callable=MockCreator
    )
    def test_setup_deap_mutate_wrapper(
        self, mock_creator, deap_setup, mock_config, mock_functions
    ):
        """突然変異ラッパーのテスト"""
        deap_setup.setup_deap(
            mock_config,
            mock_functions["create_individual"],
            mock_functions["evaluate"],
            mock_functions["crossover"],
            mock_functions["mutate"],
        )

        toolbox = deap_setup.get_toolbox()
        individual = [0.1, 0.2, 0.3]
        result = toolbox.mutate(individual)

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0] == individual
        mock_functions["mutate"].assert_called_once_with(individual, mutation_rate=0.2)

    @patch(
        "app.services.auto_strategy.core.engine.deap_setup.creator", new_callable=MockCreator
    )
    def test_get_toolbox(self, mock_creator, deap_setup, mock_config, mock_functions):
        """ツールボックスの取得テスト"""
        deap_setup.setup_deap(
            mock_config,
            mock_functions["create_individual"],
            mock_functions["evaluate"],
            mock_functions["crossover"],
            mock_functions["mutate"],
        )
        toolbox = deap_setup.get_toolbox()
        # Mock化していない場合はbase.Toolboxインスタンス
        assert toolbox is not None

    @patch(
        "app.services.auto_strategy.core.engine.deap_setup.creator", new_callable=MockCreator
    )
    def test_get_individual_class(
        self, mock_creator, deap_setup, mock_config, mock_functions
    ):
        """個体クラスの取得テスト"""
        deap_setup.setup_deap(
            mock_config,
            mock_functions["create_individual"],
            mock_functions["evaluate"],
            mock_functions["crossover"],
            mock_functions["mutate"],
        )

        individual_class = deap_setup.get_individual_class()
        assert individual_class is not None
        assert individual_class == getattr(mock_creator, deap_setup.individual_class_name)

    @patch(
        "app.services.auto_strategy.core.engine.deap_setup.creator", new_callable=MockCreator
    )
    def test_setup_deap_generates_unique_class_names(
        self, mock_creator, deap_setup, mock_config, mock_functions
    ):
        """実行ごとに一意なクラス名が採番されることを確認する"""

        deap_setup.setup_deap(
            mock_config,
            mock_functions["create_individual"],
            mock_functions["evaluate"],
            mock_functions["crossover"],
            mock_functions["mutate"],
        )
        first_fitness_name = deap_setup.fitness_class_name
        first_individual_name = deap_setup.individual_class_name
        first_fitness_class = getattr(mock_creator, first_fitness_name)
        first_individual_class = getattr(mock_creator, first_individual_name)

        deap_setup.setup_deap(
            mock_config,
            mock_functions["create_individual"],
            mock_functions["evaluate"],
            mock_functions["crossover"],
            mock_functions["mutate"],
        )

        assert first_fitness_name != deap_setup.fitness_class_name
        assert first_individual_name != deap_setup.individual_class_name
        assert first_fitness_name.startswith("FitnessMulti_")
        assert first_individual_name.startswith("Individual_")
        assert getattr(mock_creator, first_fitness_name) is first_fitness_class
        assert getattr(mock_creator, first_individual_name) is first_individual_class

    @patch(
        "app.services.auto_strategy.core.engine.deap_setup.creator", new_callable=MockCreator
    )
    def test_setup_deap_error_handling(
        self, mock_creator, deap_setup, mock_config, mock_functions
    ):
        """エラー処理テスト"""
        # createが失敗する場合
        mock_creator.create.side_effect = Exception("DEAP error")

        with pytest.raises(Exception):
            deap_setup.setup_deap(
                mock_config,
                mock_functions["create_individual"],
                mock_functions["evaluate"],
                mock_functions["crossover"],
                mock_functions["mutate"],
            )

    @patch(
        "app.services.auto_strategy.core.engine.deap_setup.creator", new_callable=MockCreator
    )
    def test_setup_deap_rejects_objective_weight_length_mismatch(
        self, mock_creator, deap_setup, mock_config, mock_functions
    ):
        """objective_weights の数が一致しない設定は早期に拒否する"""
        mock_config.objectives = ["total_return", "max_drawdown"]
        mock_config.objective_weights = [1.0]

        with pytest.raises(ValueError, match="objective_weights"):
            deap_setup.setup_deap(
                mock_config,
                mock_functions["create_individual"],
                mock_functions["evaluate"],
                mock_functions["crossover"],
                mock_functions["mutate"],
            )

    @patch(
        "app.services.auto_strategy.core.engine.deap_setup.creator", new_callable=MockCreator
    )
    def test_setup_deap_normalizes_minimize_objective_weights(
        self, mock_creator, deap_setup, mock_config, mock_functions
    ):
        """最小化目的の重みは DEAP 用に負符号へ正規化される"""
        mock_config.objectives = ["total_return", "max_drawdown"]
        mock_config.objective_weights = [1.0, 1.0]

        deap_setup.setup_deap(
            mock_config,
            mock_functions["create_individual"],
            mock_functions["evaluate"],
            mock_functions["crossover"],
            mock_functions["mutate"],
        )

        mock_creator.create.assert_any_call(
            deap_setup.fitness_class_name, base.Fitness, weights=(1.0, -1.0)
        )

    def test_get_toolbox_before_setup(self, deap_setup):
        """セットアップ前のツールボックス取得テスト"""
        toolbox = deap_setup.get_toolbox()
        assert toolbox is None

    def test_get_individual_class_before_setup(self, deap_setup):
        """セットアップ前の個体クラス取得テスト"""
        individual_class = deap_setup.get_individual_class()
        assert individual_class is None



