"""
DEAP環境初期化のテストモジュール

DEAPSetupクラスの機能をテストする。
"""

from unittest.mock import Mock, patch, call
from types import SimpleNamespace

import pytest
from deap import base

from app.services.auto_strategy.config.ga_runtime import GAConfig
from app.services.auto_strategy.core.deap_setup import DEAPSetup
from app.services.auto_strategy.models import StrategyGene


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
        config.objectives = ["sharpe_ratio", "total_return"]
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
        "app.services.auto_strategy.core.deap_setup.creator", new_callable=MockCreator
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
        mock_creator.create.assert_any_call(
            "FitnessMulti", base.Fitness, weights=(1.0, -1.0)
        )

    @patch(
        "app.services.auto_strategy.core.deap_setup.creator", new_callable=MockCreator
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
        # 引数リストの中から第1引数が"Individual"であるものを探す
        individual_call = None
        for c in calls:
            if c[0][0] == "Individual":
                individual_call = c
                break

        assert individual_call is not None
        assert individual_call[0][1] == StrategyGene
        assert "fitness" in individual_call[1]

    @patch(
        "app.services.auto_strategy.core.deap_setup.creator", new_callable=MockCreator
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
        "app.services.auto_strategy.core.deap_setup.creator", new_callable=MockCreator
    )
    def test_setup_deap_registers_nsga2_for_multi_objective(
        self, mock_creator, deap_setup, mock_config, mock_functions
    ):
        """多目的最適化時の選択関数(NSGA-II)登録テスト"""
        mock_config.enable_multi_objective = True

        # base.Toolboxをモック化してregister呼び出しを検証可能にする
        with patch(
            "app.services.auto_strategy.core.deap_setup.base.Toolbox"
        ) as MockToolbox:
            mock_toolbox_instance = MockToolbox.return_value

            with patch(
                "app.services.auto_strategy.core.deap_setup.tools"
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
        "app.services.auto_strategy.core.deap_setup.creator", new_callable=MockCreator
    )
    def test_setup_deap_registers_tournament_for_single_objective(
        self, mock_creator, deap_setup, mock_config, mock_functions
    ):
        """単一目的最適化時の選択関数(Tournament)登録テスト"""
        mock_config.enable_multi_objective = False
        mock_config.tournament_size = 5

        with patch(
            "app.services.auto_strategy.core.deap_setup.base.Toolbox"
        ) as MockToolbox:
            mock_toolbox_instance = MockToolbox.return_value

            with patch(
                "app.services.auto_strategy.core.deap_setup.tools"
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
                assert select_call[0][1] == mock_tools.selTournament
                assert select_call[1]["tournsize"] == 5

    @patch(
        "app.services.auto_strategy.core.deap_setup.creator", new_callable=MockCreator
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
        "app.services.auto_strategy.core.deap_setup.creator", new_callable=MockCreator
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
        "app.services.auto_strategy.core.deap_setup.creator", new_callable=MockCreator
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
        assert individual_class == mock_creator.Individual

    @patch(
        "app.services.auto_strategy.core.deap_setup.creator", new_callable=MockCreator
    )
    def test_setup_deap_deletes_existing_classes(
        self, mock_creator, deap_setup, mock_config, mock_functions
    ):
        """既存クラスの削除テスト"""
        # 既存のクラスをセット
        setattr(mock_creator, "FitnessMulti", Mock())
        setattr(mock_creator, "Individual", Mock())

        # 削除されたことを確認するためのside_effect再定義
        original_side_effect = mock_creator.create.side_effect

        def verifying_side_effect(name, *args, **kwargs):
            if name in ["FitnessMulti", "Individual"] and hasattr(mock_creator, name):
                pytest.fail(
                    f"Attribute '{name}' should have been deleted before create() was called."
                )
            original_side_effect(name, *args, **kwargs)

        mock_creator.create.side_effect = verifying_side_effect

        deap_setup.setup_deap(
            mock_config,
            mock_functions["create_individual"],
            mock_functions["evaluate"],
            mock_functions["crossover"],
            mock_functions["mutate"],
        )

        # 最終的に存在することを確認
        assert hasattr(mock_creator, "FitnessMulti")
        assert hasattr(mock_creator, "Individual")

    @patch(
        "app.services.auto_strategy.core.deap_setup.creator", new_callable=MockCreator
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

    def test_get_toolbox_before_setup(self, deap_setup):
        """セットアップ前のツールボックス取得テスト"""
        toolbox = deap_setup.get_toolbox()
        assert toolbox is None

    def test_get_individual_class_before_setup(self, deap_setup):
        """セットアップ前の個体クラス取得テスト"""
        individual_class = deap_setup.get_individual_class()
        assert individual_class is None


