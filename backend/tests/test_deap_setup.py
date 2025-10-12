"""
DEAP環境初期化のテストモジュール

DEAPSetupクラスの機能をテストする。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from deap import base, creator, tools

from backend.app.services.auto_strategy.core.deap_setup import DEAPSetup
from backend.app.services.auto_strategy.config.ga_runtime import GAConfig


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

    @patch('backend.app.services.auto_strategy.core.deap_setup.creator')
    def test_setup_deap_creates_fitness_class(self, mock_creator, deap_setup, mock_config, mock_functions):
        """フィットネスクラスの作成テスト"""
        # creatorのモック
        mock_fitness_class = Mock()
        mock_creator.FitnessMulti = mock_fitness_class

        deap_setup.setup_deap(
            mock_config,
            mock_functions["create_individual"],
            mock_functions["evaluate"],
            mock_functions["crossover"],
            mock_functions["mutate"],
        )

        # FitnessMultiが作成されたことを確認
        mock_creator.create.assert_called_with("FitnessMulti", base.Fitness, weights=(1.0, -1.0))

    @patch('backend.app.services.auto_strategy.core.deap_setup.creator')
    def test_setup_deap_creates_individual_class(self, mock_creator, deap_setup, mock_config, mock_functions):
        """個体クラスの作成テスト"""
        deap_setup.setup_deap(
            mock_config,
            mock_functions["create_individual"],
            mock_functions["evaluate"],
            mock_functions["crossover"],
            mock_functions["mutate"],
        )

        # Individualクラスが作成されたことを確認
        mock_creator.create.assert_any_call("Individual", list, fitness=mock_creator.FitnessMulti)

    def test_setup_deap_registers_toolbox_functions(self, deap_setup, mock_config, mock_functions):
        """ツールボックス関数の登録テスト"""
        with patch('backend.app.services.auto_strategy.core.deap_setup.creator'):
            deap_setup.setup_deap(
                mock_config,
                mock_functions["create_individual"],
                mock_functions["evaluate"],
                mock_functions["crossover"],
                mock_functions["mutate"],
            )

        toolbox = deap_setup.get_toolbox()
        assert toolbox is not None

        # 個体生成関数が登録されていることを確認
        assert hasattr(toolbox, 'individual')

        # 集団生成関数が登録されていることを確認
        assert hasattr(toolbox, 'population')

        # 評価関数が登録されていることを確認
        assert hasattr(toolbox, 'evaluate')

        # 交叉関数が登録されていることを確認
        assert hasattr(toolbox, 'mate')

        # 突然変異関数が登録されていることを確認
        assert hasattr(toolbox, 'mutate')

        # 選択関数がNSGA-IIであることを確認
        assert hasattr(toolbox, 'select')

    def test_setup_deap_mutate_wrapper(self, deap_setup, mock_config, mock_functions):
        """突然変異ラッパーのテスト"""
        with patch('backend.app.services.auto_strategy.core.deap_setup.creator'):
            deap_setup.setup_deap(
                mock_config,
                mock_functions["create_individual"],
                mock_functions["evaluate"],
                mock_functions["crossover"],
                mock_functions["mutate"],
            )

        toolbox = deap_setup.get_toolbox()
        individual = [0.1, 0.2, 0.3]

        # mutate関数を呼び出し
        result = toolbox.mutate(individual)

        # ラッパーがタプルを返すことを確認
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0] == individual

        # 元のmutate関数が呼ばれたことを確認
        mock_functions["mutate"].assert_called_once_with(individual, mutation_rate=0.2)

    def test_get_toolbox(self, deap_setup, mock_config, mock_functions):
        """ツールボックスの取得テスト"""
        with patch('backend.app.services.auto_strategy.core.deap_setup.creator'):
            deap_setup.setup_deap(
                mock_config,
                mock_functions["create_individual"],
                mock_functions["evaluate"],
                mock_functions["crossover"],
                mock_functions["mutate"],
            )

        toolbox = deap_setup.get_toolbox()
        assert isinstance(toolbox, base.Toolbox)

    def test_get_individual_class(self, deap_setup, mock_config, mock_functions):
        """個体クラスの取得テスト"""
        with patch('backend.app.services.auto_strategy.core.deap_setup.creator') as mock_creator:
            mock_individual_class = Mock()
            mock_creator.Individual = mock_individual_class

            deap_setup.setup_deap(
                mock_config,
                mock_functions["create_individual"],
                mock_functions["evaluate"],
                mock_functions["crossover"],
                mock_functions["mutate"],
            )

            individual_class = deap_setup.get_individual_class()
            assert individual_class == mock_individual_class

    @patch('backend.app.services.auto_strategy.core.deap_setup.creator')
    def test_setup_deap_deletes_existing_classes(self, mock_creator, deap_setup, mock_config, mock_functions):
        """既存クラスの削除テスト"""
        # 既存のクラスがある場合
        mock_creator.FitnessMulti = Mock()
        mock_creator.Individual = Mock()

        deap_setup.setup_deap(
            mock_config,
            mock_functions["create_individual"],
            mock_functions["evaluate"],
            mock_functions["crossover"],
            mock_functions["mutate"],
        )

        # delattrが呼ばれたことを確認
        assert mock_creator.__delattr__.called

    def test_setup_deap_error_handling(self, deap_setup, mock_config, mock_functions):
        """エラー処理テスト"""
        # creator.createが失敗する場合
        with patch('backend.app.services.auto_strategy.core.deap_setup.creator.create', side_effect=Exception("DEAP error")):
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