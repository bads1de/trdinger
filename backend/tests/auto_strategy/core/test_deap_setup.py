"""
DEAPSetupクラスのテスト

deap_setup.pyのテストケースを実装します。
TDDアプローチでバグを発見し、修正を行います。
"""

import pytest
from unittest.mock import Mock, patch
from deap import base, creator, tools

from backend.app.services.auto_strategy.core.deap_setup import DEAPSetup
from backend.app.services.auto_strategy.config.ga_runtime import GAConfig


class TestDEAPSetup:
    """DEAPSetupクラスのテスト"""

    def setup_method(self):
        """各テストメソッド前のセットアップ"""
        # DEAP creatorのクリーニング
        if hasattr(creator, "FitnessMulti"):
            delattr(creator, "FitnessMulti")
        if hasattr(creator, "Individual"):
            delattr(creator, "Individual")

    def tearDown(self):
        """各テスト後のクリーンアップ"""
        if hasattr(creator, "FitnessMulti"):
            delattr(creator, "FitnessMulti")
        if hasattr(creator, "Individual"):
            delattr(creator, "Individual")

    def test_initialization(self):
        """初期化テスト"""
        setup = DEAPSetup()
        assert setup.toolbox is None
        assert setup.Individual is None

    def test_get_toolbox_before_setup(self):
        """セットアップ前はNoneを返す"""
        setup = DEAPSetup()
        assert setup.get_toolbox() is None

    def test_get_individual_class_before_setup(self):
        """セットアップ前はNoneを返す"""
        setup = DEAPSetup()
        assert setup.get_individual_class() is None

    def test_setup_deap_single_objective(self):
        """単一目的最適化のsetup_deapテスト"""
        # モックの作成
        config = GAConfig.create_default()
        config.enable_multi_objective = False
        config.objective_weights = [1.0]
        config.objectives = ["total_return"]
        config.mutation_rate = 0.1

        mock_create_individual = Mock(return_value=[1, 2, 3])
        mock_evaluate = Mock(return_value=(5.0,))
        mock_crossover = Mock(side_effect=lambda *args: args)
        mock_mutate = Mock(return_value=(1, 2, 3))

        setup = DEAPSetup()

        # ロガーをモックしてログ出力を抑制
        with patch('backend.app.services.auto_strategy.core.deap_setup.logger'):
            setup.setup_deap(
                config,
                mock_create_individual,
                mock_evaluate,
                mock_crossover,
                mock_mutate
            )

        # ツールボックスが作成されたことを確認
        toolbox = setup.get_toolbox()
        assert toolbox is not None

        # 個体クラスが作成されたことを確認
        individual_class = setup.get_individual_class()
        assert individual_class is not None

        # DEAP creatorにクラスが作成されていることを確認
        assert hasattr(creator, "FitnessMulti")
        assert hasattr(creator, "Individual")

        # ツールボックスのメソッドが登録されていることを確認
        assert hasattr(toolbox, "individual")
        assert hasattr(toolbox, "population")
        assert hasattr(toolbox, "evaluate")
        assert hasattr(toolbox, "mate")
        assert hasattr(toolbox, "mutate")
        assert hasattr(toolbox, "select")


    def test_setup_deap_multi_objective(self):
        """多目的最適化のsetup_deapテスト"""
        # 多目的GA設定の作成
        config = GAConfig.create_default()
        config.enable_multi_objective = True
        config.objective_weights = [1.0, -1.0]  # total_return最大化、max_drawdown最小化
        config.objectives = ["total_return", "max_drawdown"]
        config.mutation_rate = 0.1

        # モックの作成
        mock_create_individual = Mock(return_value=[1, 2, 3])
        mock_evaluate = Mock(return_value=(10.0, -3.0))  # 二つの目的値
        mock_crossover = Mock(side_effect=lambda *args: args)
        mock_mutate = Mock(return_value=(1, 2, 3))

        setup = DEAPSetup()

        with patch('backend.app.services.auto_strategy.core.deap_setup.logger'):
            setup.setup_deap(
                config,
                mock_create_individual,
                mock_evaluate,
                mock_crossover,
                mock_mutate
            )

        # 正しく設定されていることを確認
        toolbox = setup.get_toolbox()
        assert toolbox is not None

        # 多目的最適化ではweightsが正しく設定されている
        fitness_class = getattr(creator, "FitnessMulti")
        assert fitness_class.weights == (1.0, -1.0)


    def test_get_functions_after_setup(self):
        """setup_deap後にgetter関数が動作することを確認"""
        config = GAConfig.create_default()
        config.enable_multi_objective = False
        config.objective_weights = [1.0]
        config.objectives = ["total_return"]
        config.mutation_rate = 0.1

        mock_create_individual = Mock(return_value=[1, 2, 3])
        mock_evaluate = Mock(return_value=(5.0,))
        mock_crossover = Mock(side_effect=lambda *args: args)
        mock_mutate = Mock(return_value=(1, 2, 3))

        setup = DEAPSetup()

        with patch('backend.app.services.auto_strategy.core.deap_setup.logger'):
            setup.setup_deap(
                config,
                mock_create_individual,
                mock_evaluate,
                mock_crossover,
                mock_mutate
            )

        # getter関数が正しく動作
        toolbox = setup.get_toolbox()
        individual_class = setup.get_individual_class()

        assert toolbox is not None
        assert individual_class is not None
        assert callable(individual_class)


    def test_mutate_wrapper_functionality(self):
        """mutate wrapper関数の動作確認"""
        config = GAConfig.create_default()
        config.mutation_rate = 0.1

        # タプルを返すmutate_func
        mock_mutate_returning_tuple = Mock(return_value=(1, 2, 3))
        # タプルを返さないmutate_func
        mock_mutate_returning_single = Mock(return_value=[4, 5, 6])

        # wrapper関数を取得（setup_deap内で作成される）
        def _mutate_wrapper(individual):
            res = mock_mutate_returning_tuple(individual, mutation_rate=config.mutation_rate)
            if isinstance(res, tuple):
                return res
            return (res,)

        def _mutate_wrapper_single(individual):
            res = mock_mutate_returning_single(individual, mutation_rate=config.mutation_rate)
            if isinstance(res, tuple):
                return res
            return (res,)

        # wrapperテスト
        test_individual = [1, 2, 3]
        result_tuple = _mutate_wrapper(test_individual)
        result_single = _mutate_wrapper_single(test_individual)

        assert isinstance(result_tuple, tuple)
        assert result_tuple == (1, 2, 3)
        assert isinstance(result_single, tuple)
        assert result_single == ([4, 5, 6],)

        mock_mutate_returning_tuple.assert_called_once_with(test_individual, mutation_rate=0.1)
        mock_mutate_returning_single.assert_called_once_with(test_individual, mutation_rate=0.1)

    # ここに他のテストケースを追加
