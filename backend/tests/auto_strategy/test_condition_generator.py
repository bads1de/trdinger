from unittest.mock import MagicMock, patch

import pytest

from app.services.auto_strategy.generators.condition_generator import (
    ConditionGenerator,
    GAConditionGenerator,
)
from app.services.auto_strategy.genes import IndicatorGene


class TestConditionGenerator:
    def setup_method(self):
        self.generator = ConditionGenerator()

    def test_ema_long_condition_right_operand_is_close(self):
        """EMAのロング条件生成でright_operandが"close"であることをテスト"""
        ema_indicator = IndicatorGene(
            type="EMA", parameters={"period": 20}, enabled=True
        )

        long_conditions = self.generator._create_trend_long_conditions(ema_indicator)

        assert len(long_conditions) == 1
        condition = long_conditions[0]
        assert condition.left_operand == "EMA"
        assert condition.operator == ">"
        assert condition.right_operand == "close"

    def test_ema_short_condition_right_operand_is_close(self):
        """EMAのショート条件生成でright_operandが"close"であることをテスト"""
        ema_indicator = IndicatorGene(
            type="EMA", parameters={"period": 20}, enabled=True
        )

        short_conditions = self.generator._create_trend_short_conditions(ema_indicator)

        assert len(short_conditions) == 1
        condition = short_conditions[0]
        assert condition.left_operand == "EMA"
        assert condition.operator == "<"
        assert condition.right_operand == "close"

    def test_sma_long_condition_uses_threshold_fallback(self):
        """SMAのロング条件生成でthresholdがない場合fallbackを使うことをテスト"""
        sma_indicator = IndicatorGene(
            type="SMA", parameters={"period": 20}, enabled=True
        )

        long_conditions = self.generator._create_trend_long_conditions(sma_indicator)

        assert len(long_conditions) == 1
        condition = long_conditions[0]
        assert condition.left_operand == "SMA"
        assert condition.operator == ">"
        assert condition.right_operand == 0

    def test_generate_balanced_conditions_success(self):
        """正常な指標リストで条件生成が成功することをテスト"""
        indicators = [
            IndicatorGene(type="EMA", parameters={"period": 20}, enabled=True)
        ]
        long_conditions, short_conditions, exit_conditions = (
            self.generator.generate_balanced_conditions(indicators)
        )

        assert isinstance(long_conditions, list)
        assert isinstance(short_conditions, list)
        assert isinstance(exit_conditions, list)

    def test_generate_balanced_conditions_raises_exception_on_error(self):
        """YAML設定読み込みでエラーが発生した場合に例外を投げることをテスト"""

        with patch(
            "app.services.auto_strategy.generators.condition_generator.YamlIndicatorUtils.load_yaml_config_for_indicators"
        ) as mock_load:
            mock_load.side_effect = Exception("YAML読み込みエラー")

            # コンストラクタで失敗するので新しいインスタンスを作成
            with pytest.raises(Exception):
                ConditionGenerator()


class TestGAConditionGenerator:
    """GAConditionGeneratorのテスト"""

    def setup_method(self):
        """各テストメソッドの前処理"""
        self.mock_backtest_service = MagicMock()
        self.generator = GAConditionGenerator(
            use_hierarchical_ga=True, backtest_service=self.mock_backtest_service
        )

    def test_initialization_with_hierarchical_ga_enabled(self):
        """階層的GA有効化フラグが正しく設定されることをテスト"""
        generator = GAConditionGenerator(use_hierarchical_ga=True)
        assert generator.use_hierarchical_ga is True
        assert generator.condition_evolver is None
        assert generator._ga_initialized is False

    def test_initialization_with_hierarchical_ga_disabled(self):
        """階層的GA無効化フラグが正しく設定されることをテスト"""
        generator = GAConditionGenerator(use_hierarchical_ga=False)
        assert generator.use_hierarchical_ga is False

    def test_initialization_with_backtest_service(self):
        """BacktestServiceが正しく設定されることをテスト"""
        mock_service = MagicMock()
        generator = GAConditionGenerator(backtest_service=mock_service)
        assert generator.backtest_service == mock_service

    def test_ga_components_initialization_success(self):
        """GAコンポーネント初期化が成功することをテスト"""
        with (
            patch(
                "app.services.auto_strategy.generators.condition_generator.CoreYamlIndicatorUtils"
            ) as mock_utils,
            patch(
                "app.services.auto_strategy.generators.condition_generator.ConditionEvolver"
            ) as mock_evolver,
        ):
            mock_yaml_utils = MagicMock()
            mock_utils.return_value = mock_yaml_utils

            mock_evolver_instance = MagicMock()
            mock_evolver.return_value = mock_evolver_instance

            result = self.generator.initialize_ga_components()

            assert result is True
            assert self.generator.condition_evolver == mock_evolver_instance
            assert self.generator._ga_initialized is True

    def test_ga_components_initialization_without_backtest_service(self):
        """BacktestServiceなしでのGAコンポーネント初期化をテスト"""
        generator = GAConditionGenerator(use_hierarchical_ga=True)
        result = generator.initialize_ga_components()
        assert result is False

    def test_ga_components_initialization_missing_yaml_file(self):
        """YAMLファイルが存在しない場合の初期化をテスト"""
        with patch(
            "app.services.auto_strategy.generators.condition_generator.CoreYamlIndicatorUtils"
        ) as mock_utils:
            mock_utils.side_effect = FileNotFoundError("ファイルが見つかりません")

            result = self.generator.initialize_ga_components()
            assert result is False

    def test_generate_hierarchical_ga_conditions_with_ga_disabled(self):
        """階層的GAが無効の場合の条件生成をテスト"""
        generator = GAConditionGenerator(use_hierarchical_ga=False)
        indicators = [
            IndicatorGene(type="EMA", parameters={"period": 20}, enabled=True)
        ]

        with patch.object(generator, "generate_balanced_conditions") as mock_fallback:
            mock_fallback.return_value = ([], [], [])

            result = generator.generate_hierarchical_ga_conditions(indicators)

            mock_fallback.assert_called_once_with(indicators)
            assert result == ([], [], [])

    def test_generate_hierarchical_ga_conditions_ga_initialization_failure(self):
        """GA初期化失敗時のフォールバックをテスト"""
        generator = GAConditionGenerator(use_hierarchical_ga=True)
        indicators = [
            IndicatorGene(type="EMA", parameters={"period": 20}, enabled=True)
        ]

        with (
            patch.object(generator, "initialize_ga_components", return_value=False),
            patch.object(generator, "generate_balanced_conditions") as mock_fallback,
        ):
            mock_fallback.return_value = ([], [], [])

            result = generator.generate_hierarchical_ga_conditions(indicators)

            mock_fallback.assert_called_once_with(indicators)
            assert result == ([], [], [])

    def test_generate_hierarchical_ga_conditions_success(self):
        """階層的GA条件生成が成功することをテスト"""
        indicators = [
            IndicatorGene(type="EMA", parameters={"period": 20}, enabled=True),
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
        ]

        with (
            patch.object(self.generator, "initialize_ga_components", return_value=True),
            patch.object(self.generator, "condition_evolver") as mock_evolver,
        ):
            # Mock ConditionEvolverの戻り値
            mock_condition = MagicMock()
            mock_condition.direction = "long"
            mock_evolver.run_evolution.return_value = {
                "best_condition": mock_condition,
                "best_fitness": 0.8,
                "generations_completed": 10,
            }

            result = self.generator.generate_hierarchical_ga_conditions(indicators)

            # 結果が返されることを確認
            assert (
                len(result) == 3
            )  # (long_conditions, short_conditions, exit_conditions)
            long_conditions, short_conditions, exit_conditions = result
            assert isinstance(long_conditions, list)
            assert isinstance(short_conditions, list)
            assert isinstance(exit_conditions, list)

    def test_generate_hierarchical_ga_conditions_evolution_failure(self):
        """GA進化失敗時のフォールバックをテスト"""
        indicators = [
            IndicatorGene(type="EMA", parameters={"period": 20}, enabled=True)
        ]

        with (
            patch.object(self.generator, "initialize_ga_components", return_value=True),
            patch.object(self.generator, "condition_evolver") as mock_evolver,
            patch.object(
                self.generator, "generate_balanced_conditions"
            ) as mock_fallback,
        ):
            mock_evolver.run_evolution.return_value = None  # 失敗をシミュレート
            mock_fallback.return_value = ([MagicMock()], [MagicMock()], [])

            self.generator.generate_hierarchical_ga_conditions(indicators)

            # フォールバックが呼ばれたことを確認
            mock_fallback.assert_called_once_with(indicators)

    def test_set_ga_config(self):
        """GA設定更新機能をテスト"""
        self.generator.set_ga_config(
            population_size=30, generations=15, crossover_rate=0.9, mutation_rate=0.1
        )

        assert self.generator.ga_config["population_size"] == 30
        assert self.generator.ga_config["generations"] == 15
        assert self.generator.ga_config["crossover_rate"] == 0.9
        assert self.generator.ga_config["mutation_rate"] == 0.1

    def test_set_ga_config_partial(self):
        """GA設定の部分更新をテスト"""
        self.generator.set_ga_config(population_size=25)

        assert self.generator.ga_config["population_size"] == 25
        # 他の設定は変更されないことを確認
        assert "generations" in self.generator.ga_config
        assert "crossover_rate" in self.generator.ga_config
        assert "mutation_rate" in self.generator.ga_config

    def test_optimize_single_condition_success(self):
        """単一条件最適化が成功することをテスト"""
        indicator = IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)
        backtest_config = {"symbol": "BTC/USDT:USDT"}

        with (
            patch.object(self.generator, "initialize_ga_components", return_value=True),
            patch.object(self.generator, "condition_evolver") as mock_evolver,
        ):
            mock_condition = MagicMock()
            mock_condition.direction = "long"
            mock_evolver.run_evolution.return_value = {
                "best_condition": mock_condition,
                "best_fitness": 0.7,
            }

            result = self.generator.optimize_single_condition(
                indicator, "long", backtest_config
            )

            assert result == mock_condition
            mock_evolver.run_evolution.assert_called_once()

    def test_optimize_single_condition_ga_disabled(self):
        """GAが無効の場合の単一条件最適化をテスト"""
        generator = GAConditionGenerator(use_hierarchical_ga=False)
        indicator = IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)
        backtest_config = {"symbol": "BTC/USDT:USDT"}

        result = generator.optimize_single_condition(indicator, "long", backtest_config)

        assert result is None

    def test_optimize_single_condition_wrong_direction(self):
        """方向が一致しない場合の最適化をテスト"""
        indicator = IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)
        backtest_config = {"symbol": "BTC/USDT:USDT"}

        with (
            patch.object(self.generator, "initialize_ga_components", return_value=True),
            patch.object(self.generator, "condition_evolver") as mock_evolver,
        ):
            mock_condition = MagicMock()
            mock_condition.direction = "short"  # 要求された方向と異なる
            mock_evolver.run_evolution.return_value = {
                "best_condition": mock_condition,
                "best_fitness": 0.7,
            }

            result = self.generator.optimize_single_condition(
                indicator, "long", backtest_config
            )

            assert result is None  # 方向が一致しないためNoneを返す

    def test_optimize_single_condition_evolution_error(self):
        """進化エラー時の処理をテスト"""
        indicator = IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)
        backtest_config = {"symbol": "BTC/USDT:USDT"}

        with (
            patch.object(self.generator, "initialize_ga_components", return_value=True),
            patch.object(self.generator, "condition_evolver") as mock_evolver,
        ):
            mock_evolver.run_evolution.side_effect = Exception("進化エラー")

            result = self.generator.optimize_single_condition(
                indicator, "long", backtest_config
            )

            assert result is None  # エラー時はNoneを返す




