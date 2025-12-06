"""
ConditionEvolver統合テスト
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from app.services.auto_strategy.core.condition_evolver import (
    Condition,
    ConditionEvolver,
    YamlIndicatorUtils,
)
from app.services.backtest.backtest_service import BacktestService


class TestYamlIndicatorUtils:
    """YAML指標ユーティリティテスト"""

    @pytest.fixture
    def mock_yaml_config(self):
        return {
            "indicators": {
                "RSI": {
                    "type": "momentum",
                    "scale_type": "oscillator_0_100",
                    "conditions": {"long": ">", "short": "<"},
                    "thresholds": {"normal": {"long_gt": 30, "short_lt": 70}},
                },
                "MACD": {
                    "type": "momentum",
                    "scale_type": "momentum_zero_centered",
                    "conditions": {"long": ">", "short": "<"},
                    "thresholds": {"normal": {"long_gt": 0, "short_lt": 0}},
                },
            },
            "scale_types": {
                "oscillator_0_100": {"range": [0, 100]},
                "momentum_zero_centered": {"range": [-10, 10]},
            },
            "default_thresholds": {
                "oscillator_0_100": {"min": 20, "max": 80},
            },
        }

    @patch("app.services.auto_strategy.core.condition_evolver.yaml.safe_load")
    @patch("builtins.open")
    @patch("pathlib.Path.exists")
    def test_initialization(self, mock_exists, mock_open, mock_yaml_load, mock_yaml_config):
        """初期化テスト"""
        mock_exists.return_value = True
        mock_yaml_load.return_value = mock_yaml_config
        
        utils = YamlIndicatorUtils("config.yaml")
        assert utils.config == mock_yaml_config

    @patch("app.services.auto_strategy.core.condition_evolver.manifest_to_yaml_dict")
    def test_initialization_with_default(self, mock_manifest, mock_yaml_config):
        """デフォルト設定での初期化テスト"""
        mock_manifest.return_value = mock_yaml_config
        
        utils = YamlIndicatorUtils()
        assert utils.config == mock_yaml_config

    @patch("app.services.auto_strategy.core.condition_evolver.manifest_to_yaml_dict")
    def test_get_available_indicators(self, mock_manifest, mock_yaml_config):
        """利用可能な指標取得テスト"""
        mock_manifest.return_value = mock_yaml_config
        utils = YamlIndicatorUtils()
        
        indicators = utils.get_available_indicators()
        assert "RSI" in indicators
        assert "MACD" in indicators
        assert len(indicators) == 2

    @patch("app.services.auto_strategy.core.condition_evolver.manifest_to_yaml_dict")
    def test_get_indicator_info(self, mock_manifest, mock_yaml_config):
        """指標情報取得テスト"""
        mock_manifest.return_value = mock_yaml_config
        utils = YamlIndicatorUtils()
        
        info = utils.get_indicator_info("RSI")
        assert info["type"] == "momentum"
        assert info["scale_type"] == "oscillator_0_100"

        with pytest.raises(ValueError):
            utils.get_indicator_info("UNKNOWN")


class TestConditionEvolver:
    """ConditionEvolverのテスト"""

    @pytest.fixture
    def mock_backtest_service(self):
        """バックテストサービスのモック"""
        mock_service = Mock(spec=BacktestService)
        mock_service.run_backtest.return_value = {
            "success": True,
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1,
                "total_trades": 25,
                "win_rate": 0.60,
            },
        }
        return mock_service

    @pytest.fixture
    def mock_yaml_utils(self):
        """YamlIndicatorUtilsのモック"""
        mock_utils = Mock(spec=YamlIndicatorUtils)
        mock_utils.get_available_indicators.return_value = ["RSI", "MACD"]
        mock_utils.get_indicator_info.side_effect = lambda name: {
            "scale_type": "oscillator_0_100" if name == "RSI" else "momentum_zero_centered",
            "thresholds": {"normal": {"long_gt": 30, "short_lt": 70}} if name == "RSI" else {}
        }
        return mock_utils

    @pytest.fixture
    def condition_evolver(self, mock_backtest_service, mock_yaml_utils):
        """ConditionEvolverインスタンス"""
        return ConditionEvolver(
            backtest_service=mock_backtest_service,
            yaml_indicator_utils=mock_yaml_utils,
        )

    def test_initialization(self, condition_evolver):
        """初期化テスト"""
        assert condition_evolver.backtest_service is not None
        assert condition_evolver.yaml_indicator_utils is not None
        assert hasattr(condition_evolver, "toolbox")

    def test_create_individual(self, condition_evolver):
        """個体生成テスト"""
        condition = condition_evolver._create_individual()
        
        assert isinstance(condition, Condition)
        assert condition.indicator_name in ["RSI", "MACD"]
        assert condition.operator in [">", "<", ">=", "<=", "==", "!="]
        assert isinstance(condition.threshold, float)
        assert condition.direction in ["long", "short"]

    def test_evaluate_fitness(self, condition_evolver):
        """適応度評価テスト"""
        condition = Condition("RSI", ">", 30.0, "long")
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "initial_capital": 10000
        }

        fitness = condition_evolver.evaluate_fitness(condition, backtest_config)
        
        assert isinstance(fitness, float)
        assert fitness > 0
        condition_evolver.backtest_service.run_backtest.assert_called_once()

    def test_crossover(self, condition_evolver):
        """交叉操作テスト"""
        parent1 = Condition("RSI", ">", 30.0, "long")
        parent2 = Condition("MACD", "<", 0.0, "short")

        # operator交叉のモック
        with patch("random.choice", return_value="operator"):
            child1, child2 = condition_evolver.crossover(parent1, parent2)
            
            # インジケータと方向は変わらないはず
            assert child1.indicator_name == parent1.indicator_name
            assert child2.indicator_name == parent2.indicator_name
            
            # オペレータが入れ替わっているか確認
            assert child1.operator == parent2.operator
            assert child2.operator == parent1.operator

        # threshold交叉のモック
        with patch("random.choice", return_value="threshold"):
            child1, child2 = condition_evolver.crossover(parent1, parent2)
            
            expected_threshold = (parent1.threshold + parent2.threshold) / 2
            assert child1.threshold == expected_threshold
            assert child2.threshold == expected_threshold

    def test_mutate(self, condition_evolver):
        """突然変異操作テスト"""
        condition = Condition("RSI", ">", 30.0, "long")

        # operator変異
        with patch("random.choice", side_effect=["operator", "<"]):
            mutated = condition_evolver.mutate(condition)
            assert mutated.operator == "<"
            assert mutated.indicator_name == condition.indicator_name

        # threshold変異
        with patch("random.choice", side_effect=["threshold"]):
            with patch("random.uniform", return_value=0.1): # 10% increase
                mutated = condition_evolver.mutate(condition)
                assert mutated.threshold == condition.threshold * 1.1

        # indicator変異
        with patch("random.choice", side_effect=["indicator", "MACD"]):
            mutated = condition_evolver.mutate(condition)
            assert mutated.indicator_name == "MACD"

    def test_run_evolution(self, condition_evolver):
        """進化プロセス実行テスト"""
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "initial_capital": 10000
        }

        # evaluate_fitnessをモック化して高速化
        condition_evolver.evaluate_fitness = Mock(return_value=1.0)

        result = condition_evolver.run_evolution(
            backtest_config,
            population_size=4,
            generations=2
        )

        assert "best_condition" in result
        assert isinstance(result["best_condition"], Condition)
        assert "best_fitness" in result
        assert "evolution_history" in result
        assert len(result["evolution_history"]) == 2
        assert result["generations_completed"] == 2

    def test_create_strategy_from_condition(self, condition_evolver):
        """条件から戦略作成テスト"""
        condition = Condition("RSI", ">", 30.0, "long")
        
        strategy = condition_evolver.create_strategy_from_condition(condition)
        
        assert "name" in strategy
        assert "conditions" in strategy
        assert strategy["conditions"]["entry"]["type"] == "single"
        
        cond_dict = strategy["conditions"]["entry"]["condition"]
        assert cond_dict["indicator"] == "RSI"
        assert cond_dict["operator"] == ">"
        assert cond_dict["threshold"] == 30.0