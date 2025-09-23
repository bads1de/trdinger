"""
ConditionEvolver 関連クラスのテスト
"""

import pytest
import yaml
from typing import Dict, List, Any
from unittest.mock import Mock, patch
from pathlib import Path

# 直接インポートして循環インポートを避ける
import sys
sys.path.append('backend')

from app.services.auto_strategy.core.condition_evolver import (
    YamlIndicatorUtils,
    Condition,
    ConditionEvolver
)


class TestYamlIndicatorUtils:
    """YamlIndicatorUtilsクラスのテスト"""

    @pytest.fixture
    def sample_yaml_config(self):
        """サンプルYAML設定"""
        return {
            "indicators": {
                "RSI": {
                    "conditions": {
                        "long": "{left_operand} > {threshold}",
                        "short": "{left_operand} < {threshold}"
                    },
                    "scale_type": "oscillator_0_100",
                    "thresholds": {
                        "aggressive": {"long_gt": 70, "short_lt": 30},
                        "normal": {"long_gt": 75, "short_lt": 25}
                    },
                    "type": "momentum"
                },
                "SMA": {
                    "conditions": {
                        "long": "close > {left_operand}",
                        "short": "close < {left_operand}"
                    },
                    "scale_type": "price_absolute",
                    "thresholds": None,
                    "type": "trend"
                },
                "ATR": {
                    "conditions": {
                        "long": "close > {left_operand}_current + {multiplier}",
                        "short": "close < {left_operand}_current - {multiplier}"
                    },
                    "scale_type": "price_absolute",
                    "thresholds": {
                        "normal": {"multiplier": 1.0}
                    },
                    "type": "volatility"
                },
                "OBV": {
                    "conditions": {
                        "long": "{left_operand} > 0",
                        "short": "{left_operand} < 0"
                    },
                    "scale_type": "momentum_zero_centered",
                    "thresholds": {"zero_cross": True},
                    "type": "volume"
                }
            },
            "scale_types": {
                "oscillator_0_100": {"range": [0, 100]},
                "momentum_zero_centered": {"range": None}
            }
        }

    @pytest.fixture
    def yaml_utils(self, sample_yaml_config, tmp_path):
        """YamlIndicatorUtilsインスタンス"""
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_yaml_config, f)
        return YamlIndicatorUtils(str(config_path))

    def test_load_yaml_config(self, yaml_utils):
        """YAML設定の読み込みテスト"""
        assert yaml_utils.config is not None
        assert "indicators" in yaml_utils.config
        assert "RSI" in yaml_utils.config["indicators"]

    def test_get_available_indicators(self, yaml_utils):
        """利用可能な指標取得テスト"""
        indicators = yaml_utils.get_available_indicators()
        expected_indicators = ["RSI", "SMA", "ATR", "OBV"]
        assert set(indicators) == set(expected_indicators)

    def test_get_indicator_info(self, yaml_utils):
        """指標情報取得テスト"""
        rsi_info = yaml_utils.get_indicator_info("RSI")
        assert rsi_info["type"] == "momentum"
        assert rsi_info["scale_type"] == "oscillator_0_100"
        assert "conditions" in rsi_info

    def test_get_indicator_types(self, yaml_utils):
        """指標タイプ別の分類テスト"""
        type_mapping = yaml_utils.get_indicator_types()
        assert "momentum" in type_mapping
        assert "trend" in type_mapping
        assert "volatility" in type_mapping
        assert "volume" in type_mapping

        assert "RSI" in type_mapping["momentum"]
        assert "SMA" in type_mapping["trend"]
        assert "ATR" in type_mapping["volatility"]
        assert "OBV" in type_mapping["volume"]

    def test_get_threshold_ranges(self, yaml_utils):
        """閾値範囲取得テスト"""
        ranges = yaml_utils.get_threshold_ranges()
        assert "oscillator_0_100" in ranges
        assert "momentum_zero_centered" in ranges

    def test_invalid_indicator(self, yaml_utils):
        """無効な指標のテスト"""
        with pytest.raises(ValueError):
            yaml_utils.get_indicator_info("INVALID_INDICATOR")


class TestCondition:
    """Conditionクラスのテスト"""

    def test_condition_creation(self):
        """Conditionクラスの作成テスト"""
        condition = Condition(
            indicator_name="RSI",
            operator=">",
            threshold=70,
            direction="long"
        )
        assert condition.indicator_name == "RSI"
        assert condition.operator == ">"
        assert condition.threshold == 70
        assert condition.direction == "long"

    def test_condition_to_dict(self):
        """辞書形式への変換テスト"""
        condition = Condition(
            indicator_name="SMA",
            operator="<",
            threshold=100,
            direction="short"
        )
        condition_dict = condition.to_dict()
        expected = {
            "indicator_name": "SMA",
            "operator": "<",
            "threshold": 100,
            "direction": "short"
        }
        assert condition_dict == expected

    def test_condition_from_dict(self):
        """辞書形式からの作成テスト"""
        condition_dict = {
            "indicator_name": "MACD",
            "operator": ">",
            "threshold": 0,
            "direction": "long"
        }
        condition = Condition.from_dict(condition_dict)
        assert condition.indicator_name == "MACD"
        assert condition.operator == ">"
        assert condition.threshold == 0
        assert condition.direction == "long"

    def test_condition_equality(self):
        """Conditionの等価性テスト"""
        condition1 = Condition("RSI", ">", 70, "long")
        condition2 = Condition("RSI", ">", 70, "long")
        condition3 = Condition("RSI", "<", 70, "long")

        assert condition1 == condition2
        assert condition1 != condition3

    def test_condition_str(self):
        """文字列表現テスト"""
        condition = Condition("SMA", "<", 100, "short")
        assert str(condition) == "SMA < 100 (short)"


class TestConditionEvolver:
    """ConditionEvolverクラスのテスト"""

    @pytest.fixture
    def mock_backtest_service(self):
        """モックバックテストサービス"""
        mock_service = Mock()
        mock_service.run_backtest.return_value = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.08,
                "win_rate": 0.65,
                "total_trades": 50,
                "profit_factor": 1.2,
                "sortino_ratio": 1.3,
                "calmar_ratio": 1.1
            },
            "trade_history": []
        }
        return mock_service

    @pytest.fixture
    def sample_yaml_config(self):
        """サンプルYAML設定"""
        return {
            "indicators": {
                "RSI": {
                    "conditions": {
                        "long": "{left_operand} > {threshold}",
                        "short": "{left_operand} < {threshold}"
                    },
                    "scale_type": "oscillator_0_100",
                    "thresholds": {
                        "normal": {"long_gt": 75, "short_lt": 25}
                    },
                    "type": "momentum"
                },
                "SMA": {
                    "conditions": {
                        "long": "close > {left_operand}",
                        "short": "close < {left_operand}"
                    },
                    "scale_type": "price_absolute",
                    "thresholds": None,
                    "type": "trend"
                }
            }
        }

    @pytest.fixture
    def yaml_utils(self, sample_yaml_config, tmp_path):
        """YamlIndicatorUtilsインスタンス"""
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_yaml_config, f)
        return YamlIndicatorUtils(str(config_path))

    @pytest.fixture
    def condition_evolver(self, mock_backtest_service, yaml_utils):
        """ConditionEvolverインスタンス"""
        return ConditionEvolver(
            backtest_service=mock_backtest_service,
            yaml_indicator_utils=yaml_utils
        )

    def test_initialization(self, condition_evolver):
        """初期化テスト"""
        assert condition_evolver.backtest_service is not None
        assert condition_evolver.yaml_indicator_utils is not None

    def test_generate_initial_population(self, condition_evolver):
        """初期個体群生成テスト"""
        population = condition_evolver.generate_initial_population(population_size=10)
        assert len(population) == 10
        assert all(isinstance(ind, Condition) for ind in population)

    def test_evaluate_fitness(self, condition_evolver, mock_backtest_service):
        """適応度評価テスト"""
        condition = Condition("RSI", ">", 70, "long")

        from app.services.auto_strategy.core.condition_evolver import create_simple_strategy
        with patch.object(ConditionEvolver, 'create_strategy_from_condition') as mock_create_strategy:
            mock_create_strategy.return_value = "test_strategy"
            fitness = condition_evolver.evaluate_fitness(condition, {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h"
            })
            assert isinstance(fitness, (int, float))
            assert fitness >= 0

    def test_tournament_selection(self, condition_evolver):
        """トーナメント選択テスト"""
        population = [
            Condition("RSI", ">", 70, "long"),
            Condition("SMA", "<", 100, "short"),
            Condition("MACD", ">", 0, "long")
        ]

        # モック適応度を設定
        fitness_values = [0.8, 0.6, 0.9]
        selected = condition_evolver.tournament_selection(population, fitness_values, k=2)
        assert len(selected) == len(population)

    def test_crossover(self, condition_evolver):
        """交叉テスト"""
        parent1 = Condition("RSI", ">", 70, "long")
        parent2 = Condition("SMA", "<", 100, "short")

        child1, child2 = condition_evolver.crossover(parent1, parent2)
        assert isinstance(child1, Condition)
        assert isinstance(child2, Condition)

    def test_mutate(self, condition_evolver):
        """突然変異テスト"""
        condition = Condition("RSI", ">", 70, "long")
        mutated = condition_evolver.mutate(condition)
        assert isinstance(mutated, Condition)

    def test_run_evolution(self, condition_evolver, mock_backtest_service):
        """進化実行テスト"""
        from app.services.auto_strategy.core.condition_evolver import create_simple_strategy
        with patch.object(condition_evolver, 'create_strategy_from_condition') as mock_create_strategy:
            mock_create_strategy.return_value = "test_strategy"

            result = condition_evolver.run_evolution(
                backtest_config={
                    "symbol": "BTC/USDT:USDT",
                    "timeframe": "1h"
                },
                population_size=10,
                generations=5
            )

            assert "best_condition" in result
            assert "best_fitness" in result
            assert "evolution_history" in result

    def test_create_strategy_from_condition(self, condition_evolver):
        """条件からの戦略作成テスト"""
        condition = Condition("RSI", ">", 70, "long")
        strategy = condition_evolver.create_strategy_from_condition(condition)
        assert strategy is not None
        assert "conditions" in strategy

    def test_condition_evolver_integration(self, mock_backtest_service, sample_yaml_config, tmp_path):
        """統合テスト"""
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_yaml_config, f)

        yaml_utils = YamlIndicatorUtils(str(config_path))
        evolver = ConditionEvolver(
            backtest_service=mock_backtest_service,
            yaml_indicator_utils=yaml_utils
        )

        # 完全な進化サイクルをテスト
        result = evolver.run_evolution(
            backtest_config={
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h"
            },
            population_size=8,
            generations=3
        )

        assert result is not None
        assert "best_fitness" in result
        assert result["best_fitness"] >= 0