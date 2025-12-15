"""
ConditionEvolver統合テスト
"""

import pytest
from unittest.mock import Mock, patch

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
    def test_initialization(
        self, mock_exists, mock_open, mock_yaml_load, mock_yaml_config
    ):
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
            "scale_type": (
                "oscillator_0_100" if name == "RSI" else "momentum_zero_centered"
            ),
            "thresholds": (
                {"normal": {"long_gt": 30, "short_lt": 70}} if name == "RSI" else {}
            ),
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
        assert condition_evolver.yaml_indicator_utils is not None

    def test_create_individual(self, condition_evolver):
        """個体生成テスト"""
        condition = condition_evolver._create_individual()

        assert isinstance(condition, Condition)
        # 新しいConditionモデルではleft_operandがインジケータ名
        assert condition.left_operand in ["RSI", "MACD"]
        assert condition.operator in [">", "<", ">=", "<=", "==", "!="]
        # 新しいConditionモデルではright_operandがしきい値
        assert isinstance(condition.right_operand, float)
        # direction属性は動的に追加される
        assert hasattr(condition, "direction")
        assert condition.direction in ["long", "short"]

    def test_evaluate_fitness(self, condition_evolver):
        """適応度評価テスト"""
        condition = Condition(left_operand="RSI", operator=">", right_operand=30.0)
        condition.direction = "long"  # 動的属性を追加
        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "initial_capital": 10000,
        }

        fitness = condition_evolver.evaluate_fitness(condition, backtest_config)

        assert isinstance(fitness, float)
        assert fitness > 0
        condition_evolver.backtest_service.run_backtest.assert_called_once()

    def test_crossover(self, condition_evolver):
        """交叉操作テスト"""
        parent1 = Condition(left_operand="RSI", operator=">", right_operand=30.0)
        parent1.direction = "long"
        parent2 = Condition(left_operand="MACD", operator="<", right_operand=0.0)
        parent2.direction = "short"

        # operator交叉のモック
        with patch("random.choice", return_value="operator"):
            child1, child2 = condition_evolver.crossover(parent1, parent2)

            # インジケータと方向は変わらないはず
            assert child1.left_operand == parent1.left_operand
            assert child2.left_operand == parent2.left_operand

            # オペレータが入れ替わっているか確認
            assert child1.operator == parent2.operator
            assert child2.operator == parent1.operator

        # threshold交叉のモック
        with patch("random.choice", return_value="threshold"):
            child1, child2 = condition_evolver.crossover(parent1, parent2)

            expected_threshold = (parent1.right_operand + parent2.right_operand) / 2
            assert child1.right_operand == expected_threshold
            assert child2.right_operand == expected_threshold

    def test_mutate(self, condition_evolver):
        """突然変異操作テスト"""
        condition = Condition(left_operand="RSI", operator=">", right_operand=30.0)
        condition.direction = "long"

        # operator変異
        with patch("random.choice", side_effect=["operator", "<"]):
            mutated = condition_evolver.mutate(condition)
            assert mutated.operator == "<"
            assert mutated.left_operand == condition.left_operand

        # threshold変異
        with patch("random.choice", side_effect=["threshold"]):
            with patch("random.uniform", return_value=0.1):  # 10% increase
                mutated = condition_evolver.mutate(condition)
                assert mutated.right_operand == condition.right_operand * 1.1

        # indicator変異
        with patch("random.choice", side_effect=["indicator", "MACD"]):
            mutated = condition_evolver.mutate(condition)
            assert mutated.left_operand == "MACD"

    def test_run_evolution(self, condition_evolver):
        """進化プロセス実行テスト"""
        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "initial_capital": 10000,
        }

        # evaluate_fitnessをモック化して高速化
        condition_evolver.evaluate_fitness = Mock(return_value=1.0)

        result = condition_evolver.run_evolution(
            backtest_config, population_size=4, generations=2
        )

        assert "best_condition" in result
        assert isinstance(result["best_condition"], Condition)
        assert "best_fitness" in result
        assert "evolution_history" in result
        assert len(result["evolution_history"]) == 2
        assert result["generations_completed"] == 2

    @pytest.mark.skip(reason="create_strategy_from_condition method has been removed")
    def test_create_strategy_from_condition(self, condition_evolver):
        """条件から戦略作成テスト"""
        condition = Condition(left_operand="RSI", operator=">", right_operand=30.0)
        condition.direction = "long"

        strategy = condition_evolver.create_strategy_from_condition(condition)

        assert "name" in strategy
        assert "conditions" in strategy
        assert strategy["conditions"]["entry"]["type"] == "single"

        cond_dict = strategy["conditions"]["entry"]["condition"]
        assert cond_dict["left_operand"] == "RSI"
        assert cond_dict["operator"] == ">"
        assert cond_dict["right_operand"] == 30.0


class TestParallelFitnessEvaluator:
    """並列適応度評価器のテスト"""

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
            },
        }
        return mock_service

    @pytest.fixture
    def mock_yaml_utils(self):
        """YamlIndicatorUtilsのモック"""
        mock_utils = Mock(spec=YamlIndicatorUtils)
        mock_utils.get_available_indicators.return_value = ["RSI", "MACD"]
        mock_utils.get_indicator_info.side_effect = lambda name: {
            "scale_type": (
                "oscillator_0_100" if name == "RSI" else "momentum_zero_centered"
            ),
            "thresholds": (
                {"normal": {"long_gt": 30, "short_lt": 70}} if name == "RSI" else {}
            ),
        }
        return mock_utils

    def test_parallel_evaluator_initialization(
        self, mock_backtest_service, mock_yaml_utils
    ):
        """並列評価器の初期化テスト"""
        from app.services.auto_strategy.core.condition_evolver import (
            ParallelFitnessEvaluator,
        )

        evaluator = ParallelFitnessEvaluator(
            backtest_service=mock_backtest_service,
            yaml_indicator_utils=mock_yaml_utils,
            max_workers=4,
        )

        assert evaluator.max_workers == 4
        assert evaluator.backtest_service is not None

    def test_parallel_evaluate_population(self, mock_backtest_service, mock_yaml_utils):
        """個体群の並列評価テスト"""
        from app.services.auto_strategy.core.condition_evolver import (
            ParallelFitnessEvaluator,
        )

        evaluator = ParallelFitnessEvaluator(
            backtest_service=mock_backtest_service,
            yaml_indicator_utils=mock_yaml_utils,
            max_workers=2,
        )

        c1 = Condition(left_operand="RSI", operator=">", right_operand=30.0)
        c1.direction = "long"
        c2 = Condition(left_operand="MACD", operator="<", right_operand=0.0)
        c2.direction = "short"
        c3 = Condition(left_operand="RSI", operator="<", right_operand=70.0)
        c3.direction = "short"
        c4 = Condition(left_operand="MACD", operator=">", right_operand=0.5)
        c4.direction = "long"
        population = [c1, c2, c3, c4]
        backtest_config = {"symbol": "BTC/USDT:USDT", "timeframe": "1h"}

        fitness_values = evaluator.evaluate_population(population, backtest_config)

        assert len(fitness_values) == len(population)
        assert all(isinstance(f, float) for f in fitness_values)

    def test_parallel_faster_than_serial(self, mock_backtest_service, mock_yaml_utils):
        """並列評価が直列評価より高速であることのテスト"""
        import time
        from app.services.auto_strategy.core.condition_evolver import (
            ParallelFitnessEvaluator,
        )

        # バックテストに遅延を追加
        def slow_backtest(*args, **kwargs):
            time.sleep(0.05)  # 50ms遅延
            return {
                "success": True,
                "performance_metrics": {
                    "total_return": 0.1,
                    "sharpe_ratio": 1.0,
                    "max_drawdown": 0.1,
                    "total_trades": 20,
                },
            }

        mock_backtest_service.run_backtest.side_effect = slow_backtest

        evaluator = ParallelFitnessEvaluator(
            backtest_service=mock_backtest_service,
            yaml_indicator_utils=mock_yaml_utils,
            max_workers=4,
        )

        population = []
        for _ in range(8):
            c = Condition(left_operand="RSI", operator=">", right_operand=30.0)
            c.direction = "long"
            population.append(c)
        backtest_config = {"symbol": "BTC/USDT:USDT", "timeframe": "1h"}

        start_time = time.time()
        evaluator.evaluate_population(population, backtest_config)
        parallel_time = time.time() - start_time

        # 並列処理では8個体 * 50ms = 400ms が理論上 ~100ms で終わるはず
        # 直列だと最低400ms、並列だと100-200ms程度を期待
        assert parallel_time < 0.35  # 350ms未満で終わること


class TestFitnessCache:
    """適応度キャッシュのテスト"""

    def test_cache_initialization(self):
        """キャッシュ初期化テスト"""
        from app.services.auto_strategy.core.condition_evolver import FitnessCache

        cache = FitnessCache(max_size=100)
        assert cache.max_size == 100
        assert len(cache) == 0

    def test_cache_get_set(self):
        """キャッシュの取得・設定テスト"""
        from app.services.auto_strategy.core.condition_evolver import FitnessCache

        cache = FitnessCache(max_size=100)
        condition = Condition(left_operand="RSI", operator=">", right_operand=30.0)
        condition.direction = "long"

        # 初期状態ではキャッシュミス
        assert cache.get(condition) is None

        # キャッシュに設定
        cache.set(condition, 0.75)

        # キャッシュヒット
        assert cache.get(condition) == 0.75

    def test_cache_eviction_on_max_size(self):
        """最大サイズ到達時のキャッシュ削除テスト"""
        from app.services.auto_strategy.core.condition_evolver import FitnessCache

        cache = FitnessCache(max_size=3)

        # 3つのエントリを追加
        c1 = Condition(left_operand="RSI", operator=">", right_operand=30.0)
        c1.direction = "long"
        c2 = Condition(left_operand="MACD", operator="<", right_operand=0.0)
        c2.direction = "short"
        c3 = Condition(left_operand="RSI", operator="<", right_operand=70.0)
        c3.direction = "short"

        cache.set(c1, 0.5)
        cache.set(c2, 0.6)
        cache.set(c3, 0.7)

        assert len(cache) == 3

        # 4つ目を追加すると古いエントリが削除される
        c4 = Condition(left_operand="MACD", operator=">", right_operand=0.5)
        c4.direction = "long"
        cache.set(c4, 0.8)

        assert len(cache) == 3
        assert cache.get(c4) == 0.8

    def test_cache_hit_rate(self):
        """キャッシュヒット率の計算テスト"""
        from app.services.auto_strategy.core.condition_evolver import FitnessCache

        cache = FitnessCache(max_size=100)
        condition = Condition(left_operand="RSI", operator=">", right_operand=30.0)
        condition.direction = "long"

        cache.set(condition, 0.75)

        # ミス1回
        c_test = Condition(left_operand="MACD", operator="<", right_operand=0.0)
        c_test.direction = "short"
        cache.get(c_test)
        # ヒット1回
        cache.get(condition)

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestEarlyStopping:
    """早期打ち切りのテスト"""

    def test_early_stopping_initialization(self):
        """早期打ち切り初期化テスト"""
        from app.services.auto_strategy.core.condition_evolver import EarlyStopping

        early_stopping = EarlyStopping(patience=5, min_delta=0.001)
        assert early_stopping.patience == 5
        assert early_stopping.min_delta == 0.001
        assert not early_stopping.should_stop

    def test_early_stopping_no_improvement(self):
        """改善がない場合の早期打ち切りテスト"""
        from app.services.auto_strategy.core.condition_evolver import EarlyStopping

        early_stopping = EarlyStopping(patience=3, min_delta=0.01)

        # 3世代改善なし
        early_stopping.update(0.5)
        assert not early_stopping.should_stop

        early_stopping.update(0.5)
        assert not early_stopping.should_stop

        early_stopping.update(0.5)
        assert not early_stopping.should_stop

        early_stopping.update(0.5)
        assert early_stopping.should_stop  # 4世代目で打ち切り

    def test_early_stopping_with_improvement(self):
        """改善がある場合は打ち切らないテスト"""
        from app.services.auto_strategy.core.condition_evolver import EarlyStopping

        early_stopping = EarlyStopping(patience=3, min_delta=0.01)

        early_stopping.update(0.5)
        early_stopping.update(0.5)
        early_stopping.update(0.52)  # 改善

        assert not early_stopping.should_stop
        assert early_stopping.counter == 0  # カウンターリセット

    def test_early_stopping_reset(self):
        """早期打ち切り状態のリセットテスト"""
        from app.services.auto_strategy.core.condition_evolver import EarlyStopping

        early_stopping = EarlyStopping(patience=2, min_delta=0.01)

        early_stopping.update(0.5)
        early_stopping.update(0.5)
        early_stopping.update(0.5)

        assert early_stopping.should_stop

        early_stopping.reset()

        assert not early_stopping.should_stop
        assert early_stopping.counter == 0


class TestConditionEvolverWithParallelization:
    """並列化機能を統合したConditionEvolverのテスト"""

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
            },
        }
        return mock_service

    @pytest.fixture
    def mock_yaml_utils(self):
        """YamlIndicatorUtilsのモック"""
        mock_utils = Mock(spec=YamlIndicatorUtils)
        mock_utils.get_available_indicators.return_value = ["RSI", "MACD"]
        mock_utils.get_indicator_info.side_effect = lambda name: {
            "scale_type": (
                "oscillator_0_100" if name == "RSI" else "momentum_zero_centered"
            ),
            "thresholds": (
                {"normal": {"long_gt": 30, "short_lt": 70}} if name == "RSI" else {}
            ),
        }
        return mock_utils

    def test_run_evolution_with_parallel(self, mock_backtest_service, mock_yaml_utils):
        """並列モードでの進化プロセステスト"""
        evolver = ConditionEvolver(
            backtest_service=mock_backtest_service,
            yaml_indicator_utils=mock_yaml_utils,
            enable_parallel=True,
            max_workers=2,
        )

        backtest_config = {"symbol": "BTC/USDT:USDT", "timeframe": "1h"}

        result = evolver.run_evolution(
            backtest_config,
            population_size=4,
            generations=2,
        )

        assert "best_condition" in result
        assert "best_fitness" in result
        assert result["parallel_enabled"] is True

    def test_run_evolution_with_early_stopping(
        self, mock_backtest_service, mock_yaml_utils
    ):
        """早期打ち切り機能付きの進化プロセステスト"""
        evolver = ConditionEvolver(
            backtest_service=mock_backtest_service,
            yaml_indicator_utils=mock_yaml_utils,
            early_stopping_patience=2,
        )

        # 常に同じフィットネスを返すようにして早期打ち切りを発動
        evolver.evaluate_fitness = Mock(return_value=0.5)

        backtest_config = {"symbol": "BTC/USDT:USDT", "timeframe": "1h"}

        result = evolver.run_evolution(
            backtest_config,
            population_size=4,
            generations=10,  # 多めに設定
        )

        # 早期打ち切りで10世代より早く終了
        assert result["generations_completed"] < 10
        assert result.get("early_stopped", False) is True

    def test_run_evolution_with_cache(self, mock_backtest_service, mock_yaml_utils):
        """キャッシュ機能付きの進化プロセステスト"""
        evolver = ConditionEvolver(
            backtest_service=mock_backtest_service,
            yaml_indicator_utils=mock_yaml_utils,
            enable_cache=True,
            cache_size=100,
        )

        # キャッシュが有効化されていることを確認
        assert evolver.cache is not None

        backtest_config = {"symbol": "BTC/USDT:USDT", "timeframe": "1h"}

        result = evolver.run_evolution(
            backtest_config,
            population_size=4,
            generations=3,
        )

        # キャッシュが存在する場合のみchache_statsをチェック
        # 実装によっては、cache_statsが含まれない場合もあるので柔軟に対応
        if evolver.cache:
            cache_stats = evolver.cache.get_stats()
            assert "hits" in cache_stats
            assert "misses" in cache_stats
        
        # 結果の基本的な構造を確認
        assert "best_condition" in result
        assert "best_fitness" in result


