"""
config_builder（共有設定ビルダー）のテスト
"""

import pytest

from app.cli.config_builder import (
    build_backtest_config_dict,
    build_ga_config_dict,
    normalize_run_options,
)


class TestNormalizeRunOptions:
    """normalize_run_options のテスト"""

    def test_valid_defaults(self):
        """デフォルト値はそのまま通る"""
        result = normalize_run_options(
            population=20,
            generations=10,
            elite_size=2,
            crossover_rate=0.8,
            mutation_rate=0.2,
            initial_capital=100000.0,
        )
        assert result["population"] == 20
        assert result["generations"] == 10
        assert result["elite_size"] == 2

    def test_smoke_mode_shrinks(self):
        """smoke モードは最小構成へ縮小する"""
        result = normalize_run_options(
            population=100,
            generations=50,
            elite_size=10,
            crossover_rate=0.8,
            mutation_rate=0.2,
            initial_capital=100000.0,
            smoke=True,
        )
        assert result["population"] == 2
        assert result["generations"] == 1
        assert result["elite_size"] == 1

    def test_population_too_small(self):
        """個体数2未満は ValueError"""
        with pytest.raises(ValueError, match="個体数"):
            normalize_run_options(
                population=1,
                generations=10,
                elite_size=0,
                crossover_rate=0.8,
                mutation_rate=0.2,
                initial_capital=100000.0,
            )

    def test_elite_not_less_than_population(self):
        """エリート数が個体数以上は ValueError"""
        with pytest.raises(ValueError, match="エリート"):
            normalize_run_options(
                population=10,
                generations=10,
                elite_size=10,
                crossover_rate=0.8,
                mutation_rate=0.2,
                initial_capital=100000.0,
            )

    def test_invalid_crossover_rate(self):
        """交叉率が範囲外は ValueError"""
        with pytest.raises(ValueError, match="交叉率"):
            normalize_run_options(
                population=20,
                generations=10,
                elite_size=2,
                crossover_rate=1.5,
                mutation_rate=0.2,
                initial_capital=100000.0,
            )

    def test_invalid_initial_capital(self):
        """初期資本が0以下は ValueError"""
        with pytest.raises(ValueError, match="初期資本"):
            normalize_run_options(
                population=20,
                generations=10,
                elite_size=2,
                crossover_rate=0.8,
                mutation_rate=0.2,
                initial_capital=0,
            )


class TestBuildGaConfigDict:
    """build_ga_config_dict のテスト"""

    def test_default_build(self):
        """デフォルトで構築できる"""
        config = build_ga_config_dict(
            population=20,
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=2,
            start_date="2024-01-01",
            end_date="2024-06-30",
        )
        assert config["population_size"] == 20
        assert config["generations"] == 10
        assert config["crossover_rate"] == 0.8
        assert config["elite_size"] == 2
        assert config["evaluation_config"]["enable_parallel"] is True
        # 検証は無効化フラグが無い限り GAConfig のデフォルトに委ねる（キー自体を入れない）
        assert "validation_config" not in config

    def test_no_validation_disables(self):
        """--no-validation で検証が無効化される"""
        config = build_ga_config_dict(
            population=20,
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=2,
            start_date="2024-01-01",
            end_date="2024-06-30",
            no_validation=True,
        )
        assert config["validation_config"]["enabled"] is False

    def test_no_parallel_disables(self):
        """--no-parallel で並列評価が無効化される"""
        config = build_ga_config_dict(
            population=20,
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=2,
            start_date="2024-01-01",
            end_date="2024-06-30",
            no_parallel=True,
        )
        assert config["evaluation_config"]["enable_parallel"] is False

    def test_smoke_sets_minimal(self):
        """smoke モードで最小構成になる"""
        config = build_ga_config_dict(
            population=100,
            generations=50,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=10,
            start_date="2024-01-01",
            end_date="2024-06-30",
            smoke=True,
        )
        assert config["population_size"] == 2
        assert config["generations"] == 1
        assert config["validation_config"]["enabled"] is False
        assert config["max_indicators"] == 3

    def test_smoke_disables_early_termination(self):
        """smoke モードでは早期終了が無効化される（全個体ペナルティ防止）"""
        config = build_ga_config_dict(
            population=20,
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=2,
            start_date="2024-01-01",
            end_date="2024-06-30",
            smoke=True,
        )
        early = config["evaluation_config"]["early_termination_settings"]
        assert early["enabled"] is False

    def test_non_smoke_keeps_early_termination_default(self):
        """通常モードでは早期終了設定を上書きしない（GAConfig デフォルトに委ねる）"""
        config = build_ga_config_dict(
            population=20,
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=2,
            start_date="2024-01-01",
            end_date="2024-06-30",
        )
        assert "early_termination_settings" not in config["evaluation_config"]

    def test_smoke_config_is_accepted_by_ga_config(self):
        """smoke 構成が GAConfig.from_dict で受け入れられる"""
        from app.services.auto_strategy.config.ga_config import (
            EarlyTerminationSettings,
            EvaluationConfig,
            GAConfig,
        )

        config_dict = build_ga_config_dict(
            population=20,
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=2,
            start_date="2024-01-01",
            end_date="2024-06-30",
            smoke=True,
        )
        ga_config = GAConfig.from_dict(config_dict)
        assert isinstance(ga_config.evaluation_config, EvaluationConfig)
        assert isinstance(
            ga_config.evaluation_config.early_termination_settings,
            EarlyTerminationSettings,
        )
        assert ga_config.evaluation_config.early_termination_settings.enabled is False

    def test_mtf_enabled(self):
        """MTF 有効化でタイムフレームが設定される"""
        config = build_ga_config_dict(
            population=20,
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=2,
            start_date="2024-01-01",
            end_date="2024-06-30",
            mtf=True,
            mtf_timeframes="1d",
        )
        assert config["enable_multi_timeframe"] is True
        assert config["available_timeframes"] == ["1d"]

    def test_mtf_invalid_timeframe(self):
        """サポート外MTFタイムフレームは ValueError"""
        with pytest.raises(ValueError, match="タイムフレーム"):
            build_ga_config_dict(
                population=20,
                generations=10,
                crossover_rate=0.8,
                mutation_rate=0.2,
                elite_size=2,
                start_date="2024-01-01",
                end_date="2024-06-30",
                mtf=True,
                mtf_timeframes="99x",
            )

    def test_no_seeds(self):
        """--no-seeds でシード注入が無効化される"""
        config = build_ga_config_dict(
            population=20,
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=2,
            start_date="2024-01-01",
            end_date="2024-06-30",
            no_seeds=True,
        )
        assert config["use_seed_strategies"] is False
        assert config["seed_injection_rate"] == 0.0

    def test_min_trades_zero(self):
        """min_trades=0 で制約が0になる"""
        config = build_ga_config_dict(
            population=20,
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=2,
            start_date="2024-01-01",
            end_date="2024-06-30",
            min_trades=0,
        )
        assert config["fitness_constraints"]["min_trades"] == 0


class TestBuildBacktestConfigDict:
    """build_backtest_config_dict のテスト"""

    def test_default_build(self):
        """デフォルトで構築できる"""
        config = build_backtest_config_dict()
        assert config["symbol"] == "BTC/USDT:USDT"
        assert config["timeframe"] == "4h"
        assert config["initial_capital"] == 100000.0
        assert config["commission_rate"] == 0.0004

    def test_custom_build(self):
        """カスタム値が反映される"""
        config = build_backtest_config_dict(
            symbol="ETH/USDT:USDT",
            timeframe="1h",
            start_date="2024-02-01",
            end_date="2024-03-01",
            initial_capital=50000.0,
        )
        assert config["symbol"] == "ETH/USDT:USDT"
        assert config["timeframe"] == "1h"
        assert config["start_date"] == "2024-02-01"
        assert config["initial_capital"] == 50000.0

    def test_invalid_capital(self):
        """初期資本0以下は ValueError"""
        with pytest.raises(ValueError, match="初期資本"):
            build_backtest_config_dict(initial_capital=-1)
