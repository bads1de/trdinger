"""
run_auto_strategy スクリプトのテスト

オートストラテジー実行スクリプトの各機能をテストします。
"""

import logging
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# スクリプトのパスをシステムパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))


class TestCreateGaConfig:
    """create_ga_config関数のテスト"""

    def test_default_config(self):
        """デフォルト設定でGAConfigを作成できる"""
        # スクリプトをインポート
        import scripts.run_auto_strategy as run_auto_strategy

        args = Namespace(
            population=20,
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=2,
            no_parallel=False,
            verbose=False,
            start_date="2024-01-01",
            end_date="2024-06-30",
            min_trades=None,
        )

        config = run_auto_strategy.create_ga_config(args)

        assert config.population_size == 20
        assert config.generations == 10
        assert config.crossover_rate == 0.8
        assert config.mutation_rate == 0.2
        assert config.elite_size == 2
        assert config.evaluation_config.enable_parallel is True

    def test_custom_config(self):
        """カスタム設定でGAConfigを作成できる"""
        import scripts.run_auto_strategy as run_auto_strategy

        args = Namespace(
            population=50,
            generations=100,
            crossover_rate=0.9,
            mutation_rate=0.1,
            elite_size=5,
            no_parallel=True,
            verbose=True,
            start_date="2023-01-01",
            end_date="2023-12-31",
            min_trades=None,
        )

        config = run_auto_strategy.create_ga_config(args)

        assert config.population_size == 50
        assert config.generations == 100
        assert config.crossover_rate == 0.9
        assert config.mutation_rate == 0.1
        assert config.elite_size == 5
        assert config.evaluation_config.enable_parallel is False
        assert config.log_level == "DEBUG"

    def test_smoke_config_disables_heavy_features(self, caplog):
        """smokeモードで高速実行向けの設定になる"""
        import scripts.run_auto_strategy as run_auto_strategy

        args = Namespace(
            population=50,
            generations=100,
            crossover_rate=0.9,
            mutation_rate=0.1,
            elite_size=5,
            no_parallel=False,
            verbose=False,
            start_date="2024-01-01",
            end_date="2024-01-31",
            min_trades=None,
            smoke=True,
        )

        with caplog.at_level(logging.WARNING):
            config = run_auto_strategy.create_ga_config(args)

        assert config.population_size == 2
        assert config.generations == 1
        assert config.elite_size == 1
        assert config.max_indicators == 3
        assert config.max_conditions == 2
        assert config.evaluation_config.enable_parallel is False
        assert config.use_seed_strategies is False
        assert config.seed_injection_rate == 0.0
        assert config.tuning_config.enabled is False
        assert config.tuning_config.n_trials == 1
        assert config.two_stage_selection_config.enabled is False
        assert config.fitness_constraints["min_trades"] == 0
        assert not any(
            "TuningConfig の未対応キー" in record.message
            for record in caplog.records
        )

    def test_invalid_population_raises_error(self):
        """無効な個体数でエラーが発生する"""
        import scripts.run_auto_strategy as run_auto_strategy

        args = Namespace(
            population=1,  # 無効（2未満）
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=2,
            no_parallel=False,
            verbose=False,
            start_date="2024-01-01",
            end_date="2024-06-30",
        )

        with pytest.raises(ValueError, match="個体数は2以上"):
            run_auto_strategy.create_ga_config(args)

    def test_elite_size_exceeds_population_raises_error(self):
        """エリート保存数が個体数以上でエラーが発生する"""
        import scripts.run_auto_strategy as run_auto_strategy

        args = Namespace(
            population=10,
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=15,  # 無効（population以上）
            no_parallel=False,
            verbose=False,
            start_date="2024-01-01",
            end_date="2024-06-30",
        )

        with pytest.raises(ValueError, match="エリート保存数は個体数未満"):
            run_auto_strategy.create_ga_config(args)

    def test_invalid_crossover_rate_raises_error(self):
        """無効な交叉率でエラーが発生する"""
        import scripts.run_auto_strategy as run_auto_strategy

        args = Namespace(
            population=20,
            generations=10,
            crossover_rate=1.5,  # 無効（1を超える）
            mutation_rate=0.2,
            elite_size=2,
            no_parallel=False,
            verbose=False,
            start_date="2024-01-01",
            end_date="2024-06-30",
        )

        with pytest.raises(ValueError, match="交叉率は0から1の範囲"):
            run_auto_strategy.create_ga_config(args)


class TestCreateBacktestConfig:
    """create_backtest_config関数のテスト"""

    def test_default_backtest_config(self):
        """デフォルトのバックテスト設定を作成できる"""
        import scripts.run_auto_strategy as run_auto_strategy

        args = Namespace(
            symbol="BTC/USDT:USDT",
            timeframe="4h",
            start_date="2024-01-01",
            end_date="2024-06-30",
            initial_capital=100000.0,
        )

        config = run_auto_strategy.create_backtest_config(args)

        assert config["symbol"] == "BTC/USDT:USDT"
        assert config["timeframe"] == "4h"
        assert config["start_date"] == "2024-01-01"
        assert config["end_date"] == "2024-06-30"
        assert config["initial_capital"] == 100000.0
        assert "commission_rate" in config
        assert "slippage" in config

    def test_invalid_initial_capital_raises_error(self):
        """無効な初期資本でエラーが発生する"""
        import scripts.run_auto_strategy as run_auto_strategy

        args = Namespace(
            symbol="BTC/USDT:USDT",
            timeframe="4h",
            start_date="2024-01-01",
            end_date="2024-06-30",
            initial_capital=-1000.0,  # 無効（0以下）
        )

        with pytest.raises(ValueError, match="初期資本は0より大きい"):
            run_auto_strategy.create_backtest_config(args)


class TestSmokeMode:
    """smokeモードのテスト"""

    def test_apply_smoke_mode_overrides_runtime_flags(self):
        """smokeモードで実行関連フラグが高速向けに切り替わる"""
        import scripts.run_auto_strategy as run_auto_strategy

        args = Namespace(
            population=20,
            generations=10,
            elite_size=2,
            no_parallel=False,
            no_save=False,
            min_trades=None,
            smoke=True,
        )

        result = run_auto_strategy.apply_smoke_mode(args)

        assert result is args
        assert args.population == 2
        assert args.generations == 1
        assert args.elite_size == 1
        assert args.no_parallel is True
        assert args.no_save is True
        assert args.min_trades == 0


class TestFormatConditions:
    """条件整形関数のテスト"""

    def test_format_empty_conditions(self):
        """空の条件リストを整形できる"""
        import scripts.run_auto_strategy as run_auto_strategy

        result = run_auto_strategy._format_conditions([])
        assert result == []

        result = run_auto_strategy._format_conditions(None)
        assert result == []

    def test_format_single_condition(self):
        """単一条件を整形できる"""
        import scripts.run_auto_strategy as run_auto_strategy

        conditions = [
            {
                "left_operand": {"indicator": "RSI"},
                "operator": ">",
                "right_operand": 70,
            }
        ]

        result = run_auto_strategy._format_conditions(conditions)
        assert len(result) == 1
        assert result[0]["operator"] == ">"

    def test_format_condition_group(self):
        """条件グループを整形できる"""
        import scripts.run_auto_strategy as run_auto_strategy

        conditions = [
            {
                "type": "group",
                "logic": "AND",
                "conditions": [
                    {"left_operand": "RSI", "operator": ">", "right_operand": 70},
                    {"left_operand": "MACD", "operator": ">", "right_operand": 0},
                ],
            }
        ]

        result = run_auto_strategy._format_conditions(conditions)
        assert len(result) == 1
        assert result[0]["type"] == "group"


class TestCreateConditionDescription:
    """条件説明生成関数のテスト"""

    def test_simple_condition_description(self):
        """シンプルな条件の説明を生成できる"""
        import scripts.run_auto_strategy as run_auto_strategy

        cond = {
            "left_operand": "RSI",
            "operator": ">",
            "right_operand": 70,
        }

        result = run_auto_strategy._create_condition_description(cond)
        assert "RSI" in result
        assert ">" in result
        assert "70" in result

    def test_indicator_operand_description(self):
        """インジケーターオペランドを含む説明を生成できる"""
        import scripts.run_auto_strategy as run_auto_strategy

        cond = {
            "left_operand": {"indicator": "SMA", "value": 20},
            "operator": "CROSS_UP",
            "right_operand": {"indicator": "EMA", "value": 50},
        }

        result = run_auto_strategy._create_condition_description(cond)
        assert "SMA" in result
        assert "CROSS_UP" in result
        assert "EMA" in result


class TestStrategyGeneToReadableDict:
    """戦略遺伝子の辞書変換テスト"""

    def test_readable_dict_structure(self):
        """可読辞書の構造が正しい"""
        import scripts.run_auto_strategy as run_auto_strategy

        # モックのStrategyGeneを作成
        mock_gene = MagicMock()

        # モックのシリアライザを作成
        mock_serializer = MagicMock()
        mock_serializer.strategy_gene_to_dict.return_value = {
            "name": "Test Strategy",
            "indicators": [
                {"type": "SMA", "parameters": {"period": 20}, "enabled": True}
            ],
            "long_entry_conditions": [],
            "short_entry_conditions": [],
            "risk_management": {},
        }

        result = run_auto_strategy.strategy_gene_to_readable_dict(
            mock_gene, mock_serializer
        )

        # 構造の確認
        assert "strategy_name" in result
        assert "description" in result
        assert "generated_at" in result
        assert "indicators" in result
        assert "entry_conditions" in result
        assert "exit_conditions" in result
        assert "raw_gene" in result

        # インジケーターの確認
        assert len(result["indicators"]) == 1
        assert result["indicators"][0]["name"] == "SMA"
        assert result["indicators"][0]["parameters"]["period"] == 20


class TestParseArgs:
    """コマンドライン引数パースのテスト"""

    def test_default_args(self):
        """デフォルト引数が正しく設定される"""
        import scripts.run_auto_strategy as run_auto_strategy

        with patch("sys.argv", ["run_auto_strategy.py"]):
            args = run_auto_strategy.parse_args()

        assert args.generations == 10
        assert args.population == 20
        assert args.symbol == "BTC/USDT:USDT"
        assert args.timeframe == "4h"

    def test_custom_args(self):
        """カスタム引数が正しくパースされる"""
        import scripts.run_auto_strategy as run_auto_strategy

        with patch(
            "sys.argv",
            [
                "run_auto_strategy.py",
                "--generations",
                "50",
                "--population",
                "100",
                "--symbol",
                "ETH/USDT:USDT",
                "--timeframe",
                "1h",
            ],
        ):
            args = run_auto_strategy.parse_args()

        assert args.generations == 50
        assert args.population == 100
        assert args.symbol == "ETH/USDT:USDT"
        assert args.timeframe == "1h"

    def test_output_arg(self):
        """出力ファイル引数が正しくパースされる"""
        import scripts.run_auto_strategy as run_auto_strategy

        with patch(
            "sys.argv",
            ["run_auto_strategy.py", "--output", "results/test.json"],
        ):
            args = run_auto_strategy.parse_args()

        assert args.output == "results/test.json"

    def test_verbose_flag(self):
        """詳細ログフラグが正しくパースされる"""
        import scripts.run_auto_strategy as run_auto_strategy

        with patch("sys.argv", ["run_auto_strategy.py", "--verbose"]):
            args = run_auto_strategy.parse_args()

        assert args.verbose is True

    def test_no_parallel_flag(self):
        """並列無効フラグが正しくパースされる"""
        import scripts.run_auto_strategy as run_auto_strategy

        with patch("sys.argv", ["run_auto_strategy.py", "--no-parallel"]):
            args = run_auto_strategy.parse_args()

        assert args.no_parallel is True

    def test_smoke_flag(self):
        """smokeフラグが正しくパースされる"""
        import scripts.run_auto_strategy as run_auto_strategy

        with patch("sys.argv", ["run_auto_strategy.py", "--smoke"]):
            args = run_auto_strategy.parse_args()

        assert args.smoke is True


class TestRunAutoStrategyExecution:
    """run_auto_strategy実行経路のテスト"""

    def test_no_save_skips_detailed_backtest_save(self):
        """no_save時に詳細バックテスト保存を呼ばないこと"""
        import scripts.run_auto_strategy as run_auto_strategy
        from app.services.auto_strategy.genes.strategy import StrategyGene

        class DummyBacktestService:
            instances = []

            def __init__(self):
                self.run_backtest_calls = []
                self.cleaned = False
                DummyBacktestService.instances.append(self)

            def run_backtest(self, config):
                self.run_backtest_calls.append(config)
                return {"success": True, "trade_history": [], "equity_curve": []}

            def cleanup(self):
                self.cleaned = True

        class DummyGeneGenerator:
            def __init__(self, config):
                self.config = config

        class DummyEngine:
            def __init__(self, backtest_service, gene_generator, hybrid_mode=False, **_):
                self.backtest_service = backtest_service
                self.gene_generator = gene_generator
                self.hybrid_mode = hybrid_mode

            def run_evolution(self, config, backtest_config, progress_callback=None):
                return {
                    "best_strategy": StrategyGene(),
                    "best_fitness": 1.23,
                    "execution_time": 0.01,
                    "generations_completed": 1,
                    "final_population_size": config.population_size,
                }

        class DummySerializer:
            def strategy_gene_to_dict(self, gene):
                return {
                    "name": "Dummy",
                    "indicators": [],
                    "long_entry_conditions": [],
                    "short_entry_conditions": [],
                    "risk_management": {},
                }

        args = Namespace(
            population=2,
            generations=1,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=1,
            no_parallel=True,
            verbose=False,
            start_date="2024-01-01",
            end_date="2024-01-31",
            min_trades=0,
            symbol="BTC/USDT:USDT",
            timeframe="4h",
            initial_capital=100000.0,
            output=None,
            no_save=True,
        )

        with patch.object(run_auto_strategy, "BacktestService", DummyBacktestService), patch.object(
            run_auto_strategy, "RandomGeneGenerator", DummyGeneGenerator
        ), patch.object(run_auto_strategy, "GeneticAlgorithmEngine", DummyEngine), patch.object(
            run_auto_strategy, "GeneSerializer", DummySerializer
        ), patch("builtins.open", create=True), patch.object(
            run_auto_strategy.Path, "mkdir", return_value=None
        ):
            result = run_auto_strategy.run_auto_strategy(args)

        assert result["success"] is True
        assert DummyBacktestService.instances[0].run_backtest_calls == []
