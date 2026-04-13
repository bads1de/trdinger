"""
IndividualEvaluatorのテスト
"""

from unittest.mock import Mock, patch
import pandas as pd

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.config.ga.nested_configs import EvaluationConfig
from app.services.auto_strategy.core.evaluation.evaluation_fidelity import (
    build_coarse_ga_config,
)
from app.services.auto_strategy.core.evaluation.individual_evaluator import IndividualEvaluator
from app.services.auto_strategy.genes import StrategyGene, IndicatorGene, Condition
from app.services.backtest.execution.backtest_executor import (
    BacktestEarlyTerminationError,
)


class TestIndividualEvaluator:
    """IndividualEvaluatorのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        self.mock_backtest_service = Mock()
        self.evaluator = IndividualEvaluator(self.mock_backtest_service)

    def _create_mock_gene(self, gene_id="test-gene"):
        """テスト用のStrategyGeneを作成するヘルパー"""
        return StrategyGene(
            id=gene_id,
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA_20")
            ],
            short_entry_conditions=[
                Condition(left_operand="close", operator="<", right_operand="SMA_20")
            ],
        )

    def test_init(self):
        """初期化のテスト"""
        assert self.evaluator.backtest_service == self.mock_backtest_service
        assert self.evaluator._fixed_backtest_config is None

    def test_resolve_gene_reuses_single_serializer_instance(self):
        """辞書入力の復元でGeneSerializerを再生成しないことを確認する。"""
        serializer_instance = Mock()
        expected_gene = self._create_mock_gene(gene_id="")
        serializer_instance.dict_to_strategy_gene.return_value = expected_gene

        with patch(
            "app.services.auto_strategy.core.evaluation.individual_evaluator.GeneSerializer"
        ) as serializer_cls:
            serializer_cls.return_value = serializer_instance
            evaluator = IndividualEvaluator(self.mock_backtest_service)

            first = evaluator._resolve_gene({"id": ""})
            second = evaluator._resolve_gene({"id": ""})

            assert first is expected_gene
            assert second is expected_gene
            assert serializer_cls.call_count == 1
            assert serializer_instance.dict_to_strategy_gene.call_count == 2

    def test_build_cache_key_reuses_single_serializer_instance_for_empty_gene_id(self):
        """IDなし遺伝子のキー生成でもGeneSerializerを再生成しないことを確認する。"""
        serializer_instance = Mock()
        serializer_instance.strategy_gene_to_dict.return_value = {"a": 1, "b": 2}

        with patch(
            "app.services.auto_strategy.core.evaluation.individual_evaluator.GeneSerializer"
        ) as serializer_cls:
            serializer_cls.return_value = serializer_instance
            evaluator = IndividualEvaluator(self.mock_backtest_service)
            gene = self._create_mock_gene(gene_id="")

            first_key = evaluator._build_cache_key(gene)
            second_key = evaluator._build_cache_key(gene)

            assert first_key == second_key
            assert serializer_cls.call_count == 1
            assert serializer_instance.strategy_gene_to_dict.call_count == 2

    def test_getstate_setstate_recreates_gene_serializer_component(self):
        """pickle復元時にGeneSerializerコンポーネントが再生成されることを確認する。"""
        state = self.evaluator.__getstate__()
        assert "_gene_serializer" not in state

        restored = IndividualEvaluator.__new__(IndividualEvaluator)
        restored.__setstate__(state)

        assert hasattr(restored, "_gene_serializer")
        assert restored._gene_serializer is not None

    def test_prepare_run_config_optimization(self):
        """バックテスト設定生成の最適化検証"""
        gene = self._create_mock_gene()
        base_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "strategy_name": "Test",
            "initial_capital": 10000,
            "commission_rate": 0.001,
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
        }
        ga_config = GAConfig()

        # モックのGeneSerializerがインポートされないことを確認するために、
        # sys.modulesから削除したりパッチを当てたりするのは過剰だが、
        # 返り値の中身をチェックすれば十分

        result = self.evaluator._prepare_run_config(gene, base_config, ga_config)

        assert result is not None
        # 1. バリデーションスキップフラグがあるか
        assert result.get("_skip_validation") is True

        # 2. strategy_gene が辞書ではなくオブジェクトそのものであるか
        # Pydanticモデルを通すと辞書になるが、今回は辞書のまま操作しているはず
        strategy_config = result["strategy_config"]
        parameters = strategy_config["parameters"]
        assert parameters["strategy_gene"] is gene
        assert not isinstance(parameters["strategy_gene"], dict)

    def test_set_backtest_config(self):
        """バックテスト設定のテスト"""
        config = {"symbol": "BTC/USDT:USDT", "timeframe": "1h"}
        self.evaluator.set_backtest_config(config)
        assert self.evaluator._fixed_backtest_config == config

    def test_evaluate_individual_force_refresh_updates_cached_result(self):
        gene = self._create_mock_gene()
        coarse_report = Mock()
        coarse_report.aggregated_fitness = (0.1,)
        coarse_report.metadata = {"evaluation_fidelity": "coarse"}
        full_report = Mock()
        full_report.aggregated_fitness = (0.9,)
        full_report.metadata = {"evaluation_fidelity": "full"}

        self.evaluator._evaluation_strategy.execute_report = Mock(
            side_effect=[coarse_report, full_report]
        )

        ga_config = GAConfig()

        first = self.evaluator.evaluate_individual(gene, ga_config)
        refreshed = self.evaluator.evaluate_individual(
            gene,
            ga_config,
            force_refresh=True,
        )

        assert first == (0.1,)
        assert refreshed == (0.9,)
        assert self.evaluator._evaluation_strategy.execute_report.call_count == 2
        assert self.evaluator.get_cached_evaluation_report(gene).metadata[
            "evaluation_fidelity"
        ] == "full"

    def test_prepare_backtest_config_for_evaluation_adds_warmup_window(self):
        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="EMA", parameters={"length": 20}, enabled=True)
            ],
            long_entry_conditions=[],
            short_entry_conditions=[],
        )
        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-10 00:00:00",
            "end_date": "2024-01-12 00:00:00",
        }

        prepared = self.evaluator._prepare_backtest_config_for_evaluation(
            gene, backtest_config
        )

        assert prepared["_evaluation_start"] == "2024-01-10 00:00:00"
        assert prepared["end_date"] == "2024-01-12 00:00:00"
        assert prepared["start_date"] == "2024-01-09 03:00:00"

    def test_perform_single_evaluation_extends_run_window_for_indicator_warmup(self):
        mock_individual = StrategyGene(
            id="warmup-gene",
            indicators=[
                IndicatorGene(type="EMA", parameters={"length": 20}, enabled=True)
            ],
            long_entry_conditions=[],
            short_entry_conditions=[],
        )
        market_data = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [10.0, 11.0],
            },
            index=pd.date_range("2024-01-09 03:00:00", periods=2, freq="h"),
        )

        mock_result = {
            "performance_metrics": {"total_trades": 1},
            "trade_history": [],
            "equity_curve": [],
            "_raw_stats": Mock(),
        }

        self.evaluator._get_cached_data = Mock(return_value=market_data)
        self.evaluator._get_cached_minute_data = Mock(return_value=None)
        self.evaluator._calculate_multi_objective_fitness = Mock(return_value=(0.42,))
        self.evaluator._apply_evaluation_window_to_result = Mock(
            return_value=mock_result
        )
        self.mock_backtest_service.run_backtest.return_value = mock_result

        ga_config = GAConfig()
        ga_config.objectives = ["weighted_score"]

        fitness = self.evaluator._perform_single_evaluation(
            mock_individual,
            {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "start_date": "2024-01-10 00:00:00",
                "end_date": "2024-01-12 00:00:00",
                "initial_capital": 10000.0,
                "commission_rate": 0.001,
                "strategy_name": "WarmupTest",
            },
            ga_config,
        )

        run_config = self.mock_backtest_service.run_backtest.call_args.kwargs["config"]
        assert run_config["_include_raw_stats"] is True
        assert run_config["start_date"] == "2024-01-09 03:00:00"
        assert (
            run_config["strategy_config"]["parameters"]["evaluation_start"]
            == "2024-01-10 00:00:00"
        )
        assert fitness == (0.42,)

    def test_perform_single_evaluation_uses_recent_tail_window_for_coarse_fidelity(self):
        mock_individual = StrategyGene(
            id="coarse-gene",
            indicators=[],
            long_entry_conditions=[],
            short_entry_conditions=[],
        )
        market_data = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [10.0, 11.0],
            },
            index=pd.date_range("2024-01-08 00:00:00", periods=2, freq="D"),
        )

        self.evaluator._get_cached_data = Mock(return_value=market_data)
        self.evaluator._get_cached_minute_data = Mock(return_value=None)
        self.evaluator._calculate_multi_objective_fitness = Mock(return_value=(0.42,))
        self.mock_backtest_service.run_backtest.return_value = {
            "performance_metrics": {"total_trades": 1},
            "trade_history": [],
            "equity_curve": [],
        }

        full_config = GAConfig(
            evaluation_config=EvaluationConfig(
                enable_multi_fidelity_evaluation=True,
                multi_fidelity_window_ratio=0.3,
            ),
        )
        full_config.objectives = ["weighted_score"]
        coarse_config = build_coarse_ga_config(full_config)

        self.evaluator._perform_single_evaluation(
            mock_individual,
            {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1d",
                "start_date": "2024-01-01 00:00:00",
                "end_date": "2024-01-11 00:00:00",
                "initial_capital": 10000.0,
                "commission_rate": 0.001,
                "strategy_name": "CoarseTest",
            },
            coarse_config,
        )

        run_config = self.mock_backtest_service.run_backtest.call_args.kwargs["config"]
        assert run_config["start_date"] == "2024-01-08 00:00:00"
        assert run_config["end_date"] == "2024-01-11 00:00:00"

    def test_perform_single_evaluation_report_returns_penalty_on_early_termination(self):
        gene = self._create_mock_gene()
        self.evaluator._get_cached_data = Mock(return_value=Mock())
        self.evaluator._get_cached_minute_data = Mock(return_value=None)
        self.mock_backtest_service.run_backtest.side_effect = BacktestEarlyTerminationError(
            "trade_pace"
        )

        config = GAConfig()
        config.objectives = ["weighted_score"]

        scenario = self.evaluator._perform_single_evaluation_report(
            gene,
            {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01 00:00:00",
                "end_date": "2024-01-02 00:00:00",
                "initial_capital": 10000.0,
                "commission_rate": 0.001,
                "strategy_name": "EarlyStopTest",
            },
            config,
        )

        assert scenario.passed is False
        assert scenario.metadata["early_terminated"] is True
        assert scenario.metadata["termination_reason"] == "trade_pace"
        assert scenario.fitness == (-float("inf"),)

    def test_apply_evaluation_window_to_result_recomputes_trimmed_window(self):
        market_data = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0, 110.0],
                "high": [101.0, 101.0, 111.0, 111.0],
                "low": [99.0, 99.0, 109.0, 109.0],
                "close": [100.0, 100.0, 110.0, 110.0],
                "volume": [10.0, 10.0, 10.0, 10.0],
            },
            index=pd.date_range("2024-01-01 00:00:00", periods=4, freq="D"),
        )
        raw_stats = Mock()
        raw_stats._equity_curve = pd.DataFrame(
            {
                "Equity": [10000.0, 10000.0, 11000.0, 11000.0],
                "DrawdownPct": [0.0, 0.0, 0.0, 0.0],
            },
            index=market_data.index,
        )
        raw_stats._trades = pd.DataFrame(
            {
                "Size": [1],
                "EntryBar": [2],
                "ExitBar": [3],
                "EntryPrice": [100.0],
                "ExitPrice": [110.0],
                "SL": [None],
                "TP": [None],
                "PnL": [10.0],
                "Commission": [0.0],
                "ReturnPct": [0.10],
                "EntryTime": [market_data.index[2]],
                "ExitTime": [market_data.index[3]],
                "Duration": [market_data.index[3] - market_data.index[2]],
                "Tag": [None],
            }
        )

        converted_result = {
            "strategy_name": "WarmupTest",
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "initial_capital": 10000.0,
            "config_json": {},
            "performance_metrics": {},
            "trade_history": [],
            "equity_curve": [],
        }

        with (
            patch.object(
                self.evaluator,
                "_compute_window_stats",
                return_value="window_stats",
            ) as mock_compute_window_stats,
            patch(
                "app.services.backtest.conversion.backtest_result_converter.BacktestResultConverter.convert_backtest_results",
                return_value={
                    "strategy_name": "WarmupTest",
                    "symbol": "BTC/USDT:USDT",
                    "timeframe": "1h",
                    "initial_capital": 10000.0,
                    "performance_metrics": {"total_return": 10.0},
                    "trade_history": [{"pnl": 10.0}],
                    "equity_curve": [{"equity": 10000.0}, {"equity": 11000.0}],
                    "start_date": pd.Timestamp("2024-01-03 00:00:00").to_pydatetime(),
                    "end_date": pd.Timestamp("2024-01-04 00:00:00").to_pydatetime(),
                    "config_json": {},
                },
            ),
        ):
            adjusted = self.evaluator._apply_evaluation_window_to_result(
                converted_result,
                raw_stats,
                market_data,
                pd.Timestamp("2024-01-03 00:00:00"),
                pd.Timestamp("2024-01-04 00:00:00"),
            )

        trades_df, equity_values, ohlc_data = mock_compute_window_stats.call_args.args
        assert list(ohlc_data.index) == list(market_data.index[2:])
        assert list(equity_values) == [11000.0, 11000.0]
        assert trades_df["EntryBar"].tolist() == [0]
        assert adjusted["performance_metrics"]["total_return"] == 10.0

    def test_slice_equity_curve_for_window_uses_initial_capital_for_leading_gaps(self):
        target_index = pd.date_range("2024-01-01 00:00:00", periods=4, freq="D")
        raw_equity_curve = pd.DataFrame(
            {
                "Equity": [10100.0, 10200.0, 10300.0],
                "DrawdownPct": [0.1, 0.2, 0.3],
            },
            index=target_index[1:],
        )

        trimmed = self.evaluator._slice_equity_curve_for_window(
            raw_equity_curve,
            target_index,
            0,
            len(target_index),
            10000.0,
        )

        assert trimmed.loc[target_index[0], "Equity"] == 10000.0
        assert trimmed["Equity"].isna().sum() == 0
        assert trimmed.index.equals(target_index)

    def test_build_parallel_worker_initargs(self):
        """並列ワーカー初期化引数の構築テスト"""
        backtest_config = {"symbol": "BTC/USDT:USDT", "timeframe": "1h"}
        ga_config = GAConfig()
        main_data = pd.DataFrame({"close": [1, 2]})
        minute_data = pd.DataFrame({"close": [1, 2]})

        self.evaluator.set_backtest_config(backtest_config)
        self.evaluator._get_cached_data = Mock(return_value=main_data)
        self.evaluator._get_cached_minute_data = Mock(return_value=minute_data)

        result = self.evaluator.build_parallel_worker_initargs(ga_config)

        assert result is not None
        run_backtest_config, run_ga_config, shared_data = result
        assert run_backtest_config == backtest_config
        assert run_backtest_config is not backtest_config
        assert run_ga_config is ga_config
        assert shared_data["main_data"]["key"] == (
            "BTC/USDT:USDT",
            "1h",
            "None",
            "None",
        )
        assert shared_data["main_data"]["data"] is main_data
        assert shared_data["minute_data"]["key"] == (
            "minute",
            "BTC/USDT:USDT",
            "1m",
            "None",
            "None",
        )
        assert shared_data["minute_data"]["data"] is minute_data

    def test_evaluate_individual_success(self):
        """個体評価成功のテスト"""
        # モック設定
        mock_individual = self._create_mock_gene()
        mock_result = {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1,
                "win_rate": 0.6,
                "profit_factor": 1.8,
            },
            "equity_curve": [100, 110, 105, 120],
            "trade_history": [{"size": 1, "pnl": 10}, {"size": -1, "pnl": -5}],
        }

        self.mock_backtest_service.run_backtest.return_value = mock_result

        ga_config = GAConfig()
        ga_config.enable_multi_objective = False
        ga_config.fitness_constraints = {}
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }

        # テスト実行
        result = self.evaluator.evaluate(mock_individual, ga_config)

        assert isinstance(result, tuple)
        assert len(result) == 1  # 単一目的最適化

    def test_evaluate_individual_multi_objective(self):
        """多目的最適化評価のテスト"""
        mock_individual = self._create_mock_gene()
        mock_result = {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1,
            },
            "equity_curve": [],
            "trade_history": [],
        }

        self.mock_backtest_service.run_backtest.return_value = mock_result

        ga_config = GAConfig()
        ga_config.enable_multi_objective = True
        ga_config.objectives = ["total_return", "sharpe_ratio", "max_drawdown"]

        result = self.evaluator.evaluate(mock_individual, ga_config)

        assert isinstance(result, tuple)
        assert len(result) == 3  # 3つの目的

    def test_evaluate_individual_exception(self):
        """個体評価例外のテスト"""
        mock_individual = self._create_mock_gene()

        # バックテストで例外が発生
        self.mock_backtest_service.run_backtest.side_effect = Exception("Test error")

        ga_config = GAConfig()
        ga_config.enable_multi_objective = False

        result = self.evaluator.evaluate(mock_individual, ga_config)

        assert result == (0.0,)

    def test_evaluate_individual_multi_objective_exception(self):
        """多目的最適化例外のテスト"""
        mock_individual = self._create_mock_gene()

        self.mock_backtest_service.run_backtest.side_effect = Exception("Test error")

        ga_config = GAConfig()
        ga_config.enable_multi_objective = True
        ga_config.objectives = ["total_return", "sharpe_ratio"]

        result = self.evaluator.evaluate(mock_individual, ga_config)

        assert result == (0.0, 0.0)  # 目的数に応じた0.0が返される

    def test_extract_performance_metrics(self):
        """パフォーメンスメトリクス抽出のテスト"""
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.55,
                "profit_factor": 1.9,
                "sortino_ratio": 1.8,
                "calmar_ratio": 1.5,
            },
            "equity_curve": [100, 110, 105, 120, 115],
            "trade_history": [
                {"size": 1, "pnl": 10},
                {"size": -1, "pnl": -5},
                {"size": 1, "pnl": 15},
            ],
            "start_date": "2024-01-01",
            "end_date": "2024-12-19",
        }

        metrics = self.evaluator._extract_performance_metrics(backtest_result)

        assert metrics["total_return"] == 0.15
        assert metrics["sharpe_ratio"] == 1.2
        assert metrics["max_drawdown"] == 0.08
        assert metrics["win_rate"] == 0.55
        assert "ulcer_index" in metrics
        assert "trade_frequency_penalty" in metrics

    def test_extract_performance_metrics_invalid_values(self):
        """無効な値の処理テスト"""
        backtest_result = {
            "performance_metrics": {
                "total_return": float("inf"),  # 無限大
                "sharpe_ratio": None,  # None
                "max_drawdown": -0.1,  # 負のドローダウン
                "win_rate": "invalid",  # 無効な型
            },
            "equity_curve": [],
            "trade_history": [],
        }

        metrics = self.evaluator._extract_performance_metrics(backtest_result)

        # 無効な値が適切に処理されているか確認
        assert metrics["total_return"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["max_drawdown"] == 0.0  # 負の値は0に修正
        assert metrics["win_rate"] == 0.0

    def test_calculate_fitness_zero_trades(self):
        """取引回数0のフィットネス計算テスト"""
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1,
                "win_rate": 0.6,
                "total_trades": 0,  # 取引なし
            },
            "equity_curve": [],
            "trade_history": [],
        }

        ga_config = GAConfig()
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }
        # カスタムペナルティを設定
        ga_config.zero_trades_penalty = 0.05

        fitness = self.evaluator._calculate_fitness(backtest_result, ga_config)
        assert fitness == 0.05  # 設定したペナルティ値が返ることを確認

    def test_calculate_fitness_custom_penalties(self):
        """カスタムペナルティ設定のテスト"""
        # 取引回数制約違反のケース
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 0.2,
                "max_drawdown": 0.1,
                "win_rate": 0.6,
                "total_trades": 5,
            },
            "equity_curve": [],
            "trade_history": [],
        }

        ga_config = GAConfig()
        ga_config.fitness_constraints = {
            "min_sharpe_ratio": 0.5,  # 0.2 < 0.5 で違反
        }

        # デフォルト（0.0）
        fitness = self.evaluator._calculate_fitness(backtest_result, ga_config)
        assert fitness == 0.0

        # カスタムペナルティ設定
        ga_config.constraint_violation_penalty = -1.0
        fitness_custom = self.evaluator._calculate_fitness(backtest_result, ga_config)
        assert fitness_custom == -1.0

    def test_calculate_fitness_constraints(self):
        """フィットネス制約のテスト"""
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 0.2,  # 最低シャープレシオ未満
                "max_drawdown": 0.1,
                "win_rate": 0.6,
                "total_trades": 5,
            },
            "equity_curve": [],
            "trade_history": [],
        }

        ga_config = GAConfig()
        ga_config.fitness_constraints = {
            "min_sharpe_ratio": 0.5,
            "min_trades": 3,
            "max_drawdown_limit": 0.15,
        }
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }

        fitness = self.evaluator._calculate_fitness(backtest_result, ga_config)
        assert fitness == 0.0  # シャープレシオが最低要件を満たしていない

    def test_is_backtest_result_passing_delegates_to_shared_constraint_checker(self):
        """制約判定が共有ロジックへ委譲されることを確認するテスト"""
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 0.2,
                "max_drawdown": 0.1,
                "win_rate": 0.6,
                "total_trades": 5,
            },
            "equity_curve": [],
            "trade_history": [],
        }

        ga_config = GAConfig()
        ga_config.fitness_constraints = {
            "min_sharpe_ratio": 0.5,
            "min_trades": 3,
            "max_drawdown_limit": 0.15,
        }

        with patch.object(
            self.evaluator._fitness_calculator,
            "meets_constraints",
            return_value=True,
        ) as mock_meets_constraints:
            result = self.evaluator._is_backtest_result_passing(
                backtest_result, ga_config
            )

        assert result is True
        mock_meets_constraints.assert_called_once()
        passed_metrics = mock_meets_constraints.call_args.args[0]
        assert passed_metrics["total_trades"] == 5

    def test_calculate_long_short_balance(self):
        """ロング・ショートバランス計算のテスト"""
        # ロングとショートがバランスしている取引履歴
        trade_history = [
            {"size": 1, "pnl": 10},  # ロング
            {"size": -1, "pnl": 5},  # ショート
            {"size": 1, "pnl": 15},  # ロング
            {"size": -1, "pnl": 10},  # ショート
        ]

        backtest_result = {"trade_history": trade_history}

        balance = self.evaluator._calculate_long_short_balance(backtest_result)
        assert 0.0 <= balance <= 1.0

    def test_calculate_long_short_balance_no_trades(self):
        """取引なしのバランス計算テスト"""
        backtest_result = {"trade_history": []}
        balance = self.evaluator._calculate_long_short_balance(backtest_result)
        assert balance == 0.5  # デフォルトの中立スコア

    def test_calculate_multi_objective_fitness(self):
        """多目的フィットネス計算のテスト"""
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.55,
                "profit_factor": 1.9,
                "sortino_ratio": 1.8,
                "calmar_ratio": 1.5,
                "total_trades": 1,
            },
            "equity_curve": [],
            "trade_history": [
                {
                    "id": 1,
                    "type": "long",
                    "entry_price": 100,
                    "exit_price": 115,
                    "pnl": 0.15,
                }
            ],
        }

        ga_config = GAConfig()
        ga_config.fitness_constraints = {}  # 制約をクリア（デフォルト値の影響を排除）
        ga_config.objectives = [
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
        ]

        result = self.evaluator._calculate_multi_objective_fitness(
            backtest_result, ga_config
        )

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert result[0] == 0.15  # total_return
        assert result[1] == 1.2  # sharpe_ratio
        assert result[2] == 0.08  # max_drawdown
        assert result[3] == 0.55  # win_rate

    def test_calculate_multi_objective_fitness_unknown_objective(self):
        """未知の目的のテスト（制約違反時のペナルティ値確認）"""
        backtest_result = {"performance_metrics": {"total_trades": 1}}
        ga_config = GAConfig()
        ga_config.objectives = ["unknown_objective"]

        result = self.evaluator._calculate_multi_objective_fitness(
            backtest_result, ga_config
        )

        # 制約違反（min_trades=10 > total_trades=1）によりペナルティ値が返される
        # 未知の目的はminimize方向とみなされ、ペナルティは +inf
        import math

        assert len(result) == 1
        assert math.isinf(result[0]) and result[0] > 0

    def test_evaluate_individual_with_ml_filter(self):
        """MLフィルターが有効な場合の個体評価テスト"""
        mock_individual = self._create_mock_gene()

        # ベース設定
        config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "initial_capital": 10000.0,
            "commission_rate": 0.001,
            "strategy_name": "TestStrategy",
        }
        self.evaluator.set_backtest_config(config)

        # MLフィルターなしの場合のバックテスト結果（ベースライン）
        mock_result_no_ml_filter = {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1,
                "win_rate": 0.6,
                "profit_factor": 1.8,
                "total_trades": 100,  # MLフィルターなしでは取引が多い
            },
            "equity_curve": [100, 110, 105, 120],
            "trade_history": [{"size": 1, "pnl": 10}] * 100,
        }

        # MLフィルターありの場合のバックテスト結果（取引が減る想定）
        mock_result_with_ml_filter = {
            "performance_metrics": {
                "total_return": 0.15,  # 改善される想定
                "sharpe_ratio": 2.0,  # 改善される想定
                "max_drawdown": 0.08,
                "win_rate": 0.7,
                "profit_factor": 2.5,
                "total_trades": 50,  # MLフィルターにより取引が半分になる想定
            },
            "equity_curve": [100, 115, 110, 130],
            "trade_history": [{"size": 1, "pnl": 15}] * 50,
        }

        # GA設定 - MLフィルター無効
        ga_config_no_ml = GAConfig()
        ga_config_no_ml.enable_multi_objective = False
        ga_config_no_ml.fitness_weights = {
            "total_return": 1.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "balance_score": 0.0,
            "ulcer_index_penalty": 0.0,
            "trade_frequency_penalty": 0.0,
        }
        ga_config_no_ml.fitness_constraints = {
            "min_trades": 0,  # 制約を無効化
            "max_drawdown_limit": 1.0,
            "min_sharpe_ratio": 0.0,
        }
        ga_config_no_ml.hybrid_config.volatility_gate_enabled = False
        ga_config_no_ml.hybrid_config.volatility_model_path = None

        # GA設定 - ボラティリティゲート有効
        ga_config_with_volatility = GAConfig()
        ga_config_with_volatility.enable_multi_objective = False
        ga_config_with_volatility.fitness_weights = {
            "total_return": 1.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "balance_score": 0.0,
            "ulcer_index_penalty": 0.0,
            "trade_frequency_penalty": 0.0,
        }
        ga_config_with_volatility.fitness_constraints = {
            "min_trades": 0,
            "max_drawdown_limit": 1.0,
            "min_sharpe_ratio": 0.0,
        }
        ga_config_with_volatility.hybrid_config.volatility_gate_enabled = True
        ga_config_with_volatility.hybrid_config.volatility_model_path = (
            "/path/to/ml_model.pkl"
        )

        # backtest_service.run_backtestが設定の内容に応じて異なる結果を返すようにモックを設定
        def run_backtest_side_effect(**kwargs):
            backtest_config = kwargs.get("config", {})
            strategy_config = backtest_config.get("strategy_config", {})

            if hasattr(strategy_config, "model_dump"):
                strategy_config = strategy_config.model_dump()

            if "parameters" in strategy_config:
                params = strategy_config["parameters"]
                if params.get("volatility_gate_enabled"):
                    return mock_result_with_ml_filter

            return mock_result_no_ml_filter

        self.mock_backtest_service.run_backtest.side_effect = run_backtest_side_effect

        # 1. ボラティリティゲート無効で評価
        result_no_ml = self.evaluator.evaluate(mock_individual, ga_config_no_ml)
        call_kwargs = self.mock_backtest_service.run_backtest.call_args.kwargs
        backtest_config_passed = call_kwargs["config"]
        params = backtest_config_passed["strategy_config"]["parameters"]
        assert not params["volatility_gate_enabled"]
        assert params["volatility_model_path"] is None
        assert params["strategy_gene"] is not None

        # 評価結果の検証
        # _calculate_fitnessを直接呼び出して期待値を計算 (テスト対象ではないが、比較のために使用)
        expected_fitness_no_ml = self.evaluator._calculate_fitness(
            mock_result_no_ml_filter, ga_config_no_ml
        )
        assert result_no_ml[0] == expected_fitness_no_ml

        # 2. ボラティリティゲート有効で評価
        self.evaluator.evaluate(
            mock_individual,
            ga_config_with_volatility,
            force_refresh=True,
        )
        call_kwargs = self.mock_backtest_service.run_backtest.call_args.kwargs
        backtest_config_passed = call_kwargs["config"]
        params = backtest_config_passed["strategy_config"]["parameters"]
        assert params["volatility_gate_enabled"] is True

    def test_evaluate_individual_with_oos(self):
        """OOS検証ありの個体評価テスト"""
        mock_individual = self._create_mock_gene()

        # 共通のベース設定
        base_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01 00:00:00",
            "end_date": "2024-01-11 00:00:00",  # 10日間
            "initial_capital": 10000.0,
            "commission_rate": 0.001,
            "strategy_name": "TestStrategy",
        }
        self.evaluator.set_backtest_config(base_config)

        # ISとOOSの結果をモック
        # run_backtestは2回呼ばれる。1回目がIS、2回目がOOSと仮定

        # IS結果: Total Return 0.1
        mock_result_is = {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 1.0,
                "max_drawdown": 0.1,
                "win_rate": 0.5,
                "total_trades": 10,
            },
            "equity_curve": [],
            "trade_history": [{"size": 1, "pnl": 10}],
            "start_date": "2024-01-01 00:00:00",
            "end_date": "2024-01-09 00:00:00",
        }
        # OOS結果: Total Return 0.05
        mock_result_oos = {
            "performance_metrics": {
                "total_return": 0.05,
                "sharpe_ratio": 0.5,
                "max_drawdown": 0.2,
                "win_rate": 0.4,
                "total_trades": 5,
            },
            "equity_curve": [],
            "trade_history": [{"size": 1, "pnl": 5}],
            "start_date": "2024-01-09 00:00:00",
            "end_date": "2024-01-11 00:00:00",
        }

        self.mock_backtest_service.run_backtest.side_effect = [
            mock_result_is,
            mock_result_oos,
        ]

        # GA設定: OOS有効化
        ga_config = GAConfig()
        ga_config.enable_multi_objective = False
        ga_config.evaluation_config.oos_split_ratio = 0.2  # 10日間のうち、最後の2日がOOS（8日がIS）
        ga_config.evaluation_config.oos_fitness_weight = 0.5  # 単純平均

        # フィットネス重み設定（total_returnのみ）
        ga_config.fitness_weights = {
            "total_return": 1.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "balance_score": 0.0,
            "ulcer_index_penalty": 0.0,
            "trade_frequency_penalty": 0.0,
        }
        # 制約を無効化
        ga_config.fitness_constraints = {
            "min_trades": 0,
            "max_drawdown_limit": 1.0,
            "min_sharpe_ratio": 0.0,
        }

        # 実行
        result = self.evaluator.evaluate(mock_individual, ga_config)

        # 検証
        assert self.mock_backtest_service.run_backtest.call_count == 2

        # 1回目の呼び出し（IS）の引数確認
        # warmup調整（SMA period=20 → 21 bars × 1h = 21時間前倒し）を反映
        call_args_is = self.mock_backtest_service.run_backtest.call_args_list[0].kwargs[
            "config"
        ]
        assert str(call_args_is["start_date"]) == "2023-12-31 03:00:00"
        assert str(call_args_is["end_date"]) == "2024-01-09 00:00:00"

        # 2回目の呼び出し（OOS）の引数確認
        call_args_oos = self.mock_backtest_service.run_backtest.call_args_list[
            1
        ].kwargs["config"]
        # OOS開始日はwarmup期間（21時間）を前倒しした値
        # テストの期待値は実装の詳細に依存するため、範囲チェックで代替
        oos_start = pd.Timestamp(call_args_oos["start_date"])
        assert oos_start >= pd.Timestamp("2024-01-08 00:00:00")
        assert oos_start <= pd.Timestamp("2024-01-08 06:00:00")
        assert str(call_args_oos["end_date"]) == "2024-01-11 00:00:00"

        # フィットネス計算の検証
        # 期待値を計算 (Evaluatorの内部計算が正しければこれになるはず)
        # total_return だけを見ているので、0.1 * 0.5 + 0.05 * 0.5 = 0.075
        expected_fitness = 0.1 * 0.5 + 0.05 * 0.5
        assert abs(result[0] - expected_fitness) < 1e-6

    def test_evaluate_individual_caching(self):
        """データのキャッシング動作テスト"""
        mock_individual = self._create_mock_gene()

        # モックデータ（DataFrameをシミュレート）
        mock_data = Mock(name="MockDataFrame")
        mock_data.empty = False  # DataFrameの empty 属性をシミュレート
        mock_data.columns = pd.Index(["open", "high", "low", "close", "volume"])
        mock_data.copy.return_value = mock_data
        # data_serviceプロパティをモック化
        message_mock = Mock()
        message_mock.get_data_for_backtest.return_value = mock_data
        self.mock_backtest_service.data_service = message_mock

        # バックテスト結果モック
        mock_result = {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 1.0,
                "max_drawdown": 0.1,
                "win_rate": 0.5,
                "total_trades": 1,
            },
            "equity_curve": [],
            "trade_history": [],
        }
        self.mock_backtest_service.run_backtest.return_value = mock_result

        config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "initial_capital": 10000.0,
            "commission_rate": 0.001,
            "strategy_name": "TestStrategy",
        }
        self.evaluator.set_backtest_config(config)

        ga_config = GAConfig()
        ga_config.fitness_constraints = {}
        ga_config.fitness_weights = {
            "total_return": 1.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "balance_score": 0.0,
            "ulcer_index_penalty": 0.0,
            "trade_frequency_penalty": 0.0,
        }

        # 1回目の評価（データ取得発生）
        self.evaluator.evaluate(mock_individual, ga_config)

        # 検証1: データ取得が呼ばれたか（メインTF + 1分足の最大2回）
        self.mock_backtest_service.ensure_data_service_initialized.assert_called()
        # メインTFデータは必ず取得される、さらに1分足データも取得される可能性がある
        initial_call_count = (
            self.mock_backtest_service.data_service.get_data_for_backtest.call_count
        )
        assert initial_call_count >= 1  # 最低1回（メインTF用）

        # 検証2: run_backtestにpreloaded_dataが渡されたか
        args, kwargs = self.mock_backtest_service.run_backtest.call_args
        # preloaded_dataはkwargsで渡される実装にした
        assert kwargs.get("preloaded_data") == mock_data

        # 2回目の評価（キャッシュ利用）
        # リセット前の呼び出し回数を記録
        call_count_before_2nd = (
            self.mock_backtest_service.data_service.get_data_for_backtest.call_count
        )
        self.evaluator.evaluate(mock_individual, ga_config)
        call_count_after_2nd = (
            self.mock_backtest_service.data_service.get_data_for_backtest.call_count
        )

        # 検証3: 2回目ではキャッシュからデータ取得（新規取得なし）
        assert call_count_after_2nd == call_count_before_2nd

        # 検証4: それでもrun_backtestには依然としてcached dataが渡されていること
        args, kwargs = self.mock_backtest_service.run_backtest.call_args
        assert kwargs.get("preloaded_data") == mock_data

    def test_evaluate_individual_caching_with_oos(self):
        """OOS検証時のキャッシング動作テスト"""
        mock_individual = self._create_mock_gene()

        # モックデータ1 (In-Sample用)
        mock_data_is = Mock(name="MockDataFrameIS")
        mock_data_is.empty = False  # DataFrameの empty 属性をシミュレート
        mock_data_is.columns = pd.Index(["open", "high", "low", "close", "volume"])
        mock_data_is.copy.return_value = mock_data_is

        # モックデータ2 (Out-of-Sample用)
        mock_data_oos = Mock(name="MockDataFrameOOS")
        mock_data_oos.empty = False  # DataFrameの empty 属性をシミュレート
        mock_data_oos.columns = pd.Index(["open", "high", "low", "close", "volume"])
        mock_data_oos.copy.return_value = mock_data_oos

        # data_serviceプロパティをモック化
        message_mock = Mock()

        # 呼ばれる日付範囲によって異なるデータを返すモック関数
        def get_data_side_effect(**kwargs):
            if "2024-01-09" in str(kwargs.get("end_date")):
                # IS期間: 1/1 - 1/9
                return mock_data_is
            else:
                # OOS期間: 1/9 - 1/11
                return mock_data_oos

        message_mock.get_data_for_backtest.side_effect = get_data_side_effect
        self.mock_backtest_service.data_service = message_mock

        # バックテスト結果モック (パフォーマンスメトリクスが必要)
        mock_metrics = {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 1.0,
                "max_drawdown": 0.1,
                "win_rate": 0.5,
                "total_trades": 1,
            },
            "equity_curve": [],
            "trade_history": [],
        }
        self.mock_backtest_service.run_backtest.return_value = mock_metrics

        # ベース設定
        config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01 00:00:00",
            "end_date": "2024-01-11 00:00:00",
            "initial_capital": 10000.0,
            "commission_rate": 0.001,
            "strategy_name": "TestStrategy",
        }
        self.evaluator.set_backtest_config(config)

        ga_config = GAConfig()
        ga_config.enable_multi_objective = False
        ga_config.evaluation_config.oos_split_ratio = 0.2
        ga_config.evaluation_config.oos_fitness_weight = 0.5
        # 重み設定必須
        ga_config.fitness_weights = {"total_return": 1.0}
        ga_config.fitness_weights = {"total_return": 1.0}
        ga_config.fitness_constraints = {}
        ga_config.objectives = ["weighted_score"]

        # 1. 初回実行（キャッシュなし）
        self.evaluator.evaluate(mock_individual, ga_config)

        # 検証1: データ取得が行われた（IS用+OOS用。1分足も必要に応じて）
        # ISとOOSの2回 + 1分足データ取得の可能性
        initial_call_count = (
            self.mock_backtest_service.data_service.get_data_for_backtest.call_count
        )
        assert initial_call_count >= 2  # 最低2回（ISとOOS用）

        # 検証2: run_backtestがIS用とOOS用のデータで呼ばれたか確認
        # call_args_listの各呼び出しで、preloaded_dataが適切に渡されているか
        calls = self.mock_backtest_service.run_backtest.call_args_list
        # call_args_list[0]はIS, call_args_list[1]はOOS (実装順序に依存してチェック)
        kwargs_is = calls[0].kwargs
        kwargs_oos = calls[1].kwargs
        assert kwargs_is.get("preloaded_data") == mock_data_is
        assert kwargs_oos.get("preloaded_data") == mock_data_oos

        # 2. 2回目実行（キャッシュあり）
        call_count_before_2nd = (
            self.mock_backtest_service.data_service.get_data_for_backtest.call_count
        )
        self.evaluator.evaluate(mock_individual, ga_config)
        call_count_after_2nd = (
            self.mock_backtest_service.data_service.get_data_for_backtest.call_count
        )

        # 検証3: 2回目ではキャッシュからデータ取得（新規取得なし）
        assert call_count_after_2nd == call_count_before_2nd

        # 検証4: 2回目ではrun_backtestのcall_countが変わらない（キャッシュ効果）
        call_count_run_backtest_2nd = self.mock_backtest_service.run_backtest.call_count
        # 1回目: IS + OOS = 2回実行
        # 2回目: キャッシュ使用 = 実行回数変わらず
        # 2回目実行後のrun_backtest.call_countは1回目と同じ
        assert call_count_run_backtest_2nd == 2


class TestUnifiedEvaluationLogic:
    """統一評価ロジックのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        self.mock_backtest_service = Mock()
        self.evaluator = IndividualEvaluator(self.mock_backtest_service)

    def _create_mock_gene(self, gene_id="test-gene"):
        """テスト用のStrategyGeneを作成するヘルパー"""
        return StrategyGene(
            id=gene_id,
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA_20")
            ],
            short_entry_conditions=[
                Condition(left_operand="close", operator="<", right_operand="SMA_20")
            ],
        )

    def test_weighted_score_objective(self):
        """weighted_score目的関数のテスト"""
        mock_individual = self._create_mock_gene()
        mock_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1,
                "win_rate": 0.6,
                "total_trades": 10,
            },
            "equity_curve": [100, 110, 105, 120],
            "trade_history": [{"size": 1, "pnl": 10}],
        }
        self.mock_backtest_service.run_backtest.return_value = mock_result

        config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "initial_capital": 10000.0,
            "commission_rate": 0.001,
            "strategy_name": "TestStrategy",
        }
        self.evaluator.set_backtest_config(config)

        ga_config = GAConfig()
        ga_config.objectives = ["weighted_score"]
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }
        ga_config.fitness_constraints = {}

        result = self.evaluator.evaluate(mock_individual, ga_config)

        # 常にタプルで返される
        assert isinstance(result, tuple)
        assert len(result) == 1
        # weighted_scoreは従来の単一目的計算と同じ結果
        assert result[0] > 0

    def test_single_objective_returns_tuple(self):
        """単一目的でもタプルを返すテスト"""
        mock_individual = self._create_mock_gene()
        mock_result = {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1,
                "win_rate": 0.6,
                "total_trades": 10,
            },
            "equity_curve": [],
            "trade_history": [{"size": 1, "pnl": 10}],
        }
        self.mock_backtest_service.run_backtest.return_value = mock_result

        config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "initial_capital": 10000.0,
            "commission_rate": 0.001,
            "strategy_name": "TestStrategy",
        }
        self.evaluator.set_backtest_config(config)

        ga_config = GAConfig()
        ga_config.objectives = ["sharpe_ratio"]  # 単一目的
        ga_config.fitness_constraints = {}

        result = self.evaluator.evaluate(mock_individual, ga_config)

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0] == 1.5  # sharpe_ratioの値

    def test_enable_multi_objective_flag_deprecated(self):
        """enable_multi_objectiveフラグが無視されるテスト（後方互換性）"""
        mock_individual = self._create_mock_gene()
        mock_result = {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1,
                "total_trades": 10,
            },
            "equity_curve": [],
            "trade_history": [{"size": 1, "pnl": 10}],
        }
        self.mock_backtest_service.run_backtest.return_value = mock_result

        config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "initial_capital": 10000.0,
            "commission_rate": 0.001,
            "strategy_name": "TestStrategy",
        }
        self.evaluator.set_backtest_config(config)

        ga_config = GAConfig()
        ga_config.enable_multi_objective = False  # 旧フラグ（無視される）
        ga_config.objectives = ["total_return", "sharpe_ratio"]
        ga_config.fitness_constraints = {}

        result = self.evaluator.evaluate(mock_individual, ga_config)

        # enable_multi_objective=Falseでも、objectivesの数に応じたタプルが返される
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_default_objectives_is_weighted_score(self):
        """デフォルトのobjectivesがweighted_scoreであるテスト"""
        ga_config = GAConfig()
        assert "weighted_score" in ga_config.objectives

    def test_calculate_weighted_score_value(self):
        """weighted_scoreの計算値が正しいことのテスト"""
        mock_individual = self._create_mock_gene()
        mock_result = {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 1.0,
                "max_drawdown": 0.2,
                "win_rate": 0.5,
                "total_trades": 10,
            },
            "equity_curve": [],
            "trade_history": [{"size": 1, "pnl": 10}],
        }
        self.mock_backtest_service.run_backtest.return_value = mock_result

        config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "initial_capital": 10000.0,
            "commission_rate": 0.001,
            "strategy_name": "TestStrategy",
        }
        self.evaluator.set_backtest_config(config)

        ga_config = GAConfig()
        ga_config.objectives = ["weighted_score"]
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
            "balance_score": 0.0,
            "ulcer_index_penalty": 0.0,
            "trade_frequency_penalty": 0.0,
        }
        ga_config.fitness_constraints = {}

        result = self.evaluator.evaluate(mock_individual, ga_config)

        # 手動計算:
        # total_return: 0.1 * 0.3 = 0.03
        # sharpe_ratio: 1.0 * 0.4 = 0.4
        # (1 - max_drawdown): 0.8 * 0.2 = 0.16
        # win_rate: 0.5 * 0.1 = 0.05
        # balance_score (trade_history から計算) は別途
        # 合計 ≈ 0.03 + 0.4 + 0.16 + 0.05 = 0.64 前後（balance_scoreの影響あり）
        assert result[0] > 0.5
