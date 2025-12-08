"""
IndividualEvaluatorのテスト
"""

from unittest.mock import Mock

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.core.individual_evaluator import IndividualEvaluator


class TestIndividualEvaluator:
    """IndividualEvaluatorのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        self.mock_backtest_service = Mock()
        self.evaluator = IndividualEvaluator(self.mock_backtest_service)

    def test_init(self):
        """初期化のテスト"""
        assert self.evaluator.backtest_service == self.mock_backtest_service
        assert self.evaluator._fixed_backtest_config is None

    def test_set_backtest_config(self):
        """バックテスト設定のテスト"""
        config = {"symbol": "BTC/USDT:USDT", "timeframe": "1h"}
        self.evaluator.set_backtest_config(config)
        assert self.evaluator._fixed_backtest_config == config

    def test_evaluate_individual_success(self):
        """個体評価成功のテスト"""
        # モック設定
        mock_individual = [1, 2, 3, 4, 5]
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
        result = self.evaluator.evaluate_individual(mock_individual, ga_config)

        assert isinstance(result, tuple)
        assert len(result) == 1  # 単一目的最適化

    def test_evaluate_individual_multi_objective(self):
        """多目的最適化評価のテスト"""
        mock_individual = [1, 2, 3, 4, 5]
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

        result = self.evaluator.evaluate_individual(mock_individual, ga_config)

        assert isinstance(result, tuple)
        assert len(result) == 3  # 3つの目的

    def test_evaluate_individual_exception(self):
        """個体評価例外のテスト"""
        mock_individual = [1, 2, 3, 4, 5]

        # バックテストで例外が発生
        self.mock_backtest_service.run_backtest.side_effect = Exception("Test error")

        ga_config = GAConfig()
        ga_config.enable_multi_objective = False

        result = self.evaluator.evaluate_individual(mock_individual, ga_config)

        assert result == (0.0,)

    def test_evaluate_individual_multi_objective_exception(self):
        """多目的最適化例外のテスト"""
        mock_individual = [1, 2, 3, 4, 5]

        self.mock_backtest_service.run_backtest.side_effect = Exception("Test error")

        ga_config = GAConfig()
        ga_config.enable_multi_objective = True
        ga_config.objectives = ["total_return", "sharpe_ratio"]

        result = self.evaluator.evaluate_individual(mock_individual, ga_config)

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

        fitness = self.evaluator._calculate_fitness(backtest_result, ga_config)
        assert fitness == 0.1  # 取引回数0の特別処理

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
        """未知の目的のテスト"""
        backtest_result = {"performance_metrics": {"total_trades": 1}}
        ga_config = GAConfig()
        ga_config.objectives = ["unknown_objective"]

        result = self.evaluator._calculate_multi_objective_fitness(
            backtest_result, ga_config
        )

        assert result == (0.0,)  # 未知の目的は0.0

    def test_evaluate_individual_with_ml_filter(self):
        """MLフィルターが有効な場合の個体評価テスト"""
        mock_individual = [1, 2, 3, 4, 5]

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
        ga_config_no_ml.ml_filter_enabled = False
        ga_config_no_ml.ml_model_path = None

        # GA設定 - MLフィルター有効
        ga_config_with_ml = GAConfig()
        ga_config_with_ml.enable_multi_objective = False
        ga_config_with_ml.fitness_weights = {
            "total_return": 1.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "balance_score": 0.0,
            "ulcer_index_penalty": 0.0,
            "trade_frequency_penalty": 0.0,
        }
        ga_config_with_ml.fitness_constraints = {
            "min_trades": 0,  # 制約を無効化
            "max_drawdown_limit": 1.0,
            "min_sharpe_ratio": 0.0,
        }
        ga_config_with_ml.ml_filter_enabled = True
        ga_config_with_ml.ml_model_path = "/path/to/ml_model.pkl"  # 仮のパス

        # backtest_service.run_backtestがGAconfigの内容に応じて異なる結果を返すようにモックを設定
        def run_backtest_side_effect(**kwargs):
            backtest_config = kwargs.get(
                "backtest_config", {}
            )  # backtest_configをkwargsから取得
            strategy_config = backtest_config.get(
                "strategy_config", {}
            )  # strategy_configはbacktest_configの中にある
            if strategy_config.get("ml_filter_enabled"):
                return mock_result_with_ml_filter
            return mock_result_no_ml_filter

        self.mock_backtest_service.run_backtest.side_effect = run_backtest_side_effect

        # 1. MLフィルター無効で評価
        result_no_ml = self.evaluator.evaluate_individual(
            mock_individual, ga_config_no_ml
        )
        # run_backtestがMLフィルターなしの引数で呼ばれたことを検証
        # call_args.kwargsからbacktest_configを取り出し、その中のstrategy_configをチェック
        call_kwargs = self.mock_backtest_service.run_backtest.call_args.kwargs
        backtest_config_passed = call_kwargs["backtest_config"]
        assert not backtest_config_passed["strategy_config"]["ml_filter_enabled"]
        assert backtest_config_passed["strategy_config"]["ml_model_path"] is None
        assert (
            backtest_config_passed["strategy_config"]["parameters"]["strategy_gene"]
            is not None
        )
        assert backtest_config_passed.get("regime_labels") is None

        # 評価結果の検証
        # _calculate_fitnessを直接呼び出して期待値を計算 (テスト対象ではないが、比較のために使用)
        expected_fitness_no_ml = self.evaluator._calculate_fitness(
            mock_result_no_ml_filter, ga_config_no_ml
        )
        assert result_no_ml[0] == expected_fitness_no_ml

        # 2. MLフィルター有効で評価
        result_with_ml = self.evaluator.evaluate_individual(
            mock_individual, ga_config_with_ml
        )
        # run_backtestがMLフィルターありの引数で呼ばれたことを検証
        call_kwargs = self.mock_backtest_service.run_backtest.call_args.kwargs
        backtest_config_passed = call_kwargs["backtest_config"]
        assert backtest_config_passed["strategy_config"]["ml_filter_enabled"]
        assert (
            backtest_config_passed["strategy_config"]["ml_model_path"]
            == "/path/to/ml_model.pkl"
        )
        assert (
            backtest_config_passed["strategy_config"]["parameters"]["strategy_gene"]
            is not None
        )
        assert backtest_config_passed.get("regime_labels") is None

        # 評価結果の検証
        expected_fitness_with_ml = self.evaluator._calculate_fitness(
            mock_result_with_ml_filter, ga_config_with_ml
        )
        assert result_with_ml[0] == expected_fitness_with_ml

        # MLフィルターによって結果が改善されたことを検証 (total_returnが改善)
        assert result_with_ml[0] > result_no_ml[0]

    def test_evaluate_individual_with_oos(self):
        """OOS検証ありの個体評価テスト"""
        mock_individual = [1, 2, 3, 4, 5]

        # 共通のベース設定
        base_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01 00:00:00",
            "end_date": "2024-01-11 00:00:00",  # 10日間
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
        ga_config.oos_split_ratio = 0.2  # 10日間のうち、最後の2日がOOS（8日がIS）
        ga_config.oos_fitness_weight = 0.5  # 単純平均

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
        result = self.evaluator.evaluate_individual(mock_individual, ga_config)

        # 検証
        assert self.mock_backtest_service.run_backtest.call_count == 2

        # 1回目の呼び出し（IS）の引数確認
        call_args_is = self.mock_backtest_service.run_backtest.call_args_list[0].kwargs[
            "backtest_config"
        ]
        assert call_args_is["start_date"] == "2024-01-01 00:00:00"
        # 10日 * 0.8 = 8日後 = 1月9日
        assert call_args_is["end_date"] == "2024-01-09 00:00:00"

        # 2回目の呼び出し（OOS）の引数確認
        call_args_oos = self.mock_backtest_service.run_backtest.call_args_list[
            1
        ].kwargs["backtest_config"]
        assert call_args_oos["start_date"] == "2024-01-09 00:00:00"
        assert call_args_oos["end_date"] == "2024-01-11 00:00:00"

        # フィットネス計算の検証
        # 期待値を計算 (Evaluatorの内部計算が正しければこれになるはず)
        # total_return だけを見ているので、0.1 * 0.5 + 0.05 * 0.5 = 0.075
        expected_fitness = 0.1 * 0.5 + 0.05 * 0.5
        assert abs(result[0] - expected_fitness) < 1e-6


class TestUnifiedEvaluationLogic:
    """統一評価ロジックのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        self.mock_backtest_service = Mock()
        self.evaluator = IndividualEvaluator(self.mock_backtest_service)

    def test_weighted_score_objective(self):
        """weighted_score目的関数のテスト"""
        mock_individual = [1, 2, 3, 4, 5]
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

        ga_config = GAConfig()
        ga_config.objectives = ["weighted_score"]
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }
        ga_config.fitness_constraints = {}

        result = self.evaluator.evaluate_individual(mock_individual, ga_config)

        # 常にタプルで返される
        assert isinstance(result, tuple)
        assert len(result) == 1
        # weighted_scoreは従来の単一目的計算と同じ結果
        assert result[0] > 0

    def test_single_objective_returns_tuple(self):
        """単一目的でもタプルを返すテスト"""
        mock_individual = [1, 2, 3, 4, 5]
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

        ga_config = GAConfig()
        ga_config.objectives = ["sharpe_ratio"]  # 単一目的
        ga_config.fitness_constraints = {}

        result = self.evaluator.evaluate_individual(mock_individual, ga_config)

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0] == 1.5  # sharpe_ratioの値

    def test_enable_multi_objective_flag_deprecated(self):
        """enable_multi_objectiveフラグが無視されるテスト（後方互換性）"""
        mock_individual = [1, 2, 3, 4, 5]
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

        ga_config = GAConfig()
        ga_config.enable_multi_objective = False  # 旧フラグ（無視される）
        ga_config.objectives = ["total_return", "sharpe_ratio"]
        ga_config.fitness_constraints = {}

        result = self.evaluator.evaluate_individual(mock_individual, ga_config)

        # enable_multi_objective=Falseでも、objectivesの数に応じたタプルが返される
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_default_objectives_is_weighted_score(self):
        """デフォルトのobjectivesがweighted_scoreであるテスト"""
        ga_config = GAConfig()
        assert "weighted_score" in ga_config.objectives

    def test_calculate_weighted_score_value(self):
        """weighted_scoreの計算値が正しいことのテスト"""
        mock_individual = [1, 2, 3, 4, 5]
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

        result = self.evaluator.evaluate_individual(mock_individual, ga_config)

        # 手動計算:
        # total_return: 0.1 * 0.3 = 0.03
        # sharpe_ratio: 1.0 * 0.4 = 0.4
        # (1 - max_drawdown): 0.8 * 0.2 = 0.16
        # win_rate: 0.5 * 0.1 = 0.05
        # balance_score (trade_history から計算) は別途
        # 合計 ≈ 0.03 + 0.4 + 0.16 + 0.05 = 0.64 前後（balance_scoreの影響あり）
        assert result[0] > 0.5
