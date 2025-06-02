"""
拡張バックテストサービスのユニットテスト

EnhancedBacktestServiceクラスの機能をテストします。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from app.core.services.enhanced_backtest_service import EnhancedBacktestService
from app.core.services.backtest_data_service import BacktestDataService


class TestEnhancedBacktestService:
    """拡張バックテストサービスのテスト"""

    @pytest.fixture
    def mock_data_service(self):
        """モックデータサービス"""
        return Mock(spec=BacktestDataService)

    @pytest.fixture
    def enhanced_service(self, mock_data_service):
        """拡張バックテストサービス"""
        return EnhancedBacktestService(data_service=mock_data_service)

    @pytest.fixture
    def sample_config(self):
        """サンプル設定"""
        return {
            "strategy_name": "SMA_CROSS_OPTIMIZED",
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "end_date": datetime(2024, 12, 31, tzinfo=timezone.utc),
            "initial_capital": 100000,
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50},
            },
        }

    @pytest.fixture
    def sample_optimization_params(self):
        """サンプル最適化パラメータ"""
        return {
            "method": "sambo",
            "max_tries": 50,
            "maximize": "Sharpe Ratio",
            "return_heatmap": True,
            "return_optimization": True,
            "random_state": 42,
            "constraint": "sma_cross",
            "parameters": {"n1": range(5, 30, 5), "n2": range(20, 100, 10)},
        }

    @pytest.fixture
    def sample_ohlcv_dataframe(self):
        """サンプルOHLCVデータフレーム"""
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        np.random.seed(42)

        # 現実的な価格データを生成
        base_price = 50000
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        df = pd.DataFrame(
            {
                "Open": prices,
                "High": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                "Low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                "Close": prices,
                "Volume": np.random.randint(1000, 10000, len(dates)),
            },
            index=dates,
        )

        # High >= max(Open, Close), Low <= min(Open, Close) を保証
        df["High"] = np.maximum(df["High"], np.maximum(df["Open"], df["Close"]))
        df["Low"] = np.minimum(df["Low"], np.minimum(df["Open"], df["Close"]))

        return df

    def test_initialization(self, enhanced_service):
        """初期化テスト"""
        assert enhanced_service is not None
        assert hasattr(enhanced_service, "constraint_functions")
        assert "sma_cross" in enhanced_service.constraint_functions
        assert "rsi" in enhanced_service.constraint_functions
        assert "macd" in enhanced_service.constraint_functions
        assert "risk_management" in enhanced_service.constraint_functions

    def test_constraint_functions(self, enhanced_service):
        """制約条件関数のテスト"""
        # SMAクロス制約のテスト
        sma_constraint = enhanced_service.constraint_functions["sma_cross"]

        # モックパラメータオブジェクト
        valid_params = Mock()
        valid_params.n1 = 10
        valid_params.n2 = 20

        invalid_params = Mock()
        invalid_params.n1 = 30
        invalid_params.n2 = 20

        assert sma_constraint(valid_params) is True
        assert sma_constraint(invalid_params) is False

    def test_validate_optimization_config_success(
        self, enhanced_service, sample_config, sample_optimization_params
    ):
        """最適化設定検証成功テスト"""
        # 例外が発生しないことを確認
        enhanced_service._validate_optimization_config(
            sample_config, sample_optimization_params
        )

    def test_validate_optimization_config_missing_parameters(
        self, enhanced_service, sample_config
    ):
        """最適化設定検証失敗テスト - パラメータ不足"""
        invalid_params = {"method": "sambo"}  # parametersが不足

        with pytest.raises(
            ValueError, match="Missing required optimization field: parameters"
        ):
            enhanced_service._validate_optimization_config(
                sample_config, invalid_params
            )

    def test_validate_optimization_config_empty_parameters(
        self, enhanced_service, sample_config
    ):
        """最適化設定検証失敗テスト - 空のパラメータ"""
        invalid_params = {"parameters": {}}  # 空のパラメータ

        with pytest.raises(
            ValueError, match="At least one parameter range must be specified"
        ):
            enhanced_service._validate_optimization_config(
                sample_config, invalid_params
            )

    def test_build_optimize_kwargs(self, enhanced_service, sample_optimization_params):
        """最適化パラメータ構築テスト"""
        kwargs = enhanced_service._build_optimize_kwargs(sample_optimization_params)

        assert kwargs["method"] == "sambo"
        assert kwargs["maximize"] == "Sharpe Ratio"
        assert kwargs["return_heatmap"] is True
        assert kwargs["return_optimization"] is True
        assert kwargs["max_tries"] == 50
        assert kwargs["random_state"] == 42
        assert "constraint" in kwargs
        assert kwargs["n1"] == sample_optimization_params["parameters"]["n1"]
        assert kwargs["n2"] == sample_optimization_params["parameters"]["n2"]

    def test_build_optimize_kwargs_with_callable_constraint(self, enhanced_service):
        """呼び出し可能制約条件での最適化パラメータ構築テスト"""

        def custom_constraint(params):
            return params.n1 < params.n2

        params = {
            "parameters": {"n1": range(5, 30), "n2": range(20, 100)},
            "constraint": custom_constraint,
        }

        kwargs = enhanced_service._build_optimize_kwargs(params)
        assert kwargs["constraint"] == custom_constraint

    def test_build_optimize_kwargs_invalid_constraint(self, enhanced_service):
        """無効な制約条件での最適化パラメータ構築テスト"""
        params = {
            "parameters": {"n1": range(5, 30), "n2": range(20, 100)},
            "constraint": "invalid_constraint",
        }

        with pytest.raises(ValueError, match="Unknown constraint: invalid_constraint"):
            enhanced_service._build_optimize_kwargs(params)

    def test_calculate_parameter_space_size(self, enhanced_service):
        """パラメータ空間サイズ計算テスト"""
        parameters = {"n1": range(5, 30, 5), "n2": range(20, 100, 10)}  # 5個  # 8個

        size = enhanced_service._calculate_parameter_space_size(parameters)
        assert size == 5 * 8  # 40

    def test_analyze_heatmap(self, enhanced_service):
        """ヒートマップ分析テスト"""
        # サンプルヒートマップデータ
        index = pd.MultiIndex.from_tuples(
            [(10, 20), (10, 30), (15, 20), (15, 30)], names=["n1", "n2"]
        )
        heatmap_data = pd.Series([0.5, 0.8, 0.3, 0.9], index=index)

        analysis = enhanced_service._analyze_heatmap(heatmap_data)

        assert analysis["best_combination"] == (15, 30)
        assert analysis["best_value"] == 0.9
        assert analysis["worst_combination"] == (15, 20)
        assert analysis["worst_value"] == 0.3
        assert analysis["total_combinations"] == 4
        assert "mean_value" in analysis
        assert "std_value" in analysis

    def test_calculate_convergence_rate(self, enhanced_service):
        """収束率計算テスト"""
        # 改善傾向のある関数値
        func_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        rate = enhanced_service._calculate_convergence_rate(func_vals)
        assert rate > 0  # 改善傾向なので正の値

        # データが少ない場合
        short_vals = [0.1, 0.2]
        rate_short = enhanced_service._calculate_convergence_rate(short_vals)
        assert rate_short == 0.0

    def test_detect_plateau(self, enhanced_service):
        """プラトー検出テスト"""
        # プラトー状態（変動が小さい）
        plateau_vals = [1.0] * 25  # 25個の同じ値
        assert enhanced_service._detect_plateau(plateau_vals) == True

        # 変動が大きい状態
        varying_vals = list(range(25))  # 0から24まで
        assert enhanced_service._detect_plateau(varying_vals) == False

        # データが少ない場合
        short_vals = [1.0] * 10
        assert enhanced_service._detect_plateau(short_vals) == False

    def test_calculate_individual_scores(self, enhanced_service):
        """個別スコア計算テスト"""
        performance_metrics = {
            "Sharpe Ratio": 1.5,
            "Return [%]": 25.0,
            "Max. Drawdown [%]": 10.0,
        }

        objectives = ["Sharpe Ratio", "Return [%]", "-Max. Drawdown [%]"]
        scores = enhanced_service._calculate_individual_scores(
            performance_metrics, objectives
        )

        assert scores["Sharpe Ratio"] == 1.5
        assert scores["Return [%]"] == 25.0
        assert scores["-Max. Drawdown [%]"] == -10.0  # 負の符号付きなので負の値

    def test_calculate_robustness_score(self, enhanced_service):
        """ロバストネススコア計算テスト"""
        performance_stats = {
            "total_return": {"consistency_score": 0.8},
            "sharpe_ratio": {"consistency_score": 0.7},
        }

        parameter_stability = {
            "n1": {"coefficient_of_variation": 0.1},
            "n2": {"coefficient_of_variation": 0.2},
        }

        score = enhanced_service._calculate_robustness_score(
            performance_stats, parameter_stability
        )

        assert 0 <= score <= 1  # スコアは0-1の範囲
        assert isinstance(score, float)

    @patch("app.core.services.enhanced_backtest_service.Backtest")
    def test_optimize_strategy_enhanced_basic(
        self,
        mock_backtest,
        enhanced_service,
        mock_data_service,
        sample_config,
        sample_optimization_params,
        sample_ohlcv_dataframe,
    ):
        """基本的な拡張最適化テスト"""
        # モックの設定
        mock_data_service.get_ohlcv_for_backtest.return_value = sample_ohlcv_dataframe

        # モックバックテストインスタンス
        mock_bt_instance = Mock()
        mock_backtest.return_value = mock_bt_instance

        # モック最適化結果
        mock_stats = pd.Series(
            {
                "Return [%]": 25.5,
                "Sharpe Ratio": 1.8,
                "Max. Drawdown [%]": -8.2,
                "Win Rate [%]": 65.0,
                "Profit Factor": 1.4,
                "# Trades": 45,
                "_strategy": Mock(n1=15, n2=40),
            }
        )

        mock_heatmap = pd.Series(
            [0.5, 0.8, 0.9],
            index=pd.MultiIndex.from_tuples([(10, 20), (15, 30), (20, 40)]),
        )

        mock_optimization_result = Mock()
        mock_optimization_result.func_vals = [0.1, 0.5, 0.8, 0.9]
        mock_optimization_result.fun = 0.9

        mock_bt_instance.optimize.return_value = (
            mock_stats,
            mock_heatmap,
            mock_optimization_result,
        )

        # テスト実行
        result = enhanced_service.optimize_strategy_enhanced(
            sample_config, sample_optimization_params
        )

        # 検証
        assert result is not None
        assert "optimized_parameters" in result
        assert "heatmap_data" in result
        assert "heatmap_summary" in result
        assert "optimization_details" in result
        assert "optimization_metadata" in result

        # 最適化されたパラメータの確認
        assert result["optimized_parameters"]["n1"] == 15
        assert result["optimized_parameters"]["n2"] == 40

        # ヒートマップサマリーの確認
        assert "best_combination" in result["heatmap_summary"]
        assert "total_combinations" in result["heatmap_summary"]

        # 最適化詳細の確認
        assert result["optimization_details"]["method"] == "sambo"
        assert result["optimization_details"]["n_calls"] == 4

        # メタデータの確認
        assert result["optimization_metadata"]["method"] == "sambo"
        assert result["optimization_metadata"]["maximize"] == "Sharpe Ratio"

    def test_multi_objective_optimization_basic(self, enhanced_service):
        """基本的なマルチ目的最適化テスト"""
        config = {
            "strategy_name": "TEST_STRATEGY",
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": datetime(2024, 1, 1),
            "end_date": datetime(2024, 12, 31),
            "initial_capital": 100000,
            "commission_rate": 0.001,
            "strategy_config": {"strategy_type": "SMA_CROSS", "parameters": {}},
        }

        objectives = ["Sharpe Ratio", "Return [%]", "-Max. Drawdown [%]"]
        weights = [0.4, 0.4, 0.2]

        # optimize_strategy_enhancedをモック
        with patch.object(
            enhanced_service, "optimize_strategy_enhanced"
        ) as mock_optimize:
            mock_result = {
                "performance_metrics": {
                    "Sharpe Ratio": 1.5,
                    "Return [%]": 25.0,
                    "Max. Drawdown [%]": 10.0,
                }
            }
            mock_optimize.return_value = mock_result

            result = enhanced_service.multi_objective_optimization(
                config, objectives, weights
            )

            # 検証
            assert "multi_objective_details" in result
            assert result["multi_objective_details"]["objectives"] == objectives
            assert result["multi_objective_details"]["weights"] == weights
            assert "individual_scores" in result["multi_objective_details"]

            # 個別スコアの確認
            scores = result["multi_objective_details"]["individual_scores"]
            assert scores["Sharpe Ratio"] == 1.5
            assert scores["Return [%]"] == 25.0
            assert scores["-Max. Drawdown [%]"] == -10.0

    def test_multi_objective_optimization_weight_mismatch(self, enhanced_service):
        """マルチ目的最適化での重み不一致テスト"""
        config = {}
        objectives = ["Sharpe Ratio", "Return [%]"]
        weights = [0.5]  # 目的関数2個に対して重み1個

        with pytest.raises(
            ValueError, match="Number of objectives must match number of weights"
        ):
            enhanced_service.multi_objective_optimization(config, objectives, weights)

    def test_robustness_test_basic(self, enhanced_service):
        """基本的なロバストネステスト"""
        config = {
            "strategy_name": "TEST_STRATEGY",
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "initial_capital": 100000,
            "commission_rate": 0.001,
            "strategy_config": {"strategy_type": "SMA_CROSS", "parameters": {}},
        }

        test_periods = [("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31")]

        optimization_params = {"parameters": {"n1": range(10, 30), "n2": range(20, 50)}}

        # optimize_strategy_enhancedをモック
        with patch.object(
            enhanced_service, "optimize_strategy_enhanced"
        ) as mock_optimize:
            mock_results = [
                {
                    "optimized_parameters": {"n1": 15, "n2": 30},
                    "performance_metrics": {
                        "total_return": 20.0,
                        "sharpe_ratio": 1.5,
                        "max_drawdown": 8.0,
                        "win_rate": 60.0,
                    },
                },
                {
                    "optimized_parameters": {"n1": 18, "n2": 35},
                    "performance_metrics": {
                        "total_return": 25.0,
                        "sharpe_ratio": 1.8,
                        "max_drawdown": 10.0,
                        "win_rate": 65.0,
                    },
                },
            ]

            mock_optimize.side_effect = mock_results

            result = enhanced_service.robustness_test(
                config, test_periods, optimization_params
            )

            # 検証
            assert "individual_results" in result
            assert "robustness_analysis" in result
            assert "test_periods" in result
            assert result["total_periods"] == 2

            # 個別結果の確認
            assert "period_1" in result["individual_results"]
            assert "period_2" in result["individual_results"]

            # ロバストネス分析の確認
            analysis = result["robustness_analysis"]
            assert "performance_statistics" in analysis
            assert "parameter_stability" in analysis
            assert "robustness_score" in analysis
            assert analysis["successful_periods"] == 2
            assert analysis["failed_periods"] == 0
