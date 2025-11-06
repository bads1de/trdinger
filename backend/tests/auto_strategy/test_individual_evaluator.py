"""
IndividualEvaluatorのテスト
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.core.individual_evaluator import IndividualEvaluator
from app.services.auto_strategy.models.strategy_models import StrategyGene
from app.services.auto_strategy.services.regime_detector import RegimeDetector


class TestIndividualEvaluator:
    """IndividualEvaluatorのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        self.mock_backtest_service = Mock()
        self.evaluator = IndividualEvaluator(self.mock_backtest_service)

    def test_init(self):
        """初期化のテスト"""
        assert self.evaluator.backtest_service == self.mock_backtest_service
        assert self.evaluator.regime_detector is None
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


class TestIndividualEvaluatorRegimeAdaptation:
    """IndividualEvaluatorのレジーム適応評価機能のテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        self.mock_backtest_service = Mock()
        self.mock_regime_detector = Mock(spec=RegimeDetector)
        self.evaluator_with_regime = IndividualEvaluator(
            self.mock_backtest_service, self.mock_regime_detector
        )
        self.evaluator_without_regime = IndividualEvaluator(self.mock_backtest_service)

    def test_init_with_regime_detector(self):
        """レジーム検知器を使用した初期化のテスト"""
        assert self.evaluator_with_regime.regime_detector == self.mock_regime_detector
        assert self.evaluator_with_regime.backtest_service == self.mock_backtest_service

    def test_init_without_regime_detector(self):
        """レジーム検知器なしの初期化のテスト"""
        assert self.evaluator_without_regime.regime_detector is None

    def test_calculate_fitness_with_regime_labels_trend_dominant(self):
        """トレンド優勢時のレジーム別重み付けテスト"""
        # トレンド優勢（レジーム0が50%以上）のregime_labels
        regime_labels = [0] * 60 + [1] * 20 + [2] * 20  # トレンド60%

        backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.55,
                "total_trades": 10,
            },
            "equity_curve": [100, 110, 105, 120],
            "trade_history": [{"size": 1, "pnl": 10}],
        }

        ga_config = GAConfig()
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }

        fitness = self.evaluator_with_regime._calculate_fitness(
            backtest_result, ga_config, regime_labels
        )

        # トレンド優勢時はsharpe_ratioとtotal_returnの重みが増加するはず
        assert fitness > 0.0
        assert isinstance(fitness, float)

    def test_calculate_fitness_with_regime_labels_range_dominant(self):
        """レンジ優勢時のレジーム別重み付けテスト"""
        # レンジ優勢（レジーム1が50%以上）のregime_labels
        regime_labels = [0] * 20 + [1] * 60 + [2] * 20  # レンジ60%

        backtest_result = {
            "performance_metrics": {
                "total_return": 0.12,
                "sharpe_ratio": 1.0,
                "max_drawdown": 0.05,
                "win_rate": 0.65,
                "total_trades": 15,
            },
            "equity_curve": [100, 105, 103, 110],
            "trade_history": [{"size": 1, "pnl": 8}],
        }

        ga_config = GAConfig()
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }

        fitness = self.evaluator_with_regime._calculate_fitness(
            backtest_result, ga_config, regime_labels
        )

        # レンジ優勢時はmax_drawdownとwin_rateの重みが増加するはず
        assert fitness > 0.0
        assert isinstance(fitness, float)

    def test_calculate_fitness_with_regime_labels_high_volatility_dominant(self):
        """高ボラティリティ優勢時のレジーム別重み付けテスト"""
        # 高ボラティリティ優勢（レジーム2が50%以上）のregime_labels
        regime_labels = [0] * 20 + [1] * 20 + [2] * 60  # 高ボラ60%

        backtest_result = {
            "performance_metrics": {
                "total_return": 0.10,
                "sharpe_ratio": 1.0,  # 制約を満たす値に変更
                "max_drawdown": 0.12,
                "win_rate": 0.50,
                "total_trades": 12,
            },
            "equity_curve": [100, 95, 105, 110],
            "trade_history": [{"size": 1, "pnl": 5}],
        }

        ga_config = GAConfig()
        ga_config.fitness_constraints = {}  # 制約を明示的に空に設定
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }

        fitness = self.evaluator_with_regime._calculate_fitness(
            backtest_result, ga_config, regime_labels
        )

        # 高ボラ優勢時はmax_drawdownの重みがさらに増加するはず
        assert fitness > 0.0
        assert isinstance(fitness, float)

    def test_calculate_fitness_with_regime_labels_mixed(self):
        """混合レジーム時の重み付けテスト"""
        # どのレジームも優勢でない（均等分布）
        regime_labels = [0] * 33 + [1] * 33 + [2] * 34

        backtest_result = {
            "performance_metrics": {
                "total_return": 0.13,
                "sharpe_ratio": 1.1,
                "max_drawdown": 0.09,
                "win_rate": 0.58,
                "total_trades": 11,
            },
            "equity_curve": [100, 108, 102, 113],
            "trade_history": [{"size": 1, "pnl": 7}],
        }

        ga_config = GAConfig()
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }

        fitness = self.evaluator_with_regime._calculate_fitness(
            backtest_result, ga_config, regime_labels
        )

        # 混合時は元の重みがほぼ維持される
        assert fitness > 0.0
        assert isinstance(fitness, float)

    def test_calculate_fitness_without_regime_labels(self):
        """regime_labelsなしでのフィットネス計算テスト"""
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.55,
                "total_trades": 10,
            },
            "equity_curve": [100, 110, 105, 120],
            "trade_history": [{"size": 1, "pnl": 10}],
        }

        ga_config = GAConfig()
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }

        # regime_labelsなしでの計算
        fitness_without = self.evaluator_with_regime._calculate_fitness(
            backtest_result, ga_config, None
        )

        # 通常のフィットネス計算と同じ結果になるはず
        assert fitness_without > 0.0
        assert isinstance(fitness_without, float)

    def test_calculate_fitness_with_empty_regime_labels(self):
        """空のregime_labelsでの処理テスト"""
        regime_labels = []  # 空のリスト

        backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.55,
                "total_trades": 10,
            },
            "equity_curve": [100, 110, 105, 120],
            "trade_history": [{"size": 1, "pnl": 10}],
        }

        ga_config = GAConfig()
        ga_config.fitness_constraints = {}  # 制約を明示的に空に設定
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }

        # 空のregime_labelsでもエラーが発生しないこと
        # 空の場合はregime_labels=Noneと同じ扱いになり、通常のフィットネス計算が実行される
        fitness = self.evaluator_with_regime._calculate_fitness(
            backtest_result, ga_config, regime_labels
        )

        assert fitness > 0.0
        assert isinstance(fitness, float)

    def test_calculate_fitness_with_single_regime_labels(self):
        """単一レジームのみのregime_labelsテスト"""
        # すべてトレンド（レジーム0）
        regime_labels = [0] * 100

        backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.55,
                "total_trades": 10,
            },
            "equity_curve": [100, 110, 105, 120],
            "trade_history": [{"size": 1, "pnl": 10}],
        }

        ga_config = GAConfig()
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }

        fitness = self.evaluator_with_regime._calculate_fitness(
            backtest_result, ga_config, regime_labels
        )

        # 単一レジームでも正しく計算される
        assert fitness > 0.0
        assert isinstance(fitness, float)

    def test_calculate_fitness_with_invalid_regime_labels(self):
        """不正なレジームラベルの処理テスト"""
        # 想定外のレジームラベル（0, 1, 2以外）
        regime_labels = [0, 1, 2, 3, 4, 5]  # 3, 4, 5は想定外

        backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.55,
                "total_trades": 10,
            },
            "equity_curve": [100, 110, 105, 120],
            "trade_history": [{"size": 1, "pnl": 10}],
        }

        ga_config = GAConfig()
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }

        # 不正なラベルでもエラーが発生せず、正常に処理される
        fitness = self.evaluator_with_regime._calculate_fitness(
            backtest_result, ga_config, regime_labels
        )

        assert fitness >= 0.0
        assert isinstance(fitness, float)

    def test_regime_weight_normalization(self):
        """レジーム別重み調整後の正規化テスト"""
        regime_labels = [0] * 70 + [1] * 20 + [2] * 10  # トレンド優勢

        backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.55,
                "total_trades": 10,
            },
            "equity_curve": [100, 110, 105, 120],
            "trade_history": [{"size": 1, "pnl": 10}],
        }

        ga_config = GAConfig()
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }

        # レジーム別重み調整が行われても、重みの合計は正規化されるべき
        fitness = self.evaluator_with_regime._calculate_fitness(
            backtest_result, ga_config, regime_labels
        )

        assert fitness > 0.0
        assert fitness <= 2.0  # 合理的な範囲内

    def test_evaluate_individual_with_regime_detector_enabled(self):
        """レジーム検知器が有効な場合の個体評価テスト"""
        mock_individual = [1, 2, 3, 4, 5]
        mock_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.55,
                "total_trades": 10,
            },
            "equity_curve": [100, 110, 105, 120],
            "trade_history": [{"size": 1, "pnl": 10}],
        }

        # バックテストサービスのモック設定
        self.mock_backtest_service.run_backtest.return_value = mock_result
        self.mock_backtest_service._ensure_data_service_initialized = Mock()
        self.mock_backtest_service.data_service = Mock()

        # OHLCVデータのモック
        import pandas as pd

        mock_ohlcv = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="1h"),
                "open": np.random.randn(100).cumsum() + 100,
                "high": np.random.randn(100).cumsum() + 101,
                "low": np.random.randn(100).cumsum() + 99,
                "close": np.random.randn(100).cumsum() + 100,
                "volume": np.random.rand(100) * 1000,
            }
        )
        self.mock_backtest_service.data_service.get_ohlcv_data.return_value = mock_ohlcv

        # レジーム検知のモック
        regime_labels = [0] * 50 + [1] * 30 + [2] * 20
        self.mock_regime_detector.detect_regimes.return_value = regime_labels

        # GA設定
        ga_config = GAConfig()
        ga_config.regime_adaptation_enabled = True
        ga_config.enable_multi_objective = False
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }

        # バックテスト設定
        self.evaluator_with_regime.set_backtest_config(
            {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-05",
            }
        )

        # 評価実行
        result = self.evaluator_with_regime.evaluate_individual(
            mock_individual, ga_config
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0] >= 0.0

        # レジーム検知が呼ばれたことを確認
        self.mock_regime_detector.detect_regimes.assert_called_once()

    def test_evaluate_individual_with_regime_detector_disabled(self):
        """レジーム検知器が無効な場合の個体評価テスト"""
        mock_individual = [1, 2, 3, 4, 5]
        mock_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.55,
                "total_trades": 10,
            },
            "equity_curve": [100, 110, 105, 120],
            "trade_history": [{"size": 1, "pnl": 10}],
        }

        self.mock_backtest_service.run_backtest.return_value = mock_result

        ga_config = GAConfig()
        ga_config.regime_adaptation_enabled = False  # 無効化
        ga_config.enable_multi_objective = False
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }

        self.evaluator_with_regime.set_backtest_config(
            {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-05",
            }
        )

        result = self.evaluator_with_regime.evaluate_individual(
            mock_individual, ga_config
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0] >= 0.0

        # レジーム検知が呼ばれていないことを確認
        self.mock_regime_detector.detect_regimes.assert_not_called()

    def test_evaluate_individual_without_regime_detector_instance(self):
        """レジーム検知器インスタンスがない場合の評価テスト"""
        mock_individual = [1, 2, 3, 4, 5]
        mock_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.55,
                "total_trades": 10,
            },
            "equity_curve": [100, 110, 105, 120],
            "trade_history": [{"size": 1, "pnl": 10}],
        }

        self.mock_backtest_service.run_backtest.return_value = mock_result

        ga_config = GAConfig()
        ga_config.regime_adaptation_enabled = True  # 有効だがインスタンスなし
        ga_config.enable_multi_objective = False
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }

        self.evaluator_without_regime.set_backtest_config(
            {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-05",
            }
        )

        # レジーム検知器がなくても正常に動作する
        result = self.evaluator_without_regime.evaluate_individual(
            mock_individual, ga_config
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0] >= 0.0

    def test_regime_detection_error_handling(self):
        """レジーム検知エラー時の処理テスト"""
        mock_individual = [1, 2, 3, 4, 5]
        mock_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.55,
                "total_trades": 10,
            },
            "equity_curve": [100, 110, 105, 120],
            "trade_history": [{"size": 1, "pnl": 10}],
        }

        self.mock_backtest_service.run_backtest.return_value = mock_result
        self.mock_backtest_service._ensure_data_service_initialized = Mock()
        self.mock_backtest_service.data_service = Mock()

        # レジーム検知でエラーが発生
        self.mock_regime_detector.detect_regimes.side_effect = Exception(
            "Regime detection failed"
        )

        import pandas as pd

        mock_ohlcv = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="1h"),
                "open": np.random.randn(100).cumsum() + 100,
                "high": np.random.randn(100).cumsum() + 101,
                "low": np.random.randn(100).cumsum() + 99,
                "close": np.random.randn(100).cumsum() + 100,
                "volume": np.random.rand(100) * 1000,
            }
        )
        self.mock_backtest_service.data_service.get_ohlcv_data.return_value = mock_ohlcv

        ga_config = GAConfig()
        ga_config.regime_adaptation_enabled = True
        ga_config.enable_multi_objective = False
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }

        self.evaluator_with_regime.set_backtest_config(
            {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-05",
            }
        )

        # エラーが発生してもregime_labels=Noneで評価が継続される
        result = self.evaluator_with_regime.evaluate_individual(
            mock_individual, ga_config
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0] >= 0.0

    def test_regime_detection_with_empty_ohlcv(self):
        """空のOHLCVデータでのレジーム検知テスト"""
        mock_individual = [1, 2, 3, 4, 5]
        mock_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.55,
                "total_trades": 10,
            },
            "equity_curve": [100, 110, 105, 120],
            "trade_history": [{"size": 1, "pnl": 10}],
        }

        self.mock_backtest_service.run_backtest.return_value = mock_result
        self.mock_backtest_service._ensure_data_service_initialized = Mock()
        self.mock_backtest_service.data_service = Mock()

        # 空のOHLCVデータ
        import pandas as pd

        empty_ohlcv = pd.DataFrame()
        self.mock_backtest_service.data_service.get_ohlcv_data.return_value = (
            empty_ohlcv
        )

        ga_config = GAConfig()
        ga_config.regime_adaptation_enabled = True
        ga_config.enable_multi_objective = False
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }

        self.evaluator_with_regime.set_backtest_config(
            {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-05",
            }
        )

        # 空のOHLCVでもエラーが発生せず、レジーム検知がスキップされる
        result = self.evaluator_with_regime.evaluate_individual(
            mock_individual, ga_config
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0] >= 0.0

        # レジーム検知が呼ばれていないことを確認
        self.mock_regime_detector.detect_regimes.assert_not_called()

    def test_calculate_multi_objective_fitness_with_regime_labels(self):
        """レジームラベル付き多目的フィットネス計算テスト"""
        regime_labels = [0] * 50 + [1] * 30 + [2] * 20

        backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.55,
                "total_trades": 10,
            },
            "equity_curve": [100, 110, 105, 120],
            "trade_history": [{"size": 1, "pnl": 10}],
        }

        ga_config = GAConfig()
        ga_config.objectives = [
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
        ]

        # 多目的最適化ではregime_labelsは現在使用されないが、
        # エラーが発生しないことを確認
        result = self.evaluator_with_regime._calculate_multi_objective_fitness(
            backtest_result, ga_config, regime_labels
        )

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert all(isinstance(v, float) for v in result)
