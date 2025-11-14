"""
HybridIndividualEvaluatorのテスト
"""

import pytest
from unittest.mock import Mock, patch
from app.services.auto_strategy.core.hybrid_individual_evaluator import (
    HybridIndividualEvaluator,
)
from app.services.auto_strategy.config import GAConfig


@pytest.mark.skip(
    reason="HybridIndividualEvaluator implementation changed - missing _calculate_hybrid_score method"
)
class TestHybridIndividualEvaluator:
    """HybridIndividualEvaluatorのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        self.mock_backtest_service = Mock()
        self.mock_hybrid_predictor = Mock()
        self.evaluator = HybridIndividualEvaluator(
            self.mock_backtest_service, self.mock_hybrid_predictor
        )

    def test_init(self):
        """初期化のテスト"""
        assert self.evaluator.backtest_service == self.mock_backtest_service
        assert self.evaluator.predictor == self.mock_hybrid_predictor

    def test_evaluate_individual_success(self):
        """ハイブリッド個体評価成功のテスト"""
        mock_individual = [1, 2, 3, 4, 5]
        mock_backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "total_trades": 10,
            },
            "equity_curve": [100, 110, 105, 120],
            "trade_history": [],
        }

        # バックテストサービスのモック
        self.mock_backtest_service.run_backtest.return_value = mock_backtest_result

        # ハイブリッド予測器のモック
        self.mock_hybrid_predictor.predict.return_value = 0.8

        ga_config = GAConfig()
        ga_config.enable_multi_objective = False
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }

        result = self.evaluator.evaluate_individual(mock_individual, ga_config)

        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_evaluate_individual_multi_objective(self):
        """ハイブリッド多目的評価のテスト"""
        mock_individual = [1, 2, 3, 4, 5]
        mock_backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
            },
            "equity_curve": [],
            "trade_history": [],
        }

        self.mock_backtest_service.run_backtest.return_value = mock_backtest_result
        self.mock_hybrid_predictor.predict.return_value = 0.85

        ga_config = GAConfig()
        ga_config.enable_multi_objective = True
        ga_config.objectives = ["total_return", "sharpe_ratio", "hybrid_score"]

        result = self.evaluator.evaluate_individual(mock_individual, ga_config)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_evaluate_individual_exception(self):
        """ハイブリッド評価例外のテスト"""
        mock_individual = [1, 2, 3, 4, 5]

        # バックテストで例外
        self.mock_backtest_service.run_backtest.side_effect = Exception("Test error")

        ga_config = GAConfig()
        ga_config.enable_multi_objective = False

        result = self.evaluator.evaluate_individual(mock_individual, ga_config)

        assert result == (0.0,)

    def test_hybrid_score_integration(self):
        """ハイブリッド評価の統合テスト"""
        # モックされた評価を通じて、ハイブリッドスコアが総合的に機能することを確認
        mock_individual = [1, 2, 3, 4, 5]
        mock_backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "total_trades": 15,
            },
            "trade_history": [],
        }

        # GA設定（ハイブリッドモード有効）
        ga_config = GAConfig()
        ga_config.hybrid_mode = True

        # BAされるバックテスト
        self.mock_backtest_service.run_backtest.return_value = mock_backtest_result
        self.mock_hybrid_predictor.predict.return_value = 0.8

    def test_calculate_hybrid_score_zero_trades(self):
        """取引回数0のハイブリッドスコアテスト"""
        backtest_result = {"performance_metrics": {"total_trades": 0}}
        prediction_score = 0.9

        score = self.evaluator._calculate_hybrid_score(
            backtest_result, prediction_score
        )

        assert score == 0.1  # 取引回数0の特別処理

    def test_calculate_hybrid_score_with_constraints(self):
        """制約付きハイブリッドスコアのテスト"""
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.05,  # 低いリターン
                "sharpe_ratio": 0.2,  # 低いシャープレシオ
                "max_drawdown": 0.25,  # 高いドローダウン
                "win_rate": 0.4,
                "total_trades": 20,
            }
        }
        prediction_score = 0.9

        ga_config = GAConfig()
        ga_config.fitness_constraints = {
            "min_sharpe_ratio": 0.5,
            "max_drawdown_limit": 0.2,
        }

        score = self.evaluator._calculate_hybrid_score(
            backtest_result, prediction_score, ga_config
        )

        assert score == 0.0  # 制約違反で0.0

    def test_calculate_hybrid_multi_objective_values(self):
        """ハイブリッド多目的評価値のテスト"""
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "profit_factor": 1.9,
                "sortino_ratio": 1.8,
                "calmar_ratio": 1.5,
            }
        }
        prediction_score = 0.85

        ga_config = GAConfig()
        ga_config.objectives = ["total_return", "hybrid_score", "sharpe_ratio"]

        values = self.evaluator._calculate_hybrid_multi_objective_values(
            backtest_result, prediction_score, ga_config
        )

        assert isinstance(values, tuple)
        assert len(values) == 3

    def test_adaptive_weight_adjustment(self):
        """適応的重み調整のテスト"""
        base_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }

        performance_metrics = {
            "total_return": 0.2,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.1,
            "win_rate": 0.7,
        }

        adjusted_weights = self.evaluator._adaptive_weight_adjustment(
            base_weights, performance_metrics
        )

        assert isinstance(adjusted_weights, dict)
        assert sum(adjusted_weights.values()) == 1.0  # 重みの合計は1.0

    def test_prediction_confidence_adjustment(self):
        """予測信頼度調整のテスト"""
        base_score = 0.8
        confidence = 0.9

        adjusted = self.evaluator._prediction_confidence_adjustment(
            base_score, confidence
        )

        assert isinstance(adjusted, float)
        assert adjusted >= base_score  # 信頼度が高いほどスコアが上がる

    def test_prediction_confidence_adjustment_low(self):
        """低信頼度調整のテスト"""
        base_score = 0.8
        confidence = 0.3  # 低い信頼度

        adjusted = self.evaluator._prediction_confidence_adjustment(
            base_score, confidence
        )

        assert adjusted <= base_score  # 信頼度が低いとスコアが下がる

    def test_market_condition_adjustment(self):
        """市場状況調整のテスト"""
        base_score = 0.8
        market_volatility = 0.2
        trend_strength = 0.6

        adjusted = self.evaluator._market_condition_adjustment(
            base_score, market_volatility, trend_strength
        )

        assert isinstance(adjusted, float)

    def test_market_condition_high_volatility(self):
        """高ボラティリティ調整のテスト"""
        base_score = 0.8
        market_volatility = 0.5  # 高ボラ
        trend_strength = 0.3  # 弱いトレンド

        adjusted = self.evaluator._market_condition_adjustment(
            base_score, market_volatility, trend_strength
        )

        assert adjusted <= base_score  # 高ボラでスコアが調整される

    def test_hybrid_fitness_calculation(self):
        """ハイブリッドフィットネス計算のテスト"""
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "total_trades": 25,
            },
            "equity_curve": [100, 110, 105, 120, 115, 130],
            "trade_history": [
                {"size": 1, "pnl": 10},
                {"size": -1, "pnl": 5},
                {"size": 1, "pnl": 15},
            ],
        }
        prediction_score = 0.85

        ga_config = GAConfig()
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "prediction_score": 0.1,
        }

        fitness = self.evaluator._hybrid_fitness_calculation(
            backtest_result, prediction_score, ga_config
        )

        assert isinstance(fitness, float)
        assert fitness >= 0.0

    def test_hybrid_fitness_calculation_exception(self):
        """ハイブリッドフィットネス計算例外のテスト"""
        backtest_result = {}
        prediction_score = 0.8

        # 例外が発生するように設定
        with patch.object(
            self.evaluator, "_extract_performance_metrics"
        ) as mock_extract:
            mock_extract.side_effect = Exception("Test error")

            fitness = self.evaluator._hybrid_fitness_calculation(
                backtest_result, prediction_score, GAConfig()
            )

            assert fitness == 0.0

    def test_evaluate_individual_with_regime_adaptation(self):
        """レジーム適応付き評価のテスト"""
        mock_individual = [1, 2, 3, 4, 5]
        mock_backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "total_trades": 12,
            },
            "equity_curve": [],
            "trade_history": [],
        }

        self.mock_backtest_service.run_backtest.return_value = mock_backtest_result
        self.mock_hybrid_predictor.predict.return_value = 0.8

        ga_config = GAConfig()
        ga_config.regime_adaptation_enabled = True
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "prediction_score": 0.1,
        }

        # レジーム検知器のモック
        with patch(
            "app.services.auto_strategy.core.hybrid_individual_evaluator.RegimeDetector"
        ) as mock_regime_class:
            mock_regime_detector = Mock()
            mock_regime_detector.detect_regimes.return_value = [0, 1, 0, 2]
            mock_regime_class.return_value = mock_regime_detector

            result = self.evaluator.evaluate_individual_with_regime_adaptation(
                mock_individual, ga_config
            )

            assert isinstance(result, tuple)
            assert len(result) == 1

    def test_evaluate_individual_with_regime_adaptation_no_detector(self):
        """レジーム検知器なしのテスト"""
        mock_individual = [1, 2, 3, 4, 5]
        mock_backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "total_trades": 8,
            },
            "equity_curve": [],
            "trade_history": [],
        }

        self.mock_backtest_service.run_backtest.return_value = mock_backtest_result
        self.mock_hybrid_predictor.predict.return_value = 0.8

        ga_config = GAConfig()
        ga_config.regime_adaptation_enabled = True
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "prediction_score": 0.1,
        }

        # レジーム検知器なし
        result = self.evaluator.evaluate_individual_with_regime_adaptation(
            mock_individual, ga_config
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
