"""
IndividualEvaluator ML統合テスト

IndividualEvaluatorがMLモデルを正しくロードし、
BacktestServiceを通じてUniversalStrategyに渡しているかを検証します。
"""

import unittest
from unittest.mock import MagicMock, patch

from app.services.auto_strategy.config.ga_runtime import GAConfig
from app.services.auto_strategy.core.individual_evaluator import IndividualEvaluator
from app.services.auto_strategy.models.strategy_models import StrategyGene
from app.services.backtest.backtest_service import BacktestService


class TestEvaluatorMLIntegration(unittest.TestCase):
    def setUp(self):
        # モックの準備
        self.mock_backtest_service = MagicMock(spec=BacktestService)
        self.evaluator = IndividualEvaluator(self.mock_backtest_service)

        # テスト用バックテスト設定
        self.backtest_config = {
            "strategy_name": "TestStrategy",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "initial_capital": 10000.0,
            "commission_rate": 0.001,
        }
        self.evaluator.set_backtest_config(self.backtest_config)

        # テスト用GA設定（MLフィルター有効）
        self.ga_config = GAConfig(
            ml_filter_enabled=True,
            ml_model_path="dummy/path/to/model",
        )

        # ダミー遺伝子
        self.gene = StrategyGene()

    @patch("app.services.auto_strategy.core.individual_evaluator.model_manager")
    def test_evaluate_individual_passes_ml_model(self, mock_model_manager):
        """
        evaluate_individualがMLモデルをロードし、
        正しいパラメータ構造でBacktestServiceに渡すことをテスト
        """
        # MLモデルのロードをモック
        mock_model = MagicMock()
        mock_model_manager.load_model.return_value = mock_model

        # _perform_single_evaluation は evaluate_individual から呼ばれるが、
        # 今回は evaluate_individual 全体のフローを通して確認する
        # ただし、データキャッシュ部分はモックが必要

        with patch.object(
            self.evaluator, "_get_cached_data", return_value=MagicMock()
        ), patch.object(
            self.evaluator, "_get_cached_minute_data", return_value=None
        ), patch.object(
            self.evaluator,
            "_calculate_multi_objective_fitness",
            return_value=(1.0,),
        ):
            self.evaluator.evaluate_individual(self.gene, self.ga_config)

        # BacktestService.run_backtest が呼ばれたか確認
        self.mock_backtest_service.run_backtest.assert_called_once()

        # 呼び出し引数を検証
        call_args = self.mock_backtest_service.run_backtest.call_args
        run_config = call_args.kwargs["config"]

        # 1. ml_filter_model がトップレベルに存在しないこと（修正前の挙動確認）
        self.assertNotIn(
            "ml_filter_model",
            run_config,
            "ml_filter_model should not be in top-level config",
        )

        # 2. strategy_config -> parameters -> ml_predictor にモデルが設定されていること
        strategy_config = run_config.get("strategy_config", {})
        parameters = strategy_config.get("parameters", {})

        self.assertIn("ml_predictor", parameters)
        self.assertEqual(parameters["ml_predictor"], mock_model)

        # 3. ml_filter_threshold が設定されていること
        self.assertIn("ml_filter_threshold", parameters)
        self.assertEqual(parameters["ml_filter_threshold"], 0.5)

    @patch("app.services.auto_strategy.core.individual_evaluator.model_manager")
    def test_evaluate_individual_handles_load_error(self, mock_model_manager):
        """モデルロードエラー時にMLフィルターが無効化されることをテスト"""
        # ロードで例外発生
        mock_model_manager.load_model.side_effect = Exception("Load failed")

        with patch.object(
            self.evaluator, "_get_cached_data", return_value=MagicMock()
        ), patch.object(
            self.evaluator, "_get_cached_minute_data", return_value=None
        ), patch.object(
            self.evaluator,
            "_calculate_multi_objective_fitness",
            return_value=(1.0,),
        ):
            self.evaluator.evaluate_individual(self.gene, self.ga_config)

        # BacktestService.run_backtest が呼ばれたか確認
        self.mock_backtest_service.run_backtest.assert_called_once()

        run_config = self.mock_backtest_service.run_backtest.call_args.kwargs["config"]
        parameters = run_config["strategy_config"]["parameters"]

        # ml_filter_enabled が False に上書きされていること
        self.assertFalse(parameters["ml_filter_enabled"])
        # ml_predictor が設定されていないこと（またはNone）
        self.assertIsNone(parameters.get("ml_predictor"))
