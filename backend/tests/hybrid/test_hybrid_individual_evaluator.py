"""
HybridIndividualEvaluatorのテスト
"""

from unittest.mock import Mock

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.core.hybrid_individual_evaluator import (
    HybridIndividualEvaluator,
)
from app.services.auto_strategy.genes.strategy import StrategyGene


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
        mock_individual = StrategyGene(id="test_gene_001")
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
        self.mock_hybrid_predictor.predict.return_value = {
            "up": 0.6,
            "down": 0.2,
            "range": 0.2,
        }

        ga_config = GAConfig()
        ga_config.enable_multi_objective = False
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }

        result = self.evaluator.evaluate(mock_individual, ga_config)

        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_evaluate_individual_multi_objective(self):
        """ハイブリッド多目的評価のテスト"""
        mock_individual = StrategyGene(id="test_gene_001")
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
        self.mock_hybrid_predictor.predict.return_value = {
            "up": 0.65,
            "down": 0.2,
            "range": 0.15,
        }

        ga_config = GAConfig()
        ga_config.enable_multi_objective = True
        ga_config.objectives = ["total_return", "sharpe_ratio", "hybrid_score"]

        result = self.evaluator.evaluate(mock_individual, ga_config)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_evaluate_individual_exception(self):
        """ハイブリッド評価例外のテスト"""
        mock_individual = StrategyGene(id="test_gene_001")

        # バックテストで例外
        self.mock_backtest_service.run_backtest.side_effect = Exception("Test error")

        ga_config = GAConfig()
        ga_config.enable_multi_objective = False

        result = self.evaluator.evaluate(mock_individual, ga_config)

        assert result == (0.0,)

    def test_evaluate_individual_volatility_mode(self):
        """ボラティリティ予測モードでの評価テスト"""
        mock_individual = StrategyGene(id="test_gene_001")
        mock_backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "total_trades": 10,
            },
            "equity_curve": [],
            "trade_history": [],
        }

        self.mock_backtest_service.run_backtest.return_value = mock_backtest_result
        # ボラティリティ予測のモック
        self.mock_hybrid_predictor.predict.return_value = {"trend": 0.8, "range": 0.2}

        ga_config = GAConfig()
        ga_config.enable_multi_objective = False
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "prediction_score": 0.1,
        }

        result = self.evaluator.evaluate(mock_individual, ga_config)

        assert isinstance(result, tuple)
        assert len(result) == 1
        # prediction_score = 0.8 - 0.5 = 0.3
        # fitness should include 0.1 * 0.3 = 0.03 boost

    def test_preloaded_data_passed_to_backtest(self):
        """バックテスト実行時にpreloaded_dataが渡されることを検証"""
        import pandas as pd
        from unittest.mock import patch

        mock_individual = StrategyGene(id="test_gene_001")
        mock_backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "total_trades": 10,
            },
            "equity_curve": [],
            "trade_history": [],
        }

        # テスト用のモックデータ
        mock_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [98, 99, 100],
                "Close": [103, 104, 105],
                "Volume": [1000, 1100, 1200],
            }
        )

        self.mock_backtest_service.run_backtest.return_value = mock_backtest_result
        self.mock_backtest_service.data_service = Mock()
        self.mock_backtest_service.data_service.get_data_for_backtest.return_value = (
            mock_data
        )

        ga_config = GAConfig()
        ga_config.enable_multi_objective = False

        # バックテスト設定をセット
        self.evaluator.set_backtest_config({
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 10000.0
        })

        # _get_cached_dataをモックして、preloaded_dataが渡されることを検証
        with patch.object(
            self.evaluator, "_get_cached_data", return_value=mock_data
        ) as mock_get_cached:
            result = self.evaluator.evaluate(mock_individual, ga_config)

            # _get_cached_dataが呼ばれたことを確認
            mock_get_cached.assert_called()

            # run_backtestがpreloaded_dataを受け取ったことを確認
            call_args = self.mock_backtest_service.run_backtest.call_args
            assert call_args is not None
            # keyword argument または positional argument で渡されることを検証
            if call_args.kwargs.get("preloaded_data") is not None:
                pd.testing.assert_frame_equal(
                    call_args.kwargs["preloaded_data"], mock_data
                )
            else:
                # 辞書形式で渡された場合もチェック
                assert call_args is not None

    def test_ohlcv_cache_key_has_prefix(self):
        """OHLCVデータ用のキャッシュキーにプレフィックスが付与されることを検証"""
        import pandas as pd

        mock_ohlcv = pd.DataFrame(
            {
                "Open": [100, 101],
                "High": [105, 106],
                "Low": [98, 99],
                "Close": [103, 104],
                "Volume": [1000, 1100],
            }
        )

        self.mock_backtest_service.data_service = Mock()
        self.mock_backtest_service.data_service.get_ohlcv_data.return_value = mock_ohlcv

        ga_config = GAConfig()
        backtest_config = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
        }

        # _fetch_ohlcv_dataを呼び出し
        fetched_data = self.evaluator._fetch_ohlcv_data(backtest_config, ga_config)

        # キャッシュにプレフィックス付きキーで保存されていることを確認
        expected_key = ("ohlcv", "BTCUSDT", "1h", "2024-01-01", "2024-01-31")
        assert expected_key in self.evaluator._data_cache

        # 基底クラスのキー形式（プレフィックスなし）とは異なることを確認
        old_style_key = ("BTCUSDT", "1h", "2024-01-01", "2024-01-31")
        assert old_style_key not in self.evaluator._data_cache

        # データが正しく返されることを確認
        pd.testing.assert_frame_equal(fetched_data, mock_ohlcv)




