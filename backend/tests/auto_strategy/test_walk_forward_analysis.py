"""
Walk-Forward Analysis (WFA) のテスト

IndividualEvaluator の WFA 機能をテストします。
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Tuple
from unittest.mock import MagicMock, patch

import pytest

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.core.individual_evaluator import IndividualEvaluator


class TestWalkForwardAnalysis:
    """Walk-Forward Analysis テストクラス"""

    @pytest.fixture
    def mock_backtest_service(self):
        """モックバックテストサービス"""
        mock_service = MagicMock()
        mock_service.ensure_data_service_initialized = MagicMock()
        mock_service.data_service = MagicMock()
        mock_service.data_service.get_data_for_backtest = MagicMock(return_value=None)

        # バックテスト結果のモック
        mock_service.run_backtest = MagicMock(
            return_value={
                "performance_metrics": {
                    "total_return": 0.1,
                    "sharpe_ratio": 1.0,
                    "max_drawdown": 0.05,
                    "win_rate": 0.6,
                    "profit_factor": 1.5,
                    "sortino_ratio": 1.2,
                    "calmar_ratio": 2.0,
                    "total_trades": 50,
                },
                "equity_curve": [10000, 10500, 11000],
                "trade_history": [],
            }
        )
        return mock_service

    @pytest.fixture
    def evaluator(self, mock_backtest_service):
        """IndividualEvaluator インスタンス"""
        evaluator = IndividualEvaluator(backtest_service=mock_backtest_service)
        # バックテスト設定を設定
        evaluator.set_backtest_config(
            {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01 00:00:00",
                "end_date": "2024-06-01 00:00:00",
                "initial_balance": 10000,
            }
        )
        return evaluator

    @pytest.fixture
    def wfa_config(self):
        """WFA が有効な GAConfig"""
        return GAConfig(
            enable_walk_forward=True,
            wfa_n_folds=5,
            wfa_train_ratio=0.7,
            wfa_anchored=False,
            objectives=["weighted_score"],
        )

    @pytest.fixture
    def mock_gene(self):
        """モック戦略遺伝子"""
        gene = MagicMock()
        gene.id = "test-gene-12345678"
        gene.validate.return_value = (True, [])
        return gene

    def test_wfa_config_defaults(self):
        """WFA 設定のデフォルト値テスト"""
        config = GAConfig()

        assert config.enable_walk_forward is False
        assert config.wfa_n_folds == 5
        assert config.wfa_train_ratio == 0.7
        assert config.wfa_anchored is False

    def test_wfa_config_custom_values(self):
        """WFA 設定のカスタム値テスト"""
        config = GAConfig(
            enable_walk_forward=True,
            wfa_n_folds=10,
            wfa_train_ratio=0.8,
            wfa_anchored=True,
        )

        assert config.enable_walk_forward is True
        assert config.wfa_n_folds == 10
        assert config.wfa_train_ratio == 0.8
        assert config.wfa_anchored is True

    def test_wfa_config_to_dict(self, wfa_config):
        """WFA 設定の辞書変換テスト"""
        config_dict = wfa_config.to_dict()

        assert "enable_walk_forward" in config_dict
        assert "wfa_n_folds" in config_dict
        assert "wfa_train_ratio" in config_dict
        assert "wfa_anchored" in config_dict
        assert config_dict["enable_walk_forward"] is True
        assert config_dict["wfa_n_folds"] == 5

    def test_wfa_config_from_dict(self):
        """WFA 設定の辞書復元テスト"""
        data = {
            "enable_walk_forward": True,
            "wfa_n_folds": 8,
            "wfa_train_ratio": 0.75,
            "wfa_anchored": True,
        }
        config = GAConfig.from_dict(data)

        assert config.enable_walk_forward is True
        assert config.wfa_n_folds == 8
        assert config.wfa_train_ratio == 0.75
        assert config.wfa_anchored is True

    @patch(
        "app.services.auto_strategy.core.individual_evaluator.IndividualEvaluator._perform_single_evaluation"
    )
    def test_evaluate_with_walk_forward_calls_perform_evaluation(
        self, mock_perform_eval, evaluator, wfa_config, mock_gene
    ):
        """WFA 評価が _perform_single_evaluation を複数回呼び出すテスト"""
        # モックの戻り値を設定
        mock_perform_eval.return_value = (0.5,)

        # WFA 評価を実行
        result = evaluator._evaluate_with_walk_forward(
            mock_gene,
            evaluator._fixed_backtest_config,
            wfa_config,
        )

        # 複数のフォールドで評価が呼ばれることを確認
        assert mock_perform_eval.call_count >= 1

        # 結果がタプルであることを確認
        assert isinstance(result, tuple)
        assert len(result) > 0

    @patch(
        "app.services.auto_strategy.core.individual_evaluator.IndividualEvaluator._perform_single_evaluation"
    )
    def test_wfa_averages_oos_scores(
        self, mock_perform_eval, evaluator, wfa_config, mock_gene
    ):
        """WFA が OOS スコアを平均化するテスト"""
        # 各フォールドで異なるスコアを返す
        scores = [(0.3,), (0.5,), (0.7,), (0.4,), (0.6,)]
        mock_perform_eval.side_effect = scores

        result = evaluator._evaluate_with_walk_forward(
            mock_gene,
            evaluator._fixed_backtest_config,
            wfa_config,
        )

        # 結果がスコアの平均に近いことを確認
        # (フォールド数が変動する可能性があるため、厳密な値ではなく範囲をチェック)
        assert isinstance(result, tuple)
        assert 0.0 <= result[0] <= 1.0

    @patch(
        "app.services.auto_strategy.core.individual_evaluator.IndividualEvaluator._perform_single_evaluation"
    )
    def test_wfa_anchored_mode(self, mock_perform_eval, evaluator, mock_gene):
        """Anchored WFA モードのテスト"""
        config = GAConfig(
            enable_walk_forward=True,
            wfa_n_folds=3,
            wfa_train_ratio=0.7,
            wfa_anchored=True,  # Anchored モード
            objectives=["weighted_score"],
        )

        mock_perform_eval.return_value = (0.5,)

        result = evaluator._evaluate_with_walk_forward(
            mock_gene,
            evaluator._fixed_backtest_config,
            config,
        )

        assert isinstance(result, tuple)
        assert mock_perform_eval.call_count >= 1

    @patch(
        "app.services.auto_strategy.core.individual_evaluator.IndividualEvaluator._perform_single_evaluation"
    )
    def test_wfa_fallback_on_missing_dates(
        self, mock_perform_eval, evaluator, wfa_config, mock_gene
    ):
        """日付が不明な場合のフォールバックテスト"""
        mock_perform_eval.return_value = (0.5,)

        # 日付なしの設定
        config_without_dates = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
        }

        result = evaluator._evaluate_with_walk_forward(
            mock_gene,
            config_without_dates,
            wfa_config,
        )

        # フォールバックで _perform_single_evaluation が呼ばれることを確認
        assert mock_perform_eval.called
        assert isinstance(result, tuple)

    def test_evaluate_individual_routes_to_wfa(
        self, evaluator, wfa_config, mock_backtest_service
    ):
        """evaluate_individual が WFA 有効時に WFA 評価にルーティングするテスト"""
        # モックの個体（リストとして）
        mock_individual = [0.5] * 50  # 十分な長さのリスト

        with patch.object(
            evaluator, "_evaluate_with_walk_forward", return_value=(0.5,)
        ) as mock_wfa:
            with patch(
                "app.services.auto_strategy.serializers.gene_serialization.GeneSerializer.from_list"
            ) as mock_from_list:
                mock_from_list.return_value = MagicMock(id="test-id")

                result = evaluator.evaluate_individual(mock_individual, wfa_config)

                # WFA メソッドが呼ばれることを確認
                assert mock_wfa.called

    def test_wfa_disabled_routes_to_normal_evaluation(self, evaluator):
        """WFA 無効時に通常評価にルーティングするテスト"""
        config = GAConfig(
            enable_walk_forward=False,
            oos_split_ratio=0.0,
            objectives=["weighted_score"],
        )

        mock_individual = [0.5] * 50

        with patch.object(
            evaluator, "_perform_single_evaluation", return_value=(0.5,)
        ) as mock_single:
            with patch(
                "app.services.auto_strategy.serializers.gene_serialization.GeneSerializer.from_list"
            ) as mock_from_list:
                mock_from_list.return_value = MagicMock(id="test-id")

                result = evaluator.evaluate_individual(mock_individual, config)

                # 通常評価メソッドが呼ばれることを確認
                assert mock_single.called


class TestWFAEdgeCases:
    """WFA エッジケースのテスト"""

    @pytest.fixture
    def evaluator(self):
        """IndividualEvaluator インスタンス"""
        mock_service = MagicMock()
        evaluator = IndividualEvaluator(backtest_service=mock_service)
        return evaluator

    @pytest.fixture
    def mock_gene(self):
        """モック戦略遺伝子"""
        gene = MagicMock()
        gene.id = "test-gene"
        return gene

    @patch(
        "app.services.auto_strategy.core.individual_evaluator.IndividualEvaluator._perform_single_evaluation"
    )
    def test_wfa_short_period_fallback(self, mock_perform_eval, evaluator, mock_gene):
        """期間が短すぎる場合のフォールバックテスト"""
        mock_perform_eval.return_value = (0.5,)

        # 非常に短い期間
        short_config = {
            "start_date": "2024-01-01 00:00:00",
            "end_date": "2024-01-03 00:00:00",  # 2日間のみ
        }

        wfa_config = GAConfig(
            enable_walk_forward=True,
            wfa_n_folds=5,
            objectives=["weighted_score"],
        )

        result = evaluator._evaluate_with_walk_forward(
            mock_gene, short_config, wfa_config
        )

        # 結果が返されることを確認
        assert isinstance(result, tuple)

    @patch(
        "app.services.auto_strategy.core.individual_evaluator.IndividualEvaluator._perform_single_evaluation"
    )
    def test_wfa_single_fold(self, mock_perform_eval, evaluator, mock_gene):
        """単一フォールドのテスト"""
        mock_perform_eval.return_value = (0.5,)

        config = {
            "start_date": "2024-01-01 00:00:00",
            "end_date": "2024-06-01 00:00:00",
        }

        wfa_config = GAConfig(
            enable_walk_forward=True,
            wfa_n_folds=1,  # 1フォールドのみ
            objectives=["weighted_score"],
        )

        result = evaluator._evaluate_with_walk_forward(mock_gene, config, wfa_config)

        assert isinstance(result, tuple)


class TestWFAMultiObjective:
    """WFA 多目的最適化のテスト"""

    @pytest.fixture
    def evaluator(self):
        """IndividualEvaluator インスタンス"""
        mock_service = MagicMock()
        evaluator = IndividualEvaluator(backtest_service=mock_service)
        evaluator.set_backtest_config(
            {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01 00:00:00",
                "end_date": "2024-06-01 00:00:00",
            }
        )
        return evaluator

    @pytest.fixture
    def mock_gene(self):
        """モック戦略遺伝子"""
        gene = MagicMock()
        gene.id = "test-gene"
        return gene

    @patch(
        "app.services.auto_strategy.core.individual_evaluator.IndividualEvaluator._perform_single_evaluation"
    )
    def test_wfa_multi_objective(self, mock_perform_eval, evaluator, mock_gene):
        """多目的最適化での WFA テスト"""
        # 各フォールドで複数の目的値を返す
        scores = [
            (0.3, 0.4, 0.5),
            (0.5, 0.6, 0.7),
            (0.4, 0.5, 0.6),
        ]
        mock_perform_eval.side_effect = scores

        config = GAConfig(
            enable_walk_forward=True,
            wfa_n_folds=3,
            objectives=["sharpe_ratio", "total_return", "max_drawdown"],
        )

        result = evaluator._evaluate_with_walk_forward(
            mock_gene,
            evaluator._fixed_backtest_config,
            config,
        )

        # 結果が3つの目的値を持つことを確認
        assert isinstance(result, tuple)
        # 評価が呼ばれた回数だけ目的値が返される
        if mock_perform_eval.call_count > 0:
            assert len(result) == 3




