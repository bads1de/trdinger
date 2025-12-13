"""
StrategyParameterTuner のテスト
"""

import pytest
from unittest.mock import Mock, patch

from app.services.auto_strategy.config.ga_runtime import GAConfig
from app.services.auto_strategy.models.indicator_gene import IndicatorGene
from app.services.auto_strategy.models.strategy_gene import StrategyGene
from app.services.auto_strategy.models.tpsl_gene import TPSLGene
from app.services.auto_strategy.optimization.strategy_parameter_tuner import (
    StrategyParameterTuner,
)


class TestStrategyParameterTuner:
    """StrategyParameterTuner のテスト"""

    @pytest.fixture
    def mock_evaluator(self):
        """モック IndividualEvaluator"""
        evaluator = Mock()
        evaluator.evaluate_individual.return_value = (0.5,)  # フィットネスタプル
        return evaluator

    @pytest.fixture
    def sample_config(self):
        """テスト用の GAConfig"""
        return GAConfig(
            population_size=10,
            generations=5,
            enable_walk_forward=False,
        )

    @pytest.fixture
    def sample_gene(self):
        """テスト用の StrategyGene"""
        return StrategyGene(
            id="test-gene-001",
            indicators=[
                IndicatorGene(type="RSI", parameters={"length": 14}),
            ],
            tpsl_gene=TPSLGene(
                stop_loss_pct=0.03,
                take_profit_pct=0.06,
            ),
        )

    def test_init(self, mock_evaluator, sample_config):
        """初期化のテスト"""
        tuner = StrategyParameterTuner(
            evaluator=mock_evaluator,
            config=sample_config,
            n_trials=10,
        )

        assert tuner.evaluator == mock_evaluator
        assert tuner.config == sample_config
        assert tuner.n_trials == 10

    def test_init_with_options(self, mock_evaluator, sample_config):
        """オプション付き初期化のテスト"""
        tuner = StrategyParameterTuner(
            evaluator=mock_evaluator,
            config=sample_config,
            n_trials=20,
            use_wfa=False,
            include_indicators=True,
            include_tpsl=False,
            include_thresholds=True,
        )

        assert tuner.n_trials == 20
        assert tuner.use_wfa is False
        assert tuner.include_indicators is True
        assert tuner.include_tpsl is False
        assert tuner.include_thresholds is True

    @patch(
        "app.services.auto_strategy.optimization.strategy_parameter_tuner.OptunaOptimizer"
    )
    def test_tune_basic(
        self, mock_optimizer_class, mock_evaluator, sample_config, sample_gene
    ):
        """基本的なチューニングのテスト"""
        # モックの設定
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer

        mock_result = Mock()
        mock_result.best_params = {"ind_0_length": 21, "tpsl_stop_loss_pct": 0.04}
        mock_result.best_score = 0.75
        mock_result.total_evaluations = 10
        mock_result.optimization_time = 5.0
        mock_optimizer.optimize.return_value = mock_result

        tuner = StrategyParameterTuner(
            evaluator=mock_evaluator,
            config=sample_config,
            n_trials=10,
        )

        result = tuner.tune(sample_gene)

        # Optuna 最適化が呼ばれた
        mock_optimizer.optimize.assert_called_once()
        mock_optimizer.cleanup.assert_called_once()

        # 結果に最適化情報が含まれる
        assert result.metadata.get("optuna_tuned") is True
        assert result.metadata.get("optuna_best_score") == 0.75
        assert result.metadata.get("optuna_trials") == 10

    def test_tune_empty_parameter_space(self, mock_evaluator, sample_config):
        """パラメータ空間が空の場合のテスト"""
        empty_gene = StrategyGene(id="empty-gene")

        tuner = StrategyParameterTuner(
            evaluator=mock_evaluator,
            config=sample_config,
            n_trials=10,
            include_indicators=True,
            include_tpsl=False,
            include_thresholds=False,
        )

        result = tuner.tune(empty_gene)

        # 元の遺伝子がそのまま返される
        assert result.id == "empty-gene"

    @patch(
        "app.services.auto_strategy.optimization.strategy_parameter_tuner.OptunaOptimizer"
    )
    def test_tune_error_handling(
        self, mock_optimizer_class, mock_evaluator, sample_config, sample_gene
    ):
        """エラー時の処理テスト"""
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        mock_optimizer.optimize.side_effect = Exception("Optimization failed")

        tuner = StrategyParameterTuner(
            evaluator=mock_evaluator,
            config=sample_config,
            n_trials=10,
        )

        result = tuner.tune(sample_gene)

        # エラー時は元の遺伝子が返される
        assert result.id == sample_gene.id
        # クリーンアップは呼ばれる
        mock_optimizer.cleanup.assert_called_once()

    def test_evaluate_gene(self, mock_evaluator, sample_config, sample_gene):
        """遺伝子評価のテスト"""
        mock_evaluator.evaluate_individual.return_value = (0.65,)

        tuner = StrategyParameterTuner(
            evaluator=mock_evaluator,
            config=sample_config,
            n_trials=10,
            use_wfa=False,
        )

        fitness = tuner._evaluate_gene(sample_gene)

        assert fitness == 0.65
        mock_evaluator.evaluate_individual.assert_called_once()

    def test_evaluate_gene_error(self, mock_evaluator, sample_config, sample_gene):
        """遺伝子評価エラー時のテスト"""
        mock_evaluator.evaluate_individual.side_effect = Exception("Evaluation failed")

        tuner = StrategyParameterTuner(
            evaluator=mock_evaluator,
            config=sample_config,
            n_trials=10,
        )

        fitness = tuner._evaluate_gene(sample_gene)

        # エラー時は 0.0 が返される
        assert fitness == 0.0

    def test_create_wfa_config(self, mock_evaluator, sample_config):
        """WFA 設定作成のテスト"""
        sample_config.enable_walk_forward = True
        sample_config.wfa_n_folds = 5

        tuner = StrategyParameterTuner(
            evaluator=mock_evaluator,
            config=sample_config,
            n_trials=10,
            use_wfa=True,
        )

        wfa_config = tuner._create_wfa_config()

        assert wfa_config.enable_walk_forward is True
        # フォールド数は高速化のため 3 以下に制限される
        assert wfa_config.wfa_n_folds <= 3

    @patch(
        "app.services.auto_strategy.optimization.strategy_parameter_tuner.OptunaOptimizer"
    )
    def test_tune_multiple(
        self, mock_optimizer_class, mock_evaluator, sample_config, sample_gene
    ):
        """複数遺伝子のチューニングテスト"""
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer

        mock_result = Mock()
        mock_result.best_params = {"ind_0_length": 21}
        mock_result.best_score = 0.75
        mock_result.total_evaluations = 10
        mock_result.optimization_time = 5.0
        mock_optimizer.optimize.return_value = mock_result

        tuner = StrategyParameterTuner(
            evaluator=mock_evaluator,
            config=sample_config,
            n_trials=10,
        )

        genes = [sample_gene, sample_gene, sample_gene]
        results = tuner.tune_multiple(genes, top_n=2)

        # 上位 2 個のみチューニングされる
        assert len(results) == 2
        assert mock_optimizer.optimize.call_count == 2

    def test_tune_multiple_no_limit(self, mock_evaluator, sample_config, sample_gene):
        """全遺伝子をチューニングするテスト"""
        with patch(
            "app.services.auto_strategy.optimization.strategy_parameter_tuner.OptunaOptimizer"
        ) as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer

            mock_result = Mock()
            mock_result.best_params = {}
            mock_result.best_score = 0.5
            mock_result.total_evaluations = 10
            mock_result.optimization_time = 1.0
            mock_optimizer.optimize.return_value = mock_result

            tuner = StrategyParameterTuner(
                evaluator=mock_evaluator,
                config=sample_config,
                n_trials=10,
            )

            genes = [sample_gene] * 3
            results = tuner.tune_multiple(genes, top_n=None)

            # 全て（3 個）チューニングされる
            assert len(results) == 3
