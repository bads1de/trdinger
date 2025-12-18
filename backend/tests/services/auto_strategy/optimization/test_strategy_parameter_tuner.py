"""
StrategyParameterTunerのユニットテスト
"""

import pytest
from unittest.mock import Mock, MagicMock

from app.services.auto_strategy.optimization.strategy_parameter_tuner import StrategyParameterTuner
from app.services.auto_strategy.genes.strategy import StrategyGene
from app.services.auto_strategy.config.ga import GAConfig


class TestStrategyParameterTuner:
    """StrategyParameterTunerのテストクラス"""

    @pytest.fixture
    def mock_evaluator(self):
        evaluator = Mock()
        # フィットネススコアを返す
        evaluator.evaluate_individual.return_value = (0.5,)
        return evaluator

    @pytest.fixture
    def tuner(self, mock_evaluator):
        config = GAConfig()
        # 試行回数を減らしてテストを高速化
        return StrategyParameterTuner(evaluator=mock_evaluator, config=config, n_trials=2)

    def test_tune_basic(self, tuner):
        """基本的なチューニング実行テスト"""
        # サンプル遺伝子（ID付き）
        gene = StrategyGene(id="test_tune")
        
        # モックの設定
        mock_result = Mock()
        mock_result.best_score = 0.8
        mock_result.best_params = {"param1": 10}
        mock_result.total_evaluations = 2
        mock_result.optimization_time = 1.0
        
        tuner.optimizer = Mock()
        tuner.optimizer.optimize.return_value = mock_result
        
        # parameter_space_builderをモックして、予測可能な結果を返す
        tuner.parameter_space_builder = Mock()
        tuner.parameter_space_builder.build_parameter_space.return_value = {"param1": Mock()}
        # apply_params_to_gene が新しい遺伝子を返すように設定
        mock_gene = StrategyGene(id="tuned_gene")
        mock_gene.metadata = {} # 初期化
        tuner.parameter_space_builder.apply_params_to_gene.return_value = mock_gene
        
        tuned_gene = tuner.tune(gene)
        
        assert tuned_gene.id == "tuned_gene"
        assert tuned_gene.metadata.get("optuna_tuned") is True
        assert tuned_gene.metadata.get("optuna_best_score") == 0.8

    def test_evaluate_gene_handles_wfa(self, tuner, mock_evaluator):
        """WFA有効時の評価テスト"""
        gene = StrategyGene()
        tuner.use_wfa = True
        tuner.config.enable_walk_forward = True
        
        score = tuner._evaluate_gene(gene)
        
        assert score == 0.5
        # evaluate_individualが呼ばれた際、wfaが有効なconfigが渡されているか
        args, kwargs = mock_evaluator.evaluate_individual.call_args
        assert args[1].enable_walk_forward is True