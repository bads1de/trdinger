import pytest
from unittest.mock import MagicMock, call, ANY
from app.services.auto_strategy.optimization.strategy_parameter_tuner import StrategyParameterTuner
from app.services.auto_strategy.genes.strategy import StrategyGene
from app.services.auto_strategy.config.ga import GAConfig

class TestStrategyParameterTuner:
    @pytest.fixture
    def mock_evaluator(self):
        evaluator = MagicMock()
        # 評価結果はタプル (fitness, ...)
        evaluator.evaluate_individual.return_value = (1.5, )
        return evaluator

    @pytest.fixture
    def mock_config(self):
        config = MagicMock(spec=GAConfig)
        config.enable_walk_forward = False
        config.wfa_n_folds = 5
        return config

    @pytest.fixture
    def mock_gene(self):
        gene = MagicMock(spec=StrategyGene)
        gene.metadata = {}
        return gene

    @pytest.fixture
    def tuner(self, mock_evaluator, mock_config):
        # 内部コンポーネントをモック化するために継承またはパッチを使用するが、
        # ここではインスタンス生成後に属性を差し替える手法をとる
        tuner = StrategyParameterTuner(mock_evaluator, mock_config)
        tuner.parameter_space_builder = MagicMock()
        tuner.optimizer = MagicMock()
        return tuner

    def test_tune_success(self, tuner, mock_gene):
        # Arrange
        tuner.parameter_space_builder.build_parameter_space.return_value = {"param1": [1, 10]}
        
        # モックされたoptimizerの結果
        mock_result = MagicMock()
        mock_result.best_params = {"param1": 5}
        mock_result.best_score = 2.0
        mock_result.total_evaluations = 10
        mock_result.optimization_time = 1.0
        tuner.optimizer.optimize.return_value = mock_result
        
        # apply_params_to_gene は新しい遺伝子（または更新された遺伝子）を返す
        tuned_gene = MagicMock(spec=StrategyGene)
        tuned_gene.metadata = {}
        tuner.parameter_space_builder.apply_params_to_gene.return_value = tuned_gene

        # Act
        result = tuner.tune(mock_gene)

        # Assert
        assert result == tuned_gene
        assert result.metadata["optuna_tuned"] is True
        assert result.metadata["optuna_best_score"] == 2.0
        
        tuner.parameter_space_builder.build_parameter_space.assert_called_once()
        tuner.optimizer.optimize.assert_called_once()
        tuner.parameter_space_builder.apply_params_to_gene.assert_called_with(mock_gene, {"param1": 5})
        tuner.optimizer.cleanup.assert_called_once()

    def test_tune_no_params(self, tuner, mock_gene):
        # パラメータ空間が空の場合
        tuner.parameter_space_builder.build_parameter_space.return_value = {}
        
        result = tuner.tune(mock_gene)
        
        # 最適化は実行されず、元の遺伝子が返る
        tuner.optimizer.optimize.assert_not_called()
        assert result == mock_gene

    def test_tune_exception(self, tuner, mock_gene):
        # 最適化中に例外
        tuner.parameter_space_builder.build_parameter_space.return_value = {"p": [1, 2]}
        tuner.optimizer.optimize.side_effect = Exception("Optimization Failed")
        
        result = tuner.tune(mock_gene)
        
        # 例外はログ出力され、元の遺伝子が返る（クラッシュしない）
        assert result == mock_gene
        tuner.optimizer.cleanup.assert_called_once()

    def test_evaluate_gene_basic(self, tuner, mock_gene):
        # WFAなし
        tuner.use_wfa = False
        
        score = tuner._evaluate_gene(mock_gene)
        
        assert score == 1.5
        tuner.evaluator.evaluate_individual.assert_called_with(mock_gene, tuner.config)

    def test_evaluate_gene_wfa(self, tuner, mock_gene):
        # WFAあり
        tuner.use_wfa = True
        tuner.config.enable_walk_forward = True
        
        score = tuner._evaluate_gene(mock_gene)
        
        assert score == 1.5
        # 呼び出し時のconfigはWFA設定が書き換わったもの（コピー）であるはず
        # assert_called_with で完全一致を確認するのは難しいので、呼び出しが行われたことだけ確認
        args, _ = tuner.evaluator.evaluate_individual.call_args
        assert args[0] == mock_gene
        assert args[1].enable_walk_forward is True
        # フォールド数が制限されているか
        assert args[1].wfa_n_folds <= 3

    def test_objective_function_integration(self, tuner, mock_gene):
        # optimizeに渡されるobjective関数が正しく動作するか検証
        tuner.parameter_space_builder.build_parameter_space.return_value = {"p": [1, 2]}
        
        # モックの中で、optimizeの第一引数（objective関数）を実行するような仕掛け
        def side_effect_optimize(objective, space, trials):
            # objective関数を試しに呼んでみる
            params = {"p": 1}
            score = objective(params)
            # 結果オブジェクトを返す
            res = MagicMock()
            res.best_params = params
            res.best_score = score
            return res
            
        tuner.optimizer.optimize.side_effect = side_effect_optimize
        
        # apply_params_to_geneの戻り値設定
        tuned_gene = MagicMock(spec=StrategyGene)
        tuner.parameter_space_builder.apply_params_to_gene.return_value = tuned_gene
        
        tuner.tune(mock_gene)
        
        # 評価が行われたはず
        tuner.parameter_space_builder.apply_params_to_gene.assert_any_call(mock_gene, {"p": 1})
        tuner.evaluator.evaluate_individual.assert_called()

    def test_tune_multiple(self, tuner):
        genes = [MagicMock(), MagicMock(), MagicMock()]
        tuner.tune = MagicMock()
        tuner.tune.side_effect = lambda g: g # そのまま返す
        
        results = tuner.tune_multiple(genes, top_n=2)
        
        assert len(results) == 2
        assert tuner.tune.call_count == 2
