import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from app.services.ml.optimization.optimization_service import OptimizationService, OptimizationSettings
from app.services.ml.ensemble.ensemble_trainer import EnsembleTrainer

class TestOptimizationService:
    @pytest.fixture
    def service(self):
        return OptimizationService()

    @pytest.fixture
    def sample_data(self):
        n = 100
        df = pd.DataFrame({
            "open": np.random.randn(n) + 100,
            "high": np.random.randn(n) + 101,
            "low": np.random.randn(n) + 99,
            "close": np.random.randn(n) + 100,
            "volume": np.random.rand(n) * 1000
        }, index=pd.date_range("2023-01-01", periods=n, freq="h"))
        return df

    def test_optimize_parameters_success(self, service, sample_data):
        """パラメータ最適化の成功フロー"""
        mock_trainer = MagicMock(spec=EnsembleTrainer)
        mock_trainer.ensemble_config = {"method": "stacking", "models": ["lightgbm"]}
        
        # 1. 最適化設定
        settings = OptimizationSettings(enabled=True, n_calls=2)
        
        # 2. Optunaの結果をシミュレート
        mock_res = MagicMock()
        mock_res.best_params = {"lgb_num_leaves": 50}
        mock_res.best_score = 0.8
        mock_res.total_evaluations = 2
        mock_res.optimization_time = 1.5
        
        with patch.object(service.optimizer, 'optimize', return_value=mock_res):
            # 目的関数の作成をパッチして実際の学習を避ける
            with patch.object(service, '_create_objective_function'):
                result = service.optimize_parameters(mock_trainer, sample_data, settings)
                
                assert result["best_params"] == {"lgb_num_leaves": 50}
                assert result["best_score"] == 0.8
                assert result["method"] == "optuna"

    def test_prepare_parameter_space(self, service):
        """パラメータ空間の準備テスト"""
        # トレーナーに基づく自動生成
        mock_trainer = MagicMock()
        mock_trainer.ensemble_config = {"method": "stacking", "models": ["lightgbm"]}
        
        settings = OptimizationSettings(enabled=True)
        ps = service._prepare_parameter_space(mock_trainer, settings)
        
        assert "lgb_num_leaves" in ps
        assert "stacking_meta_C" in ps

    def test_create_temp_trainer(self, service):
        """一時的なトレーナーの作成"""
        original_trainer = MagicMock()
        original_trainer.ensemble_config = {
            "method": "stacking",
            "models": ["lightgbm"],
            "stacking_params": {"cv_folds": 5}
        }
        
        temp_trainer = service._create_temp_trainer(original_trainer, {"lgb_num_leaves": 30})
        
        assert temp_trainer.ensemble_config["method"] == "stacking"
        # CV foldsが削減されていること（最適化高速化のため）
        assert temp_trainer.ensemble_config["stacking_params"]["cv_folds"] == 3

    def test_objective_function_wrapper(self, service, sample_data):
        """目的関数のラップ処理テスト"""
        mock_trainer = MagicMock()
        mock_trainer.ensemble_config = {"method": "stacking"}
        settings = OptimizationSettings(enabled=True, n_calls=1)
        
        # 内部で呼ばれる temp_trainer をモック化
        with patch.object(service, '_create_temp_trainer') as mock_create:
            temp_mock = mock_create.return_value
            # 学習結果としてF1スコアを返すように設定
            temp_mock.train_model.return_value = {"f1_score": 0.75}
            
            obj_func = service._create_objective_function(mock_trainer, sample_data, settings)
            score = obj_func({"lgb_num_leaves": 30})
            
            assert score == 0.75
            assert temp_mock.train_model.called

    def test_objective_function_error_fallback(self, service, sample_data):
        """目的関数内でエラーが起きた際のスコア返却"""
        mock_trainer = MagicMock()
        settings = OptimizationSettings(enabled=True, n_calls=1)
        
        with patch.object(service, '_create_temp_trainer', side_effect=Exception("Fatal")):
            obj_func = service._create_objective_function(mock_trainer, sample_data, settings)
            score = obj_func({})
            # エラー時は 0.0 を返して最適化を止めない
            assert score == 0.0
