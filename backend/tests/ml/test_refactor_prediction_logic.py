
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from backend.app.services.ml.ml_training_service import MLTrainingService
from backend.app.services.ml.base_ml_trainer import BaseMLTrainer
from backend.app.services.ml.single_model.single_model_trainer import SingleModelTrainer

class TestRefactorPredictionLogic:
    """予測ロジックのリファクタリングを検証するテスト"""

    @pytest.fixture
    def mock_features(self):
        """テスト用の特徴量DataFrame"""
        return pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [0.5, 0.6, 0.7],
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })

    @pytest.fixture
    def mock_service(self):
        """MLTrainingServiceのモック"""
        service = MLTrainingService(trainer_type="single")
        # トレーナーをモック化
        service.trainer = MagicMock(spec=SingleModelTrainer)
        service.trainer.is_trained = True
        service.trainer.feature_columns = ['feature1', 'feature2']
        service.trainer.scaler = MagicMock()
        # スケーラーの動作定義
        service.trainer.scaler.transform.side_effect = lambda x: x.values
        
        # モデルのモック
        service.trainer.model = MagicMock()
        # best_iterationの設定
        service.trainer.model.best_iteration = 100
        
        return service

    def test_generate_signals_delegation(self, mock_service, mock_features):
        """
        generate_signalsがBaseMLTrainer.predict_signalに委譲しているか確認
        """
        # predict_signalのモック戻り値
        expected_result = {'down': 0.1, 'range': 0.2, 'up': 0.7}
        mock_service.trainer.predict_signal = MagicMock(return_value=expected_result)
        
        # 実行
        result = mock_service.generate_signals(mock_features)
        
        # 検証
        assert result == expected_result
        
        # predict_signalが正しく呼ばれたか確認
        mock_service.trainer.predict_signal.assert_called_once_with(mock_features)

    def test_base_ml_trainer_predict_signal_implementation(self):
        """BaseMLTrainerに実装されたpredict_signalのテスト"""
        
        # テスト用の具象クラス
        class ConcreteTrainer(BaseMLTrainer):
            def _train_model_impl(self, *args, **kwargs): pass
            def predict(self, features_df): 
                # ダミーの確率を返す (scaler適用後の値をチェックしたい場合はここでassert可能)
                return np.array([[0.1, 0.2, 0.7]])
                
        trainer = ConcreteTrainer()
        trainer.is_trained = True
        trainer.feature_columns = ['f1', 'f2']
        trainer.scaler = MagicMock()
        trainer.scaler.transform.return_value = np.array([[1.0, 2.0]])
        
        # 設定オブジェクトのモック (get_default_predictionsが必要)
        trainer.config = MagicMock()
        trainer.config.prediction.get_default_predictions.return_value = {'down': 0.33, 'range': 0.34, 'up': 0.33}

        # テストデータ (カラム不足あり)
        features = pd.DataFrame({'f1': [10]})
        
        # 実行
        result = trainer.predict_signal(features)
        
        # 検証
        assert result == {'down': 0.1, 'range': 0.2, 'up': 0.7}
        
        # スケーラーが呼ばれたか（欠損補完後のデータで呼ばれるはず）
        assert trainer.scaler.transform.called
        # 呼ばれた時の引数のshape確認 (1行2列になっているはず)
        args, _ = trainer.scaler.transform.call_args
        assert args[0].shape == (1, 2)
        assert 'f2' in args[0].columns # 欠損カラムが追加されていること

