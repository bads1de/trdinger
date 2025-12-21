import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from app.services.ml.trainers.base_ml_trainer import BaseMLTrainer
from app.utils.error_handler import DataError
from app.services.ml.common.exceptions import MLModelError

# テスト用の具象クラス
class MockTrainer(BaseMLTrainer):
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        # クラス確率を返す (2クラス分類を想定)
        n = len(features_df)
        probs = np.zeros((n, 2))
        probs[:, 0] = 0.3  # class 0
        probs[:, 1] = 0.7  # class 1
        return probs

    def _train_model_impl(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        **training_params,
    ) -> dict:
        self._model = MagicMock()
        return {
            "accuracy": 0.8,
            "balanced_accuracy": 0.75,
            "f1_score": 0.78,
        }

class TestBaseMLTrainer:
    @pytest.fixture
    def trainer(self):
        return MockTrainer()

    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start="2023-01-01", periods=150, freq="h")
        df = pd.DataFrame({
            "open": np.random.randn(150) + 100,
            "high": np.random.randn(150) + 101,
            "low": np.random.randn(150) + 99,
            "close": np.random.randn(150) + 100,
            "volume": np.random.rand(150) * 1000
        }, index=dates)
        return df

    def test_train_model_insufficient_data(self, trainer):
        """データ不足の場合のテスト"""
        short_data = pd.DataFrame(np.random.randn(50, 5), columns=["open", "high", "low", "close", "volume"])
        # @safe_ml_operation によって例外はキャッチされ、デフォルト値が返される
        result = trainer.train_model(short_data)
        assert result["success"] is False

    def test_train_model_success(self, trainer, sample_data):
        """正常な学習フローのテスト"""
        with patch.object(trainer.feature_service, 'calculate_advanced_features', return_value=pd.DataFrame(np.random.randn(150, 10), index=sample_data.index)):
            with patch.object(trainer.label_service, 'prepare_labels', return_value=(pd.DataFrame(np.random.randn(140, 10)), pd.Series(np.random.randint(0, 2, 140)))):
                with patch('app.services.ml.trainers.base_ml_trainer.model_manager.save_model', return_value="/path/to/model"):
                    result = trainer.train_model(sample_data, save_model=True)
                    
                    assert result["success"] is True
                    assert trainer.is_trained is True
                    assert "accuracy" in result
                    assert result["model_path"] == "/path/to/model"

    def test_predict_signal_not_trained(self, trainer, sample_data):
        """未学習状態での予測"""
        # 未学習時はデフォルト値を返すべき
        result = trainer.predict_signal(sample_data)
        assert "is_valid" in result
        # デフォルト値を確認（configの設定によるが、通常は0.5や0.0など）

    def test_predict_signal_success(self, trainer, sample_data):
        """学習後の予測シグナル取得"""
        trainer.is_trained = True
        trainer.feature_columns = ["feat1", "feat2"]
        trainer._model = MagicMock()
        
        features = pd.DataFrame(np.random.randn(10, 2), columns=["feat1", "feat2"], index=sample_data.index[:10])
        
        with patch('app.services.ml.trainers.base_ml_trainer.prepare_data_for_prediction', return_value=features):
            result = trainer.predict_signal(features)
            assert "is_valid" in result
            assert result["is_valid"] == 0.7  # MockTrainerのpredictが返す値

    def test_load_model_failure(self, trainer):
        """モデル読み込み失敗"""
        with patch('app.services.ml.trainers.base_ml_trainer.model_manager.load_model', return_value=None):
            result = trainer.load_model("/invalid/path")
            assert result is False
            assert trainer.is_trained is False

    def test_load_model_success(self, trainer):
        """モデル読み込み成功"""
        model_data = {
            "model": MagicMock(),
            "scaler": MagicMock(),
            "feature_columns": ["f1", "f2"],
            "metadata": {"type": "test"}
        }
        with patch('app.services.ml.trainers.base_ml_trainer.model_manager.load_model', return_value=model_data):
            result = trainer.load_model("/path/to/model")
            assert result is True
            assert trainer.is_trained is True
            assert trainer.feature_columns == ["f1", "f2"]

    def test_calculate_features_fallback(self, trainer, sample_data):
        """特徴量計算エラー時のフォールバック"""
        with patch.object(trainer.feature_service, 'calculate_advanced_features', side_effect=Exception("Calc error")):
            # エラー時は元のデータをコピーして返すべき
            result = trainer._calculate_features(sample_data)
            pd.testing.assert_frame_equal(result, sample_data)

    def test_cross_validation_flow(self, trainer, sample_data):
        """クロスバリデーションのフロー確認"""
        X = pd.DataFrame(np.random.randn(150, 5), index=sample_data.index)
        y = pd.Series(np.random.randint(0, 2, 150), index=sample_data.index)
        
        with patch('app.services.ml.trainers.base_ml_trainer.PurgedKFold') as mock_kfold:
            mock_kfold.return_value.split.return_value = [
                (np.arange(100), np.arange(100, 120)),
                (np.arange(20, 120), np.arange(120, 140))
            ]
            
            # 各フォールドの結果をシミュレート
            trainer._train_model_impl = MagicMock(return_value={"accuracy": 0.8})
            
            result = trainer._time_series_cross_validate(X, y, cv_splits=2)
            
            assert "cv_scores" in result
            assert len(result["cv_scores"]) == 2
            assert result["mean_score"] == 0.8

    def test_cleanup_resources(self, trainer):
        """リソースクリーンアップのテスト"""
        trainer.is_trained = True
        trainer._model = MagicMock()
        
        trainer.cleanup_resources()
        
        assert trainer._model is None
        assert trainer.is_trained is False
