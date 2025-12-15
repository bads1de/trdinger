import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from app.services.ml.base_ml_trainer import BaseMLTrainer


class ConcreteMLTrainer(BaseMLTrainer):
    def _train_model_impl(self, X_train, X_test, y_train, y_test, **training_params):
        return {"accuracy": 0.5}

    def predict(self, features_df):
        return np.zeros(len(features_df))


class TestPurgedCVUnification:
    """PurgedKFold統合のテストクラス"""

    def setup_method(self):
        # ML設定のモックを設定
        with patch("app.config.unified_config.unified_config.ml.training") as mock_ml_training_config:
            # label_generation属性を適切に設定
            label_gen_mock = MagicMock()
            label_gen_mock.timeframe = "1h"
            mock_ml_training_config.label_generation = label_gen_mock
            mock_ml_training_config.prediction_horizon = 4
            mock_ml_training_config.pct_embargo = 0.01
            
            self.trainer = ConcreteMLTrainer()
        
        # モックデータ
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
        self.X = pd.DataFrame(
            np.random.rand(100, 5), index=dates, columns=[f"feat_{i}" for i in range(5)]
        )
        self.y = pd.Series(np.random.randint(0, 2, 100), index=dates)

    def test_time_series_cross_validate_uses_purged_kfold(self):
        """統合されている場合、設定フラグに関わらず_time_series_cross_validateがPurgedKFoldを使用することをテスト"""
        # PurgedKFoldが使用されていることを検証したい。
        # 現在は設定に依存している。常に使用するようにコードを変更する予定。

        # PurgedKFoldをモック化して呼び出しを確認
        with patch("app.config.unified_config.unified_config.ml.training") as mock_ml_training_config:
            label_gen_mock = MagicMock()
            label_gen_mock.timeframe = "1h"
            mock_ml_training_config.label_generation = label_gen_mock
            mock_ml_training_config.prediction_horizon = 4
            mock_ml_training_config.pct_embargo = 0.01
            
            with patch("app.services.ml.base_ml_trainer.PurgedKFold") as MockPurgedKFold:
                MockPurgedKFold.return_value.split.return_value = (
                    []
                )  # 空のジェネレータを返す

                # まずはデフォルト(True)で実行して動作確認
                self.trainer._time_series_cross_validate(self.X, self.y)

                MockPurgedKFold.assert_called()

    def test_purged_kfold_integration(self):
        """BaseMLTrainerにおけるPurgedKFoldの統合テスト"""
        # 実際の分割ロジックをテスト
        with patch("app.config.unified_config.unified_config.ml.training") as mock_ml_training_config:
            label_gen_mock = MagicMock()
            label_gen_mock.timeframe = "1h"
            mock_ml_training_config.label_generation = label_gen_mock
            mock_ml_training_config.prediction_horizon = 4
            mock_ml_training_config.pct_embargo = 0.01
            
            with patch.object(self.trainer, "_train_model_impl", return_value={"accuracy": 0.5}):
                cv_result = self.trainer._time_series_cross_validate(
                    self.X, self.y, cv_splits=3
                )

                assert "cv_scores" in cv_result
                assert "fold_results" in cv_result
                assert len(cv_result["fold_results"]) == 3



