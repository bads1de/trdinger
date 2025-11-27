"""
BaseMLTrainerのTimeSeriesSplit対応テスト

TDDアプローチにより、TimeSeriesSplitをデフォルトのCV手法とする
新機能をテストファーストで実装します。
BaseMLTrainerは抽象クラスのため、ConcreteMLTrainerを使用してテストします。
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from backend.app.services.ml.base_ml_trainer import BaseMLTrainer

# テスト用具象クラス
class ConcreteMLTrainer(BaseMLTrainer):
    def _train_model_impl(self, X_train, y_train, X_test=None, y_test=None, **kwargs):
        return {
            "accuracy": 0.85, 
            "f1_score": 0.82,
            "model": "dummy"
        }
    
    def predict(self, X):
        return np.zeros(len(X))
    
    def save_model(self, filepath):
        pass
        
    def load_model(self, filepath):
        pass

class TestBaseMLTrainerTimeSeriesCV:
    """BaseMLTrainerのTimeSeriesSplit対応テスト"""

    @pytest.fixture
    def sample_timeseries_data(self):
        """時系列データのサンプル（200サンプル）"""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=200, freq="h")

        data = pd.DataFrame(
            {
                "open": 10000 + np.random.randn(200) * 100,
                "high": 10100 + np.random.randn(200) * 100,
                "low": 9900 + np.random.randn(200) * 100,
                "close": 10000 + np.random.randn(200) * 100,
                "volume": 500 + np.random.randint(100, 500, 200),
            },
            index=dates,
        )

        return data

    def test_default_split_is_timeseries(self, sample_timeseries_data):
        """デフォルトでTimeSeriesSplitが使用されることをテスト"""
        trainer = ConcreteMLTrainer()

        # データ準備や特徴量計算をモック化してテスト時間を短縮
        with patch.object(trainer, "_calculate_features", return_value=sample_timeseries_data), \
             patch.object(trainer, "_prepare_training_data", return_value=(sample_timeseries_data, pd.Series(np.zeros(200), index=sample_timeseries_data.index))):
            
            result = trainer.train_model(
                sample_timeseries_data,
                save_model=False,
            )

        assert result["success"] is True

    def test_timeseries_split_parameter_explicit(self, sample_timeseries_data):
        """use_time_series_split=Trueを明示的に指定"""
        trainer = ConcreteMLTrainer()
        
        # Splitメソッドを直接呼んで確認
        X = sample_timeseries_data
        y = pd.Series(np.zeros(len(X)), index=X.index)
        
        X_train, X_test, _, _ = trainer._split_data(X, y, use_time_series_split=True)
        
        # 時系列順序確認
        assert X_train.index[-1] < X_test.index[0]

    def test_random_split_with_flag(self, sample_timeseries_data):
        """use_random_split=Trueでランダム分割"""
        trainer = ConcreteMLTrainer()
        X = sample_timeseries_data
        y = pd.Series(np.zeros(len(X)), index=X.index)
        
        X_train, _, _, _ = trainer._split_data(X, y, use_random_split=True)
        assert len(X_train) > 0

    def test_cross_validation_with_timeseries(self, sample_timeseries_data):
        """use_cross_validation=TrueでCV実行"""
        trainer = ConcreteMLTrainer()
        
        # 内部メソッドをモック
        with patch.object(trainer, "_calculate_features", return_value=sample_timeseries_data), \
             patch.object(trainer, "_prepare_training_data", return_value=(sample_timeseries_data, pd.Series(np.zeros(200), index=sample_timeseries_data.index))):
            
            result = trainer.train_model(
                sample_timeseries_data,
                save_model=False,
                use_cross_validation=True,
                cv_splits=3,
            )

        assert result["success"] is True
        assert "cv_scores" in result
        assert len(result["cv_scores"]) == 3

    def test_cv_splits_parameter(self, sample_timeseries_data):
        """cv_splitsパラメータ動作確認"""
        trainer = ConcreteMLTrainer()
        
        with patch.object(trainer, "_calculate_features", return_value=sample_timeseries_data), \
             patch.object(trainer, "_prepare_training_data", return_value=(sample_timeseries_data, pd.Series(np.zeros(200), index=sample_timeseries_data.index))):
            
            result = trainer.train_model(
                sample_timeseries_data,
                save_model=False,
                use_cross_validation=True,
                cv_splits=2,
            )
            
        assert len(result["cv_scores"]) == 2

    def test_max_train_size_parameter(self, sample_timeseries_data):
        """max_train_sizeパラメータ動作確認"""
        trainer = ConcreteMLTrainer()
        # 内部実装のTimeSerisSplitがmax_train_sizeを受け取るか確認するのは難しいので
        # ここではエラーにならないことだけ確認
        with patch.object(trainer, "_calculate_features", return_value=sample_timeseries_data), \
             patch.object(trainer, "_prepare_training_data", return_value=(sample_timeseries_data, pd.Series(np.zeros(200), index=sample_timeseries_data.index))):
            
            result = trainer.train_model(
                sample_timeseries_data,
                save_model=False,
                use_cross_validation=True,
                max_train_size=50,
            )
        assert result["success"] is True

    def test_config_integration(self, sample_timeseries_data):
        """ml_config統合テスト"""
        # 省略: configのモックが必要になるため、統合テスト側でカバー済みとするか
        # ここではデフォルト値が使われることを確認
        trainer = ConcreteMLTrainer()
        
        with patch.object(trainer, "_calculate_features", return_value=sample_timeseries_data), \
             patch.object(trainer, "_prepare_training_data", return_value=(sample_timeseries_data, pd.Series(np.zeros(200), index=sample_timeseries_data.index))):
            
            result = trainer.train_model(
                sample_timeseries_data,
                save_model=False,
                use_cross_validation=True,
            )
            
        # デフォルトは5
        assert len(result["cv_scores"]) == 5

    def test_backward_compatibility_use_time_series_split_false(self, sample_timeseries_data):
        """use_time_series_split=Falseでランダム分割"""
        trainer = ConcreteMLTrainer()
        X = sample_timeseries_data
        y = pd.Series(np.zeros(len(X)), index=X.index)
        
        # _split_dataを直接呼ぶ
        X_train, _, _, _ = trainer._split_data(X, y, use_time_series_split=False)
        assert len(X_train) > 0

    def test_split_data_method_default_timeseries(self):
        """_split_dataデフォルト"""
        trainer = ConcreteMLTrainer()
        dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
        X = pd.DataFrame(np.random.randn(100, 2), index=dates)
        y = pd.Series(np.zeros(100), index=dates)
        
        X_train, X_test, _, _ = trainer._split_data(X, y)
        assert X_train.index[-1] < X_test.index[0]

    def test_split_data_method_with_random_flag(self):
        """_split_data random flag"""
        trainer = ConcreteMLTrainer()
        dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
        X = pd.DataFrame(np.random.randn(100, 2), index=dates)
        y = pd.Series(np.zeros(100), index=dates)
        
        X_train, X_test, _, _ = trainer._split_data(X, y, use_random_split=True)
        assert len(X_train) > 0