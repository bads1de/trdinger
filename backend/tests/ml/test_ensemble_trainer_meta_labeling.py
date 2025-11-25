import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from backend.app.services.ml.ensemble.ensemble_trainer import EnsembleTrainer

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 200
    X = pd.DataFrame(np.random.rand(n_samples, 5), columns=[f'feature_{i}' for i in range(5)], index=pd.RangeIndex(n_samples))
    # 2値分類 (0, 1)
    y = pd.Series(np.random.randint(0, 2, n_samples), name='target', index=pd.RangeIndex(n_samples))
    return X, y

def test_ensemble_trainer_meta_labeling_integration(sample_data):
    X, y = sample_data
    
    # アンサンブル設定
    config = {
        "method": "stacking",
        "stacking_params": {
            "base_models": ["lightgbm", "xgboost"], # 軽量なモデルのみ
            "meta_model": "logistic_regression",
            "cv_folds": 2,
            "stack_method": "predict_proba",
            "n_jobs": 1,
            "passthrough": False
        }
    }
    
    trainer = EnsembleTrainer(config)
    
    # 学習
    # X_train, X_test, y_train, y_test を用意
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # モックを使わず、実際に学習させてみる (軽量な設定で)
    result = trainer._train_model_impl(X_train, X_test, y_train, y_test, random_state=42)
    
    # メタラベリングサービスが初期化され、学習されているか確認
    assert trainer.meta_labeling_service is not None
    assert trainer.meta_labeling_service.is_trained
    
    # 予測
    # メタラベリングが適用された予測結果 (0/1) が返るはず
    predictions = trainer.predict(X_test)
    assert isinstance(predictions, np.ndarray)
    assert set(np.unique(predictions)).issubset({0, 1})
    
    # predict_proba は生の確率を返すはず
    proba = trainer.predict_proba(X_test)
    assert isinstance(proba, np.ndarray)
    assert proba.shape == (len(X_test), 2) # 2クラス分類の確率
    
    # メタモデルの保存・ロードテストは、BaseMLTrainerのsave_model/load_modelに依存するため、
    # ここでは簡易的にtrainer.meta_labeling_serviceが存在することを確認するに留める
