import pytest
import pandas as pd
import numpy as np
from backend.app.services.ml.ensemble.stacking import StackingEnsemble

@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100), name='target')
    return X, y

def test_stacking_ensemble_oof_predictions(sample_data):
    X, y = sample_data
    config = {
        "base_models": ["lightgbm", "xgboost"],
        "meta_model": "logistic_regression",
        "cv_folds": 3,
        "stack_method": "predict_proba",
        "n_jobs": 1,
        "passthrough": False,
        "cv_strategy": "kfold",
    }
    
    ensemble = StackingEnsemble(config)
    ensemble.fit(X, y)
    
    # OOF予測値が取得できるか確認
    oof_preds = ensemble.get_oof_predictions()
    assert oof_preds is not None
    assert isinstance(oof_preds, np.ndarray)
    assert len(oof_preds) == len(X)
    
    # 各ベースモデルのOOF予測値が取得できるか確認
    oof_base_preds = ensemble.get_oof_base_model_predictions()
    assert oof_base_preds is not None
    assert isinstance(oof_base_preds, pd.DataFrame)
    assert len(oof_base_preds) == len(X)
    assert "lightgbm" in oof_base_preds.columns
    assert "xgboost" in oof_base_preds.columns
    
    # オリジナルデータが取得できるか確認
    X_original = ensemble.get_X_train_original()
    y_original = ensemble.get_y_train_original()
    
    assert X_original.equals(X)
    assert y_original.equals(y)


