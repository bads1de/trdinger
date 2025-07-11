"""
MLSignalGeneratorの特徴量重要度テスト

MLSignalGeneratorのget_feature_importanceメソッドが、
学習済みモデルから正しく特徴量重要度を返すことを検証します。
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 既存のテストファイルからcreate_sample_features_dfをコピー
def create_sample_features_df(num_records=100):
    """サンプル特徴量DataFrameを作成"""
    dates = pd.date_range(start='2024-01-01', periods=num_records, freq='h')
    np.random.seed(42)

    data = {
        'timestamp': dates,
        'feature_1': np.random.rand(num_records),
        'feature_2': np.random.rand(num_records) * 100,
        'close': np.random.rand(num_records) * 50000 + 10000,
        'ML_UP_PROB': np.random.rand(num_records),
        'ML_DOWN_PROB': np.random.rand(num_records),
        'ML_RANGE_PROB': np.random.rand(num_records),
        'target_up': (np.random.rand(num_records) > 0.5).astype(int),
        'target_down': (np.random.rand(num_records) > 0.5).astype(int),
        'target_range': (np.random.rand(num_records) > 0.5).astype(int),
    }
    df = pd.DataFrame(data)
    return df

def test_get_feature_importance_after_training():
    """学習後に特徴量重要度が取得できることをテスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        features_df = create_sample_features_df(num_records=200)
        X, y = generator.prepare_training_data(features_df)
        generator.train(X, y)
        
        feature_importance = generator.get_feature_importance()
        
        assert isinstance(feature_importance, dict), "Feature importance should be a dictionary"
        assert len(feature_importance) > 0, "Feature importance dictionary should not be empty"
        
        # 特徴量名が正しいことを確認
        expected_features = [col for col in X.columns if col not in ['timestamp', 'close']]
        for feature in expected_features:
            assert feature in feature_importance, f"Feature '{feature}' not found in importance"
            assert isinstance(feature_importance[feature], (float, int)), f"Importance for '{feature}' should be numeric"
            
        print("✅ Feature importance can be retrieved after training.")
        
    except Exception as e:
        pytest.fail(f"Feature importance test failed: {e}")

def test_get_feature_importance_before_training():
    """学習前に特徴量重要度が空であることをテスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        feature_importance = generator.get_feature_importance()
        
        assert isinstance(feature_importance, dict), "Feature importance should be a dictionary"
        assert len(feature_importance) == 0, "Feature importance should be empty before training"
        
        print("✅ Feature importance is empty before training as expected.")
        
    except Exception as e:
        pytest.fail(f"Feature importance before training test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
