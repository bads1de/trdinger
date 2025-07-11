"""
MLSignalGeneratorのモデル永続化テスト

MLSignalGeneratorのsave_modelおよびload_modelメソッドが
正しく機能することを検証します。
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import os

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

def test_model_save_and_load():
    """モデルの保存と読み込みのテスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        # モデルを学習
        generator = MLSignalGenerator()
        features_df = create_sample_features_df(num_records=200)
        X, y = generator.prepare_training_data(features_df)
        generator.train(X, y)
        
        # モデルを保存
        model_path = generator.save_model("test_ml_model")
        assert os.path.exists(model_path), "Model file should exist after saving"
        
        # 新しいジェネレータでモデルを読み込み
        new_generator = MLSignalGenerator()
        load_success = new_generator.load_model(model_path)
        
        assert load_success, "Model should be loaded successfully"
        assert new_generator.is_trained, "New generator should be marked as trained"
        assert new_generator.model is not None, "New generator model should not be None"
        assert new_generator.scaler is not None, "New generator scaler should not be None"
        assert new_generator.feature_columns is not None, "New generator feature columns should be set"
        
        # 読み込んだモデルで予測を実行し、元のモデルと同じ結果が得られることを確認
        test_features_df = create_sample_features_df(num_records=10)
        original_predictions = generator.predict(test_features_df)
        loaded_predictions = new_generator.predict(test_features_df)
        
        for key in original_predictions:
            assert original_predictions[key] == pytest.approx(loaded_predictions[key], abs=1e-6), f"Prediction mismatch for {key}"
            
        print("✅ Model save and load functionality works correctly.")
        
    except Exception as e:
        pytest.fail(f"Model persistence test failed: {e}")

def test_load_non_existent_model():
    """存在しないモデルの読み込みテスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        non_existent_path = "non_existent_model.pkl"
        
        load_success = generator.load_model(non_existent_path)
        assert not load_success, "Loading non-existent model should fail"
        
        print("✅ Loading non-existent model fails as expected.")
        
    except Exception as e:
        pytest.fail(f"Non-existent model load test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
