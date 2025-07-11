"""
MLSignalGeneratorの学習テスト

MLSignalGeneratorのtrainメソッドがモデルを正しく学習し、
予測を実行できることを検証します。
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch

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

def test_ml_signal_generator_training_and_prediction():
    """MLSignalGeneratorの学習と予測のテスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        features_df = create_sample_features_df(num_records=200) # 十分なデータ量
        
        # 学習データの準備
        X, y = generator.prepare_training_data(features_df)
        
        # モデルの学習
        training_result = generator.train(X, y)
        
        assert generator.is_trained, "Model should be marked as trained"
        assert generator.model is not None, "Model object should not be None"
        assert generator.scaler is not None, "Scaler object should not be None"
        assert generator.feature_columns is not None, "Feature columns should be set"
        
        # 学習結果の検証
        assert 'accuracy' in training_result
        assert training_result['accuracy'] >= 0.0 # 精度は0以上
        
        # 予測のテスト
        test_features_df = create_sample_features_df(num_records=10) # テスト用の新しいデータ
        predictions = generator.predict(test_features_df)
        
        assert 'up' in predictions
        assert 'down' in predictions
        assert 'range' in predictions
        
        # 予測確率の合計が1に近いことを確認
        prob_sum = predictions['up'] + predictions['down'] + predictions['range']
        assert abs(prob_sum - 1.0) < 0.1
        
        print("✅ MLSignalGenerator training and prediction successful.")
        
    except Exception as e:
        pytest.fail(f"MLSignalGenerator training and prediction test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
