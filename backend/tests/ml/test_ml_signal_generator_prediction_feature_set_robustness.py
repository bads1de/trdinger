"""
MLSignalGeneratorの予測特徴量セットロバストネス（頑健性）テスト

MLSignalGeneratorのpredictメソッドが、学習時と異なる特徴量セット
（一部の特徴量が追加または削除された場合）にどのように振る舞うかを検証します。
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
def create_sample_features_df(num_records=100, num_features=5):
    """サンプル特徴量DataFrameを作成"""
    dates = pd.date_range(start='2024-01-01', periods=num_records, freq='h')
    np.random.seed(42)

    data = {
        'timestamp': dates,
        'close': np.random.rand(num_records) * 50000 + 10000,
        'target_up': (np.random.rand(num_records) > 0.5).astype(int),
        'target_down': (np.random.rand(num_records) > 0.5).astype(int),
        'target_range': (np.random.rand(num_records) > 0.5).astype(int),
    }
    for i in range(num_features):
        data[f'feature_{i+1}'] = np.random.rand(num_records) * 100

    df = pd.DataFrame(data)
    return df

def test_predict_with_fewer_features_than_trained():
    """学習時より少ない特徴量での予測テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        # モデルを学習（多めの特徴量で）
        features_df_trained = create_sample_features_df(num_records=200, num_features=10)
        X_trained, y_trained = generator.prepare_training_data(features_df_trained)
        generator.train(X_trained, y_trained)
        
        # 学習時より少ない特徴量を持つDataFrameを作成
        features_df_fewer = create_sample_features_df(num_records=10, num_features=5)
        
        # 予測を実行
        predictions = generator.predict(features_df_fewer)
        
        # エラーが発生せず、デフォルト値またはエラーハンドリングされた値が返されることを確認
        assert 'up' in predictions
        assert 'down' in predictions
        assert 'range' in predictions
        
        # 予測確率の合計が1に近いことを確認
        prob_sum = predictions['up'] + predictions['down'] + predictions['range']
        assert abs(prob_sum - 1.0) < 0.1
        
        print("✅ Prediction with fewer features handled gracefully.")
        
    except Exception as e:
        pytest.fail(f"Fewer features prediction test failed: {e}")

def test_predict_with_more_features_than_trained():
    """学習時より多い特徴量での予測テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        # モデルを学習（少なめの特徴量で）
        features_df_trained = create_sample_features_df(num_records=200, num_features=5)
        X_trained, y_trained = generator.prepare_training_data(features_df_trained)
        generator.train(X_trained, y_trained)
        
        # 学習時より多い特徴量を持つDataFrameを作成
        features_df_more = create_sample_features_df(num_records=10, num_features=10)
        
        # 予測を実行
        predictions = generator.predict(features_df_more)
        
        # エラーが発生せず、デフォルト値またはエラーハンドリングされた値が返されることを確認
        assert 'up' in predictions
        assert 'down' in predictions
        assert 'range' in predictions
        
        # 予測確率の合計が1に近いことを確認
        prob_sum = predictions['up'] + predictions['down'] + predictions['range']
        assert abs(prob_sum - 1.0) < 0.1
        
        print("✅ Prediction with more features handled gracefully.")
        
    except Exception as e:
        pytest.fail(f"More features prediction test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])