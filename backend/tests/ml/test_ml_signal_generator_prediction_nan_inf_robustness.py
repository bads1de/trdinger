"""
MLSignalGeneratorの予測NaN/Infロバストネス（頑健性）テスト

MLSignalGeneratorのpredictメソッドが、特徴量カラムにNaNや無限大などの
非数値データが含まれる場合にどのように振る舞うかを検証します。
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
def create_sample_features_df(num_records=100, num_features=5, include_nan=False, include_inf=False):
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
        feature_data = np.random.rand(num_records) * 100
        if include_nan:
            nan_indices = np.random.choice(num_records, size=int(num_records * 0.1), replace=False)
            feature_data[nan_indices] = np.nan
        if include_inf:
            inf_indices = np.random.choice(num_records, size=int(num_records * 0.05), replace=False)
            feature_data[inf_indices] = np.inf
        data[f'feature_{i+1}'] = feature_data

    df = pd.DataFrame(data)
    return df

def test_predict_with_nan_features():
    """NaNを含む特徴量での予測テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        # モデルを学習
        features_df_trained = create_sample_features_df(num_records=200, num_features=5)
        X_trained, y_trained = generator.prepare_training_data(features_df_trained)
        generator.train(X_trained, y_trained)
        
        # NaNを含む特徴量を持つDataFrameを作成
        features_df_nan = create_sample_features_df(num_records=10, num_features=5, include_nan=True)
        
        # 予測を実行
        predictions = generator.predict(features_df_nan)
        
        # エラーが発生せず、デフォルト値またはエラーハンドリングされた値が返されることを確認
        assert 'up' in predictions
        assert 'down' in predictions
        assert 'range' in predictions
        
        # 予測確率の合計が1に近いことを確認
        prob_sum = predictions['up'] + predictions['down'] + predictions['range']
        assert abs(prob_sum - 1.0) < 0.1
        
        print("✅ Prediction with NaN features handled gracefully.")
        
    except Exception as e:
        pytest.fail(f"NaN features prediction test failed: {e}")

def test_predict_with_inf_features():
    """無限大を含む特徴量での予測テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        # モデルを学習
        features_df_trained = create_sample_features_df(num_records=200, num_features=5)
        X_trained, y_trained = generator.prepare_training_data(features_df_trained)
        generator.train(X_trained, y_trained)
        
        # 無限大を含む特徴量を持つDataFrameを作成
        features_df_inf = create_sample_features_df(num_records=10, num_features=5, include_inf=True)
        
        # 予測を実行
        predictions = generator.predict(features_df_inf)
        
        # エラーが発生せず、デフォルト値またはエラーハンドリングされた値が返されることを確認
        assert 'up' in predictions
        assert 'down' in predictions
        assert 'range' in predictions
        
        # 予測確率の合計が1に近いことを確認
        prob_sum = predictions['up'] + predictions['down'] + predictions['range']
        assert abs(prob_sum - 1.0) < 0.1
        
        print("✅ Prediction with Inf features handled gracefully.")
        
    except Exception as e:
        pytest.fail(f"Inf features prediction test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
