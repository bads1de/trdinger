"""
MLSignalGeneratorの予測極値ロバストネス（頑健性）テスト

MLSignalGeneratorのpredictメソッドが、特徴量カラムに非常に大きな値や
非常に小さな値が含まれる場合にどのように振る舞うかを検証します。
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
def create_sample_features_df(num_records=100, num_features=5, value_type='normal'):
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
        if value_type == 'normal':
            data[f'feature_{i+1}'] = np.random.rand(num_records) * 100
        elif value_type == 'large':
            data[f'feature_{i+1}'] = np.random.rand(num_records) * 1e9 # 非常に大きな値
        elif value_type == 'small':
            data[f'feature_{i+1}'] = np.random.rand(num_records) * 1e-9 # 非常に小さな値

    df = pd.DataFrame(data)
    return df

def test_predict_with_large_feature_values():
    """非常に大きな特徴量値での予測テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        # モデルを学習（標準値で）
        features_df_normal = create_sample_features_df(num_records=200, num_features=5, value_type='normal')
        X_normal, y_normal = generator.prepare_training_data(features_df_normal)
        generator.train(X_normal, y_normal)
        
        # 非常に大きな特徴量値を持つDataFrameを作成
        features_df_large = create_sample_features_df(num_records=10, num_features=5, value_type='large')
        
        # 予測を実行
        predictions = generator.predict(features_df_large)
        
        # エラーが発生せず、デフォルト値またはエラーハンドリングされた値が返されることを確認
        assert 'up' in predictions
        assert 'down' in predictions
        assert 'range' in predictions
        
        # 予測確率の合計が1に近いことを確認
        prob_sum = predictions['up'] + predictions['down'] + predictions['range']
        assert abs(prob_sum - 1.0) < 0.1
        
        print("✅ Prediction with large feature values handled gracefully.")
        
    except Exception as e:
        pytest.fail(f"Large feature values prediction test failed: {e}")

def test_predict_with_small_feature_values():
    """非常に小さな特徴量値での予測テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        # モデルを学習（標準値で）
        features_df_normal = create_sample_features_df(num_records=200, num_features=5, value_type='normal')
        X_normal, y_normal = generator.prepare_training_data(features_df_normal)
        generator.train(X_normal, y_normal)
        
        # 非常に小さな特徴量値を持つDataFrameを作成
        features_df_small = create_sample_features_df(num_records=10, num_features=5, value_type='small')
        
        # 予測を実行
        predictions = generator.predict(features_df_small)
        
        # エラーが発生せず、デフォルト値またはエラーハンドリングされた値が返されることを確認
        assert 'up' in predictions
        assert 'down' in predictions
        assert 'range' in predictions
        
        # 予測確率の合計が1に近いことを確認
        prob_sum = predictions['up'] + predictions['down'] + predictions['range']
        assert abs(prob_sum - 1.0) < 0.1
        
        print("✅ Prediction with small feature values handled gracefully.")
        
    except Exception as e:
        pytest.fail(f"Small feature values prediction test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
