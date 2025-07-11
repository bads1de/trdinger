"""
MLSignalGeneratorの予測データ型ロバストネス（頑健性）テスト

MLSignalGeneratorのpredictメソッドが、特徴量カラムのデータ型が
学習時と異なる場合にどのように振る舞うかを検証します。
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
def create_sample_features_df(num_records=100, num_features=5, feature_type='numeric'):
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
        if feature_type == 'numeric':
            data[f'feature_{i+1}'] = np.random.rand(num_records) * 100
        elif feature_type == 'string':
            data[f'feature_{i+1}'] = [f'val_{j}' for j in range(num_records)]
        elif feature_type == 'boolean':
            data[f'feature_{i+1}'] = np.random.choice([True, False], num_records)

    df = pd.DataFrame(data)
    return df

def test_predict_with_non_numeric_features():
    """非数値特徴量を含む場合の予測テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        # モデルを学習（数値特徴量で）
        features_df_numeric = create_sample_features_df(num_records=200, num_features=5, feature_type='numeric')
        X_numeric, y_numeric = generator.prepare_training_data(features_df_numeric)
        generator.train(X_numeric, y_numeric)
        
        # 文字列特徴量を含むDataFrameを作成
        features_df_string = create_sample_features_df(num_records=10, num_features=5, feature_type='string')
        
        # 予測を実行
        predictions = generator.predict(features_df_string)
        
        # エラーが発生せず、デフォルト値またはエラーハンドリングされた値が返されることを確認
        assert 'up' in predictions
        assert 'down' in predictions
        assert 'range' in predictions
        
        # 予測確率の合計が1に近いことを確認
        prob_sum = predictions['up'] + predictions['down'] + predictions['range']
        assert abs(prob_sum - 1.0) < 0.1
        
        print("✅ Prediction with non-numeric features handled gracefully.")
        
    except Exception as e:
        pytest.fail(f"Non-numeric features prediction test failed: {e}")

def test_predict_with_boolean_features():
    """真偽値特徴量を含む場合の予測テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        # モデルを学習（数値特徴量で）
        features_df_numeric = create_sample_features_df(num_records=200, num_features=5, feature_type='numeric')
        X_numeric, y_numeric = generator.prepare_training_data(features_df_numeric)
        generator.train(X_numeric, y_numeric)
        
        # 真偽値特徴量を含むDataFrameを作成
        features_df_boolean = create_sample_features_df(num_records=10, num_features=5, feature_type='boolean')
        
        # 予測を実行
        predictions = generator.predict(features_df_boolean)
        
        # エラーが発生せず、デフォルト値またはエラーハンドリングされた値が返されることを確認
        assert 'up' in predictions
        assert 'down' in predictions
        assert 'range' in predictions
        
        # 予測確率の合計が1に近いことを確認
        prob_sum = predictions['up'] + predictions['down'] + predictions['range']
        assert abs(prob_sum - 1.0) < 0.1
        
        print("✅ Prediction with boolean features handled gracefully.")
        
    except Exception as e:
        pytest.fail(f"Boolean features prediction test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
