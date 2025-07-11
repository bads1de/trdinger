"""
MLSignalGeneratorの予測特徴量順序ロバストネス（頑健性）テスト

MLSignalGeneratorのpredictメソッドが、特徴量カラムの順序が
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

def test_predict_with_reordered_features_only():
    """特徴量カラムの順序が異なる場合のみの予測テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        # モデルを学習
        features_df_original = create_sample_features_df(num_records=200, num_features=10)
        X_original, y_original = generator.prepare_training_data(features_df_original)
        generator.train(X_original, y_original)
        
        # 特徴量カラムの順序を入れ替えたDataFrameを作成
        # timestamp, close, target_up, target_down, target_range は特徴量ではないので除外
        feature_cols = [col for col in X_original.columns if col.startswith('feature_')]
        reordered_feature_cols = feature_cols[::-1] # 逆順にする
        
        # 予測用DataFrameを作成
        test_features_df_reordered = features_df_original[reordered_feature_cols + [col for col in features_df_original.columns if col not in feature_cols]].head(1)
        
        # 予測を実行
        predictions_reordered = generator.predict(test_features_df_reordered)
        
        # 元の順序のDataFrameで予測を実行
        predictions_original = generator.predict(features_df_original[feature_cols + [col for col in features_df_original.columns if col not in feature_cols]].head(1))
        
        # 予測結果がほぼ同じであることを確認
        for key in predictions_original:
            assert predictions_original[key] == pytest.approx(predictions_reordered[key], abs=1e-6), f"Prediction mismatch for {key} with reordered features"
            
        print("✅ Prediction with reordered features only works correctly.")
        
    except Exception as e:
        pytest.fail(f"Reordered features only prediction test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
