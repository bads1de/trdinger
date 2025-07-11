"""
MLSignalGeneratorの予測ロバストネス（頑健性）テスト

MLSignalGeneratorのpredictメソッドが、特徴量カラムが不足している場合や
順序が異なる場合にどのように振る舞うかを検証します。
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

def test_predict_with_missing_features():
    """特徴量カラムが不足している場合の予測テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        # モデルを学習（完全な特徴量セットで）
        features_df_full = create_sample_features_df(num_records=200, num_features=10)
        X_full, y_full = generator.prepare_training_data(features_df_full)
        generator.train(X_full, y_full)
        
        # 一部の特徴量を欠損させたDataFrameを作成
        features_df_missing = features_df_full.drop(columns=['feature_1', 'feature_5']).head(1)
        
        # 予測を実行
        predictions = generator.predict(features_df_missing)
        
        # エラーが発生せず、デフォルト値またはエラーハンドリングされた値が返されることを確認
        # （signal_generator.pyのpredictメソッドのエラーハンドリングに依存）
        assert 'up' in predictions
        assert 'down' in predictions
        assert 'range' in predictions
        
        # 予測確率の合計が1に近いことを確認
        prob_sum = predictions['up'] + predictions['down'] + predictions['range']
        assert abs(prob_sum - 1.0) < 0.1
        
        print("✅ Prediction with missing features handled gracefully.")
        
    except Exception as e:
        pytest.fail(f"Missing features prediction test failed: {e}")

def test_predict_with_reordered_features():
    """特徴量カラムの順序が異なる場合の予測テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        # モデルを学習
        features_df_original = create_sample_features_df(num_records=200, num_features=10)
        X_original, y_original = generator.prepare_training_data(features_df_original)
        generator.train(X_original, y_original)
        
        # 特徴量カラムの順序を入れ替えたDataFrameを作成
        original_cols = [col for col in features_df_original.columns if col.startswith('feature_')]
        reordered_cols = original_cols[::-1] # 逆順にする
        
        # timestamp, close, target_up, target_down, target_range は特徴量ではないので除外
        non_feature_cols = ['timestamp', 'close', 'target_up', 'target_down', 'target_range']
        other_cols = [col for col in features_df_original.columns if col not in original_cols and col not in non_feature_cols]

        # 予測用DataFrameを作成
        test_features_df_reordered = features_df_original[reordered_cols + other_cols].head(1)
        
        # 予測を実行
        predictions_reordered = generator.predict(test_features_df_reordered)
        
        # 元の順序のDataFrameで予測を実行
        predictions_original = generator.predict(features_df_original[original_cols + other_cols].head(1))
        
        # 予測結果がほぼ同じであることを確認
        for key in predictions_original:
            assert predictions_original[key] == pytest.approx(predictions_reordered[key], abs=1e-6), f"Prediction mismatch for {key} with reordered features"
            
        print("✅ Prediction with reordered features works correctly.")
        
    except Exception as e:
        pytest.fail(f"Reordered features prediction test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
