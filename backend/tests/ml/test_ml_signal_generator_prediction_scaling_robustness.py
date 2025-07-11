"""
MLSignalGeneratorの予測スケーリングロバストネス（頑健性）テスト

MLSignalGeneratorのpredictメソッドが、特徴量カラムのスケールが
学習時と大きく異なる場合にどのように振る舞うかを検証します。
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
def create_sample_features_df(num_records=100, num_features=5, scale_factor=1.0):
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
        data[f'feature_{i+1}'] = np.random.rand(num_records) * 100 * scale_factor

    df = pd.DataFrame(data)
    return df

def test_predict_with_different_feature_scaling():
    """異なる特徴量スケールでの予測テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        # モデルを学習（標準スケールで）
        features_df_normal = create_sample_features_df(num_records=200, num_features=5, scale_factor=1.0)
        X_normal, y_normal = generator.prepare_training_data(features_df_normal)
        generator.train(X_normal, y_normal)
        
        # スケールが異なる特徴量DataFrameを作成（例: 10倍）
        features_df_scaled = create_sample_features_df(num_records=10, num_features=5, scale_factor=10.0)
        
        # 予測を実行
        predictions_scaled = generator.predict(features_df_scaled)
        
        # 予測確率の合計が1に近いことを確認
        prob_sum = predictions_scaled['up'] + predictions_scaled['down'] + predictions_scaled['range']
        assert abs(prob_sum - 1.0) < 0.1
        
        # 予測結果がデフォルト値から有意に逸脱していることを確認
        default_prob = 1/3
        assert abs(predictions_scaled['up'] - default_prob) > 0.05
        assert abs(predictions_scaled['down'] - default_prob) > 0.05
        assert abs(predictions_scaled['range'] - default_prob) > 0.05
        
        print("✅ Prediction with different feature scaling handled gracefully.")
        
    except Exception as e:
        pytest.fail(f"Different feature scaling prediction test failed: {e}")

def test_predict_with_zero_scaled_features():
    """ゼロスケール特徴量での予測テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        # モデルを学習（標準スケールで）
        features_df_normal = create_sample_features_df(num_records=200, num_features=5, scale_factor=1.0)
        X_normal, y_normal = generator.prepare_training_data(features_df_normal)
        generator.train(X_normal, y_normal)
        
        # 全ての特徴量がゼロのDataFrameを作成
        features_df_zero = create_sample_features_df(num_records=10, num_features=5, scale_factor=0.0)
        
        # 予測を実行
        predictions_zero = generator.predict(features_df_zero)
        
        # 予測確率の合計が1に近いことを確認
        prob_sum = predictions_zero['up'] + predictions_zero['down'] + predictions_zero['range']
        assert abs(prob_sum - 1.0) < 0.1
        
        # 予測結果がデフォルト値から有意に逸脱していることを確認
        default_prob = 1/3
        assert abs(predictions_zero['up'] - default_prob) > 0.05
        assert abs(predictions_zero['down'] - default_prob) > 0.05
        assert abs(predictions_zero['range'] - default_prob) > 0.05
        
        print("✅ Prediction with zero-scaled features handled gracefully.")
        
    except Exception as e:
        pytest.fail(f"Zero-scaled features prediction test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
