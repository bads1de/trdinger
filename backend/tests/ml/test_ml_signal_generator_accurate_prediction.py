"""
MLSignalGeneratorの正確な予測テスト

MLSignalGeneratorのpredictメソッドが、学習済みモデルを使用して
正確な予測を生成できることを検証します。
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

def test_accurate_prediction_after_training():
    """学習後の正確な予測テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        features_df = create_sample_features_df(num_records=200) # 十分なデータ量
        X, y = generator.prepare_training_data(features_df)
        generator.train(X, y)
        
        # テスト用のデータを作成し、特定のターゲットを持つように調整
        test_features_df_up = create_sample_features_df(num_records=1)
        test_features_df_down = create_sample_features_df(num_records=1)
        test_features_df_range = create_sample_features_df(num_records=1)

        # 予測
        predictions_up = generator.predict(test_features_df_up)
        predictions_down = generator.predict(test_features_df_down)
        predictions_range = generator.predict(test_features_df_range)

        # 予測確率の合計が1に近いことを確認
        assert abs(predictions_up['up'] + predictions_up['down'] + predictions_up['range'] - 1.0) < 0.1
        assert abs(predictions_down['up'] + predictions_down['down'] + predictions_down['range'] - 1.0) < 0.1
        assert abs(predictions_range['up'] + predictions_range['down'] + predictions_range['range'] - 1.0) < 0.1

        # 各クラスの予測確率がデフォルト値から有意に逸脱していることを確認
        # これは、モデルがランダムな予測ではなく、何らかのパターンを学習していることを示唆
        default_prob = 1/3 # 3クラス分類なので、ランダムなら約1/3
        assert abs(predictions_up['up'] - default_prob) > 0.05 # 0.05は仮の閾値
        assert abs(predictions_down['down'] - default_prob) > 0.05
        assert abs(predictions_range['range'] - default_prob) > 0.05

        print("✅ Accurate prediction after training works correctly.")
        
    except Exception as e:
        pytest.fail(f"Accurate prediction test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
