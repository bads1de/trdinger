"""
MLSignalGeneratorの学習ラベルロバストネス（頑健性）テスト

MLSignalGeneratorのtrainメソッドが、ターゲットクラスの分布が極端に偏っている場合や
一部のクラスのサンプル数が非常に少ない場合にどのように振る舞うかを検証します。
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
def create_sample_features_df(num_records=100, num_features=5, target_distribution=None):
    """サンプル特徴量DataFrameを作成"""
    dates = pd.date_range(start='2024-01-01', periods=num_records, freq='h')
    np.random.seed(42)

    data = {
        'timestamp': dates,
        'close': np.random.rand(num_records) * 50000 + 10000,
    }
    for i in range(num_features):
        data[f'feature_{i+1}'] = np.random.rand(num_records) * 100

    # ターゲットを生成
    if target_distribution == 'all_up':
        target_labels = np.full(num_records, 2) # 全て上昇
    elif target_distribution == 'all_down':
        target_labels = np.full(num_records, 0) # 全て下降
    elif target_distribution == 'all_range':
        target_labels = np.full(num_records, 1) # 全てレンジ
    elif target_distribution == 'sparse_up':
        target_labels = np.full(num_records, 1) # ほとんどレンジ
        target_labels[np.random.choice(num_records, size=int(num_records * 0.05), replace=False)] = 2 # 5%だけ上昇
    elif target_distribution == 'sparse_down':
        target_labels = np.full(num_records, 1) # ほとんどレンジ
        target_labels[np.random.choice(num_records, size=int(num_records * 0.05), replace=False)] = 0 # 5%だけ下降
    else:
        # ランダムな分布
        target_labels = np.random.choice([0, 1, 2], num_records)

    data['target_up'] = (target_labels == 2).astype(int)
    data['target_down'] = (target_labels == 0).astype(int)
    data['target_range'] = (target_labels == 1).astype(int)

    df = pd.DataFrame(data)
    return df

def test_training_with_single_class_labels():
    """単一クラスラベルでの学習テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        # 全て上昇のラベルを持つデータ
        features_df = create_sample_features_df(num_records=100, target_distribution='all_up')
        X, y = generator.prepare_training_data(features_df)
        
        # 訓練データが十分に存在することを確認
        assert len(X) > 0, "Training data should not be empty"
        
        training_result = generator.train(X, y)
        
        assert generator.is_trained, "Model should be trained"
        # 精度が1.0に近いことを期待（単一クラスなので）
        assert training_result['accuracy'] > 0.0, "Accuracy should be greater than 0 for single class data"
        
        print("✅ Training with single class labels handled gracefully.")
        
    except Exception as e:
        pytest.fail(f"Single class labels training test failed: {e}")

def test_training_with_sparse_class_labels():
    """スパースなクラスラベルでの学習テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        # ほとんどレンジで、一部だけ上昇のラベルを持つデータ
        features_df = create_sample_features_df(num_records=200, target_distribution='sparse_up')
        X, y = generator.prepare_training_data(features_df)
        
        # 訓練データが十分に存在することを確認
        assert len(X) > 0, "Training data should not be empty"
        
        training_result = generator.train(X, y)
        
        assert generator.is_trained, "Model should be trained"
        # 精度が0.0より大きいことを確認（完全にランダムではない）
        assert training_result['accuracy'] > 0.0, "Accuracy should be greater than 0 for sparse data"
        
        print("✅ Training with sparse class labels handled gracefully.")
        
    except Exception as e:
        pytest.fail(f"Sparse class labels training test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
