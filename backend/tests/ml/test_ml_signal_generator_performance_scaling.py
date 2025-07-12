"""
MLSignalGeneratorのパフォーマンススケーリングテスト

MLSignalGeneratorのtrainメソッドにおけるLightGBMモデルの学習が、
データセットのサイズや特徴量の数によってどのようにパフォーマンスが変化するかを検証します。
"""

import pytest
import numpy as np
import pandas as pd
import sys
import time
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

def test_training_performance_with_varying_data_size():
    """データサイズが変化した場合の学習パフォーマンステスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        data_sizes = [100, 500, 1000] # 小、中、大のデータサイズ
        training_times = []
        
        for size in data_sizes:
            generator = MLSignalGenerator()
            features_df = create_sample_features_df(num_records=size)
            X, y = generator.prepare_training_data(features_df)
            
            start_time = time.time()
            generator.train(X, y)
            end_time = time.time()
            
            training_times.append(end_time - start_time)
            print(f"✅ Training with {size} records took {training_times[-1]:.2f} seconds.")
            
        # 学習時間がデータサイズに対して線形または準線形に増加することを確認
        # 厳密な閾値ではなく、傾向を確認
        assert training_times[0] > 0, "Small data training time should be positive"
        assert training_times[1] > 0, "Medium data training time should be positive"
        assert training_times[2] > 0, "Large data training time should be positive"
        
        # 最大許容時間（仮の閾値）
        assert training_times[2] < 10.0, "Training with large data took too long"
        
    except Exception as e:
        pytest.fail(f"Training performance with varying data size test failed: {e}")

def test_training_performance_with_varying_feature_count():
    """特徴量数が変化した場合の学習パフォーマンステスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        feature_counts = [5, 20, 50] # 少ない、中程度、多い特徴量数
        training_times = []
        
        for count in feature_counts:
            generator = MLSignalGenerator()
            features_df = create_sample_features_df(num_records=500, num_features=count)
            X, y = generator.prepare_training_data(features_df)
            
            start_time = time.time()
            generator.train(X, y)
            end_time = time.time()
            
            training_times.append(end_time - start_time)
            print(f"✅ Training with {count} features took {training_times[-1]:.2f} seconds.")
            
        # 学習時間が特徴量数に対して線形または準線形に増加することを確認
        # 厳密な線形性を期待するのではなく、学習が完了し、エラーが発生しないことを確認する程度に留める
        assert training_times[0] > 0, "Training time should be positive"
        assert training_times[1] > 0, "Training time should be positive"
        assert training_times[2] > 0, "Training time should be positive"
        
        # 最大許容時間（仮の閾値）
        assert training_times[2] < 10.0, "Training with many features took too long"
        
    except Exception as e:
        pytest.fail(f"Training performance with varying feature count test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
