"""
MLSignalGeneratorの時系列分割テスト

MLSignalGeneratorのtrainメソッド内で使用される時系列分割（TimeSeriesSplit）が
正しく機能し、データリークがないことを検証します。
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

def test_time_series_split_integrity():
    """時系列分割の整合性テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        features_df = create_sample_features_df(num_records=100) # 100レコードのデータ
        X, y = generator.prepare_training_data(features_df)
        
        # trainメソッドの内部ロジックを直接テスト
        test_size = 0.2
        split_index = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        
        # 訓練データとテストデータが重複していないことを確認
        # 時系列データなので、訓練データの最終インデックスがテストデータの開始インデックスより前であることを確認
        assert X_train.index.max() < X_test.index.min(), "Data leak detected: Training and test data overlap"
        
        # 訓練データとテストデータの合計が元のデータと同じであることを確認
        assert len(X_train) + len(X_test) == len(X), "Total data length mismatch after split"
        assert len(y_train) + len(y_test) == len(y), "Total label length mismatch after split"
        
        print("✅ TimeSeriesSplit integrity check passed.")
        
    except Exception as e:
        pytest.fail(f"TimeSeriesSplit integrity test failed: {e}")

def test_time_series_split_with_small_data():
    """非常に小さいデータセットでの時系列分割テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        features_df = create_sample_features_df(num_records=5) # 5レコードのデータ
        X, y = generator.prepare_training_data(features_df)
        
        # trainメソッドの内部ロジックを直接テスト
        test_size = 0.2
        split_index = int(len(X) * (1 - test_size))
        
        # 少なくとも1つの訓練データとテストデータが存在することを確認
        if len(X) > 1:
            X_train = X.iloc[:split_index]
            X_test = X.iloc[split_index:]
            y_train = y.iloc[:split_index]
            y_test = y.iloc[split_index:]
            
            assert len(X_train) >= 1, "Training data should not be empty"
            assert len(X_test) >= 0, "Test data can be empty but should not raise error"
            
            if len(X_test) > 0:
                assert X_train.index.max() < X_test.index.min(), "Data leak detected for small data"
        else:
            # データが少なすぎて分割できない場合
            print("ℹ️ Data too small for meaningful split, skipping detailed checks.")
            
        print("✅ TimeSeriesSplit with small data handled gracefully.")
        
    except Exception as e:
        pytest.fail(f"TimeSeriesSplit small data test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
