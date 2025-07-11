"""
MLSignalGeneratorのデータ準備テスト

MLSignalGeneratorのprepare_training_dataメソッドが、特徴量DataFrameから
正しくX（特徴量）とy（ターゲット）を生成するかを検証します。
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 既存のテストファイルからcreate_comprehensive_test_dataをコピー
def create_sample_features_df(num_records=100):
    """サンプル特徴量DataFrameを作成"""
    dates = pd.date_range(start='2024-01-01', periods=num_records, freq='h')
    np.random.seed(42)

    data = {
        'timestamp': dates,
        'feature_1': np.random.rand(num_records),
        'feature_2': np.random.rand(num_records) * 100,
        'close': np.random.rand(num_records) * 50000 + 10000, # closeカラムを追加
        'ML_UP_PROB': np.random.rand(num_records),
        'ML_DOWN_PROB': np.random.rand(num_records),
        'ML_RANGE_PROB': np.random.rand(num_records),
        'target_up': (np.random.rand(num_records) > 0.5).astype(int),
        'target_down': (np.random.rand(num_records) > 0.5).astype(int),
        'target_range': (np.random.rand(num_records) > 0.5).astype(int),
    }
    df = pd.DataFrame(data)
    return df

def test_prepare_training_data_output_shape():
    """prepare_training_dataの出力形状テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        features_df = create_sample_features_df(num_records=50)
        
        X, y = generator.prepare_training_data(features_df)
        
        # Xは特徴量、yはターゲット
        assert isinstance(X, pd.DataFrame), "X should be a pandas DataFrame"
        assert isinstance(y, pd.Series), "y should be a pandas Series"
        
        # 行数が一致することを確認
        expected_rows = len(features_df) - 24 # prediction_horizonのデフォルト値24を考慮
        assert len(X) == expected_rows, "X row count mismatch"
        assert len(y) == expected_rows, "y row count mismatch"
        
        # 特徴量カラムがXに含まれており、ターゲットカラムとtimestamp、closeが含まれていないことを確認
        expected_target_columns = ['target_up', 'target_down', 'target_range']
        for col in features_df.columns:
            if col not in expected_target_columns and col not in ['timestamp', 'close']:
                assert col in X.columns, f"Feature column '{col}' not found in X"
            elif col in expected_target_columns or col == 'close':
                assert col not in X.columns, f"Column '{col}' found in X when it should not be"
        
        print("✅ prepare_training_data output shape and columns are correct.")
        
    except Exception as e:
        pytest.fail(f"prepare_training_data output shape test failed: {e}")

def test_prepare_training_data_with_missing_targets():
    """ターゲットカラムが欠損している場合のprepare_training_dataテスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        # ターゲットカラムを含まないDataFrameを作成
        features_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='h'),
            'feature_1': np.random.rand(10),
            'feature_2': np.random.rand(10),
            'close': np.random.rand(10) * 50000 + 10000,
        })
        
        with pytest.raises(ValueError, match="Target columns not found in features_df"):
            generator.prepare_training_data(features_df)
            
        print("✅ prepare_training_data handles missing target columns correctly.")
        
    except Exception as e:
        pytest.fail(f"Missing target columns test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
