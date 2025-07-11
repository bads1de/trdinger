"""
MLSignalGeneratorの予測エッジケーステスト

MLSignalGeneratorのpredictメソッドが、モデルが学習されていない場合に
デフォルト値を返すこと、および予測結果の確率の合計が1に近いことを検証します。
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch

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

def test_predict_uninitialized_model_returns_default_values():
    """未初期化モデルがデフォルト値を返すことをテスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        features_df = create_sample_features_df(num_records=10)
        
        # モデルが学習されていない状態であることを確認
        assert not generator.is_trained, "Model should not be trained initially"
        
        predictions = generator.predict(features_df)
        
        # デフォルト値が返されることを確認
        assert predictions['up'] == pytest.approx(0.33, abs=0.01)
        assert predictions['down'] == pytest.approx(0.33, abs=0.01)
        assert predictions['range'] == pytest.approx(0.34, abs=0.01)
        
        prob_sum = predictions['up'] + predictions['down'] + predictions['range']
        assert prob_sum == pytest.approx(1.0, abs=0.01)
        
        print("✅ Uninitialized model returns default prediction values.")
        
    except Exception as e:
        pytest.fail(f"Uninitialized model prediction test failed: {e}")

def test_predict_with_empty_features_df():
    """空の特徴量DataFrameに対する予測テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        empty_features_df = pd.DataFrame()
        
        # モデルが学習されていない状態であることを確認
        assert not generator.is_trained, "Model should not be trained initially"
        
        predictions = generator.predict(empty_features_df)
        
        # 空のDataFrameでもデフォルト値が返されることを確認
        assert predictions['up'] == pytest.approx(0.33, abs=0.01)
        assert predictions['down'] == pytest.approx(0.33, abs=0.01)
        assert predictions['range'] == pytest.approx(0.34, abs=0.01)
        
        prob_sum = predictions['up'] + predictions['down'] + predictions['range']
        assert prob_sum == pytest.approx(1.0, abs=0.01)
        
        print("✅ Prediction with empty features DataFrame returns default values.")
        
    except Exception as e:
        pytest.fail(f"Empty features DataFrame prediction test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
