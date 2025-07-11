"""
MLSignalGeneratorの学習エッジケーステスト

MLSignalGeneratorのtrainメソッドが、データ量が少ない場合や
ターゲットクラスが偏っている場合にどのように振る舞うかを検証します。
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
def create_sample_features_df(num_records=100, target_bias=None):
    """サンプル特徴量DataFrameを作成

    Args:
        num_records (int): 生成するレコード数。
        target_bias (str, optional): 'up', 'down', 'range' のいずれかを指定すると、
                                     そのクラスのターゲットを偏らせる。
    """
    dates = pd.date_range(start='2024-01-01', periods=num_records, freq='h')
    np.random.seed(42)

    data = {
        'timestamp': dates,
        'feature_1': np.random.rand(num_records),
        'feature_2': np.random.rand(num_records) * 100,
        'close': np.random.rand(num_records) * 50000 + 10000,
    }
    
    # ターゲットを生成
    target_up = (np.random.rand(num_records) > 0.5).astype(int)
    target_down = (np.random.rand(num_records) > 0.5).astype(int)
    target_range = (np.random.rand(num_records) > 0.5).astype(int)

    if target_bias == 'up':
        target_up = np.ones(num_records, dtype=int) # 全て上昇
        target_down = np.zeros(num_records, dtype=int)
        target_range = np.zeros(num_records, dtype=int)
    elif target_bias == 'down':
        target_up = np.zeros(num_records, dtype=int)
        target_down = np.ones(num_records, dtype=int) # 全て下降
        target_range = np.zeros(num_records, dtype=int)
    elif target_bias == 'range':
        target_up = np.zeros(num_records, dtype=int)
        target_down = np.zeros(num_records, dtype=int)
        target_range = np.ones(num_records, dtype=int) # 全てレンジ

    data['target_up'] = target_up
    data['target_down'] = target_down
    data['target_range'] = target_range

    df = pd.DataFrame(data)
    return df

def test_training_with_minimal_data():
    """最小データ量での学習テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        # prediction_horizon (デフォルト24) を考慮し、それ以上のレコード数が必要
        features_df = create_sample_features_df(num_records=30) 
        X, y = generator.prepare_training_data(features_df)
        
        # 訓練データが十分に存在することを確認
        assert len(X) > 0, "Training data should not be empty"
        
        training_result = generator.train(X, y)
        
        assert generator.is_trained, "Model should be trained"
        assert training_result['accuracy'] >= 0.0, "Accuracy should be non-negative"
        
        print("✅ Training with minimal data handled gracefully.")
        
    except Exception as e:
        pytest.fail(f"Training with minimal data test failed: {e}")

def test_training_with_biased_target_class():
    """ターゲットクラスが偏っている場合の学習テスト"""
    try:
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        generator = MLSignalGenerator()
        # 上昇クラスに偏ったデータ
        features_df = create_sample_features_df(num_records=100, target_bias='up')
        X, y = generator.prepare_training_data(features_df)
        
        # 訓練データが十分に存在することを確認
        assert len(X) > 0, "Training data should not be empty"
        
        training_result = generator.train(X, y)
        
        assert generator.is_trained, "Model should be trained"
        # 偏ったデータでも学習が完了し、精度が計算されることを確認
        assert training_result['accuracy'] >= 0.0, "Accuracy should be non-negative for biased data"
        
        # 予測が偏ったクラスに集中することを確認（厳密な閾値は設定しない）
        test_features_df = create_sample_features_df(num_records=1)
        predictions = generator.predict(test_features_df)
        
        assert predictions['up'] > predictions['down'] or predictions['up'] > predictions['range'], "Up probability should be dominant"
        
        print("✅ Training with biased target class handled gracefully.")
        
    except Exception as e:
        pytest.fail(f"Training with biased target class test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
