"""
ML特徴量と信号生成の統合テスト

FeatureEngineeringServiceによって生成された特徴量が、
MLSignalGeneratorによって正しく消費・処理されることを検証します。
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
def create_comprehensive_ohlcv_data(num_records=200):
    """包括的なOHLCVデータを作成"""
    dates = pd.date_range(start='2024-01-01', periods=num_records, freq='h')
    np.random.seed(42)

    price_base = 50000
    prices = [price_base]
    for i in range(1, num_records):
        prices.append(prices[-1] * (1 + np.random.randn() * 0.005))
    prices = np.array(prices)

    ohlcv_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(num_records)) * 0.001),
        'low': prices * (1 - np.abs(np.random.randn(num_records)) * 0.001),
        'close': prices * (1 + np.random.randn(num_records) * 0.0005),
        'volume': np.random.rand(num_records) * 1000000
    })
    return ohlcv_data

def test_feature_engineering_to_signal_generator_flow():
    """特徴量エンジニアリングから信号生成へのフローテスト"""
    try:
        from app.core.services.feature_engineering import FeatureEngineeringService
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        # 1. OHLCVデータを生成
        ohlcv_data = create_comprehensive_ohlcv_data(num_records=200)
        
        # 2. FeatureEngineeringServiceで特徴量を計算
        feature_service = FeatureEngineeringService()
        features_df = feature_service.calculate_advanced_features(ohlcv_data)
        
        # 必要なターゲットカラムを追加（テスト用）
        features_df['target_up'] = (np.random.rand(len(features_df)) > 0.5).astype(int)
        features_df['target_down'] = (np.random.rand(len(features_df)) > 0.5).astype(int)
        features_df['target_range'] = (np.random.rand(len(features_df)) > 0.5).astype(int)

        # 3. MLSignalGeneratorで学習データを準備
        signal_generator = MLSignalGenerator()
        X, y = signal_generator.prepare_training_data(features_df)
        
        # 準備されたデータの妥当性を確認
        assert not X.empty, "Prepared features (X) should not be empty"
        assert not y.empty, "Prepared labels (y) should not be empty"
        assert len(X) == len(y), "X and y should have the same number of rows"
        
        # 4. MLSignalGeneratorでモデルを学習
        training_result = signal_generator.train(X, y)
        assert signal_generator.is_trained, "Model should be trained"
        assert training_result['accuracy'] >= 0.0, "Accuracy should be non-negative"
        
        # 5. MLSignalGeneratorで予測を実行
        # 予測には、学習時と同じ特徴量セットを持つデータが必要
        # prepare_training_dataで生成されたXを使用
        predictions = signal_generator.predict(X.head(1))
        
        # 予測結果の妥当性を確認
        assert 'up' in predictions
        assert 'down' in predictions
        assert 'range' in predictions
        prob_sum = predictions['up'] + predictions['down'] + predictions['range']
        assert abs(prob_sum - 1.0) < 0.1, "Sum of probabilities should be close to 1.0"
        
        print("✅ Feature Engineering and Signal Generator integration flow works correctly.")
        
    except Exception as e:
        pytest.fail(f"Feature Engineering to Signal Generator integration test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])