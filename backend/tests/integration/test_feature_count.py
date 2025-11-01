#!/usr/bin/env python
"""
特徴量数の実際の変化をテストするスクリプト
"""
import sys
sys.path.append('.')

from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
from app.services.ml.feature_engineering.crypto_features import CryptoFeatures
from app.services.ml.feature_engineering.advanced_features import AdvancedFeatureEngineer
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def main():
    print("=" * 60)
    print("特徴量数テスト - 実際の動作確認")
    print("=" * 60)

    # サンプルデータを作成
    dates = pd.date_range(start='2024-10-01', end='2024-10-15', freq='1h')
    data = {
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 102,
        'low': np.random.randn(len(dates)).cumsum() + 98,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }
    df = pd.DataFrame(data, index=dates)

    print(f"\n1. 元データ:")
    print(f"   行数: {len(df)}")
    print(f"   列数: {len(df.columns)}")
    print(f"   列名: {list(df.columns)}")

    # CryptoFeatures単体テスト
    print(f"\n2. CryptoFeatures単体テスト:")
    try:
        crypto = CryptoFeatures()
        result_crypto = crypto.create_crypto_features(df)
        print(f"   ✓ 成功")
        print(f"   追加特徴量: {len(result_crypto.columns) - len(df.columns)}個")
        crypto_cols = [col for col in result_crypto.columns if not col in df.columns]
        print(f"   新規列: {crypto_cols[:5]}")
    except Exception as e:
        print(f"   ERROR: エラー: {e}")

    # AdvancedFeatureEngineer単体テスト
    print(f"\n3. AdvancedFeatureEngineer単体テスト:")
    try:
        advanced = AdvancedFeatureEngineer()
        result_advanced = advanced.create_advanced_features(df)
        print(f"   ✓ 成功")
        print(f"   追加特徴量: {len(result_advanced.columns) - len(df.columns)}個")
    except Exception as e:
        print(f"   ERROR: エラー: {e}")

    # FeatureEngineeringService全体テスト
    print(f"\n4. FeatureEngineeringService全体テスト:")
    try:
        service = FeatureEngineeringService()
        result_service = service.calculate_advanced_features(df)
        print(f"   ✓ 成功")
        print(f"   総特徴量数: {len(result_service.columns)}")
        print(f"   追加特徴量: {len(result_service.columns) - len(df.columns)}個")

        # 特殊特徴量の確認
        special_features = []
        for col in result_service.columns:
            if col.startswith(('volume_', 'vwap_', 'bb_', 'rsi_', 'price_', 'crypto_', 'advanced_')):
                special_features.append(col)

        print(f"   特殊特徴量の数: {len(special_features)}個")
        if special_features:
            print(f"   特殊特徴量の例: {special_features[:10]}")

    except Exception as e:
        print(f"   ERROR: エラー: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)

if __name__ == '__main__':
    main()
