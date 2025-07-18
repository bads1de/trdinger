#!/usr/bin/env python3
"""
ML統合テスト

MLOrchestratorの統合テスト（MLIndicatorServiceは廃止済み）
"""

import warnings
import pandas as pd
import numpy as np

from app.core.services.auto_strategy.services.ml_orchestrator import MLOrchestrator


def test_integration():
    """統合テスト実行"""

    # テストデータ作成
    test_data = pd.DataFrame(
        {
            "open": np.random.rand(100) * 100,
            "high": np.random.rand(100) * 100,
            "low": np.random.rand(100) * 100,
            "close": np.random.rand(100) * 100,
            "volume": np.random.rand(100) * 1000,
        }
    )

    print("=== ML統合テスト ===")

    # MLOrchestratorのテスト
    print("MLOrchestrator テスト:")
    orchestrator = MLOrchestrator()
    result1 = orchestrator.calculate_ml_indicators(test_data)
    print(f"  結果キー: {list(result1.keys())}")
    print(f'  配列長: {len(result1["ML_UP_PROB"])}')

    # 結果の妥当性確認
    print("結果検証:")
    required_keys = ["ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"]
    for key in required_keys:
        if key in result1:
            print(f"  {key}: ✓ 存在")
            print(f"    配列長: {len(result1[key])}")
            print(f"    データ型: {type(result1[key])}")
        else:
            print(f"  {key}: ✗ 不存在")

    # 単一指標テスト
    print("単一指標テスト:")
    single_result = orchestrator.calculate_single_ml_indicator("ML_UP_PROB", test_data)
    print(f"  単一指標配列長: {len(single_result)}")
    print(f"  単一指標データ型: {type(single_result)}")

    # モデル状態テスト
    print("モデル状態テスト:")
    status = orchestrator.get_model_status()
    print(f'  モデル読み込み: {status.get("is_model_loaded", False)}')
    print(f'  最終予測: {status.get("last_predictions", {})}')

    # 特徴量重要度テスト
    print("特徴量重要度テスト:")
    importance = orchestrator.get_feature_importance(top_n=5)
    print(f"  特徴量重要度: {len(importance)}個の特徴量")

    print("統合テスト完了")

    return True


if __name__ == "__main__":
    test_integration()
