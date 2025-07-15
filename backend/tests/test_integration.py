#!/usr/bin/env python3
"""
ML統合テスト

MLOrchestratorとMLIndicatorServiceの統合テスト
"""

import warnings
import pandas as pd
import numpy as np

# 非推奨警告を無視
warnings.filterwarnings('ignore', category=DeprecationWarning)

from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
from app.core.services.auto_strategy.services.ml_orchestrator import MLOrchestrator

def test_integration():
    """統合テスト実行"""
    
    # テストデータ作成
    test_data = pd.DataFrame({
        'open': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 100,
        'low': np.random.rand(100) * 100,
        'close': np.random.rand(100) * 100,
        'volume': np.random.rand(100) * 1000
    })

    print('=== ML統合テスト ===')

    # MLOrchestratorのテスト
    print('MLOrchestrator テスト:')
    orchestrator = MLOrchestrator()
    result1 = orchestrator.calculate_ml_indicators(test_data)
    print(f'  結果キー: {list(result1.keys())}')
    print(f'  配列長: {len(result1["ML_UP_PROB"])}')

    # MLIndicatorServiceのテスト（プロキシ）
    print('MLIndicatorService テスト:')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        indicator_service = MLIndicatorService()
        result2 = indicator_service.calculate_ml_indicators(test_data)
        print(f'  結果キー: {list(result2.keys())}')
        print(f'  配列長: {len(result2["ML_UP_PROB"])}')

    # 結果の一致確認
    print('結果比較:')
    for key in result1.keys():
        match = np.array_equal(result1[key], result2[key])
        print(f'  {key}: {"一致" if match else "不一致"}')

    # 単一指標テスト
    print('単一指標テスト:')
    single1 = orchestrator.calculate_single_ml_indicator("ML_UP_PROB", test_data)
    single2 = indicator_service.calculate_single_ml_indicator("ML_UP_PROB", test_data)
    single_match = np.array_equal(single1, single2)
    print(f'  単一指標一致: {"一致" if single_match else "不一致"}')

    # モデル状態テスト
    print('モデル状態テスト:')
    status1 = orchestrator.get_model_status()
    status2 = indicator_service.get_model_status()
    print(f'  MLOrchestrator モデル読み込み: {status1.get("is_model_loaded", False)}')
    print(f'  MLIndicatorService モデル読み込み: {status2.get("is_model_loaded", False)}')

    print('統合テスト完了')
    
    return True

if __name__ == "__main__":
    test_integration()
