#!/usr/bin/env python3
"""
PVOL指標修正のテストファイル
"""

import numpy as np
import pandas as pd
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

def test_pvol_indicator_fix():
    """PVOL指標の修正をテストする"""

    # テスト用の合成データを生成
    np.random.seed(42)
    n_bars = 100
    close_prices = 100 + np.random.randn(n_bars).cumsum() * 0.5
    volumes = np.random.randint(1000, 10000, n_bars)

    # OHLCデータをDataFrameにまとめる
    test_data = pd.DataFrame({
        'close': close_prices,
        'volume': volumes,
        'open': close_prices + np.random.normal(0, 1, n_bars),
        'high': close_prices + np.random.uniform(0.5, 2, n_bars),
        'low': close_prices - np.random.uniform(0.5, 2, n_bars)
    })

    print("=== PVOL指標修正テスト開始 ===")
    print(f"テストデータサイズ: {len(test_data)} bars")

    # TechnicalIndicatorServiceの初期化
    orchestrator = TechnicalIndicatorService()

    try:
        # PVOL指標を計算（正しいパラメータのみ）
        pvol_params = {'signed': True}
        pvol_result = orchestrator.calculate_indicator(test_data, "PVOL", pvol_params)

        print("\n=== PVOL計算成功！ ===")
        print(f"結果タイプ: {type(pvol_result)}")
        print(f"結果形状: {pvol_result.shape if hasattr(pvol_result, 'shape') else 'N/A'}")

        # 結果の内容を確認
        if isinstance(pvol_result, (pd.DataFrame, pd.Series)):
            print(f"結果の列数: {pvol_result.shape[1] if len(pvol_result.shape) > 1 else 1}")
            print(f"  データ: {pvol_result.iloc[0]:.6f}")
        # NaNの数をチェック
        if hasattr(pvol_result, 'isna'):
            nan_count = pvol_result.isna().sum().sum() if len(pvol_result.shape) > 1 else pvol_result.isna().sum()
            print(f"NaNの数: {nan_count}")

        print("\n=== PVOL指標修正テストが成功しました！ ===")
        return True

    except Exception as e:
        print(f"\n=== PVOL指標計算エラー: {e} ===")
        print(f"エラー詳細: {type(e).__name__}: {str(e)}")
        return False

def test_pvol_with_wrong_parameters():
    """PVOLに対して誤ったパラメータ（length, period）を渡した場合のテスト"""

    print("\n=== 誤ったパラメータテスト開始 ===")

    # テストデータを生成
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500]
    })

    orchestrator = TechnicalIndicatorService()

    # 間違ったパラメータをテスト
    wrong_params = [
        {'length': 10},  # 不正なパラメータ
        {'period': 10, 'signed': True},  # periodパラメータを含む
        {'length': 10, 'period': 20, 'signed': True}  # 両方
    ]

    any_failed = False
    for i, params in enumerate(wrong_params):
        print(f"パラメータセット {i+1}: {params}")
        try:
            result = orchestrator.calculate_indicator(data, "PVOL", params)
            if result is not None:
                print(f"  ✅ 成功: 誤ったパラメータが無視されました")
            else:
                print(f"  ⚠️  警告: 結果がNoneです")
        except Exception as e:
            print(f"  ❌ エラー: {e}")
            any_failed = True

    return not any_failed

if __name__ == "__main__":
    print("PVOL指標修正テストを開始します...")

    # 基本テスト
    basic_success = test_pvol_indicator_fix()

    # 誤ったパラメータテスト
    wrong_param_success = test_pvol_with_wrong_parameters()

    print("\n=== 最終結果 ===")
    print(f"基本テスト: {'SUCCESS' if basic_success else 'FAILED'}")
    print(f"誤パラメータテスト: {'SUCCESS' if wrong_param_success else 'FAILED'}")

    if basic_success and wrong_param_success:
        print("すべてのテストが成功しました！ PVOL指標修正完了です。")
    else:
        print("一部のテストが失敗しました。修正が必要です。")