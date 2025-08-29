#!/usr/bin/env python3
"""
HILO指標修正のテストファイル
"""

import numpy as np
import pandas as pd
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
from app.services.indicators.config.indicator_definitions import setup_trend_indicators

def test_hilo_indicator_fix():
    """HILO指標の修正をテストする"""

    # テスト用の合成データを生成
    np.random.seed(42)
    n_bars = 100
    open_prices = 100 + np.random.randn(n_bars).cumsum() * 0.5
    close_prices = open_prices + np.random.randn(n_bars) * 2
    high_prices = np.maximum(open_prices, close_prices) + np.random.uniform(0, 2, n_bars)
    low_prices = np.minimum(open_prices, close_prices) - np.random.uniform(0, 2, n_bars)

    # OHLCデータをDataFrameにまとめる
    test_data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n_bars)
    })

    print("=== HILO指標修正テスト開始 ===")
    print(f"テストデータサイズ: {len(test_data)} bars")
    print(f"最高値範囲: {high_prices.min():.2f} - {high_prices.max():.2f}")
    print(f"最価格範囲: {close_prices.min():.2f} - {close_prices.max():.2f}")

    # TechnicalIndicatorServiceの初期化
    orchestrator = TechnicalIndicatorService()

    try:
        # HILO指標を計算
        hilo_result = orchestrator.calculate_indicator(test_data, "HILO", {})

        print("\n=== HILO計算成功！ ===")
        print(f"結果タイプ: {type(hilo_result)}")
        print(f"結果形状: {hilo_result.shape if hasattr(hilo_result, 'shape') else 'N/A'}")

        # 結果の内容を確認
        if isinstance(hilo_result, (pd.DataFrame, pd.Series)):
            print(f"結果の列数: {hilo_result.shape[1] if len(hilo_result.shape) > 1 else 1}")
            print(f"最初の行（最初の10個）:")
            if len(hilo_result.shape) > 1:
                for col in hilo_result.columns[:10]:
                    print(f"  {col}: {hilo_result[col].iloc[0]:.6f}")
            else:
                print(f"  データ: {hilo_result.iloc[0]:.6f}")

        # NaNの数をチェック
        if hasattr(hilo_result, 'isna'):
            nan_count = hilo_result.isna().sum().sum() if len(hilo_result.shape) > 1 else hilo_result.isna().sum()
            print(f"NaNの数: {nan_count}")

        print("\n=== HILO指標修正テストが成功しました！ ===")
        return True

    except Exception as e:
        print(f"\n=== HILO指標計算エラー: {e} ===")
        print(f"エラー詳細: {type(e).__name__}: {str(e)}")
        return False

def test_hilo_with_autostrategy():
    """オートストラテジー内でHILOを使用したテスト"""

    print("\n=== オートストラテジー連携テスト開始 ===")

    # 基本的なテストデータの作成 (HILO 계산을 위해 충분한 데이터를 확보 - 최소 22개 필요)
    base_data = {
        'open': [], 'high': [], 'low': [], 'close': [], 'volume': []
    }

    # 보다 충분한 테스트 데이터 생성
    np.random.seed(123)  # 다른 seed로 더다양한 데이터 생성
    for i in range(50):  # 22개 이상의 데이터 포인트 생성
        price = 100 + i*0.5 + np.random.normal(0, 1)
        base_data['open'].append(price)
        base_data['high'].append(price + np.random.uniform(0.5, 2))
        base_data['low'].append(price - np.random.uniform(0.5, 2))
        base_data['close'].append(price + np.random.normal(0, 0.8))
        base_data['volume'].append(1000 + i*50 + np.random.randint(-100, 100))

    test_df = pd.DataFrame(base_data)

    orchestrator = TechnicalIndicatorService()

    try:
        # HILO指標計算（オートストラテジーで使用されるパラメータ）
        params = {
            'high_length': 13,
            'low_length': 21
        }

        hilo_result = orchestrator.calculate_indicator(test_df, "HILO", params)
        print("オートストラテジー連携テスト成功！")
        print(f"HILO結果タイプ: {type(hilo_result)}")
        return True

    except Exception as e:
        print(f"オートストラテジー連携テスト失敗: {e}")
        return False

if __name__ == "__main__":
    print("HILO指標修正テストを開始します...")

    # 基本テスト
    basic_success = test_hilo_indicator_fix()

    # オートストラテジー連携テスト
    autostrategy_success = test_hilo_with_autostrategy()

    print("\n=== 最終結果 ===")
    print(f"基本テスト: {'SUCCESS' if basic_success else 'FAILED'}")
    print(f"連携テスト: {'SUCCESS' if autostrategy_success else 'FAILED'}")

    if basic_success and autostrategy_success:
        print("すべてのテストが成功しました！ HILO指標修正完了です。")
    else:
        print("一部のテストが失敗しました。修正が必要です。")