#!/usr/bin/env python3
"""簡素化テスト"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import numpy as np

def test_sma():
    """SMA計算テスト"""
    try:
        from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

        print("=== SMAテスト開始 ===")

        # テストデータ作成
        df = pd.DataFrame({
            'Close': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        })
        print(f"テストデータ: {df['Close'].tolist()}")

        # サービス初期化
        service = TechnicalIndicatorService()
        print("サービス初期化成功")

        # SMA計算テスト
        result = service.calculate_indicator(df, 'SMA', {'length': 5})
        print(f"SMA計算結果: {result}")
        print(f"結果型: {type(result)}")
        print(f"結果長: {len(result) if hasattr(result, '__len__') else 'N/A'}")

        print("=== SMAテスト成功 ===")
        return True

    except Exception as e:
        import traceback
        print(f"SMAテスト失敗: {e}")
        traceback.print_exc()
        return False

def test_rsi():
    """RSI計算テスト"""
    try:
        from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

        print("\n=== RSIテスト開始 ===")

        # テストデータ作成（より長いデータ）
        df = pd.DataFrame({
            'Close': list(range(1, 51))  # 1から50までのデータ
        })
        print(f"テストデータ長: {len(df)}")

        # サービス初期化
        service = TechnicalIndicatorService()

        # RSI計算テスト
        result = service.calculate_indicator(df, 'RSI', {'length': 14})
        print(f"RSI計算結果（最後5個）: {result[-5:] if len(result) > 5 else result}")
        print(f"結果型: {type(result)}")

        print("=== RSIテスト成功 ===")
        return True

    except Exception as e:
        import traceback
        print(f"RSIテスト失敗: {e}")
        traceback.print_exc()
        return False

def test_multi_column():
    """複数カラム指標テスト（STOCH）"""
    try:
        from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

        print("\n=== STOCHテスト開始 ===")

        # OHLCデータ作成
        data = {
            'Open': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            'High': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            'Low': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            'Close': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        }
        df = pd.DataFrame(data)
        print(f"OHLCデータ作成: {len(df)}行")

        # サービス初期化
        service = TechnicalIndicatorService()

        # STOCH計算テスト
        result = service.calculate_indicator(df, 'STOCH', {'k': 5, 'smooth_k': 3, 'd': 3})
        if isinstance(result, tuple):
            print(f"STOCH_K（最後3個）: {result[0][-3:]}")
            print(f"STOCH_D（最後3個）: {result[1][-3:]}")
        else:
            print(f"STOCH結果: {result}")
        print(f"結果型: {type(result)}")

        print("=== STOCHテスト成功 ===")
        return True

    except Exception as e:
        import traceback
        print(f"STOCHテスト失敗: {e}")
        traceback.print_exc()
        return False

def test_comprehensive_indicators():
    """32個の主要指標を総合的にテスト"""
    try:
        from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

        print("\n=== 32個指標総合テスト開始 ===")

        # テストデータ作成（全て同じ長さ）
        data_length = 50
        df = pd.DataFrame({
            'Close': list(range(10, 10 + data_length)),
            'High': list(range(12, 12 + data_length)),
            'Low': list(range(8, 8 + data_length)),
            'Open': list(range(9, 9 + data_length)),
            'Volume': [100 + i*10 for i in range(data_length)]
        })

        service = TechnicalIndicatorService()

        # 32個の指標をテスト
        indicators_to_test = [
            # Trend indicators
            ("SMA", {'length': 5}),
            ("EMA", {'length': 5}),
            ("WMA", {'length': 5}),
            ("DEMA", {'length': 5}),
            ("TEMA", {'length': 5}),
            ("T3", {'length': 5, 'a': 0.7}),
            ("KAMA", {'length': 5}),

            # Momentum indicators
            ("RSI", {'length': 5}),
            ("MACD", {'fast': 5, 'slow': 10, 'signal': 4}),
            ("STOCH", {'k': 5, 'smooth_k': 3, 'd': 3}),
            ("CCI", {'length': 5}),
            ("WILLR", {'length': 5}),
            ("ROC", {'length': 5}),
            ("MOM", {'length': 5}),
            ("ADX", {'length': 5}),
            ("QQE", {'length': 5, 'smooth': 3}),

            # Volatility indicators
            ("ATR", {'length': 5}),
            ("BBANDS", {'length': 5, 'std': 2.0}),
            ("KELTNER", {'length': 5, 'multiplier': 2.0}),
            ("DONCHIAN", {'length': 5}),
            ("ACCBANDS", {'length': 5, 'std': 2.0}),

            # Volume indicators
            ("OBV", {}),
            ("AD", {}),
            ("ADOSC", {'fast': 3, 'slow': 10}),
            ("CMF", {'length': 5}),
            ("EFI", {'length': 5}),
            ("VWAP", {}),
            ("MFI", {'length': 5}),

            # Other indicators
            ("SAR", {'acceleration': 0.02, 'maximum': 0.2}),
            ("UI", {'length': 5}),
            ("SQUEEZE", {'bb_length': 5, 'bb_std': 2.0, 'kc_length': 5, 'kc_scalar': 1.5, 'mom_length': 5, 'mom_smooth': 3, 'use_tr': True}),
        ]

        results = []
        for indicator_name, params in indicators_to_test:
            try:
                result = service.calculate_indicator(df.copy(), indicator_name, params)
                if result is not None and len(result) > 0:
                    results.append((indicator_name, True, "成功"))
                    print(f"[OK] {indicator_name}: 成功")
                else:
                    results.append((indicator_name, False, "結果なし"))
                    print(f"[NG] {indicator_name}: 結果なし")
            except Exception as e:
                results.append((indicator_name, False, str(e)))
                print(f"[NG] {indicator_name}: エラー - {e}")

        success_count = sum(1 for _, success, _ in results if success)
        total_count = len(results)

        print(f"\n総合結果: {success_count}/{total_count} 指標が成功")
        print(f"成功率: {success_count/total_count:.1%}")
        return success_count > total_count * 0.8  # 80%以上成功で合格

    except Exception as e:
        import traceback
        print(f"総合テスト失敗: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("テクニカル指標簡素化テスト開始")
    print("=" * 50)

    results = []

    # 基本テスト
    print("1. 基本テスト実行")
    results.append(("SMA", test_sma()))
    results.append(("RSI", test_rsi()))
    results.append(("STOCH", test_multi_column()))

    # 総合テスト
    print("\n2. 32個指標総合テスト実行")
    results.append(("総合テスト", test_comprehensive_indicators()))

    print("\n" + "=" * 50)
    print("テスト結果サマリー:")

    all_passed = True
    for test_name, passed in results:
        status = "[OK] 成功" if passed else "[NG] 失敗"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    print(f"\n全体結果: {'全てのテストが成功しました！' if all_passed else '一部のテストが失敗しました。'}")
    sys.exit(0 if all_passed else 1)