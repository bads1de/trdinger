#!/usr/bin/env python3
"""
MINMAX 問題解決確認テスト
"""

import sys
import os

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from app.services.indicators.config.indicator_definitions import indicator_registry, initialize_all_indicators

def test_invalid_metrics():
    """無効な指標が登録されていないことを確認"""
    print("=== Invalid Indicators Check Test ===")

    # 無効な指標一覧
    invalid_indicators = ["MINMAX", "MINMAXINDEX", "MAXINDEX", "MININDEX", "MAMA"]

    print(f"Invalid indicators list: {invalid_indicators}")
    registered_indicators = indicator_registry.list_indicators()
    print(f"Currently registered indicators: {len(registered_indicators)}")

    # 無効な指標が登録されていないことを確認
    registered_indicators = indicator_registry.list_indicators()
    invalid_found = []
    for indicator_name in invalid_indicators:
        if indicator_name in registered_indicators:
            invalid_found.append(indicator_name)
            print(f"[WARNING] {indicator_name} is still registered")
        else:
            print(f"[OK] {indicator_name} is properly excluded")

    if invalid_found:
        print(f"\n[WARNING] Unresolved invalid indicators: {invalid_found}")
        print("These cause implementation errors if included")
        return False
    else:
        print(f"\n[SUCCESS] All invalid indicators properly excluded!")
        return True

def test_valid_metrics():
    """有効な指標が正常に登録されていることを確認"""
    print("\n=== 有効な指標確認テスト ===")

    # 有効な指標例
    valid_indicators = [
        "SMA", "EMA", "RSI", "MACD", "ATR",
        "BBANDS", "STOCH", "CCI", "ADX"
    ]

    registered_indicators = indicator_registry.list_indicators()
    valid_found = []
    for indicator_name in valid_indicators:
        if indicator_name in registered_indicators:
            valid_found.append(indicator_name)
            print(f"[OK] {indicator_name} is properly registered")
        else:
            print(f"[WARNING] {indicator_name} is not registered")

    print(f"\nValid indicators check: {len(valid_found)}/{len(valid_indicators)} indicators OK")
    return len(valid_found) == len(valid_indicators)

if __name__ == "__main__":
    try:
        # 全インジケーターの初期化
        initialize_all_indicators()

        # テスト実行
        invalid_test_ok = test_invalid_metrics()
        valid_test_ok = test_valid_metrics()

        if invalid_test_ok and valid_test_ok:
            print("\n[SUCCESS] All tests passed! MINMAX problem is resolved.")
        else:
            print("\n[FAILURE] Tests failed. Valid/invalid indicator check needed.")
            sys.exit(1)

    except Exception as e:
        print(f"[ERROR] Test execution error: {e}")
        sys.exit(1)