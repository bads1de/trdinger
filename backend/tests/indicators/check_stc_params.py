#!/usr/bin/env python3
"""
STC関数のパラメータを確認するスクリプト
"""

import pandas_ta as ta
import inspect

def check_stc_signature():
    """STC関数のシグネチャを確認"""
    print("=== STC Function Signature Check ===")

    try:
        # STC関数を取得
        stc_func = getattr(ta, 'stc', None)
        if stc_func is None:
            print("ERROR: STC function not found in pandas_ta")
            return

        # シグネチャを取得
        sig = inspect.signature(stc_func)
        print(f"STC function signature: STC{sig}")
        print()

        # パラメータの詳細を表示
        print("Parameters:")
        for name, param in sig.parameters.items():
            if name != 'data':  # dataパラメータは除外
                print(f"  {name}: {param}")
                if param.default != inspect.Parameter.empty:
                    print(f"    Default: {param.default}")

        print()

        # STCのヘルプを表示
        print("=== STC Help ===")
        help(ta.stc)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_stc_signature()