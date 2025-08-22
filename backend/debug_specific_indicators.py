#!/usr/bin/env python3

import inspect
from unittest.mock import Mock
from app.services.indicators.parameter_manager import normalize_params
from app.services.indicators.technical_indicators import momentum, trend

def debug_specific_indicators():
    """SMAとRSIのパラメータ処理を詳細にデバッグ"""
    print("=== SPECIFIC INDICATORS DEBUG ===\n")

    config = Mock()
    config.indicator_name = None
    config.parameters = {}
    config.param_map = {}

    # RSI関数のシグネチャを確認
    print("[+] RSI FUNCTION SIGNATURE:")
    try:
        rsi_sig = inspect.signature(momentum.MomentumIndicators.rsi)
        print(f"  Parameters: {list(rsi_sig.parameters.keys())}")
        for name, param in rsi_sig.parameters.items():
            print(f"    {name}: {param}")
    except Exception as e:
        print(f"  Error: {e}")

    # SMA関数のシグネチャを確認
    print("\n[+] SMA FUNCTION SIGNATURE:")
    try:
        sma_sig = inspect.signature(trend.TrendIndicators.ma)
        print(f"  Parameters: {list(sma_sig.parameters.keys())}")
        for name, param in sma_sig.parameters.items():
            print(f"    {name}: {param}")
    except Exception as e:
        print(f"  Error: {e}")

    # RSIのテスト
    print("\n[+] RSI PARAMETER TEST:")
    config.indicator_name = "RSI"
    params = {"period": 14}
    try:
        result = normalize_params("RSI", params, config)
        print(f"  Input: {params}")
        print(f"  Output: {result}")
    except Exception as e:
        print(f"  Error: {e}")

    # SMAのテスト
    print("\n[+] SMA PARAMETER TEST:")
    config.indicator_name = "SMA"
    params = {"period": 20}
    try:
        result = normalize_params("SMA", params, config)
        print(f"  Input: {params}")
        print(f"  Output: {result}")
    except Exception as e:
        print(f"  Error: {e}")

    # period_basedセットの内容を確認（関数内で取得）
    print("\n[+] CHECKING PERIOD_BASED SET:")
    try:
        # parameter_managerモジュールのソースを確認
        import app.services.indicators.parameter_manager as pm
        source = inspect.getsource(pm.normalize_params)

        # period_basedセットを探す
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'period_based' in line and '=' in line:
                print(f"  Found period_based definition at line {i+1}: {line}")
                # 次の数行も表示
                for j in range(1, min(10, len(lines)-i)):
                    if lines[i+j].strip():
                        print(f"    {lines[i+j]}")
                    else:
                        break
                break
    except Exception as e:
        print(f"  Error reading source: {e}")

if __name__ == "__main__":
    debug_specific_indicators()