#!/usr/bin/env python3

import inspect
from unittest.mock import Mock
from app.services.indicators.parameter_manager import normalize_params
from app.services.indicators.technical_indicators import momentum, statistics, trend, volatility, volume

def test_indicator_parameter_handling():
    """各指標のパラメータ処理をテスト"""
    modules = [
        ("momentum", momentum),
        ("statistics", statistics),
        ("trend", trend),
        ("volatility", volatility),
        ("volume", volume),
    ]

    print("=== INDICATOR PARAMETER TEST ===\n")

    config = Mock()
    config.indicator_name = None
    config.parameters = {}
    config.param_map = {}

    for module_name, module in modules:
        print(f"[+] {module_name.upper()} MODULE:")
        print("-" * 50)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr) and not attr_name.startswith('_'):
                try:
                    sig = inspect.signature(attr)
                    params = list(sig.parameters.keys())

                    # テスト用のパラメータを準備
                    test_params = {}
                    if "length" in params:
                        test_params["period"] = 14  # lengthパラメータが必要な場合はperiodでテスト
                    elif "period" in params:
                        test_params["period"] = 14
                    elif "k" in params or "fastk_period" in params:
                        # ストキャスティクス系
                        test_params["fastk_period"] = 5
                        test_params["fastd_period"] = 3
                    elif "r1" in params:
                        # KST系
                        test_params["r1"] = 10
                        test_params["r2"] = 15
                        test_params["r3"] = 20
                        test_params["r4"] = 30
                    else:
                        # デフォルトパラメータ
                        test_params["period"] = 14

                    config.indicator_name = attr_name

                    # normalize_paramsをテスト
                    try:
                        result = normalize_params(attr_name, test_params, config)
                        print(f"  [OK] {attr_name}: {test_params} -> {result}")
                    except Exception as e:
                        print(f"  [ERROR] {attr_name}: {test_params} -> Error: {e}")

                except Exception as e:
                    print(f"  [WARN] {attr_name}: Signature error - {e}")

        print()

def check_common_issues():
    """よくある問題をチェック"""
    print("=== COMMON ISSUE CHECK ===\n")

    # 代表的な指標をテスト
    test_cases = [
        ("RSI", {"period": 14}, "lengthパラメータが必要"),
        ("MACD", {"fast": 12, "slow": 26, "signal": 9}, "lengthパラメータ不要"),
        ("STOCH", {"fastk_period": 5, "fastd_period": 3}, "lengthパラメータ不要"),
        ("KST", {"r1": 10, "r2": 15, "r3": 20, "r4": 30}, "lengthパラメータ不要"),
        ("LINEARREG", {"period": 14}, "period->length変換"),
        ("SMA", {"period": 20}, "period->length変換"),
    ]

    config = Mock()
    config.indicator_name = None
    config.parameters = {}
    config.param_map = {}

    for indicator_name, params, expected in test_cases:
        config.indicator_name = indicator_name
        try:
            result = normalize_params(indicator_name, params, config)
            print(f"  {indicator_name}: {params} -> {result} ({expected})")
        except Exception as e:
            print(f"  [ERROR] {indicator_name}: Error - {e}")

if __name__ == "__main__":
    test_indicator_parameter_handling()
    check_common_issues()