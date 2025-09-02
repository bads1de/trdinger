import sys
import os
sys.path.append('backend')

# constants.py単独テスト
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "constants",
        os.path.join("backend", "app", "services", "auto_strategy", "config", "constants.py")
    )
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)

    print("constants.py インポートOK")

    # INDICATOR_CHARACTERISTICSの確認
    if hasattr(constants, 'INDICATOR_CHARACTERISTICS'):
        print("INDICATOR_CHARACTERISTICS exists")

        # キー指標の確認
        key_indicators = ['RSI', 'MACD', 'EMA', 'AO', 'ATR', 'ICHIMOKU', 'ADX', 'SUPERTREND']
        for ind in key_indicators:
            if ind in constants.INDICATOR_CHARACTERISTICS:
                print(f"{ind}: 特性存在 OK")
                # タイプを表示
                ind_type = constants.INDICATOR_CHARACTERISTICS[ind].get('type', 'unknown')
                print(f"  タイプ: {ind_type}")
            else:
                print(f"{ind}: 特性存在 NG")
    else:
        print("INDICATOR_CHARACTERISTICS not found")

    # generate_characteristics_from_yamlの確認
    if hasattr(constants, 'generate_characteristics_from_yaml'):
        print("generate_characteristics_from_yaml function exists")
    else:
        print("generate_characteristics_from_yaml function not found")

    print("テスト完了")

except Exception as e:
    print(f"エラー: {e}")
    import traceback
    traceback.print_exc()