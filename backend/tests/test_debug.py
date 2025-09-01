import sys
import os
sys.path.append('backend')

# constantsのYAML読み込み関数をテスト
try:
    spec = __import__('importlib.util').util.spec_from_file_location(
        "constants",
        os.path.join("backend", "app", "services", "auto_strategy", "config", "constants.py")
    )
    constants = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(constants)

    # シンプルな呼び出しテスト
    import sys
    print(f"Generated characteristics: {len(result)} items", file=sys.stderr)

    if len(result) > 0:
        # 最初の5つのキー表示
        keys = list(result.keys())[:5]
        print("First 5 characteristics keys:", keys)

        # AOがあれば表示
        if 'AO' in result:
            print("AO found!")
            print("AO characteristics:", result['AO'])
        else:
            print("AO not found")

        # いくつかだけ表示
        count = 0
        for k, v in result.items():
            print(f"{k}: {v.get('type', '?')}")
            count += 1
            if count >= 3:  # 3つだけ
                print("... (showing first 3)")
                break
    else:
        print("No characteristics generated", file=sys.stderr)
        print("Checking YAML file exists:", os.path.exists(constants.YAML_CONFIG_PATH))

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()