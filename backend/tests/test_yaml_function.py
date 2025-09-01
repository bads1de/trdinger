import sys
import os
sys.path.append('backend')

# constantsのYAML読み込み関数を単独テスト
try:
    spec = __import__('importlib.util').util.spec_from_file_location(
        "constants",
        os.path.join("backend", "app", "services", "auto_strategy", "config", "constants.py")
    )
    constants = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(constants)

    # YAMLファイルパスを確認
    config_dir = constants.CONFIG_DIR
    yaml_path = constants.YAML_CONFIG_PATH
    print(f"CONFIG_DIR: {config_dir}")
    print(f"YAML_CONFIG_PATH: {yaml_path}")
    print(f"File exists: {os.path.exists(yaml_path)}")

    # generate_characteristics_from_yaml関数をテスト
    if hasattr(constants, 'generate_characteristics_from_yaml'):
        result = constants.generate_characteristics_from_yaml(yaml_path)
        print(f"Generated characteristics: {len(result)} items")

        # AO, ICHIMOKU, SUPERTRENDを確認
        for ind in ['AO', 'ICHIMOKU', 'SUPERTREND']:
            if ind in result:
                print(f"{ind}: Found - {result[ind]}")
            else:
                print(f"{ind}: Not found")
    else:
        print("generate_characteristics_from_yaml function not found")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()