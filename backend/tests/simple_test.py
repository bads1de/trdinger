import yaml

yaml_file_path = 'backend/app/services/auto_strategy/config/technical_indicators_config.yaml'

try:
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    print("Parsed YAML config keys:", list(config.keys()))

    if 'indicators' in config:
        indicators = config['indicators']
        if isinstance(indicators, dict):
            print("Number of indicators:", len(indicators))

            # AO, ICHIMOKU, SUPERTRENDを確認
            for ind in ['AO', 'ICHIMOKU', 'SUPERTREND']:
                if ind in indicators:
                    print(f"{ind}: Found")
                    ind_config = indicators[ind]
                    if isinstance(ind_config, dict):
                        print(f"  Type: {ind_config.get('type')}")
                        print(f"  Scale type: {ind_config.get('scale_type')}")
                        thresholds = ind_config.get('thresholds', {})
                        print(f"  Thresholds keys: {list(thresholds.keys()) if isinstance(thresholds, dict) else 'Not dict'}")
                else:
                    print(f"{ind}: Not found")
        else:
            print("'indicators' is not a dict")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()