import yaml

try:
    with open('backend/app/services/auto_strategy/config/technical_indicators_config.yaml', 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    print("YAML構文OK")
    print(f"indicatorsセクションの数: {len(data.get('indicators', {}))}")

    # 新しく追加された指標を確認
    new_indicators = ['AO', 'ATR', 'ICHIMOKU', 'DX', 'SUPERTREND']
    indicators = data.get('indicators', {})
    for ind in new_indicators:
        if ind in indicators:
            print(f"{ind}: OK")
            # riskレベルを確認
            thresholds = indicators[ind].get('thresholds', {})
            if 'aggressive' in thresholds and 'normal' in thresholds and 'conservative' in thresholds:
                print(f"  {ind} riskレベル: OK")
            else:
                print(f"  {ind} riskレベル: NG - missing levels")
        else:
            print(f"{ind}: NG - not found")

except Exception as e:
    print(f"エラー: {e}")