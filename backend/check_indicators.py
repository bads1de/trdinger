#!/usr/bin/env python3
from app.services.indicators.config import indicator_registry

# adapter_function があるインジケータのみを取得
supported_names = [
    name for name in indicator_registry.get_supported_indicator_names()
    if (
        indicator_registry.get_indicator_config(name)
        and indicator_registry.get_indicator_config(name).adapter_function
    )
]

print(f'Total indicators with adapter_function: {len(supported_names)}')
print('\nFirst 20 indicators:')
for i, name in enumerate(supported_names[:20]):
    print(f'{i+1:2d}. {name}')

print('\nLast 20 indicators:')
for i, name in enumerate(supported_names[-20:]):
    print(f'{i+95:3d}. {name}')

# period_based セットに含まれていない length パラメータを使うインジケータをチェック
period_based = {"MA", "MAVP", "MAX", "MIN", "SUM", "BETA", "CORREL", "LINEARREG", "LINEARREG_SLOPE", "STDDEV", "VAR", "SAR"}

length_based_indicators = []
for name in supported_names:
    config = indicator_registry.get_indicator_config(name)
    if config and 'length' in config.parameters and name not in period_based:
        length_based_indicators.append(name)

print(f'\n\nIndicators using length parameter (not in period_based): {len(length_based_indicators)}')
for name in length_based_indicators:
    print(f'- {name}')

# 実験的インジケータをチェック
experimental = getattr(indicator_registry, 'experimental_indicators', set())
period_based = {"MA", "MAVP", "MAX", "MIN", "SUM", "BETA", "CORREL", "LINEARREG", "LINEARREG_SLOPE", "STDDEV", "VAR", "SAR"}

print(f'\n\nExperimental indicators: {len(experimental)}')
for name in sorted(experimental):
    if name in supported_names:
        config = indicator_registry.get_indicator_config(name)
        param_type = 'none'
        if config and 'length' in config.parameters and name not in period_based:
            param_type = 'length'
        elif config and 'period' in config.parameters and name in period_based:
            param_type = 'period'
        print(f'- {name} ({param_type})')