#!/usr/bin/env python3
"""
登録されているすべてのテクニカル指標を取得
"""

from app.services.indicators.config import indicator_registry
from backend.app.services.auto_strategy.config.constants import (
    VALID_INDICATOR_TYPES,
    ML_INDICATOR_TYPES,
)

print("=== 登録されているすべてのテクニカル指標 ===")

# indicator_registryから登録されている指標を取得
registered_indicators = indicator_registry.get_supported_indicator_names()
print(f"indicator_registryに登録されている指標数: {len(registered_indicators)}")
print(f"登録指標: {sorted(registered_indicators)}")

print(f"\n=== VALID_INDICATOR_TYPESの指標 ===")
print(f"VALID_INDICATOR_TYPES数: {len(VALID_INDICATOR_TYPES)}")
print(f"VALID_INDICATOR_TYPES: {sorted(VALID_INDICATOR_TYPES)}")

print(f"\n=== ML指標 ===")
print(f"ML指標数: {len(ML_INDICATOR_TYPES)}")
print(f"ML指標: {sorted(ML_INDICATOR_TYPES)}")

print(f"\n=== 差分分析 ===")

# indicator_registryにあるがVALID_INDICATOR_TYPESにない指標
registry_only = (
    set(registered_indicators) - set(VALID_INDICATOR_TYPES) - set(ML_INDICATOR_TYPES)
)
print(f"registryにあるがVALID_INDICATOR_TYPESにない指標 ({len(registry_only)}個):")
for indicator in sorted(registry_only):
    print(f"  - {indicator}")

# VALID_INDICATOR_TYPESにあるがregistryにない指標
valid_only = set(VALID_INDICATOR_TYPES) - set(registered_indicators)
print(f"\nVALID_INDICATOR_TYPESにあるがregistryにない指標 ({len(valid_only)}個):")
for indicator in sorted(valid_only):
    print(f"  - {indicator}")

# 共通の指標
common = set(registered_indicators) & set(VALID_INDICATOR_TYPES)
print(f"\n共通の指標 ({len(common)}個):")
for indicator in sorted(common):
    print(f"  - {indicator}")

print(f"\n=== 総計 ===")
all_indicators = (
    set(registered_indicators) | set(VALID_INDICATOR_TYPES) | set(ML_INDICATOR_TYPES)
)
print(f"すべての指標の総数: {len(all_indicators)}")

# 各指標の詳細情報を取得
print(f"\n=== 指標詳細情報 ===")
for indicator in sorted(registered_indicators):
    config = indicator_registry.get_indicator_config(indicator)
    if config:
        print(f"{indicator}:")
        print(f"  - カテゴリ: {config.category}")
        print(f"  - 結果タイプ: {config.result_type}")
        print(f"  - 必要データ: {config.required_data}")
        print(f"  - パラメータ数: {len(config.parameters)}")
        if config.parameters:
            param_names = [p.name for p in config.parameters.values()]
            print(f"  - パラメータ: {param_names}")
    else:
        print(f"{indicator}: 設定なし")
