#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/c:/Users/buti3/trading/backend')

try:
    from app.services.indicators.config import indicator_registry

    # Get all supported indicators
    all_names = indicator_registry.get_supported_indicator_names()
    supported_names = [
        name
        for name in all_names
        if (
            indicator_registry.get_indicator_config(name)
            and indicator_registry.get_indicator_config(name).adapter_function
        )
    ]

    print(f"Total indicators in registry: {len(all_names)}")
    print(f"Supported indicators (with adapter_function): {len(supported_names)}")
    print("\nSupported indicators:")
    for name in sorted(supported_names):
        print(f"  - {name}")

    # Check if SAR is in the list
    if 'SAR' in supported_names:
        print(f"\nSAR is supported")
        config = indicator_registry.get_indicator_config('SAR')
        print(f"SAR parameters: {list(config.parameters.keys())}")
    else:
        print(f"\nSAR is NOT in supported list")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()