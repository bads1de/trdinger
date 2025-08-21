#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/c:/Users/buti3/trading/backend')

import pandas as pd
import numpy as np
from app.services.indicators import TechnicalIndicatorService
from app.services.indicators.config import indicator_registry
from tests.indicators.test_all_indicator_initialization import make_df, _default_params

# Create test data
df = make_df()

# Get SAR config
config = indicator_registry.get_indicator_config('SAR')
print(f"SAR config parameters: {list(config.parameters.keys())}")

# Get default parameters
params = _default_params(config)
print(f"Default params: {params}")

# Create service and test
svc = TechnicalIndicatorService()

try:
    result = svc.calculate_indicator(df, 'SAR', params)
    print(f"SAR calculation successful: {type(result)}")
except Exception as e:
    print(f"SAR calculation failed: {e}")
    import traceback
    traceback.print_exc()