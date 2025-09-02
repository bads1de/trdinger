import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))  # Add backend directory to path
import pytest

import pandas as pd
import numpy as np
from backend.app.services.indicators.technical_indicators.volatility import VolatilityIndicators

# Sample data
high = pd.Series([10, 11, 12, 13, 14, 15])
low = pd.Series([8, 9, 10, 11, 12, 13])
close = pd.Series([9, 10, 11, 12, 13, 14])
series = pd.Series([1, 2, 3, 4, 5])


