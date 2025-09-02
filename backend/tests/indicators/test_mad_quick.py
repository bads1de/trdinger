#!/usr/bin/env python3
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
import pandas as pd

# Create test data
dates = pd.date_range(start='2024-01-01', periods=20, freq='1H')
close_prices = [100 + i*0.5 for i in range(20)]
df = pd.DataFrame({'timestamp': dates, 'close': close_prices})

# Test MAD
service = TechnicalIndicatorService()
result = service.calculate_indicator(df, 'MAD', {})
print('MAD result:', result)
if result is not None:
    print('MAD SUCCESS: result shape', result.shape)
    print('MAD values (first 5):', result[:5] if len(result) > 5 else result)
else:
    print('MAD FAILED: None result')