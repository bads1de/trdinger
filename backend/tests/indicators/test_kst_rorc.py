#!/usr/bin/env python
"""KST rorcパラメータの簡単テスト"""
import sys
sys.path.append(r'c:\Users\buti3\trading\backend')

try:
    import pandas as pd
    import numpy as np
    from app.services.indicators.technical_indicators.momentum import MomentumIndicators

    print("Testing KST with rorc parameters...")

    # テストデータ生成
    np.random.seed(42)
    trend = np.linspace(100, 200, 100) + np.random.randn(100) * 5
    data = pd.Series(trend, name='close')

    # rorcパラメータでテスト
    kst_value, signal_value = MomentumIndicators.kst(
        data, rorc1=10, rorc2=15, rorc3=20, rorc4=30,
        sma1=10, sma2=10, sma3=10, sma4=15, signal=9
    )

    print("Success! KST with rorc parameters works")
    print(f"KST type: {type(kst_value)}")
    print(f"Signal type: {type(signal_value)}")
    print(f"Valid KST values: {~kst_value.isna().all()}")
    print(f"Valid Signal values: {~signal_value.isna().all()}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()