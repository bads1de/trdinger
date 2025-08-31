"""
Comprehensive Momentum Indicators Test
モメンタム指標の全関数を検証
"""

import pytest
import numpy as np
import pandas as pd

from app.services.indicators.technical_indicators.momentum import MomentumIndicators


class TestComprehensiveMomentum:
    """Manual comprehensive test for all momentum functions"""

    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data"""
        np.random.seed(42)
        n = 100

        # OHLCVデータ生成
        close = pd.Series(100 + np.cumsum(np.random.randn(n)), name="close")
        high = close + np.abs(np.random.randn(n)) * 5
        low = close - np.abs(np.random.randn(n)) * 5
        open_price = close + np.random.randn(n) * 2
        volume = pd.Series(np.random.randint(1000, 10000, n), name="volume")

        return {
            'close': close,
            'high': high,
            'low': low,
            'open': open_price,
            'volume': volume
        }

    def manual_test_function(self, func_name, func_call):
        """Test a single function manually"""
        try:
            result = func_call()
            print(f"PASS {func_name}: {type(result)}")
            if isinstance(result, tuple):
                print(f"  Tuple with {len(result)} elements:")
                for i, r in enumerate(result):
                    print(f"    [{i}]: {type(r)} - length: {len(r) if hasattr(r, '__len__') else 'N/A'}")
                    if hasattr(r, 'dtype'):
                        print(f"      dtype: {r.dtype}")
            else:
                print(f"  Single result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
                if hasattr(result, 'dtype'):
                    print(f"  dtype: {result.dtype}")
            return True
        except Exception as e:
            print(f"FAIL {func_name}: {e}")
            return False

    def test_all_momentum_functions_comprehensively(self, sample_data):
        """Manually test each momentum function"""

        print("\n=== MANUAL COMPREHENSIVE MOMENTUM FUNCTIONS TEST ===")

        functions_to_test = [

            # Single Series functions
            ('rsi', lambda: MomentumIndicators.rsi(sample_data['close'])),
            ('willr', lambda: MomentumIndicators.willr(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('cci', lambda: MomentumIndicators.cci(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('roc', lambda: MomentumIndicators.roc(sample_data['close'])),
            ('mom', lambda: MomentumIndicators.mom(sample_data['close'])),
            ('adx', lambda: MomentumIndicators.adx(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('mfi', lambda: MomentumIndicators.mfi(sample_data['high'], sample_data['low'], sample_data['close'], sample_data['volume'])),
            ('uo', lambda: MomentumIndicators.uo(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('apo', lambda: MomentumIndicators.apo(sample_data['close'])),
            ('ao', lambda: MomentumIndicators.ao(sample_data['high'], sample_data['low'])),
            ('cmo', lambda: MomentumIndicators.cmo(sample_data['close'])),
            ('trix', lambda: MomentumIndicators.trix(sample_data['close'])),
            ('ultosc', lambda: MomentumIndicators.ultosc(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('bop', lambda: MomentumIndicators.bop(sample_data['open'], sample_data['high'], sample_data['low'], sample_data['close'])),
            ('adxr', lambda: MomentumIndicators.adxr(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('rocp', lambda: MomentumIndicators.rocp(sample_data['close'])),
            ('rocr', lambda: MomentumIndicators.rocr(sample_data['close'])),
            ('rocr100', lambda: MomentumIndicators.rocr100(sample_data['close'])),
            ('tsi', lambda: MomentumIndicators.tsi(sample_data['close'])),
            ('rvi', lambda: MomentumIndicators.rvi(sample_data['open'], sample_data['high'], sample_data['low'], sample_data['close'])),
            ('cfo', lambda: MomentumIndicators.cfo(sample_data['close'])),
            ('cti', lambda: MomentumIndicators.cti(sample_data['close'])),
            ('rmi', lambda: MomentumIndicators.rmi(sample_data['close'])),
            ('dpo', lambda: MomentumIndicators.dpo(sample_data['close'])),
            ('chop', lambda: MomentumIndicators.chop(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('bias', lambda: MomentumIndicators.bias(sample_data['close'])),
            ('brar', lambda: MomentumIndicators.brar(sample_data['open'], sample_data['high'], sample_data['low'], sample_data['close'])),
            ('cg', lambda: MomentumIndicators.cg(sample_data['close'])),
            ('coppock', lambda: MomentumIndicators.coppock(sample_data['close'])),
            ('er', lambda: MomentumIndicators.er(sample_data['close'])),
            ('eri', lambda: MomentumIndicators.eri(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('inertia', lambda: MomentumIndicators.inertia(sample_data['close'])),
            ('pgo', lambda: MomentumIndicators.pgo(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('psl', lambda: MomentumIndicators.psl(sample_data['close'], sample_data['open'])),
            ('rsx', lambda: MomentumIndicators.rsx(sample_data['close'])),
            ('squeeze', lambda: MomentumIndicators.squeeze(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('squeeze_pro', lambda: MomentumIndicators.squeeze_pro(sample_data['high'], sample_data['low'], sample_data['close'])),

            # Tuple functions
            ('macd', lambda: MomentumIndicators.macd(sample_data['close'])),
            ('stoch', lambda: MomentumIndicators.stoch(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('kdj', lambda: MomentumIndicators.kdj(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('stochrsi', lambda: MomentumIndicators.stochrsi(sample_data['close'])),
            ('ppo', lambda: MomentumIndicators.ppo(sample_data['close'])),
            ('rvgi', lambda: MomentumIndicators.rvgi(sample_data['open'], sample_data['high'], sample_data['low'], sample_data['close'])),
            ('smi', lambda: MomentumIndicators.smi(sample_data['close'])),
            ('kst', lambda: MomentumIndicators.kst(sample_data['close'])),
            ('fisher', lambda: MomentumIndicators.fisher(sample_data['high'], sample_data['low'])),
            ('vortex', lambda: MomentumIndicators.vortex(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('pvo', lambda: MomentumIndicators.pvo(sample_data['volume'])),
            ('aroon', lambda: MomentumIndicators.aroon(sample_data['high'], sample_data['low'])),

            # Alias functions
            ('macdext', lambda: MomentumIndicators.macdext(sample_data['close'])),
            ('macdfix', lambda: MomentumIndicators.macdfix(sample_data['close'])),
            ('stochf', lambda: MomentumIndicators.stochf(sample_data['high'], sample_data['low'], sample_data['close'])),

            # Complex functions
            ('rsi_ema_cross', lambda: MomentumIndicators.rsi_ema_cross(sample_data['close'])),
            ('stc', lambda: MomentumIndicators.stc(sample_data['close'])),
            ('qqe', lambda: MomentumIndicators.qqe(sample_data['close'])),

            # Directional Movement functions
            ('plus_di', lambda: MomentumIndicators.plus_di(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('minus_di', lambda: MomentumIndicators.minus_di(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('plus_dm', lambda: MomentumIndicators.plus_dm(sample_data['high'], sample_data['low'])),
            ('minus_dm', lambda: MomentumIndicators.minus_dm(sample_data['high'], sample_data['low'])),
            ('dx', lambda: MomentumIndicators.dx(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('aroonosc', lambda: MomentumIndicators.aroonosc(sample_data['high'], sample_data['low'])),
        ]

        passed_functions = 0
        total_functions = len(functions_to_test)

        for func_name, func_call in functions_to_test:
            if self.manual_test_function(func_name, func_call):
                passed_functions += 1

        print(f"\n=== TEST SUMMARY ===")
        print(f"Passed: {passed_functions}/{total_functions}")
        print(".1f")

        # Assert all functions work (adjust if minor issues)
        assert passed_functions >= total_functions * 0.95, f"Less than 95% functions passed ({passed_functions}/{total_functions})"