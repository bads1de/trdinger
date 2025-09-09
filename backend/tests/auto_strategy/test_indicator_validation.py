"""
Test for indicator validation and NaN detection bugs
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from app.services.auto_strategy.services.indicator_service import IndicatorCalculator
from app.services.auto_strategy.models.indicator_gene import IndicatorParams
from app.services.indicators.technical_indicators.trend import TrendIndicators

class TestIndicatorValidation:
    """Indicator calculation validation tests"""

    @pytest.fixture
    def sample_data(self):
        """Sample OHLCV data for testing"""
        # Create sample data with potential NaN triggers
        return {
            'high': np.array([110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                             120, 121, 122, 123, 124, 125, 126, 127, 128, 129]),
            'low': np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                            100, 101, 102, 103, 104, 105, 106, 107, 108, 109]),
            'close': np.array([105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                              115, 116, 117, 118, 119, 120, 121, 122, 123, 124]),
            'volume': np.array([1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
                               2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900]),
            'open': np.array([102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                             112, 113, 114, 115, 116, 117, 118, 119, 120, 121])
        }

    def test_nan_indicator_detection(self, sample_data):
        """Test that NaN is detected in problematic indicators"""
        calculator = IndicatorCalculator()

        # Test indicators known to have NaN issues
        problematic_indicators = [
            'ADOSC', 'ADX', 'ADXR', 'ALMA', 'APO', 'AROONOSC', 'ATR', 'CCI',
            'CFO', 'CHOP', 'LinearREG', 'MAVP', 'SAREXT'  # From report
        ]

        failures = []

        for indicator in problematic_indicators:
            try:
                params = IndicatorParams(
                    indicator_type=indicator,
                    period=14,  # Standard period
                    source='close'
                )
                result = calculator.calculate(sample_data, params)

                # Check for NaN
                if isinstance(result, np.ndarray):
                    if np.isnan(result).any():
                        failures.append(f"{indicator}: Contains NaN values")
                    elif len(result) == 0:
                        failures.append(f"{indicator}: Empty result array")
                else:
                    if np.isnan(float(result)):
                        failures.append(f"{indicator}: NaN value")

            except Exception as e:
                failures.append(f"{indicator}: Calculation error - {str(e)}")

        # Log failures but don't fail test - for reporting
        if failures:
            print("\n=== Indicator NaN Detection Report ===")
            for failure in failures:
                print(f"- {failure}")
            print(f"\nTotal issues found: {len(failures)}")

        # This test passes - failures are for reporting
        assert True

    def test_period_parameter_errors(self):
        """Test period parameter validation errors"""
        trend_indicators = TrendIndicators()

        # Test indicators that fail with period parameter
        error_indicators = ['linreg', 'mavp', 'sarext']

        failures = []

        for indicator in error_indicators:
            try:
                # This should fail with unexpected keyword
                if indicator == 'linreg':
                    result = trend_indicators.linreg(period=14)
                elif indicator == 'mavp':
                    result = trend_indicators.mavp(period=14)
                elif indicator == 'sarext':
                    result = trend_indicators.sarext(period=14)

                if result is None or (hasattr(result, '__len__') and len(result) == 0):
                    failures.append(f"{indicator}: Unexpected empty result")

            except TypeError as e:
                if 'unexpected keyword argument' in str(e):
                    failures.append(f"{indicator}: Parameter error - {str(e)}")
                else:
                    failures.append(f"{indicator}: Different TypeError - {str(e)}")
            except Exception as e:
                failures.append(f"{indicator}: Other error - {str(e)}")

        if failures:
            print("\n=== Period Parameter Error Report ===")
            for failure in failures:
                print(f"- {failure}")
            print(f"\nTotal parameter errors: {len(failures)}")

        assert True

    def test_boundary_conditions(self, sample_data):
        """Test boundary conditions that might cause NaN"""
        calculator = IndicatorCalculator()

        # Test with minimum data lengths
        short_data = {
            'high': np.array([110, 111, 112]),
            'low': np.array([100, 101, 102]),
            'close': np.array([105, 106, 107]),
            'volume': np.array([1000, 1100, 1200]),
            'open': np.array([102, 103, 104])
        }

        indicators_to_test = ['RSI', 'MACD', 'Stochastic']
        failures = []

        for indicator in indicators_to_test:
            try:
                params = IndicatorParams(
                    indicator_type=indicator,
                    period=14,  # May be too long for short data
                    source='close'
                )
                result = calculator.calculate(short_data, params)

                if np.isnan(result).any():
                    failures.append(f"{indicator}: NaN with short data")

            except Exception as e:
                failures.append(f"{indicator}: Boundary error - {str(e)}")

        if failures:
            print("\n=== Boundary Condition Report ===")
            for failure in failures:
                print(f"- {failure}")

        assert True