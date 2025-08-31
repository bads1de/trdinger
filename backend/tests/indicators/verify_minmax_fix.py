#!/usr/bin/env python3
"""
Verify MINMAX issue fix
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from app.services.indicators.config.indicator_definitions import indicator_registry, initialize_all_indicators

def test_invalid_metrics():
    """Verify invalid indicators are not registered"""
    print("=== Invalid Indicators Check ===")

    # Invalid indicators list
    invalid_indicators = ["MINMAX", "MINMAXINDEX", "MAXINDEX", "MININDEX", "MAMA"]

    print(f"Invalid indicators: {invalid_indicators}")
    registered_indicators = indicator_registry.list_indicators()
    print(f"Currently registered indicators: {len(registered_indicators)}")

    # Check for invalid indicators
    invalid_found = []
    for indicator_name in invalid_indicators:
        if indicator_name in registered_indicators:
            invalid_found.append(indicator_name)
            print(f"[WARNING] {indicator_name} still registered")
        else:
            print(f"[OK] {indicator_name} excluded")

    if invalid_found:
        print(f"\n[FAIL] Unresolved invalid indicators: {invalid_found}")
        return False
    else:
        print(f"\n[SUCCESS] All invalid indicators properly excluded!")
        return True

def test_valid_metrics():
    """Verify valid indicators are registered"""
    print("\n=== Valid Indicators Check ===")

    # Valid indicator examples
    valid_indicators = [
        "SMA", "EMA", "RSI", "MACD", "ATR",
        "BBANDS", "STOCH", "CCI", "ADX"
    ]

    registered_indicators = indicator_registry.list_indicators()
    valid_found = []
    for indicator_name in valid_indicators:
        if indicator_name in registered_indicators:
            valid_found.append(indicator_name)
            print(f"[OK] {indicator_name} registered")
        else:
            print(f"[WARNING] {indicator_name} not registered")

    print(f"\nValid indicators check: {len(valid_found)}/{len(valid_indicators)} OK")
    return len(valid_found) == len(valid_indicators)

if __name__ == "__main__":
    try:
        # Initialize all indicators
        initialize_all_indicators()

        # Run tests
        invalid_test_ok = test_invalid_metrics()
        valid_test_ok = test_valid_metrics()

        if invalid_test_ok and valid_test_ok:
            print("\n[SUCCESS] MINMAX problem resolved!")
            sys.exit(0)
        else:
            print("\n[FAILURE] Tests failed.")
            sys.exit(1)

    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)