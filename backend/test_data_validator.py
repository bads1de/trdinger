#!/usr/bin/env python3
"""
Data validation testing script for improved data_validator.py
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timezone

# Add backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

try:
    from app.utils.data_processing.validators.data_validator import (
        validate_ohlcv_data,
        validate_extended_data,
        validate_data_integrity
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running this from the backend directory")
    sys.exit(1)


def test_validate_ohlcv_data():
    """Test OHLCV data validation"""
    print("Testing validate_ohlcv_data...")
    
    # Test valid data
    valid_data = pd.DataFrame({
        'open': [100, 105, 103],
        'high': [110, 115, 108],
        'low': [95, 100, 98],
        'close': [105, 108, 102],
        'volume': [1000, 1200, 1100]
    })
    try:
        result = validate_ohlcv_data(valid_data)
        assert result == True
        print("[PASS] Valid OHLCV data test passed")
    except Exception as e:
        print(f"[FAIL] Valid OHLCV data test failed: {e}")
        return False
    
    # Test invalid OHLC data
    invalid_ohlc = pd.DataFrame({
        'open': [100, 105, 103],
        'high': [110, 115, 108],  
        'low': [120, 100, 98],   # low > open
        'close': [105, 108, 102],
        'volume': [1000, 1200, 1100]
    })
    try:
        validate_ohlcv_data(invalid_ohlc)
        print("[FAIL] Invalid OHLC test failed - should have raised ValueError")
        return False
    except ValueError:
        print("[PASS] Invalid OHLC test passed")
    
    # Test negative volume
    invalid_volume = pd.DataFrame({
        'open': [100, 105, 103],
        'high': [110, 115, 108],
        'low': [95, 100, 98],
        'close': [105, 108, 102],
        'volume': [1000, -1200, 1100]  # negative volume
    })
    try:
        validate_ohlcv_data(invalid_volume)
        print("[FAIL] Negative volume test failed - should have raised ValueError")
        return False
    except ValueError:
        print("[PASS] Negative volume test passed")
    
    # Test empty DataFrame
    empty_df = pd.DataFrame()
    try:
        result = validate_ohlcv_data(empty_df)
        assert result == True
        print("[PASS] Empty DataFrame test passed")
    except Exception as e:
        print(f"[FAIL] Empty DataFrame test failed: {e}")
        return False
    
    return True


def test_validate_extended_data():
    """Test extended data validation"""
    print("Testing validate_extended_data...")
    
    # Test valid funding rate
    valid_extended = pd.DataFrame({
        'funding_rate': [-0.01, 0.005, 0.0001, -0.5],
        'open_interest': [1000, 1200, 1100, 1300]
    })
    try:
        result = validate_extended_data(valid_extended)
        assert result == True
        print("[PASS] Valid extended data test passed")
    except Exception as e:
        print(f"[FAIL] Valid extended data test failed: {e}")
        return False
    
    # Test invalid funding rate
    invalid_funding = pd.DataFrame({
        'funding_rate': [-0.01, 5.0, 0.0001],  # 5.0 > 1
    })
    try:
        validate_extended_data(invalid_funding)
        print("[FAIL] Invalid funding rate test failed - should have raised ValueError")
        return False
    except ValueError:
        print("[PASS] Invalid funding rate test passed")
    
    # Test no extended columns
    no_extended = pd.DataFrame({
        'price': [100, 105, 103]
    })
    try:
        result = validate_extended_data(no_extended)
        assert result == True
        print("[PASS] No extended columns test passed")
    except Exception as e:
        print(f"[FAIL] No extended columns test failed: {e}")
        return False
    
    return True


def test_validate_data_integrity():
    """Test data integrity validation"""
    print("Testing validate_data_integrity...")
    
    # Test valid timestamp data
    valid_integrity = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=3, freq='1H'),
        'price': [100, 105, 103]
    })
    try:
        result = validate_data_integrity(valid_integrity)
        assert result == True
        print("[PASS] Valid timestamp data test passed")
    except Exception as e:
        print(f"[FAIL] Valid timestamp data test failed: {e}")
        return False
    
    # Test unsorted timestamps
    unsorted_ts = pd.DataFrame({
        'timestamp': [datetime.now(timezone.utc), 
                     datetime(2023, 1, 1, tzinfo=timezone.utc),
                     datetime(2023, 1, 2, tzinfo=timezone.utc)],
        'price': [100, 105, 103]
    })
    try:
        validate_data_integrity(unsorted_ts)
        print("[FAIL] Unsorted timestamps test failed - should have raised ValueError")
        return False
    except ValueError:
        print("[PASS] Unsorted timestamps test passed")
    
    # Test no timestamp column
    no_timestamp = pd.DataFrame({
        'price': [100, 105, 103]
    })
    try:
        result = validate_data_integrity(no_timestamp)
        assert result == True
        print("[PASS] No timestamp column test passed")
    except Exception as e:
        print(f"[FAIL] No timestamp column test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests"""
    print("Data Validator Tests")
    print("=" * 50)
    
    all_passed = True
    
    # Run all tests
    tests = [
        test_validate_ohlcv_data,
        test_validate_extended_data,
        test_validate_data_integrity
    ]
    
    for test_func in tests:
        if not test_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("[SUCCESS] All tests passed!")
        return 0
    else:
        print("[ERROR] Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())