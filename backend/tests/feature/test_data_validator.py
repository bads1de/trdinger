#!/usr/bin/env python3
"""
Data validation testing script for improved data_validator.py
"""

import os
import sys
from datetime import datetime, timezone

import pandas as pd
import pytest

# Add backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

try:
    from app.utils.data_processing.validators.data_validator import (
        validate_data_integrity,
        validate_extended_data,
        validate_ohlcv_data,
    )
except ImportError:
    pass


def test_validate_ohlcv_data():
    """Test OHLCV data validation"""
    print("Testing validate_ohlcv_data...")

    # Test valid data
    valid_data = pd.DataFrame(
        {
            "open": [100, 105, 103],
            "high": [110, 115, 108],
            "low": [95, 100, 98],
            "close": [105, 108, 102],
            "volume": [1000, 1200, 1100],
        }
    )
    
    result = validate_ohlcv_data(valid_data)
    assert result
    print("[PASS] Valid OHLCV data test passed")

    # Test invalid OHLC data
    invalid_ohlc = pd.DataFrame(
        {
            "open": [100, 105, 103],
            "high": [110, 115, 108],
            "low": [120, 100, 98],  # low > open
            "close": [105, 108, 102],
            "volume": [1000, 1200, 1100],
        }
    )
    with pytest.raises(ValueError):
        validate_ohlcv_data(invalid_ohlc)
    print("[PASS] Invalid OHLC test passed")

    # Test negative volume
    invalid_volume = pd.DataFrame(
        {
            "open": [100, 105, 103],
            "high": [110, 115, 108],
            "low": [95, 100, 98],
            "close": [105, 108, 102],
            "volume": [1000, -1200, 1100],  # negative volume
        }
    )
    with pytest.raises(ValueError):
        validate_ohlcv_data(invalid_volume)
    print("[PASS] Negative volume test passed")

    # Test empty DataFrame
    empty_df = pd.DataFrame()
    result = validate_ohlcv_data(empty_df)
    assert result
    print("[PASS] Empty DataFrame test passed")


def test_validate_extended_data():
    """Test extended data validation"""
    print("Testing validate_extended_data...")

    # Test valid funding rate
    valid_extended = pd.DataFrame(
        {
            "funding_rate": [-0.01, 0.005, 0.0001, -0.5],
            "open_interest": [1000, 1200, 1100, 1300],
        }
    )
    
    result = validate_extended_data(valid_extended)
    assert result
    print("[PASS] Valid extended data test passed")

    # Test invalid funding rate
    invalid_funding = pd.DataFrame(
        {
            "funding_rate": [-0.01, 5.0, 0.0001],  # 5.0 > 1
        }
    )
    with pytest.raises(ValueError):
        validate_extended_data(invalid_funding)
    print("[PASS] Invalid funding rate test passed")

    # Test no extended columns
    no_extended = pd.DataFrame({"price": [100, 105, 103]})
    
    result = validate_extended_data(no_extended)
    assert result
    print("[PASS] No extended columns test passed")


def test_validate_data_integrity():
    """Test data integrity validation"""
    print("Testing validate_data_integrity...")

    # Test valid timestamp data
    valid_integrity = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=3, freq="1h"),
            "price": [100, 105, 103],
        }
    )
    
    result = validate_data_integrity(valid_integrity)
    assert result
    print("[PASS] Valid timestamp data test passed")

    # Test unsorted timestamps
    unsorted_ts = pd.DataFrame(
        {
            "timestamp": [
                datetime.now(timezone.utc),
                datetime(2023, 1, 1, tzinfo=timezone.utc),
                datetime(2023, 1, 2, tzinfo=timezone.utc),
            ],
            "price": [100, 105, 103],
        }
    )
    with pytest.raises(ValueError):
        validate_data_integrity(unsorted_ts)
    print("[PASS] Unsorted timestamps test passed")

    # Test no timestamp column
    no_timestamp = pd.DataFrame({"price": [100, 105, 103]})
    
    result = validate_data_integrity(no_timestamp)
    assert result
    print("[PASS] No timestamp column test passed")


if __name__ == "__main__":
    # Manual execution
    try:
        test_validate_ohlcv_data()
        test_validate_extended_data()
        test_validate_data_integrity()
        print("\n" + "=" * 50)
        print("[SUCCESS] All tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Tests failed: {e}")
        sys.exit(1)