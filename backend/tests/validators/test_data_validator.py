import pandas as pd
import pytest
from backend.app.utils.data_processing.validators.data_validator import (
    validate_ohlcv_data,
    validate_extended_data,
    validate_data_integrity
)

def test_validate_ohlcv_data():
    """OHLCVデータの検証テスト"""
    df = pd.DataFrame({
        'open': [100.0, 101.0],
        'high': [105.0, 106.0],
        'low': [95.0, 96.0],
        'close': [102.0, 103.0],
        'volume': [1000, 2000]
    })
    result = validate_ohlcv_data(df)
    assert result is True

def test_validate_ohlcv_data_missing_columns():
    """必須カラムが欠落した場合のテスト"""
    df = pd.DataFrame({'open': [1], 'high': [1]})  # low, close, volume が欠落
    with pytest.raises(ValueError):
        validate_ohlcv_data(df)

def test_validate_extended_data():
    """拡張データの検証テスト"""
    df = pd.DataFrame({
        'fear_greed': [50, 60],
        'funding_rate': [0.01, 0.02]
    })
    result = validate_extended_data(df)
    assert result is True

def test_validate_data_integrity():
    """データ整合性の検証テスト"""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=3),
        'close': [100, 101, 102]
    })
    result = validate_data_integrity(df)
    assert result is True

def test_validate_data_integrity_unsorted_timestamps():
    """タイムスタンプがソートされていない場合のテスト"""
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-03', '2023-01-01', '2023-01-02']),
        'close': [100, 101, 102]
    })
    with pytest.raises(ValueError):
        validate_data_integrity(df)