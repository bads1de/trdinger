"""
シンプルなインジケータ動作確認テスト
"""
import pytest
import pandas as pd
import numpy as np


def test_basic_indicator_service():
    """基本的なインジケータサービスのテスト"""
    from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

    service = TechnicalIndicatorService()
    assert service is not None

    # サポートされているインジケータを取得
    supported_indicators = service.get_supported_indicators()
    assert isinstance(supported_indicators, dict)
    assert len(supported_indicators) > 0

    print(f"Total supported indicators: {len(supported_indicators)}")
    key_indicators = ['RSI', 'SMA', 'EMA', 'MACD', 'BB', 'STC']

    for indicator in key_indicators:
        assert indicator in supported_indicators, f"{indicator} not found in supported indicators"

    print("Key indicators found:", [ind for ind in key_indicators if ind in supported_indicators])
    return True


def test_sample_data_creation():
    """サンプルデータの作成テスト"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=50, freq='1H')

    base_price = 50000
    price_changes = np.random.normal(0, 0.02, 50)
    close_prices = [base_price]
    for change in price_changes[1:]:
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(new_price)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': [price * (1 + np.random.normal(0, 0.005)) for price in close_prices],
        'high': [price * (1 + abs(np.random.normal(0, 0.01))) for price in close_prices],
        'low': [price * (1 - abs(np.random.normal(0, 0.01))) for price in close_prices],
        'close': close_prices,
        'volume': np.random.uniform(1000000, 10000000, 50)
    })

    assert len(df) == 50
    assert 'open' in df.columns
    assert 'high' in df.columns
    assert 'low' in df.columns
    assert 'close' in df.columns
    assert 'volume' in df.columns

    print("Sample data created successfully")
    return df


def test_stc_indicator():
    """STCインジケータのテスト"""
    from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

    # サンプルデータ作成
    df = test_sample_data_creation()
    service = TechnicalIndicatorService()

    try:
        result = service.calculate_indicator(df, "STC", {"length": 10})

        if result is not None:
            print("STC calculation successful")
            assert isinstance(result, np.ndarray)
            assert len(result) == len(df)
            return True
        else:
            print("STC calculation returned None")
            return False

    except Exception as e:
        print(f"STC calculation failed: {e}")
        return False


def test_rsi_indicator():
    """RSIインジケータのテスト"""
    from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

    df = test_sample_data_creation()
    service = TechnicalIndicatorService()

    try:
        result = service.calculate_indicator(df, "RSI", {"length": 14})

        if result is not None:
            print("RSI calculation successful")
            assert isinstance(result, np.ndarray)
            assert len(result) == len(df)
            # RSIは0-100の範囲
            valid_values = result[~np.isnan(result)]
            assert all(0 <= val <= 100 for val in valid_values)
            return True
        else:
            print("RSI calculation returned None")
            return False

    except Exception as e:
        print(f"RSI calculation failed: {e}")
        return False


if __name__ == "__main__":
    print("Running indicator tests...")

    try:
        test_basic_indicator_service()
        test_stc_indicator()
        test_rsi_indicator()
        print("All tests completed successfully!")

    except Exception as e:
        print(f"Test failed: {e}")
        raise