"""
総合的な指標修正確認テスト
"""
import pandas as pd
import numpy as np

def test_comprehensive_indicators():
    """包括的な指標テスト"""
    from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

    # 生成テストデータ
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
        'open': [price * (1 + np.random.normal(0, 0.005)) for price in close_prices[:50]],
        'high': [price * (1 + abs(np.random.normal(0, 0.01))) for price in close_prices[:50]],
        'low': [price * (1 - abs(np.random.normal(0, 0.01))) for price in close_prices[:50]],
        'close': close_prices[:50],
        'volume': np.random.uniform(1000000, 10000000, 50)
    })

    service = TechnicalIndicatorService()

    # テスト対象の指標リスト
    test_indicators = [
        # 期間不要な指標
        ('HLC3', {}),
        ('HL2', {}),
        ('OHLC4', {}),
        ('VP', {'width': 10}),
        ('AOBV', {'fast': 5, 'slow': 10}),
        ('WCP', {}),

        # volume_indicators グループの指標
        ('NVI', {}),
        ('PVI', {}),
        ('PVT', {}),
        ('AD', {}),
        ('PVR', {}),

        # パターン認識指標
        ('CDL_DOJI', {}),
        ('CDL_HAMMER', {}),
        ('CDL_ENGULFING', {}),
        ('HAMMER', {}),
        ('ENGULFING_PATTERN', {}),
        ('MORNING_STAR', {}),
        ('EVENING_STAR', {}),

        # NO_LENGTH_INDICATORS グループの他の指標
        ('SAR', {}),
        ('OBV', {}),
        ('VWAP', {}),
        ('BOP', {}),
        ('ICHIMOKU', {}),
        ('BB', {}),
        ('STC', {}),
    ]

    success_count = 0
    failure_count = 0
    results = {}

    for indicator_name, params in test_indicators:
        print(f'Testing {indicator_name}...')
        try:
            result = service.calculate_indicator(df.copy(), indicator_name, params)
            if result is not None:
                print(f'  [SUCCESS] {type(result).__name__}')
                if isinstance(result, tuple):
                    print(f'    Output count: {len(result)}')
                    for i, arr in enumerate(result):
                        if hasattr(arr, 'shape'):
                            print(f'    [{i}]: shape={arr.shape}')
                success_count += 1
                results[indicator_name] = 'SUCCESS'
            else:
                print(f'  [FAIL] None result')
                failure_count += 1
                results[indicator_name] = 'FAILED - None result'

        except Exception as e:
            print(f'  [ERROR] {e}')
            failure_count += 1
            results[indicator_name] = f'ERROR - {e}'

    print("\n" + "="*60)
    print("テスト結果サマリー")
    print("="*60)
    print(f"総テスト数: {len(test_indicators)}")
    print(f"✓ 成功: {success_count}")
    print(f"✗ 失敗: {failure_count}")
    print(".1f")

    if failure_count > 0:
        print("\n失敗した指標詳細:")
        for name, status in results.items():
            if status.startswith(('FAILED', 'ERROR')):
                print(f"  • {name}: {status}")

    return success_count, failure_count, results

if __name__ == "__main__":
    success_count, failure_count, results = test_comprehensive_indicators()
    print(f"\n総合テスト完了: {'成功' if failure_count == 0 else '一部失敗'}")