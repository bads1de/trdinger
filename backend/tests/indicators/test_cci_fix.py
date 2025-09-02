import pandas as pd
import numpy as np
import sys
import os

from app.services.indicators.technical_indicators.momentum import MomentumIndicators

def test_cci():
    """CCI修正テスト"""
    print("=== CCI修正テスト開始 ===")

    # テストデータ生成（OHLCV）
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')

    # OHLCVデータの生成
    close = np.random.uniform(100, 200, n).cumsum() + 1000
    high = close * np.random.uniform(1.005, 1.03, n)
    low = close * np.random.uniform(0.97, 0.995, n)
    open_price = (high + low) / 2

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close
    }, index=dates)

    # CCI計算
    try:
        cci_result = MomentumIndicators.cci(
            high=pd.Series(df['high'].values, index=df.index),
            low=pd.Series(df['low'].values, index=df.index),
            close=pd.Series(df['close'].values, index=df.index),
            period=20
        )

        print(f"CCI計算結果: {cci_result is not None}")
        if cci_result is not None and not cci_result.isna().all():
            # CCI範囲確認 (-200 to +200)
            valid_values = cci_result.dropna()
            if len(valid_values) > 0:
                cci_min = valid_values.min()
                cci_max = valid_values.max()
                cci_mean = valid_values.mean()

                print(f"CCI範囲チェック:")
                print(".2f")
                print(".2f")
                print(".2f")
                print(f"CCI値が-200から+200の範囲内: {(cci_min >= -200) and (cci_max <= 200)}")

                # 有効なCCI値の数
                total_values = len(cci_result)
                nan_count = cci_result.isna().sum()
                valid_count = total_values - nan_count
                print(f"有効なCCI値数: {valid_count}/{total_values}")

                # 実際の値サンプル表示（NaN以外）
                print("\n有効なCCI値サンプル:")
                valid_sample = cci_result.dropna().head(10)
                if len(valid_sample) > 0:
                    print(valid_sample.to_frame('CCI'))
                    print("\nCCIの基本統計:")
                    print(valid_values.describe())
                else:
                    print("有効なサンプル値なし")
            else:
                print("有効なCCI値が見つからない")
        else:
            print("CCI計算失敗または全NaN")

    except Exception as e:
        print(f"エラー発生: {str(e)}")
        import traceback
        traceback.print_exc()

    print("=== CCI修正テスト完了 ===")

if __name__ == "__main__":
    test_cci()