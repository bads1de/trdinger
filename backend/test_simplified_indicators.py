#!/usr/bin/env python3
"""
簡素化されたテクニカル指標の動作確認テスト

pandas-taを直接使用した簡素化実装の動作を確認します。
"""

import numpy as np
import pandas as pd
import sys
import os

# パスを追加してモジュールをインポート
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))


def test_simplified_trend_indicators():
    """簡素化されたトレンド指標のテスト"""
    print("=== 簡素化されたトレンド指標のテスト ===")

    try:
        from app.services.indicators.technical_indicators.trend import TrendIndicators

        # テストデータ作成
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

        # SMAテスト
        sma_result = TrendIndicators.sma(prices, 20)
        print(f"SMA(20) - 最初の5値: {sma_result[:5]}")
        print(f"SMA(20) - 最後の5値: {sma_result[-5:]}")

        # EMAテスト
        ema_result = TrendIndicators.ema(prices, 20)
        print(f"EMA(20) - 最初の5値: {ema_result[:5]}")
        print(f"EMA(20) - 最後の5値: {ema_result[-5:]}")

        print("✅ トレンド指標テスト成功")

    except Exception as e:
        print(f"❌ トレンド指標テストエラー: {e}")
        import traceback

        traceback.print_exc()


def test_simplified_data_conversion():
    """簡素化されたデータ変換のテスト"""
    print("\n=== 簡素化されたデータ変換のテスト ===")

    try:
        from app.utils.data_conversion import ensure_series, ensure_array, ensure_list

        # テストデータ
        test_data = [1, 2, 3, 4, 5]

        # ensure_seriesテスト
        series_result = ensure_series(test_data)
        print(f"ensure_series結果: {type(series_result)}, 値: {series_result.tolist()}")

        # ensure_arrayテスト
        array_result = ensure_array(test_data)
        print(f"ensure_array結果: {type(array_result)}, 値: {array_result}")

        # ensure_listテスト
        list_result = ensure_list(np.array(test_data))
        print(f"ensure_list結果: {type(list_result)}, 値: {list_result}")

        print("✅ データ変換テスト成功")

    except Exception as e:
        print(f"❌ データ変換テストエラー: {e}")
        import traceback

        traceback.print_exc()


def test_ohlcv_standardization():
    """OHLCV列名標準化のテスト"""
    print("\n=== OHLCV列名標準化のテスト ===")

    try:
        from app.utils.data_conversion import standardize_ohlcv_columns

        # テストデータフレーム（小文字列名）
        test_df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [99, 100, 101],
                "close": [104, 105, 106],
                "volume": [1000, 1100, 1200],
            }
        )

        print(f"変換前の列名: {test_df.columns.tolist()}")

        # 標準化実行
        standardized_df = standardize_ohlcv_columns(test_df)
        print(f"変換後の列名: {standardized_df.columns.tolist()}")

        # 必要な列が存在することを確認
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [
            col for col in required_cols if col not in standardized_df.columns
        ]

        if not missing_cols:
            print("✅ OHLCV標準化テスト成功")
        else:
            print(f"❌ 不足している列: {missing_cols}")

    except Exception as e:
        print(f"❌ OHLCV標準化テストエラー: {e}")
        import traceback

        traceback.print_exc()


def test_indicator_service():
    """指標サービスのテスト"""
    print("\n=== 指標サービスのテスト ===")

    try:
        # テスト用のOHLCVデータ作成
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        df = pd.DataFrame(
            {
                "Open": 100 + np.cumsum(np.random.randn(100) * 0.3),
                "High": 100 + np.cumsum(np.random.randn(100) * 0.3) + 2,
                "Low": 100 + np.cumsum(np.random.randn(100) * 0.3) - 2,
                "Close": 100 + np.cumsum(np.random.randn(100) * 0.3),
                "Volume": np.random.randint(1000, 5000, 100),
            },
            index=dates,
        )

        print(f"テストデータ形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")

        # pandas-taを直接使用したテスト
        import pandas_ta as ta

        # RSI計算
        rsi = ta.rsi(df["Close"], length=14)
        print(f"RSI計算成功 - 最後の5値: {rsi.tail().values}")

        # SMA計算
        sma = ta.sma(df["Close"], length=20)
        print(f"SMA計算成功 - 最後の5値: {sma.tail().values}")

        print("✅ 指標サービステスト成功")

    except Exception as e:
        print(f"❌ 指標サービステストエラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 簡素化されたバックエンド機能のテスト開始")

    test_simplified_data_conversion()
    test_ohlcv_standardization()
    test_simplified_trend_indicators()
    test_indicator_service()

    print("\n🎉 テスト完了")
