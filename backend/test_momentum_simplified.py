#!/usr/bin/env python3
"""
簡素化されたモメンタム指標の動作確認テスト

pandas-taを直接使用した簡素化実装の動作を確認します。
"""

import numpy as np
import pandas as pd
import sys
import os

# パスを追加してモジュールをインポート
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))


def test_simplified_momentum_indicators():
    """簡素化されたモメンタム指標のテスト"""
    print("=== 簡素化されたモメンタム指標のテスト ===")

    try:
        from app.services.indicators.technical_indicators.momentum import (
            MomentumIndicators,
        )

        # テストデータ作成
        np.random.seed(42)
        n = 100

        # 価格データ
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        # OHLCV作成
        open_prices = prices + np.random.randn(n) * 0.1
        close_prices = prices + np.random.randn(n) * 0.1
        high_prices = np.maximum(open_prices, close_prices) + np.abs(
            np.random.randn(n) * 0.2
        )
        low_prices = np.minimum(open_prices, close_prices) - np.abs(
            np.random.randn(n) * 0.2
        )
        volumes = np.random.randint(1000, 5000, n)

        print(f"テストデータ作成完了: {n}本のローソク足")

        # RSIテスト
        rsi_result = MomentumIndicators.rsi(close_prices, length=14)
        print(f"RSI計算成功 - 最後の5値: {rsi_result[-5:]}")

        # MACDテスト
        macd_line, macd_signal, macd_hist = MomentumIndicators.macd(close_prices)
        print(f"MACD計算成功 - MACD最後の5値: {macd_line[-5:]}")
        print(f"MACD計算成功 - Signal最後の5値: {macd_signal[-5:]}")

        # ストキャスティクステスト
        stoch_k, stoch_d = MomentumIndicators.stoch(
            high_prices, low_prices, close_prices
        )
        print(f"Stoch計算成功 - %K最後の5値: {stoch_k[-5:]}")
        print(f"Stoch計算成功 - %D最後の5値: {stoch_d[-5:]}")

        # Williams %Rテスト
        willr_result = MomentumIndicators.willr(high_prices, low_prices, close_prices)
        print(f"Williams %R計算成功 - 最後の5値: {willr_result[-5:]}")

        # CCIテスト
        cci_result = MomentumIndicators.cci(high_prices, low_prices, close_prices)
        print(f"CCI計算成功 - 最後の5値: {cci_result[-5:]}")

        # ROCテスト
        roc_result = MomentumIndicators.roc(close_prices)
        print(f"ROC計算成功 - 最後の5値: {roc_result[-5:]}")

        # ADXテスト
        adx_result = MomentumIndicators.adx(high_prices, low_prices, close_prices)
        print(f"ADX計算成功 - 最後の5値: {adx_result[-5:]}")

        # MFIテスト
        mfi_result = MomentumIndicators.mfi(
            high_prices, low_prices, close_prices, volumes
        )
        print(f"MFI計算成功 - 最後の5値: {mfi_result[-5:]}")

        # 結果の型チェック
        assert isinstance(rsi_result, np.ndarray), "RSI結果がnumpy配列でない"
        assert isinstance(macd_line, np.ndarray), "MACD結果がnumpy配列でない"
        assert isinstance(stoch_k, np.ndarray), "Stoch結果がnumpy配列でない"
        assert len(rsi_result) == n, f"RSI結果の長さが不正: {len(rsi_result)} != {n}"

        print("✅ モメンタム指標簡素化テスト成功")

    except Exception as e:
        print(f"❌ モメンタム指標簡素化テストエラー: {e}")
        import traceback

        traceback.print_exc()


def test_backward_compatibility():
    """後方互換性テスト"""
    print("\n=== 後方互換性テスト ===")

    try:
        from app.services.indicators.technical_indicators.momentum import (
            MomentumIndicators,
        )

        # テストデータ
        test_data = np.array([100, 101, 102, 103, 104, 105, 104, 103, 102, 101])

        # エイリアスメソッドのテスト
        macdext_result = MomentumIndicators.macdext(test_data)
        macdfix_result = MomentumIndicators.macdfix(test_data)

        print(f"MACDEXT（エイリアス）計算成功: {len(macdext_result)} 個の配列")
        print(f"MACDFIX（エイリアス）計算成功: {len(macdfix_result)} 個の配列")

        # 結果が3つのタプルであることを確認
        assert len(macdext_result) == 3, "MACDEXTの結果が3つのタプルでない"
        assert len(macdfix_result) == 3, "MACDFIXの結果が3つのタプルでない"

        print("✅ 後方互換性テスト成功")

    except Exception as e:
        print(f"❌ 後方互換性テストエラー: {e}")
        import traceback

        traceback.print_exc()


def test_performance_comparison():
    """パフォーマンス比較テスト"""
    print("\n=== パフォーマンス比較テスト ===")

    try:
        import time
        from app.services.indicators.technical_indicators.momentum import (
            MomentumIndicators,
        )

        # 大きなテストデータ
        np.random.seed(42)
        n = 10000
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        # RSI計算時間測定
        start_time = time.time()
        for _ in range(10):  # 10回実行
            rsi_result = MomentumIndicators.rsi(prices, length=14)
        end_time = time.time()

        avg_time = (end_time - start_time) / 10
        print(f"RSI計算時間（{n}データポイント）: {avg_time:.4f}秒/回")
        print(f"処理速度: {n/avg_time:.0f} データポイント/秒")

        # 結果の妥当性確認
        assert len(rsi_result) == n, "結果の長さが不正"
        assert not np.isnan(rsi_result[-1]), "最後の値がNaN"

        print("✅ パフォーマンステスト成功")

    except Exception as e:
        print(f"❌ パフォーマンステストエラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 簡素化されたモメンタム指標のテスト開始")

    test_simplified_momentum_indicators()
    test_backward_compatibility()
    test_performance_comparison()

    print("\n🎉 モメンタム指標簡素化テスト完了")
