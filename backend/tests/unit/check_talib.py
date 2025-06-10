#!/usr/bin/env python3
"""
TA-Libのインストール状況確認スクリプト
"""


def check_talib_installation():
    """TA-Libのインストール状況を確認"""
    try:
        import talib

        print("✅ TA-Lib インポート成功")
        print(f"📦 TA-Lib バージョン: {talib.__version__}")

        # 利用可能な関数数を確認
        functions = talib.get_functions()
        print(f"🔧 利用可能な関数数: {len(functions)}")

        # サンプル関数を表示
        print(f"📋 サンプル関数: {functions[:10]}")

        # 基本的な指標をテスト
        import numpy as np

        test_data = np.random.random(100)

        # SMAテスト
        sma_result = talib.SMA(test_data, timeperiod=20)
        print(f"🧮 SMA計算テスト: 成功 (最後の値: {sma_result[-1]:.4f})")

        # EMAテスト
        ema_result = talib.EMA(test_data, timeperiod=20)
        print(f"🧮 EMA計算テスト: 成功 (最後の値: {ema_result[-1]:.4f})")

        # RSIテスト
        rsi_result = talib.RSI(test_data, timeperiod=14)
        print(f"🧮 RSI計算テスト: 成功 (最後の値: {rsi_result[-1]:.4f})")

        # MACDテスト
        macd, signal, hist = talib.MACD(test_data)
        print(
            f"🧮 MACD計算テスト: 成功 (MACD: {macd[-1]:.4f}, Signal: {signal[-1]:.4f})"
        )

        print("\n🎉 TA-Lib は正常にインストールされ、動作しています！")
        return True

    except ImportError as e:
        print(f"❌ TA-Lib インポートエラー: {e}")
        print("💡 TA-Libをインストールしてください:")
        print("   pip install TA-Lib")
        print("   または")
        print("   conda install -c conda-forge ta-lib")
        return False

    except Exception as e:
        print(f"❌ TA-Lib テストエラー: {e}")
        return False


if __name__ == "__main__":
    print("🔍 TA-Lib インストール状況確認")
    print("=" * 50)
    check_talib_installation()
