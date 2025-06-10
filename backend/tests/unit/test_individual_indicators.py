#!/usr/bin/env python3
"""
個別指標クラスの包括的テスト

各指標クラス（SMA, EMA, RSI等）の動作を個別にテストします。
"""

import sys
import os
import pandas as pd
import numpy as np
import traceback

# バックエンドのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def create_test_data(periods=100):
    """テスト用のOHLCVデータを作成"""
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    np.random.seed(42)

    base_price = 50000
    returns = np.random.normal(0, 0.02, periods)
    close_prices = base_price * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {
            "open": close_prices * (1 + np.random.normal(0, 0.001, periods)),
            "high": close_prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
            "low": close_prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
            "close": close_prices,
            "volume": np.random.randint(1000, 10000, periods),
        },
        index=dates,
    )


def test_trend_indicators():
    """トレンド指標のテスト"""
    print("🧪 トレンド指標テスト")
    print("=" * 50)

    try:
        from app.core.services.indicators.trend_indicators import (
            SMAIndicator,
            EMAIndicator,
            MACDIndicator,
            KAMAIndicator,
            T3Indicator,
            TEMAIndicator,
        )

        test_data = create_test_data(100)
        print(f"📊 テストデータ作成: {len(test_data)}件")

        # SMAテスト
        print("\n1. SMA指標テスト")
        sma_indicator = SMAIndicator()
        sma_result = sma_indicator.calculate(test_data, period=20)

        assert isinstance(sma_result, pd.Series)
        assert len(sma_result) == len(test_data)
        assert sma_result.name == "SMA_20"
        print(f"   ✅ SMA計算成功: 最後の値 {sma_result.iloc[-1]:.2f}")

        # EMAテスト
        print("\n2. EMA指標テスト")
        ema_indicator = EMAIndicator()
        ema_result = ema_indicator.calculate(test_data, period=20)

        assert isinstance(ema_result, pd.Series)
        assert ema_result.name == "EMA_20"
        print(f"   ✅ EMA計算成功: 最後の値 {ema_result.iloc[-1]:.2f}")

        # MACDテスト
        print("\n3. MACD指標テスト")
        macd_indicator = MACDIndicator()
        macd_result = macd_indicator.calculate(test_data, period=12)

        assert isinstance(macd_result, dict)
        assert "macd_line" in macd_result
        assert "signal_line" in macd_result
        assert "histogram" in macd_result
        print(f"   ✅ MACD計算成功: MACD {macd_result['macd_line'].iloc[-1]:.4f}")

        # KAMAテスト
        print("\n4. KAMA指標テスト")
        kama_indicator = KAMAIndicator()
        kama_result = kama_indicator.calculate(test_data, period=20)

        assert isinstance(kama_result, pd.Series)
        assert kama_result.name == "KAMA_20"
        print(f"   ✅ KAMA計算成功: 最後の値 {kama_result.iloc[-1]:.2f}")

        # T3テスト
        print("\n5. T3指標テスト")
        t3_indicator = T3Indicator()
        t3_result = t3_indicator.calculate(test_data, period=5)

        assert isinstance(t3_result, pd.Series)
        assert t3_result.name == "T3_5"
        print(f"   ✅ T3計算成功: 最後の値 {t3_result.iloc[-1]:.2f}")

        # TEMAテスト
        print("\n6. TEMA指標テスト")
        tema_indicator = TEMAIndicator()
        tema_result = tema_indicator.calculate(test_data, period=14)

        assert isinstance(tema_result, pd.Series)
        assert tema_result.name == "TEMA_14"
        print(f"   ✅ TEMA計算成功: 最後の値 {tema_result.iloc[-1]:.2f}")

        print("✅ トレンド指標 全テスト成功")
        return True

    except Exception as e:
        print(f"❌ トレンド指標テスト失敗: {e}")
        traceback.print_exc()
        return False


def test_momentum_indicators():
    """モメンタム指標のテスト"""
    print("\n🧪 モメンタム指標テスト")
    print("=" * 50)

    try:
        from app.core.services.indicators.momentum_indicators import (
            RSIIndicator,
            StochasticIndicator,
            WilliamsRIndicator,
            ROCIndicator,
            MomentumIndicator,
        )

        test_data = create_test_data(100)

        # RSIテスト
        print("\n1. RSI指標テスト")
        rsi_indicator = RSIIndicator()
        rsi_result = rsi_indicator.calculate(test_data, period=14)

        assert isinstance(rsi_result, pd.Series)
        assert rsi_result.name == "RSI_14"

        # RSIの範囲チェック
        valid_values = rsi_result.dropna()
        if len(valid_values) > 0:
            assert (valid_values >= 0).all()
            assert (valid_values <= 100).all()
        print(f"   ✅ RSI計算成功: 最後の値 {rsi_result.iloc[-1]:.2f}")

        # Stochasticテスト
        print("\n2. Stochastic指標テスト")
        stoch_indicator = StochasticIndicator()
        stoch_result = stoch_indicator.calculate(test_data, period=14)

        assert isinstance(stoch_result, dict)
        assert "slowk" in stoch_result
        assert "slowd" in stoch_result
        print(f"   ✅ Stochastic計算成功: %K {stoch_result['slowk'].iloc[-1]:.2f}")

        # Williams %Rテスト
        print("\n3. Williams %R指標テスト")
        willr_indicator = WilliamsRIndicator()
        willr_result = willr_indicator.calculate(test_data, period=14)

        assert isinstance(willr_result, pd.Series)
        assert willr_result.name == "WILLR_14"
        print(f"   ✅ Williams %R計算成功: 最後の値 {willr_result.iloc[-1]:.2f}")

        # ROCテスト
        print("\n4. ROC指標テスト")
        roc_indicator = ROCIndicator()
        roc_result = roc_indicator.calculate(test_data, period=10)

        assert isinstance(roc_result, pd.Series)
        assert roc_result.name == "ROC_10"
        print(f"   ✅ ROC計算成功: 最後の値 {roc_result.iloc[-1]:.2f}")

        # Momentumテスト
        print("\n5. Momentum指標テスト")
        mom_indicator = MomentumIndicator()
        mom_result = mom_indicator.calculate(test_data, period=10)

        assert isinstance(mom_result, pd.Series)
        assert mom_result.name == "MOM_10"
        print(f"   ✅ Momentum計算成功: 最後の値 {mom_result.iloc[-1]:.2f}")

        print("✅ モメンタム指標 全テスト成功")
        return True

    except Exception as e:
        print(f"❌ モメンタム指標テスト失敗: {e}")
        traceback.print_exc()
        return False


def test_volatility_indicators():
    """ボラティリティ指標のテスト"""
    print("\n🧪 ボラティリティ指標テスト")
    print("=" * 50)

    try:
        from app.core.services.indicators.volatility_indicators import (
            ATRIndicator,
            BollingerBandsIndicator,
        )

        test_data = create_test_data(100)

        # ATRテスト
        print("\n1. ATR指標テスト")
        atr_indicator = ATRIndicator()
        atr_result = atr_indicator.calculate(test_data, period=14)

        assert isinstance(atr_result, pd.Series)
        assert atr_result.name == "ATR_14"
        print(f"   ✅ ATR計算成功: 最後の値 {atr_result.iloc[-1]:.2f}")

        # Bollinger Bandsテスト
        print("\n2. Bollinger Bands指標テスト")
        bb_indicator = BollingerBandsIndicator()
        bb_result = bb_indicator.calculate(test_data, period=20)

        assert isinstance(bb_result, dict)
        assert "upper" in bb_result
        assert "middle" in bb_result
        assert "lower" in bb_result
        print(f"   ✅ Bollinger Bands計算成功: Upper {bb_result['upper'].iloc[-1]:.2f}")

        print("✅ ボラティリティ指標 全テスト成功")
        return True

    except Exception as e:
        print(f"❌ ボラティリティ指標テスト失敗: {e}")
        traceback.print_exc()
        return False


def test_volume_indicators():
    """ボリューム指標のテスト"""
    print("\n🧪 ボリューム指標テスト")
    print("=" * 50)

    try:
        from app.core.services.indicators.volume_indicators import (
            OBVIndicator,
            ADIndicator,
        )

        test_data = create_test_data(100)

        # OBVテスト
        print("\n1. OBV指標テスト")
        obv_indicator = OBVIndicator()
        obv_result = obv_indicator.calculate(test_data, period=1)  # OBVは期間を使わない

        assert isinstance(obv_result, pd.Series)
        assert obv_result.name == "OBV"
        print(f"   ✅ OBV計算成功: 最後の値 {obv_result.iloc[-1]:.0f}")

        # A/D Lineテスト
        print("\n2. A/D Line指標テスト")
        ad_indicator = ADIndicator()
        ad_result = ad_indicator.calculate(
            test_data, period=1
        )  # A/D Lineは期間を使わない

        assert isinstance(ad_result, pd.Series)
        assert ad_result.name == "AD"
        print(f"   ✅ A/D Line計算成功: 最後の値 {ad_result.iloc[-1]:.2f}")

        print("✅ ボリューム指標 全テスト成功")
        return True

    except Exception as e:
        print(f"❌ ボリューム指標テスト失敗: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔬 個別指標クラス 包括的テスト")
    print("=" * 60)

    # テスト実行
    results = {
        "trend": test_trend_indicators(),
        "momentum": test_momentum_indicators(),
        "volatility": test_volatility_indicators(),
        "volume": test_volume_indicators(),
    }

    # 結果サマリー
    print("\n📋 テスト結果サマリー")
    print("=" * 60)
    for test_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"{test_name.capitalize()}指標テスト: {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"\n📊 総合結果: {passed_tests}/{total_tests} 成功")

    if passed_tests == total_tests:
        print("🎉 全ての個別指標テストが成功しました！")
    else:
        print("⚠️ 一部の指標テストが失敗しました。修正が必要です。")
