"""
包括的なテクニカル指標テスト

すべてのTA-Lib実装指標の動作確認を行います。
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import talib
from typing import Dict, Any, List


# TALibAdapterクラスのコードを直接コピーしてテスト
class TALibCalculationError(Exception):
    """TA-Lib計算エラー"""

    pass


class TALibAdapter:
    """TA-Libと既存システムの橋渡しクラス"""

    @staticmethod
    def _validate_input(data: pd.Series, period: int) -> None:
        """入力データとパラメータの検証"""
        if data is None or len(data) == 0:
            raise TALibCalculationError("入力データが空です")
        if period <= 0:
            raise TALibCalculationError(f"期間は正の整数である必要があります: {period}")
        if len(data) < period:
            raise TALibCalculationError(
                f"データ長({len(data)})が期間({period})より短いです"
            )

    @staticmethod
    def _safe_talib_calculation(func, *args, **kwargs):
        """TA-Lib計算の安全な実行"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise TALibCalculationError(f"TA-Lib計算エラー: {e}")

    # トレンド系指標
    @staticmethod
    def sma(data: pd.Series, period: int = 20) -> pd.Series:
        TALibAdapter._validate_input(data, period)
        result = TALibAdapter._safe_talib_calculation(
            talib.SMA, data.values, timeperiod=period
        )
        return pd.Series(result, index=data.index, name=f"SMA_{period}")

    @staticmethod
    def ema(data: pd.Series, period: int = 20) -> pd.Series:
        TALibAdapter._validate_input(data, period)
        result = TALibAdapter._safe_talib_calculation(
            talib.EMA, data.values, timeperiod=period
        )
        return pd.Series(result, index=data.index, name=f"EMA_{period}")

    @staticmethod
    def kama(data: pd.Series, period: int = 30) -> pd.Series:
        TALibAdapter._validate_input(data, period)
        result = TALibAdapter._safe_talib_calculation(
            talib.KAMA, data.values, timeperiod=period
        )
        return pd.Series(result, index=data.index, name=f"KAMA_{period}")

    @staticmethod
    def t3(data: pd.Series, period: int = 5, vfactor: float = 0.7) -> pd.Series:
        TALibAdapter._validate_input(data, period)
        result = TALibAdapter._safe_talib_calculation(
            talib.T3, data.values, timeperiod=period, vfactor=vfactor
        )
        return pd.Series(result, index=data.index, name=f"T3_{period}")

    @staticmethod
    def tema(data: pd.Series, period: int = 30) -> pd.Series:
        TALibAdapter._validate_input(data, period)
        result = TALibAdapter._safe_talib_calculation(
            talib.TEMA, data.values, timeperiod=period
        )
        return pd.Series(result, index=data.index, name=f"TEMA_{period}")

    @staticmethod
    def dema(data: pd.Series, period: int = 30) -> pd.Series:
        TALibAdapter._validate_input(data, period)
        result = TALibAdapter._safe_talib_calculation(
            talib.DEMA, data.values, timeperiod=period
        )
        return pd.Series(result, index=data.index, name=f"DEMA_{period}")

    # モメンタム系指標
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        TALibAdapter._validate_input(data, period)
        result = TALibAdapter._safe_talib_calculation(
            talib.RSI, data.values, timeperiod=period
        )
        return pd.Series(result, index=data.index, name=f"RSI_{period}")

    @staticmethod
    def adx(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        if not (len(high) == len(low) == len(close)):
            raise TALibCalculationError("高値、安値、終値のデータ長が一致しません")
        TALibAdapter._validate_input(close, period)
        result = TALibAdapter._safe_talib_calculation(
            talib.ADX, high.values, low.values, close.values, timeperiod=period
        )
        return pd.Series(result, index=close.index, name=f"ADX_{period}")

    @staticmethod
    def mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        if not (len(high) == len(low) == len(close) == len(volume)):
            raise TALibCalculationError(
                "高値、安値、終値、出来高のデータ長が一致しません"
            )
        TALibAdapter._validate_input(close, period)
        result = TALibAdapter._safe_talib_calculation(
            talib.MFI,
            high.values,
            low.values,
            close.values,
            volume.values,
            timeperiod=period,
        )
        return pd.Series(result, index=close.index, name=f"MFI_{period}")

    # ボラティリティ系指標
    @staticmethod
    def atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        if not (len(high) == len(low) == len(close)):
            raise TALibCalculationError("高値、安値、終値のデータ長が一致しません")
        TALibAdapter._validate_input(close, period)
        result = TALibAdapter._safe_talib_calculation(
            talib.ATR, high.values, low.values, close.values, timeperiod=period
        )
        return pd.Series(result, index=close.index, name=f"ATR_{period}")

    @staticmethod
    def natr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        if not (len(high) == len(low) == len(close)):
            raise TALibCalculationError("高値、安値、終値のデータ長が一致しません")
        TALibAdapter._validate_input(close, period)
        result = TALibAdapter._safe_talib_calculation(
            talib.NATR, high.values, low.values, close.values, timeperiod=period
        )
        return pd.Series(result, index=close.index, name=f"NATR_{period}")

    @staticmethod
    def trange(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        if not (len(high) == len(low) == len(close)):
            raise TALibCalculationError("高値、安値、終値のデータ長が一致しません")
        if len(close) == 0:
            raise TALibCalculationError("入力データが空です")
        result = TALibAdapter._safe_talib_calculation(
            talib.TRANGE, high.values, low.values, close.values
        )
        return pd.Series(result, index=close.index, name="TRANGE")

    # 出来高系指標
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        if not (len(close) == len(volume)):
            raise TALibCalculationError("終値、出来高のデータ長が一致しません")
        if len(close) == 0:
            raise TALibCalculationError("入力データが空です")
        result = TALibAdapter._safe_talib_calculation(
            talib.OBV, close.values, volume.values
        )
        return pd.Series(result, index=close.index, name="OBV")

    @staticmethod
    def ad(
        high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        if not (len(high) == len(low) == len(close) == len(volume)):
            raise TALibCalculationError(
                "高値、安値、終値、出来高のデータ長が一致しません"
            )
        if len(close) == 0:
            raise TALibCalculationError("入力データが空です")
        result = TALibAdapter._safe_talib_calculation(
            talib.AD, high.values, low.values, close.values, volume.values
        )
        return pd.Series(result, index=close.index, name="AD")

    @staticmethod
    def adosc(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        fast_period: int = 3,
        slow_period: int = 10,
    ) -> pd.Series:
        if not (len(high) == len(low) == len(close) == len(volume)):
            raise TALibCalculationError(
                "高値、安値、終値、出来高のデータ長が一致しません"
            )
        TALibAdapter._validate_input(close, slow_period)
        result = TALibAdapter._safe_talib_calculation(
            talib.ADOSC,
            high.values,
            low.values,
            close.values,
            volume.values,
            fastperiod=fast_period,
            slowperiod=slow_period,
        )
        return pd.Series(
            result, index=close.index, name=f"ADOSC_{fast_period}_{slow_period}"
        )


def generate_test_data(sample_size: int = 200) -> Dict[str, pd.Series]:
    """テスト用のOHLCVデータを生成"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=sample_size, freq="D")

    # より現実的な価格データを生成
    base_price = 100
    trend = np.linspace(0, 10, sample_size)
    noise = np.random.normal(0, 1, sample_size)
    close_prices = base_price + trend + noise

    # OHLC データを生成
    high_prices = close_prices + np.random.uniform(0, 2, sample_size)
    low_prices = close_prices - np.random.uniform(0, 2, sample_size)
    open_prices = close_prices + np.random.uniform(-1, 1, sample_size)
    volume_data = np.random.uniform(1000, 10000, sample_size)

    return {
        "open": pd.Series(open_prices, index=dates),
        "high": pd.Series(high_prices, index=dates),
        "low": pd.Series(low_prices, index=dates),
        "close": pd.Series(close_prices, index=dates),
        "volume": pd.Series(volume_data, index=dates),
    }


def test_comprehensive_technical_indicators():
    """包括的なテクニカル指標テスト"""
    print("=== 包括的なテクニカル指標テスト ===")

    # テストデータ生成
    data = generate_test_data(200)
    print(f"テストデータ生成完了: {len(data['close'])}日分")
    print(f"価格範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")

    # テスト結果を格納
    test_results = {}

    # トレンド系指標のテスト
    print("\n=== トレンド系指標テスト ===")
    trend_tests = [
        ("SMA", lambda: TALibAdapter.sma(data["close"], 20)),
        ("EMA", lambda: TALibAdapter.ema(data["close"], 20)),
        ("KAMA", lambda: TALibAdapter.kama(data["close"], 30)),
        ("T3", lambda: TALibAdapter.t3(data["close"], 14)),
        ("TEMA", lambda: TALibAdapter.tema(data["close"], 21)),
        ("DEMA", lambda: TALibAdapter.dema(data["close"], 21)),
    ]

    for name, test_func in trend_tests:
        try:
            result = test_func()
            valid_count = len(result.dropna())
            test_results[name] = {
                "status": "OK",
                "valid_count": valid_count,
                "total_count": len(result),
                "latest_value": (
                    result.iloc[-1] if not pd.isna(result.iloc[-1]) else None
                ),
            }
            print(f"OK {name}: {valid_count}/{len(result)} 有効値")
        except Exception as e:
            test_results[name] = {"status": "NG", "error": str(e)}
            print(f"NG {name}: {e}")

    # モメンタム系指標のテスト
    print("\n=== モメンタム系指標テスト ===")
    momentum_tests = [
        ("RSI", lambda: TALibAdapter.rsi(data["close"], 14)),
        ("ADX", lambda: TALibAdapter.adx(data["high"], data["low"], data["close"], 14)),
        (
            "MFI",
            lambda: TALibAdapter.mfi(
                data["high"], data["low"], data["close"], data["volume"], 14
            ),
        ),
    ]

    for name, test_func in momentum_tests:
        try:
            result = test_func()
            valid_count = len(result.dropna())
            test_results[name] = {
                "status": "✅",
                "valid_count": valid_count,
                "total_count": len(result),
                "latest_value": (
                    result.iloc[-1] if not pd.isna(result.iloc[-1]) else None
                ),
            }
            print(f"✅ {name}: {valid_count}/{len(result)} 有効値")
        except Exception as e:
            test_results[name] = {"status": "❌", "error": str(e)}
            print(f"❌ {name}: {e}")

    # ボラティリティ系指標のテスト
    print("\n=== ボラティリティ系指標テスト ===")
    volatility_tests = [
        ("ATR", lambda: TALibAdapter.atr(data["high"], data["low"], data["close"], 14)),
        (
            "NATR",
            lambda: TALibAdapter.natr(data["high"], data["low"], data["close"], 14),
        ),
        (
            "TRANGE",
            lambda: TALibAdapter.trange(data["high"], data["low"], data["close"]),
        ),
    ]

    for name, test_func in volatility_tests:
        try:
            result = test_func()
            valid_count = len(result.dropna())
            test_results[name] = {
                "status": "✅",
                "valid_count": valid_count,
                "total_count": len(result),
                "latest_value": (
                    result.iloc[-1] if not pd.isna(result.iloc[-1]) else None
                ),
            }
            print(f"✅ {name}: {valid_count}/{len(result)} 有効値")
        except Exception as e:
            test_results[name] = {"status": "❌", "error": str(e)}
            print(f"❌ {name}: {e}")

    # 出来高系指標のテスト
    print("\n=== 出来高系指標テスト ===")
    volume_tests = [
        ("OBV", lambda: TALibAdapter.obv(data["close"], data["volume"])),
        (
            "AD",
            lambda: TALibAdapter.ad(
                data["high"], data["low"], data["close"], data["volume"]
            ),
        ),
        (
            "ADOSC",
            lambda: TALibAdapter.adosc(
                data["high"], data["low"], data["close"], data["volume"], 3, 10
            ),
        ),
    ]

    for name, test_func in volume_tests:
        try:
            result = test_func()
            valid_count = len(result.dropna())
            test_results[name] = {
                "status": "✅",
                "valid_count": valid_count,
                "total_count": len(result),
                "latest_value": (
                    result.iloc[-1] if not pd.isna(result.iloc[-1]) else None
                ),
            }
            print(f"✅ {name}: {valid_count}/{len(result)} 有効値")
        except Exception as e:
            test_results[name] = {"status": "❌", "error": str(e)}
            print(f"❌ {name}: {e}")

    return test_results


if __name__ == "__main__":
    results = test_comprehensive_technical_indicators()

    # 結果サマリー
    print("\n=== テスト結果サマリー ===")
    success_count = sum(1 for r in results.values() if r["status"] == "✅")
    total_count = len(results)

    print(f"成功: {success_count}/{total_count} 指標")
    print(f"成功率: {success_count/total_count*100:.1f}%")

    if success_count == total_count:
        print("🎉 すべてのテクニカル指標が正常に動作しています！")
    else:
        print("⚠️ 一部の指標でエラーが発生しました。")
        for name, result in results.items():
            if result["status"] == "❌":
                print(f"   {name}: {result.get('error', 'Unknown error')}")
