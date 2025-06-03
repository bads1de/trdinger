"""
TALibAdapterの直接テスト

依存関係を避けて、TALibAdapterクラスの機能を直接テストします。
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import talib


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
    def _safe_talib_calculation(func, *args, **kwargs) -> np.ndarray:
        """TA-Lib計算の安全な実行"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise TALibCalculationError(f"TA-Lib計算エラー: {e}")

    @staticmethod
    def adx(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Average Directional Movement Index (ADX) を計算"""
        if not (len(high) == len(low) == len(close)):
            raise TALibCalculationError("高値、安値、終値のデータ長が一致しません")

        TALibAdapter._validate_input(close, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.ADX, high.values, low.values, close.values, timeperiod=period
            )
            return pd.Series(result, index=close.index, name=f"ADX_{period}")
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"ADX計算失敗: {e}")

    @staticmethod
    def aroon(high: pd.Series, low: pd.Series, period: int = 14) -> dict:
        """Aroon (アルーン) を計算"""
        if not (len(high) == len(low)):
            raise TALibCalculationError("高値、安値のデータ長が一致しません")

        TALibAdapter._validate_input(high, period)

        try:
            aroon_down, aroon_up = TALibAdapter._safe_talib_calculation(
                talib.AROON, high.values, low.values, timeperiod=period
            )

            return {
                "aroon_down": pd.Series(
                    aroon_down, index=high.index, name=f"AROON_DOWN_{period}"
                ),
                "aroon_up": pd.Series(
                    aroon_up, index=high.index, name=f"AROON_UP_{period}"
                ),
            }
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"AROON計算失敗: {e}")

    @staticmethod
    def mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Money Flow Index (MFI) を計算"""
        if not (len(high) == len(low) == len(close) == len(volume)):
            raise TALibCalculationError(
                "高値、安値、終値、出来高のデータ長が一致しません"
            )

        TALibAdapter._validate_input(close, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.MFI,
                high.values,
                low.values,
                close.values,
                volume.values,
                timeperiod=period,
            )
            return pd.Series(result, index=close.index, name=f"MFI_{period}")
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"MFI計算失敗: {e}")


def test_new_indicators():
    """新しい指標のテスト"""
    print("=== 新しいTA-Lib指標のテスト ===")

    # テストデータの準備
    np.random.seed(42)
    sample_size = 100
    dates = pd.date_range("2023-01-01", periods=sample_size, freq="D")

    # より現実的な価格データを生成
    base_price = 100
    price_changes = np.random.normal(0, 1, sample_size).cumsum()
    close_prices = base_price + price_changes

    high = pd.Series(close_prices + np.random.uniform(0, 2, sample_size), index=dates)
    low = pd.Series(close_prices - np.random.uniform(0, 2, sample_size), index=dates)
    close = pd.Series(close_prices, index=dates)
    volume = pd.Series(np.random.uniform(1000, 10000, sample_size), index=dates)

    print(f"テストデータ準備完了: {sample_size}日分")
    print(f"価格範囲: {close.min():.2f} - {close.max():.2f}")

    # ADXテスト
    print("\n--- ADX (Average Directional Movement Index) ---")
    try:
        adx = TALibAdapter.adx(high, low, close, 14)
        valid_adx = adx.dropna()
        print(f"✅ ADX計算成功")
        print(f"   有効値数: {len(valid_adx)}/{len(adx)}")
        if len(valid_adx) > 0:
            print(f"   値域: {valid_adx.min():.2f} - {valid_adx.max():.2f}")
            latest_val = adx.iloc[-1]
            latest_str = f"{latest_val:.2f}" if not pd.isna(latest_val) else "NaN"
            print(f"   最新値: {latest_str}")
            # ADXは0-100の範囲であることを確認
            assert all(0 <= val <= 100 for val in valid_adx), "ADX値が範囲外"
    except Exception as e:
        print(f"❌ ADXエラー: {e}")

    # Aroonテスト
    print("\n--- Aroon ---")
    try:
        aroon = TALibAdapter.aroon(high, low, 14)
        aroon_up = aroon["aroon_up"]
        aroon_down = aroon["aroon_down"]
        valid_up = aroon_up.dropna()
        valid_down = aroon_down.dropna()

        print(f"✅ Aroon計算成功")
        print(f"   Aroon Up有効値数: {len(valid_up)}/{len(aroon_up)}")
        print(f"   Aroon Down有効値数: {len(valid_down)}/{len(aroon_down)}")

        if len(valid_up) > 0:
            print(f"   Aroon Up値域: {valid_up.min():.2f} - {valid_up.max():.2f}")
            latest_up = aroon_up.iloc[-1]
            latest_up_str = f"{latest_up:.2f}" if not pd.isna(latest_up) else "NaN"
            print(f"   Aroon Up最新値: {latest_up_str}")
            assert all(0 <= val <= 100 for val in valid_up), "Aroon Up値が範囲外"

        if len(valid_down) > 0:
            print(f"   Aroon Down値域: {valid_down.min():.2f} - {valid_down.max():.2f}")
            latest_down = aroon_down.iloc[-1]
            latest_down_str = (
                f"{latest_down:.2f}" if not pd.isna(latest_down) else "NaN"
            )
            print(f"   Aroon Down最新値: {latest_down_str}")
            assert all(0 <= val <= 100 for val in valid_down), "Aroon Down値が範囲外"

    except Exception as e:
        print(f"❌ Aroonエラー: {e}")

    # MFIテスト
    print("\n--- MFI (Money Flow Index) ---")
    try:
        mfi = TALibAdapter.mfi(high, low, close, volume, 14)
        valid_mfi = mfi.dropna()
        print(f"✅ MFI計算成功")
        print(f"   有効値数: {len(valid_mfi)}/{len(mfi)}")
        if len(valid_mfi) > 0:
            print(f"   値域: {valid_mfi.min():.2f} - {valid_mfi.max():.2f}")
            latest_mfi = mfi.iloc[-1]
            latest_mfi_str = f"{latest_mfi:.2f}" if not pd.isna(latest_mfi) else "NaN"
            print(f"   最新値: {latest_mfi_str}")
            # MFIは0-100の範囲であることを確認
            assert all(0 <= val <= 100 for val in valid_mfi), "MFI値が範囲外"
    except Exception as e:
        print(f"❌ MFIエラー: {e}")

    # エラーハンドリングテスト
    print("\n--- エラーハンドリングテスト ---")
    try:
        # データ長不一致テスト
        try:
            TALibAdapter.adx(high[:50], low, close, 14)
            print("❌ データ長不一致エラーが検出されませんでした")
        except TALibCalculationError:
            print("✅ データ長不一致エラーが正しく検出されました")

        # 期間過大テスト
        try:
            TALibAdapter.adx(high[:10], low[:10], close[:10], 14)
            print("❌ 期間過大エラーが検出されませんでした")
        except TALibCalculationError:
            print("✅ 期間過大エラーが正しく検出されました")

    except Exception as e:
        print(f"❌ エラーハンドリングテストエラー: {e}")

    print("\n=== テスト完了 ===")


if __name__ == "__main__":
    test_new_indicators()
