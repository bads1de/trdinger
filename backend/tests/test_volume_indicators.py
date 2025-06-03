"""
出来高系指標のテスト

OBV、AD、ADOSC指標のテストを実行します。
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
    def _safe_talib_calculation(func, *args, **kwargs) -> np.ndarray:
        """TA-Lib計算の安全な実行"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise TALibCalculationError(f"TA-Lib計算エラー: {e}")

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume (OBV) を計算"""
        if not (len(close) == len(volume)):
            raise TALibCalculationError("終値、出来高のデータ長が一致しません")

        if len(close) == 0:
            raise TALibCalculationError("入力データが空です")

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.OBV, close.values, volume.values
            )
            return pd.Series(result, index=close.index, name="OBV")
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"OBV計算失敗: {e}")

    @staticmethod
    def ad(
        high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """Chaikin A/D Line (Accumulation/Distribution Line) を計算"""
        if not (len(high) == len(low) == len(close) == len(volume)):
            raise TALibCalculationError(
                "高値、安値、終値、出来高のデータ長が一致しません"
            )

        if len(close) == 0:
            raise TALibCalculationError("入力データが空です")

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.AD, high.values, low.values, close.values, volume.values
            )
            return pd.Series(result, index=close.index, name="AD")
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"A/D Line計算失敗: {e}")

    @staticmethod
    def adosc(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        fast_period: int = 3,
        slow_period: int = 10,
    ) -> pd.Series:
        """Chaikin A/D Oscillator (ADOSC) を計算"""
        if not (len(high) == len(low) == len(close) == len(volume)):
            raise TALibCalculationError(
                "高値、安値、終値、出来高のデータ長が一致しません"
            )

        if len(close) < slow_period:
            raise TALibCalculationError(
                f"データ長({len(close)})が期間({slow_period})より短いです"
            )

        try:
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
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"ADOSC計算失敗: {e}")


def test_volume_indicators():
    """出来高系指標のテスト"""
    print("=== 出来高系TA-Lib指標のテスト ===")

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
    print(f"出来高範囲: {volume.min():.0f} - {volume.max():.0f}")

    # OBVテスト
    print("\n--- OBV (On Balance Volume) ---")
    try:
        obv = TALibAdapter.obv(close, volume)
        valid_obv = obv.dropna()
        print(f"✅ OBV計算成功")
        print(f"   有効値数: {len(valid_obv)}/{len(obv)}")
        if len(valid_obv) > 0:
            print(f"   値域: {valid_obv.min():.0f} - {valid_obv.max():.0f}")
            latest_obv = obv.iloc[-1]
            latest_str = f"{latest_obv:.0f}" if not pd.isna(latest_obv) else "NaN"
            print(f"   最新値: {latest_str}")
            initial_obv = obv.iloc[0]
            initial_str = f"{initial_obv:.0f}" if not pd.isna(initial_obv) else "NaN"
            print(f"   初期値: {initial_str}")
    except Exception as e:
        print(f"❌ OBVエラー: {e}")

    # A/D Lineテスト
    print("\n--- AD (Chaikin A/D Line) ---")
    try:
        ad = TALibAdapter.ad(high, low, close, volume)
        valid_ad = ad.dropna()
        print(f"✅ A/D Line計算成功")
        print(f"   有効値数: {len(valid_ad)}/{len(ad)}")
        if len(valid_ad) > 0:
            print(f"   値域: {valid_ad.min():.0f} - {valid_ad.max():.0f}")
            latest_ad = ad.iloc[-1]
            latest_str = f"{latest_ad:.0f}" if not pd.isna(latest_ad) else "NaN"
            print(f"   最新値: {latest_str}")
            initial_ad = ad.iloc[0]
            initial_str = f"{initial_ad:.0f}" if not pd.isna(initial_ad) else "NaN"
            print(f"   初期値: {initial_str}")
    except Exception as e:
        print(f"❌ A/D Lineエラー: {e}")

    # ADOSCテスト
    print("\n--- ADOSC (Chaikin A/D Oscillator) ---")
    try:
        adosc = TALibAdapter.adosc(high, low, close, volume, 3, 10)
        valid_adosc = adosc.dropna()
        print(f"✅ ADOSC計算成功")
        print(f"   有効値数: {len(valid_adosc)}/{len(adosc)}")
        if len(valid_adosc) > 0:
            print(f"   値域: {valid_adosc.min():.2f} - {valid_adosc.max():.2f}")
            latest_adosc = adosc.iloc[-1]
            latest_str = f"{latest_adosc:.2f}" if not pd.isna(latest_adosc) else "NaN"
            print(f"   最新値: {latest_str}")
    except Exception as e:
        print(f"❌ ADOSCエラー: {e}")

    # エラーハンドリングテスト
    print("\n--- エラーハンドリングテスト ---")
    try:
        # データ長不一致テスト（OBV）
        try:
            TALibAdapter.obv(close[:50], volume)
            print("❌ OBVデータ長不一致エラーが検出されませんでした")
        except TALibCalculationError:
            print("✅ OBVデータ長不一致エラーが正しく検出されました")

        # データ長不一致テスト（AD）
        try:
            TALibAdapter.ad(high[:50], low, close, volume)
            print("❌ ADデータ長不一致エラーが検出されませんでした")
        except TALibCalculationError:
            print("✅ ADデータ長不一致エラーが正しく検出されました")

        # 期間過大テスト（ADOSC）
        try:
            TALibAdapter.adosc(high[:5], low[:5], close[:5], volume[:5], 3, 10)
            print("❌ ADOSC期間過大エラーが検出されませんでした")
        except TALibCalculationError:
            print("✅ ADOSC期間過大エラーが正しく検出されました")

    except Exception as e:
        print(f"❌ エラーハンドリングテストエラー: {e}")

    # 相関テスト
    print("\n--- 指標間の相関テスト ---")
    try:
        obv_result = TALibAdapter.obv(close, volume)
        ad_result = TALibAdapter.ad(high, low, close, volume)
        adosc_result = TALibAdapter.adosc(high, low, close, volume, 3, 10)

        # 有効なデータのみで相関を計算
        valid_data = pd.DataFrame(
            {"obv": obv_result, "ad": ad_result, "adosc": adosc_result}
        ).dropna()

        if len(valid_data) > 10:
            correlation_matrix = valid_data.corr()
            print(f"✅ 指標間相関計算成功（{len(valid_data)}データポイント）")
            print(f"   OBV vs AD: {correlation_matrix.loc['obv', 'ad']:.3f}")
            print(f"   OBV vs ADOSC: {correlation_matrix.loc['obv', 'adosc']:.3f}")
            print(f"   AD vs ADOSC: {correlation_matrix.loc['ad', 'adosc']:.3f}")
        else:
            print("⚠️ 相関計算に十分なデータがありません")

    except Exception as e:
        print(f"❌ 相関テストエラー: {e}")

    print("\n=== テスト完了 ===")


if __name__ == "__main__":
    test_volume_indicators()
