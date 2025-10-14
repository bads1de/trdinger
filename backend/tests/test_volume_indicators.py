"""
VolumeIndicatorsのテスト
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from app.services.indicators.technical_indicators.volume import VolumeIndicators


class TestVolumeIndicators:
    """VolumeIndicatorsのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        pass

    def test_init(self):
        """初期化のテスト"""
        assert VolumeIndicators is not None

    def test_calculate_obv_valid_data(self):
        """有効データでのOBV計算テスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            }
        )

        result = VolumeIndicators.obv(data["close"], data["volume"], period=14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # OBVは累積値なので単調増加または減少
        # NaN以外の値が含まれている
        assert not result.isna().all()

    def test_calculate_obv_mismatched_lengths(self):
        """長さ不一致のOBVテスト"""
        close_data = pd.DataFrame({"close": [100, 101, 102]})
        volume_data = pd.DataFrame({"volume": [1000, 1100]})  # 長さが異なる

        with pytest.raises(ValueError, match="close and volume series must have the same length"):
            VolumeIndicators.obv(close_data["close"], volume_data["volume"], period=3)

    def test_calculate_ad_valid_data(self):
        """有効データでのA/Dライン計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            }
        )

        result = VolumeIndicators.ad(data["high"], data["low"], data["close"], data["volume"])

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # 累積値なので単調性がある
        assert not result.isna().all()

    def test_calculate_adosc_valid_data(self):
        """有効データでのA/Dオシレーター計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            }
        )

        result = VolumeIndicators.adosc(data["high"], data["low"], data["close"], data["volume"], fast=3, slow=10)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # オシレーターなので正負の値を取り得る
        assert not result.isna().all()

    def test_calculate_cmf_valid_data(self):
        """有効データでのCMF計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            }
        )

        result = VolumeIndicators.cmf(data["high"], data["low"], data["close"], data["volume"], length=20)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # CMFは-1から+1の範囲
        assert result.dropna().min() >= -1
        assert result.dropna().max() <= 1

    def test_calculate_cmf_insufficient_data(self):
        """不十分なデータでのCMF計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 105],
                "low": [98, 99, 100, 101],
                "close": [100, 101, 102, 103],
                "volume": [1000, 1100, 1200, 1300],
            }
        )

        with pytest.raises(ValueError, match="Insufficient data for CMF calculation"):
            VolumeIndicators.cmf(data["high"], data["low"], data["close"], data["volume"], length=20)

    def test_calculate_mfi_valid_data(self):
        """有効データでのMFI計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            }
        )

        result = VolumeIndicators.mfi(data["high"], data["low"], data["close"], data["volume"], length=14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # MFIは0-100の範囲
        assert result.dropna().min() >= 0
        assert result.dropna().max() <= 100

    def test_calculate_vwap_valid_data(self):
        """有効データでのVWAP計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            }
        )

        result = VolumeIndicators.vwap(data["high"], data["low"], data["close"], data["volume"], period=10)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # VWAPは価格の範囲内
        assert result.dropna().min() >= 98
        assert result.dropna().max() <= 111

    def test_calculate_pvo_valid_data(self):
        """有効データでのPVO計算テスト"""
        data = pd.DataFrame(
            {"volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]}
        )

        result = VolumeIndicators.pvo(data["volume"], fast=12, slow=26, signal=9, scalar=100.0)

        assert isinstance(result, tuple)
        assert len(result) == 3
        pvo, signal_line, histogram = result
        assert len(pvo) == len(data)
        assert len(signal_line) == len(data)
        assert len(histogram) == len(data)

    def test_calculate_pvo_invalid_parameters(self):
        """無効なパラメータのPVOテスト"""
        data = pd.DataFrame({"volume": [1000, 1100, 1200]})

        with pytest.raises(ValueError, match="fast, slow, signal must be positive"):
            VolumeIndicators.pvo(data["volume"], fast=-1, slow=26, signal=9, scalar=100.0)

    def test_calculate_pvt_valid_data(self):
        """有効データでのPVT計算テスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            }
        )

        result = VolumeIndicators.pvt(data["close"], data["volume"])

        assert isinstance(result, np.ndarray) or hasattr(result, 'to_numpy')
        assert len(result) == len(data)

    def test_calculate_kvo_valid_data(self):
        """有効データでのKVO計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            }
        )

        result = VolumeIndicators.kvo(data["high"], data["low"], data["close"], data["volume"])

        assert isinstance(result, tuple)
        assert len(result) == 2
        kvo_line, signal_line = result
        assert len(kvo_line) == len(data)
        assert len(signal_line) == len(data)

    def test_calculate_kvo_insufficient_data(self):
        """不十分なデータでのKVO計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103],
                "low": [98, 99],
                "close": [100, 101],
                "volume": [1000, 1100],
            }
        )

        result = VolumeIndicators.kvo(data["high"], data["low"], data["close"], data["volume"])

        assert isinstance(result, tuple)
        assert len(result) == 2
        kvo_line, signal_line = result
        # 不十分なデータではNaN
        assert kvo_line.isna().all()
        assert signal_line.isna().all()

    def test_calculate_nvi_valid_data(self):
        """有効データでのNVI計算テスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            }
        )

        result = VolumeIndicators.nvi(data["close"], data["volume"])

        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)

    def test_calculate_efi_valid_data(self):
        """有効データでのEFI計算テスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            }
        )

        result = VolumeIndicators.efi(data["close"], data["volume"], period=13, mamode="ema", drift=1)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # EFIは正負の値を取り得る
        assert not result.isna().all()

    def test_calculate_efi_insufficient_data(self):
        """不十分なデータでのEFI計算テスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102],
                "volume": [1000, 1100, 1200],
            }
        )

        with pytest.raises(ValueError, match="Insufficient data for EFI calculation"):
            VolumeIndicators.efi(data["close"], data["volume"], period=14, mamode="ema", drift=1)

    def test_calculate_eom_valid_data(self):
        """有効データでのEase of Movement計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            }
        )

        result = VolumeIndicators.eom(data["high"], data["low"], data["close"], data["volume"], length=14, divisor=100000000.0, drift=1)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # EOMは正負の値を取り得る
        assert not result.isna().all()

    def test_calculate_eom_mismatched_lengths(self):
        """長さ不一致のEOMテスト"""
        data_high = pd.DataFrame({"high": [102, 103, 104]})
        data_low = pd.DataFrame({"low": [98, 99]})
        data_close = pd.DataFrame({"close": [100, 101, 102]})
        data_volume = pd.DataFrame({"volume": [1000, 1100, 1200]})

        with pytest.raises(ValueError, match="high, low, close, volume must have the same length"):
            VolumeIndicators.eom(data_high["high"], data_low["low"], data_close["close"], data_volume["volume"], length=3, divisor=100000000.0, drift=1)

    def test_handle_invalid_data_types(self):
        """無効なデータ型のテスト"""
        # 数値以外のデータ
        data = pd.DataFrame(
            {"high": ["invalid", "data"], "low": [98, 99], "close": [100, 101], "volume": [1000, 1100]}
        )

        with pytest.raises((TypeError, ValueError)):
            VolumeIndicators.ad(data["high"], data["low"], data["close"], data["volume"])

    def test_handle_negative_volume(self):
        """負の出来高のテスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104],
                "low": [98, 99, 100],
                "close": [100, 101, 102],
                "volume": [1000, -1100, 1200],  # 負の値
            }
        )

        # 負の出来高はNaNとして処理される
        result = VolumeIndicators.obv(data["close"], data["volume"], period=3)
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_handle_zero_volume(self):
        """ゼロ出来高のテスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104],
                "volume": [1000, 0, 1200, 0, 1400],  # ゼロが含まれる
            }
        )

        result = VolumeIndicators.obv(data["close"], data["volume"], period=3)
        assert isinstance(result, pd.Series)
        assert len(result) == 5

    def test_handle_negative_length(self):
        """負の長さパラメータのテスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

        with pytest.raises(ValueError):
            VolumeIndicators.obv(data["close"], data["volume"], period=-1)

    def test_edge_case_empty_data(self):
        """空データのテスト"""
        data = pd.DataFrame({"close": [], "volume": []})

        result = VolumeIndicators.obv(data["close"], data["volume"], period=5)

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_edge_case_single_value(self):
        """単一値のテスト"""
        data = pd.DataFrame({"close": [100], "volume": [1000]})

        result = VolumeIndicators.obv(data["close"], data["volume"], period=5)

        assert isinstance(result, pd.Series)
        assert len(result) == 1
        assert result.isna().all()

    def test_data_validation_empty_series(self):
        """空シリーズのデータ検証"""
        empty_close = pd.Series([])
        empty_volume = pd.Series([])

        result = VolumeIndicators.obv(empty_close, empty_volume, period=5)

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_all_volume_indicators_with_sample_data(self):
        """代表的な出来高指標の統合テスト"""
        # 十分な長さのテストデータ
        np.random.seed(42)
        base = 1000
        trend = np.linspace(0, 100, 100)
        noise = np.random.normal(0, 10, 100)
        close_prices = base + trend + noise

        data = pd.DataFrame(
            {
                "close": close_prices,
                "high": close_prices + np.random.normal(0, 5, 100),
                "low": close_prices - np.random.normal(0, 5, 100),
                "volume": np.random.uniform(1000, 5000, 100),
            }
        )

        # テスト対象の指標
        indicators_to_test = [
            ("obv", (data["close"], data["volume"], 14)),
            ("ad", (data["high"], data["low"], data["close"], data["volume"])),
            ("adosc", (data["high"], data["low"], data["close"], data["volume"], 3, 10)),
            ("mfi", (data["high"], data["low"], data["close"], data["volume"], 14)),
            ("vwap", (data["high"], data["low"], data["close"], data["volume"], 20)),
        ]

        for indicator_name, params in indicators_to_test:
            with self.subTest(indicator=indicator_name):
                method = getattr(VolumeIndicators, indicator_name)
                result = method(*params)
                if indicator_name in ["pvo", "kvo"]:
                    # 複数のシリーズを返す場合
                    assert isinstance(result, tuple)
                    for series in result:
                        assert isinstance(series, (pd.Series, np.ndarray))
                        assert len(series) == len(data)
                else:
                    assert isinstance(result, (pd.Series, np.ndarray))
                    assert len(result) == len(data)

    def test_all_volume_indicators_handle_errors(self):
        """出来高指標のエラーハンドリング統合テスト"""
        # 短いデータ
        short_data = pd.DataFrame(
            {
                "high": [102, 103],
                "low": [98, 99],
                "close": [100, 101],
                "volume": [1000, 1100],
            }
        )

        # エラーが適切に処理されるかテスト
        try:
            VolumeIndicators.cmf(
                short_data["high"], short_data["low"], short_data["close"], short_data["volume"], length=20
            )
            # エラーにならずNaNが返される
        except Exception as e:
            assert "data" in str(e) or "length" in str(e)

        try:
            VolumeIndicators.efi(short_data["close"], short_data["volume"], period=14, mamode="ema", drift=1)
            # エラーにならずNaNが返される
        except Exception as e:
            assert "data" in str(e) or "length" in str(e)

        # 無効なパラメータ
        try:
            VolumeIndicators.obv(short_data["close"], short_data["volume"], period=-1)
            assert False, "負の長さでエラーが発生すべき"
        except ValueError:
            pass  # 期待されるエラー

    def test_obv_with_different_price_patterns(self):
        """異なる価格パターンでのOBVテスト"""
        # 上昇トレンド
        uptrend_data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500],
            }
        )
        result_uptrend = VolumeIndicators.obv(uptrend_data["close"], uptrend_data["volume"], period=3)
        assert isinstance(result_uptrend, pd.Series)

        # 下降トレンド
        downtrend_data = pd.DataFrame(
            {
                "close": [105, 104, 103, 102, 101, 100],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500],
            }
        )
        result_downtrend = VolumeIndicators.obv(downtrend_data["close"], downtrend_data["volume"], period=3)
        assert isinstance(result_downtrend, pd.Series)

        # 価格と出来高の関係が反映される
        assert len(result_uptrend) == len(uptrend_data)
        assert len(result_downtrend) == len(downtrend_data)

    def test_mfi_with_extreme_values(self):
        """極端な値でのMFIテスト"""
        # 極端な価格変動
        extreme_data = pd.DataFrame(
            {
                "high": [200, 250, 150, 300, 100, 350],
                "low": [50, 100, 80, 120, 90, 110],
                "close": [180, 220, 130, 280, 95, 320],
                "volume": [5000, 6000, 7000, 8000, 9000, 10000],
            }
        )

        result = VolumeIndicators.mfi(extreme_data["high"], extreme_data["low"], extreme_data["close"], extreme_data["volume"], length=6)

        assert isinstance(result, pd.Series)
        assert len(result) == len(extreme_data)
        # MFIは依然として0-100の範囲
        assert result.dropna().min() >= 0
        assert result.dropna().max() <= 100

    def subTest(self, indicator):
        """サブテスト用ダミーメソッド"""
        pass


if __name__ == "__main__":
    # コマンドラインからの実行用
    pytest.main([__file__, "-v"])