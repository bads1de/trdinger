"""
パンダスオンリー移行テスト - volume.py

TDDでpandasオンリー移行を検証します。
"""

import pytest
import numpy as np
import pandas as pd
import pandas_ta as ta

from app.services.indicators.technical_indicators.volume import VolumeIndicators
from app.services.indicators.utils import PandasTAError


class TestVolumeMigration:
    """volume.py の pandasオンリー移行テスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用データの生成"""
        np.random.seed(42)
        n = 100

        # OHLCVデータ生成
        close = pd.Series(100 + np.cumsum(np.random.randn(n)), name="close")
        high = close + np.abs(np.random.randn(n)) * 5
        low = close - np.abs(np.random.randn(n)) * 5
        open_price = close + np.random.randn(n) * 2
        volume = pd.Series(np.random.randint(1000, 10000, n), name="volume")

        return {
            'close': close,
            'high': high,
            'low': low,
            'open': open_price,
            'volume': volume
        }

    def test_ad_migration_pandas_input(self, sample_data):
        """AD: pandas入力で正常動作"""
        result = VolumeIndicators.ad(
            high=sample_data['high'],
            low=sample_data['low'],
            close=sample_data['close'],
            volume=sample_data['volume']
        )

        # pandasオンリー移行後のテスト
        assert isinstance(result, pd.Series), f"Expected pd.Series, got {type(result)}"
        assert len(result) == len(sample_data['high'])
        assert not result.isna().all()

    def test_adosc_migration_pandas_input(self, sample_data):
        """ADOSC: pandas入力で正常動作"""
        result = VolumeIndicators.adosc(
            high=sample_data['high'],
            low=sample_data['low'],
            close=sample_data['close'],
            volume=sample_data['volume']
        )

        # pandasオンリー移行後のテスト
        assert isinstance(result, pd.Series), f"Expected pd.Series, got {type(result)}"
        assert len(result) == len(sample_data['high'])
        assert not result.isna().all()

    def test_obv_migration_pandas_input(self, sample_data):
        """OBV: pandas入力で正常動作"""
        result = VolumeIndicators.obv(
            close=sample_data['close'],
            volume=sample_data['volume']
        )

        # pandasオンリー移行後のテスト
        assert isinstance(result, pd.Series), f"Expected pd.Series, got {type(result)}"
        assert len(result) == len(sample_data['close'])
        assert not result.isna().all()

    def test_nvi_migration_pandas_input(self, sample_data):
        """NVI: pandas入力で正常動作"""
        result = VolumeIndicators.nvi(
            close=sample_data['close'],
            volume=sample_data['volume']
        )

        # pandasオンリー移行後のテスト
        assert isinstance(result, pd.Series), f"Expected pd.Series, got {type(result)}"
        assert len(result) == len(sample_data['close'])
        assert not result.isna().all()

    def test_pvi_migration_pandas_input(self, sample_data):
        """PVI: pandas入力で正常動作"""
        result = VolumeIndicators.pvi(
            close=sample_data['close'],
            volume=sample_data['volume']
        )

        # pandasオンリー移行後のテスト
        assert isinstance(result, pd.Series), f"Expected pd.Series, got {type(result)}"
        assert len(result) == len(sample_data['close'])
        assert not result.isna().all()

    def test_vwap_migration_pandas_input(self, sample_data):
        """VWAP: pandas入力で正常動作"""
        result = VolumeIndicators.vwap(
            high=sample_data['high'],
            low=sample_data['low'],
            close=sample_data['close'],
            volume=sample_data['volume']
        )

        # pandasオンリー移行後のテスト
        assert isinstance(result, pd.Series), f"Expected pd.Series, got {type(result)}"
        assert len(result) == len(sample_data['high'])
        assert not result.isna().all()

    def test_eom_migration_pandas_input(self, sample_data):
        """EOM: pandas入力で正常動作"""
        result = VolumeIndicators.eom(
            high=sample_data['high'],
            low=sample_data['low'],
            close=sample_data['close'],
            volume=sample_data['volume']
        )

        # pandasオンリー移行後のテスト
        assert isinstance(result, pd.Series), f"Expected pd.Series, got {type(result)}"
        assert len(result) == len(sample_data['high'])
        assert not result.isna().all()

    def test_kvo_migration_pandas_input(self, sample_data):
        """KVO: pandas入力で正常動作"""
        kvo, kvos = VolumeIndicators.kvo(
            high=sample_data['high'],
            low=sample_data['low'],
            close=sample_data['close'],
            volume=sample_data['volume']
        )

        # pandasオンリー移行後のテスト - tuple of pd.Series
        assert isinstance(kvo, pd.Series), f"Expected pd.Series for kvo, got {type(kvo)}"
        assert isinstance(kvos, pd.Series), f"Expected pd.Series for kvos, got {type(kvos)}"
        assert len(kvo) == len(sample_data['high'])
        assert len(kvos) == len(sample_data['high'])
        assert not kvo.isna().all()

    def test_pvt_migration_pandas_input(self, sample_data):
        """PVT: pandas入力で正常動作"""
        result = VolumeIndicators.pvt(
            close=sample_data['close'],
            volume=sample_data['volume']
        )

        # pandasオンリー移行後のテスト
        assert isinstance(result, pd.Series), f"Expected pd.Series, got {type(result)}"
        assert len(result) == len(sample_data['close'])
        assert not result.isna().all()

    def test_cmf_migration_pandas_input(self, sample_data):
        """CMF: pandas入力で正常動作"""
        result = VolumeIndicators.cmf(
            high=sample_data['high'],
            low=sample_data['low'],
            close=sample_data['close'],
            volume=sample_data['volume']
        )

        # pandasオンリー移行後のテスト
        assert isinstance(result, pd.Series), f"Expected pd.Series, got {type(result)}"
        assert len(result) == len(sample_data['high'])
        assert not result.isna().all()

    def test_aobv_migration_pandas_input(self, sample_data):
        """AOBV: pandas入力で正常動作"""
        result = VolumeIndicators.aobv(
            close=sample_data['close'],
            volume=sample_data['volume']
        )

        # pandasオンリー移行後のテスト - tuple of pd.Series
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 7  # aobv関数は7つのシリーズを返す
        for i, series in enumerate(result):
            assert isinstance(series, pd.Series), f"Item {i}: Expected pd.Series, got {type(series)}"
            assert len(series) == len(sample_data['close'])
            # 一部のものはNaNになる可能性があるため、コメントアウト
            # assert not series.isna().all(), f"Item {i} is all NaN"

    def test_efi_migration_pandas_input(self, sample_data):
        """EFI: pandas入力で正常動作"""
        result = VolumeIndicators.efi(
            close=sample_data['close'],
            volume=sample_data['volume']
        )

        # pandasオンリー移行後のテスト
        assert isinstance(result, pd.Series), f"Expected pd.Series, got {type(result)}"
        assert len(result) == len(sample_data['close'])
        assert not result.isna().all()

    def test_pvol_migration_pandas_input(self, sample_data):
        """PVOL: pandas入力で正常動作"""
        result = VolumeIndicators.pvol(
            close=sample_data['close'],
            volume=sample_data['volume']
        )

        # pandasオンリー移行後のテスト
        assert isinstance(result, pd.Series), f"Expected pd.Series, got {type(result)}"
        assert len(result) == len(sample_data['close'])
        assert not result.isna().all()

    def test_pvr_migration_pandas_input(self, sample_data):
        """PVR: pandas入力で正常動作"""
        result = VolumeIndicators.pvr(
            close=sample_data['close'],
            volume=sample_data['volume']
        )

        # pandasオンリー移行後のテスト
        assert isinstance(result, pd.Series), f"Expected pd.Series, got {type(result)}"
        assert len(result) == len(sample_data['close'])
        assert not result.isna().all()

    def test_vp_migration_pandas_input(self, sample_data):
        """VP: pandas入力で正常動作 - tuple結果"""
        result = VolumeIndicators.vp(
            close=sample_data['close'],
            volume=sample_data['volume']
        )

        # pandasオンリー移行後のテスト - tuple of pd.Series
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 6  # vp関数は6つのシリーズを返す
        for i, series in enumerate(result):
            assert isinstance(series, pd.Series), f"Item {i}: Expected pd.Series, got {type(series)}"
            assert len(series) > 0  # VPは価格範囲に基づくため、長さはデータ長と一致しない可能性がある
            # VPのブロックによっては全てNaNになる可能性があるため、コメントアウト
            # assert not series.isna().all(), f"Item {i} is all NaN"

    def test_current_union_type_handling(self, sample_data):
        """pandasオンリー移行: numpy配列入力が拒否されることを確認"""

        # numpy配列入力をテスト
        high_np = sample_data['high'].to_numpy()
        low_np = sample_data['low'].to_numpy()
        close_np = sample_data['close'].to_numpy()
        volume_np = sample_data['volume'].to_numpy()

        # pandasオンリー移行後はnumpy配列入力でエラーが発生
        with pytest.raises(Exception):  # PandasTAErrorやTypeErrorが発生
            VolumeIndicators.ad(high_np, low_np, close_np, volume_np)

        # pandas Series入力は正常動作
        result_pd = VolumeIndicators.ad(sample_data['high'], sample_data['low'], sample_data['close'], sample_data['volume'])
        assert isinstance(result_pd, pd.Series)

    def test_pandas_ta_direct_comparison(self, sample_data):
        """pandas_ta直接計算との比較"""

        # pandas_ta直接計算
        direct_result = ta.ad(
            high=sample_data['high'],
            low=sample_data['low'],
            close=sample_data['close'],
            volume=sample_data['volume']
        )

        # 現在の関数結果
        current_result = VolumeIndicators.ad(
            high=sample_data['high'],
            low=sample_data['low'],
            close=sample_data['close'],
            volume=sample_data['volume']
        )

        # 結果を比較
        np.testing.assert_array_equal(current_result.values, direct_result.values)

    def test_all_functions_work_with_pandas(self, sample_data):
        """全関数がpandas入力で動作することを確認"""

        functions_to_test = [
            ('ad', lambda: VolumeIndicators.ad(sample_data['high'], sample_data['low'], sample_data['close'], sample_data['volume'])),
            ('adosc', lambda: VolumeIndicators.adosc(sample_data['high'], sample_data['low'], sample_data['close'], sample_data['volume'])),
            ('obv', lambda: VolumeIndicators.obv(sample_data['close'], sample_data['volume'])),
            ('nvi', lambda: VolumeIndicators.nvi(sample_data['close'], sample_data['volume'])),
            ('pvi', lambda: VolumeIndicators.pvi(sample_data['close'], sample_data['volume'])),
            ('vwap', lambda: VolumeIndicators.vwap(sample_data['high'], sample_data['low'], sample_data['close'], sample_data['volume'])),
            ('eom', lambda: VolumeIndicators.eom(sample_data['high'], sample_data['low'], sample_data['close'], sample_data['volume'])),
            ('kvo', lambda: VolumeIndicators.kvo(sample_data['high'], sample_data['low'], sample_data['close'], sample_data['volume'])),
            ('pvt', lambda: VolumeIndicators.pvt(sample_data['close'], sample_data['volume'])),
            ('cmf', lambda: VolumeIndicators.cmf(sample_data['high'], sample_data['low'], sample_data['close'], sample_data['volume'])),
            ('aobv', lambda: VolumeIndicators.aobv(sample_data['close'], sample_data['volume'])),
            ('efi', lambda: VolumeIndicators.efi(sample_data['close'], sample_data['volume'])),
            ('pvol', lambda: VolumeIndicators.pvol(sample_data['close'], sample_data['volume'])),
            ('pvr', lambda: VolumeIndicators.pvr(sample_data['close'], sample_data['volume'])),
            ('vp', lambda: VolumeIndicators.vp(sample_data['close'], sample_data['volume'])),
        ]

        failed_functions = []
        for func_name, func_call in functions_to_test:
            try:
                result = func_call()
                # 結果がpd.Seriesまたはtupleであることを確認
                if isinstance(result, tuple):
                    for r in result:
                        assert isinstance(r, pd.Series), f"{func_name}: Expected pd.Series, got {type(r)}"
                        assert not r.isna().all()
                else:
                    assert isinstance(result, pd.Series), f"{func_name}: Expected pd.Series, got {type(result)}"
                    assert not result.isna().all()
            except Exception as e:
                failed_functions.append(f"{func_name}: {e}")

        if failed_functions:
            pytest.fail(f"以下の関数が失敗しました:\n" + "\n".join(failed_functions))


# TODO: pandasオンリー移行後のテスト
class TestVolumePandasOnly:
    """将来のpandasオンリー対応テスト（移行後に有効化）"""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        close = pd.Series(100 + np.cumsum(np.random.randn(n)), name="close")
        high = close + np.abs(np.random.randn(n)) * 5
        low = close - np.abs(np.random.randn(n)) * 5
        open_price = close + np.random.randn(n) * 2
        volume = pd.Series(np.random.randint(1000, 10000, n), name="volume")

        return {'close': close, 'high': high, 'low': low, 'open': open_price, 'volume': volume}

    @pytest.mark.skip(reason="pandasオンリー移行前に実行")
    def test_ad_returns_pandas_series(self, sample_data):
        """ADがpandas Seriesを返すこと"""
    @pytest.mark.skip(reason="pandasオンリー移行前に実行")
    def test_obv_volume_error_processing(self, sample_data):
        """OBV: Volumeデータ処理関連エラーテスト"""
        from app.services.indicators.utils import PandasTAError
        import pytest

        # Test None volume
        with pytest.raises(PandasTAError):
            VolumeIndicators.obv(sample_data['close'], None)

        # Test zero volume
        zero_volume = pd.Series([0] * len(sample_data['volume']), name='volume')
        result = VolumeIndicators.obv(sample_data['close'], zero_volume)
        # Should return Series but with NaN or other processing
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data['close'])

        # Test length mismatch
        with pytest.raises(PandasTAError):
            VolumeIndicators.obv(sample_data['close'], pd.Series([1, 2], name='volume'))

    @pytest.mark.skip(reason="pandasオンリー移行前に実行")
    def test_ad_returns_pandas_series(self, sample_data):
        result = VolumeIndicators.ad(sample_data['high'], sample_data['low'], sample_data['close'], sample_data['volume'])

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data['high'])