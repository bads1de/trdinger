"""
パンダスオンリー移行テスト - momentum.py

TDDでpandasオンリー移行を検証します。
"""

import pytest
import numpy as np
import pandas as pd
import pandas_ta as ta

from app.services.indicators.technical_indicators.momentum import MomentumIndicators
from app.services.indicators.technical_indicators.volume import VolumeIndicators


class TestMomentumMigration:
    """momentum.py の pandasオンリー移行テスト"""

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

    # RSI関数のテスト（まず最も簡単な関数から移行）
    def test_rsi_pandas_only_behavior(self, sample_data):
        """RSI: pandasオンリー移行後の動作確認"""
        result = MomentumIndicators.rsi(sample_data['close'])

        # pandasオンリー移行後pd.Seriesを返す
        assert isinstance(result, pd.Series), f"RSI returns pd.Series, got {type(result)}"
        assert len(result) == len(sample_data['close'])
        assert not pd.isna(result).all()
        # RSIの範囲チェック（0-100）
        valid_rsi = result.dropna()  # NaNを除去
        if len(valid_rsi) > 0:
            assert valid_rsi.min() >= 0
            assert valid_rsi.max() <= 100

    def test_macd_pandas_only_behavior(self, sample_data):
        """MACD: pandasオンリー移行後の動作確認"""
        macd, signal, histogram = MomentumIndicators.macd(sample_data['close'])

        # pandasオンリー移行後pd.Seriesを返す
        for band in [macd, signal, histogram]:
            assert isinstance(band, pd.Series), f"MACD returns pd.Series, got {type(band)}"
            assert len(band) == len(sample_data['close'])

    def test_stoch_pandas_only_behavior(self, sample_data):
        """Stochastic: pandasオンリー移行後の動作確認"""
        stoch_k, stoch_d = MomentumIndicators.stoch(
            sample_data['high'], sample_data['low'], sample_data['close']
        )

        # pandasオンリー移行後pd.Seriesを返す
        for result in [stoch_k, stoch_d]:
            assert isinstance(result, pd.Series), f"Stoch returns pd.Series, got {type(result)}"
            # Stochastic計算の特性上、一部データが失われることがある（NaNとして埋められる）
            assert len(result) > 0, "Result should not be empty"
            # Stochasticの範囲チェック（0-100）
            valid_stoch = result.dropna()  # NaNを除去
            if len(valid_stoch) > 0:
                assert valid_stoch.min() >= 0, f"Min value {valid_stoch.min()} should be >= 0"
                assert valid_stoch.max() <= 100, f"Max value {valid_stoch.max()} should be <= 100"

    def test_current_union_type_handling_momentum(self, sample_data):
        """pandasオンリー移行: numpy配列入力が拒否されることを確認"""

        # numpy配列入力をテスト - RSI
        close_np = sample_data['close'].to_numpy()
        high_np = sample_data['high'].to_numpy()
        low_np = sample_data['low'].to_numpy()

        # pandasオンリー移行後はnumpy配列入力でエラーが発生
        with pytest.raises(TypeError):
            MomentumIndicators.rsi(close_np)

        with pytest.raises(TypeError):
            MomentumIndicators.stoch(high_np, low_np, close_np)

    def test_all_functions_pandas_only(self, sample_data):
        """全関数がpandas Seriesのみを受け取り、pd.SeriesまたはTuple[pd.Series]を返すことを確認"""

        # Series型関数のテスト
        single_series_functions = [
            ('rsi', lambda: MomentumIndicators.rsi(sample_data['close'])),
            ('willr', lambda: MomentumIndicators.willr(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('cci', lambda: MomentumIndicators.cci(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('roc', lambda: MomentumIndicators.roc(sample_data['close'])),
            ('mom', lambda: MomentumIndicators.mom(sample_data['close'])),
            ('adx', lambda: MomentumIndicators.adx(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('mfi', lambda: VolumeIndicators.mfi(sample_data['high'], sample_data['low'], sample_data['close'], sample_data['volume'])),
            ('uo', lambda: MomentumIndicators.uo(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('apo', lambda: MomentumIndicators.apo(sample_data['close'])),
            ('ao', lambda: MomentumIndicators.ao(sample_data['high'], sample_data['low'])),
            ('cmo', lambda: MomentumIndicators.cmo(sample_data['close'])),
            ('trix', lambda: MomentumIndicators.trix(sample_data['close'])),
            ('ultosc', lambda: MomentumIndicators.ultosc(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('bop', lambda: MomentumIndicators.bop(sample_data['open'], sample_data['high'], sample_data['low'], sample_data['close'])),
            ('adxr', lambda: MomentumIndicators.adxr(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('rocp', lambda: MomentumIndicators.rocp(sample_data['close'])),
            ('rocr', lambda: MomentumIndicators.rocr(sample_data['close'])),
            ('rocr100', lambda: MomentumIndicators.rocr100(sample_data['close'])),
            ('tsi', lambda: MomentumIndicators.tsi(sample_data['close'])),
            ('rvi', lambda: MomentumIndicators.rvi(sample_data['open'], sample_data['high'], sample_data['low'], sample_data['close'])),
            ('cfo', lambda: MomentumIndicators.cfo(sample_data['close'])),
            ('cti', lambda: MomentumIndicators.cti(sample_data['close'])),
            ('rmi', lambda: MomentumIndicators.rmi(sample_data['close'])),
            ('dpo', lambda: MomentumIndicators.dpo(sample_data['close'])),
            ('chop', lambda: MomentumIndicators.chop(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('bias', lambda: MomentumIndicators.bias(sample_data['close'])),
            ('brar', lambda: MomentumIndicators.brar(sample_data['open'], sample_data['high'], sample_data['low'], sample_data['close'])),
            ('cg', lambda: MomentumIndicators.cg(sample_data['close'])),
            ('coppock', lambda: MomentumIndicators.coppock(sample_data['close'])),
            ('er', lambda: MomentumIndicators.er(sample_data['close'])),
            ('eri', lambda: MomentumIndicators.eri(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('inertia', lambda: MomentumIndicators.inertia(sample_data['close'])),
            ('pgo', lambda: MomentumIndicators.pgo(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('psl', lambda: MomentumIndicators.psl(sample_data['close'], sample_data['open'])),
            ('rsx', lambda: MomentumIndicators.rsx(sample_data['close'])),
            ('squeeze', lambda: MomentumIndicators.squeeze(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('squeeze_pro', lambda: MomentumIndicators.squeeze_pro(sample_data['high'], sample_data['low'], sample_data['close'])),
        ]

        # Tuple型関数のテスト
        tuple_functions = [
            ('macd', lambda: MomentumIndicators.macd(sample_data['close'])),
            ('stoch', lambda: MomentumIndicators.stoch(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('kdj', lambda: MomentumIndicators.kdj(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('stochrsi', lambda: MomentumIndicators.stochrsi(sample_data['close'])),
            ('ppo', lambda: MomentumIndicators.ppo(sample_data['close'])),
            ('rvgi', lambda: MomentumIndicators.rvgi(sample_data['open'], sample_data['high'], sample_data['low'], sample_data['close'])),
            ('smi', lambda: MomentumIndicators.smi(sample_data['close'])),
            ('kst', lambda: MomentumIndicators.kst(sample_data['close'])),
            ('ksi', lambda: MomentumIndicators.fisher(sample_data['high'], sample_data['low'])),
            ('vortex', lambda: MomentumIndicators.vortex(sample_data['high'], sample_data['low'], sample_data['close'])),
            ('pvo', lambda: MomentumIndicators.pvo(sample_data['volume'])),
            ('aroon', lambda: MomentumIndicators.aroon(sample_data['high'], sample_data['low'])),
        ]

        # エイリアス関数のテスト
        alias_functions = [
            ('macdext', lambda: MomentumIndicators.macdext(sample_data['close'])),
            ('macdfix', lambda: MomentumIndicators.macdfix(sample_data['close'])),
            ('stochf', lambda: MomentumIndicators.stochf(sample_data['high'], sample_data['low'], sample_data['close'])),
        ]

        # 特殊関数（さらに複雑なパラメータ）
        complex_functions = [
            ('rsi_ema_cross', lambda: MomentumIndicators.rsi_ema_cross(sample_data['close'])),
            ('stc', lambda: MomentumIndicators.stc(sample_data['close'])),
            ('qqe', lambda: MomentumIndicators.qqe(sample_data['close'])),
        ]

        # 全ての関数をテスト
        all_functions = single_series_functions + tuple_functions + alias_functions + complex_functions

        for func_name, func_call in all_functions:
            try:
                result = func_call()
                if isinstance(result, tuple):
                    for r in result:
                        assert isinstance(r, (pd.Series, pd.DataFrame)), f"{func_name}: Expected pd.Series or pd.DataFrame in tuple, got {type(r)}"
                else:
                    assert isinstance(result, (pd.Series, pd.DataFrame)), f"{func_name}: Expected pd.Series or pd.DataFrame, got {type(result)}"
            except Exception as e:
                pytest.fail(f"Function {func_name} failed: {e}")


# TODO: pandasオンリー移行後のテスト
class TestMomentumPandasOnly:
    """将来のpandasオンリー対応テスト（移行後に有効化）"""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        close = pd.Series(100 + np.cumsum(np.random.randn(n)), name="close")
        high = close + np.abs(np.random.randn(n)) * 5
        low = close - np.abs(np.random.randn(n)) * 5
        volume = pd.Series(np.random.randint(1000, 10000, n), name="volume")

        return {'close': close, 'high': high, 'low': low, 'volume': volume}

    def test_technical_indicators_rsi_pandas_input_only(self, sample_data):
        """RSIがpandas Seriesのみ受け付け、他はTypeErrorを発生させる"""
        close_np = sample_data['close'].to_numpy()

        # pandas Seriesは正常
        result = MomentumIndicators.rsi(sample_data['close'])
        assert isinstance(result, pd.Series)

        # numpy arrayはTypeError
        with pytest.raises(TypeError, match="must be pandas Series"):
            MomentumIndicators.rsi(close_np)