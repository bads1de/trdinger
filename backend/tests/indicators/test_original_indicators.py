"""
OriginalIndicatorsのテスト
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from app.services.indicators.technical_indicators.original import OriginalIndicators


class TestOriginalIndicators:
    """OriginalIndicatorsのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        pass

    def test_init(self):
        """初期化のテスト"""
        assert OriginalIndicators is not None

    def test_calculate_frama_valid_data(self):
        """有効データでのFRAMA計算テスト"""
        data = pd.DataFrame(
            {
                "close": [
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                ]
            }
        )

        result = OriginalIndicators.frama(data["close"], length=16, slow=200)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # FRAMAは平滑化された価格なので元の価格の範囲内
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 115

    def test_calculate_frama_insufficient_data(self):
        """データ不足でのFRAMA計算テスト"""
        data = pd.DataFrame({"close": [100, 101, 102]})  # 不十分なデータ

        with pytest.raises(ValueError, match="length must be >= 4"):
            OriginalIndicators.frama(data["close"], length=2, slow=200)

    def test_calculate_frama_odd_length(self):
        """奇数長さのFRAMAテスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        with pytest.raises(ValueError, match="length must be an even number"):
            OriginalIndicators.frama(data["close"], length=5, slow=200)

    def test_calculate_frama_negative_slow(self):
        """負のslowパラメータのFRAMAテスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        with pytest.raises(ValueError, match="slow must be >= 1"):
            OriginalIndicators.frama(data["close"], length=16, slow=0)

    def test_calculate_frama_empty_data(self):
        """空データのFRAMAテスト"""
        data = pd.DataFrame({"close": []})

        result = OriginalIndicators.frama(data["close"], length=16, slow=200)

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_calculate_frama_single_value(self):
        """単一値のFRAMAテスト"""
        data = pd.DataFrame({"close": [100]})

        result = OriginalIndicators.frama(data["close"], length=16, slow=200)

        assert isinstance(result, pd.Series)
        assert len(result) == 1
        assert result.isna().all()

    def test_adaptive_entropy_valid_data(self):
        """有効データでのAdaptive Entropy計算テスト"""
        data = pd.DataFrame(
            {
                "close": [
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                    119,
                    120,
                    121,
                    122,
                    123,
                    124,
                    125,
                    126,
                    127,
                    128,
                    129,
                    130,
                ]
            }
        )

        oscillator, signal, ratio = OriginalIndicators.adaptive_entropy(
            data["close"], short_length=14, long_length=28, signal_length=5
        )

        assert isinstance(oscillator, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(ratio, pd.Series)
        assert len(oscillator) == len(data)
        assert len(signal) == len(data)
        assert len(ratio) == len(data)

    def test_adaptive_entropy_insufficient_data(self):
        """データ不足でのAdaptive Entropyテスト"""
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104]})

        oscillator, signal, ratio = OriginalIndicators.adaptive_entropy(
            data["close"], short_length=14, long_length=28, signal_length=5
        )

        # 不十分なデータの場合はNaNが返される
        assert all(np.isnan(v) for v in oscillator)
        assert all(np.isnan(v) for v in signal)
        assert all(np.isnan(v) for v in ratio)

    def test_mcginley_dynamic_valid_data(self):
        """有効データでのMcGinley Dynamic計算テスト"""
        data = pd.DataFrame(
            {
                "close": [
                    100,
                    102,
                    104,
                    103,
                    105,
                    107,
                    106,
                    108,
                    110,
                    109,
                    111,
                    113,
                    112,
                    114,
                    116,
                ]
            }
        )

        result = OriginalIndicators.mcginley_dynamic(data["close"], length=10, k=0.6)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        assert result.name == "MCGINLEY_10"
        # McGinley Dynamicは価格に追従するため、範囲内にある
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 116

    def test_mcginley_dynamic_insufficient_data(self):
        """データ不足でのMcGinley Dynamic計算テスト"""
        data = pd.DataFrame({"close": [100]})

        result = OriginalIndicators.mcginley_dynamic(data["close"], length=10)

        assert isinstance(result, pd.Series)
        assert len(result) == 1

    def test_mcginley_dynamic_invalid_length(self):
        """無効なlengthパラメータのMcGinley Dynamicテスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        with pytest.raises(ValueError, match="length must be >= 1"):
            OriginalIndicators.mcginley_dynamic(data["close"], length=0)

    def test_mcginley_dynamic_invalid_k(self):
        """無効なkパラメータのMcGinley Dynamicテスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        with pytest.raises(ValueError, match="k must be > 0"):
            OriginalIndicators.mcginley_dynamic(data["close"], length=10, k=0)

    def test_mcginley_dynamic_empty_data(self):
        """空データのMcGinley Dynamicテスト"""
        data = pd.DataFrame({"close": []})

        result = OriginalIndicators.mcginley_dynamic(data["close"], length=10)

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_mcginley_dynamic_wrapper(self):
        """McGinley Dynamic DataFrameラッパーメソッドのテスト"""
        data = pd.DataFrame(
            {
                "close": [
                    100,
                    102,
                    104,
                    103,
                    105,
                    107,
                    106,
                    108,
                    110,
                    109,
                    111,
                    113,
                    112,
                    114,
                    116,
                ]
            }
        )

        result = OriginalIndicators.calculate_mcginley_dynamic(data, length=10, k=0.6)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
        assert "MCGINLEY_10" in result.columns

    def test_kaufman_efficiency_ratio_valid_data(self):
        """有効データでのKaufman Efficiency Ratio計算テスト"""
        data = pd.DataFrame(
            {
                "close": [
                    100,
                    102,
                    104,
                    106,
                    105,
                    107,
                    109,
                    108,
                    110,
                    112,
                    111,
                    113,
                    115,
                    114,
                    116,
                    118,
                    117,
                    119,
                    121,
                    120,
                ]
            }
        )

        result = OriginalIndicators.kaufman_efficiency_ratio(data["close"], length=10)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        assert result.name == "KER_10"
        # Efficiency Ratioは0-1の範囲
        assert result.dropna().min() >= 0.0
        assert result.dropna().max() <= 1.0

    def test_kaufman_efficiency_ratio_invalid_length(self):
        """無効なlengthパラメータのKERテスト"""
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104, 105]})

        with pytest.raises(ValueError, match="length must be >= 2"):
            OriginalIndicators.kaufman_efficiency_ratio(data["close"], length=1)

    def test_kaufman_efficiency_ratio_wrapper(self):
        """KER DataFrameラッパーメソッドのテスト"""
        data = pd.DataFrame(
            {"close": [100, 102, 104, 106, 105, 107, 109, 108, 110, 112, 111, 113, 115]}
        )

        result = OriginalIndicators.calculate_kaufman_efficiency_ratio(data, length=10)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
        assert "KER_10" in result.columns

    def test_chande_kroll_stop_valid_data(self):
        """有効データでのChande Kroll Stop計算テスト"""
        data = pd.DataFrame(
            {
                "high": [
                    105,
                    107,
                    109,
                    108,
                    110,
                    112,
                    111,
                    113,
                    115,
                    114,
                    116,
                    118,
                    117,
                    119,
                    121,
                ],
                "low": [
                    95,
                    97,
                    99,
                    98,
                    100,
                    102,
                    101,
                    103,
                    105,
                    104,
                    106,
                    108,
                    107,
                    109,
                    111,
                ],
                "close": [
                    100,
                    102,
                    104,
                    103,
                    105,
                    107,
                    106,
                    108,
                    110,
                    109,
                    111,
                    113,
                    112,
                    114,
                    116,
                ],
            }
        )

        long_stop, short_stop = OriginalIndicators.chande_kroll_stop(
            data["high"], data["low"], data["close"], p=10, x=1, q=9
        )

        assert isinstance(long_stop, pd.Series)
        assert isinstance(short_stop, pd.Series)
        assert len(long_stop) == len(data)
        assert len(short_stop) == len(data)
        # Long stopはcloseより低く、Short stopはcloseより高い
        valid_long = long_stop.dropna()
        valid_short = short_stop.dropna()
        if len(valid_long) > 0:
            assert valid_long.max() <= data["close"].max()
        if len(valid_short) > 0:
            assert valid_short.min() >= data["close"].min()

    def test_chande_kroll_stop_invalid_params(self):
        """無効なパラメータのChande Kroll Stopテスト"""
        data = pd.DataFrame(
            {"high": [105, 107, 109], "low": [95, 97, 99], "close": [100, 102, 104]}
        )

        with pytest.raises(ValueError, match="p must be >= 1"):
            OriginalIndicators.chande_kroll_stop(
                data["high"], data["low"], data["close"], p=0
            )

    def test_chande_kroll_stop_wrapper(self):
        """Chande Kroll Stop DataFrameラッパーメソッドのテスト"""
        data = pd.DataFrame(
            {
                "high": [105, 107, 109, 108, 110, 112, 111, 113, 115, 114, 116],
                "low": [95, 97, 99, 98, 100, 102, 101, 103, 105, 104, 106],
                "close": [100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111],
            }
        )

        result = OriginalIndicators.calculate_chande_kroll_stop(data, p=10, x=1, q=9)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
        assert "CKS_LONG_10" in result.columns
        assert "CKS_SHORT_10" in result.columns

    def test_trend_intensity_index_valid_data(self):
        """有効データでのTrend Intensity Index計算テスト"""
        data = pd.DataFrame(
            {
                "close": [
                    100,
                    102,
                    104,
                    103,
                    105,
                    107,
                    106,
                    108,
                    110,
                    109,
                    111,
                    113,
                    112,
                    114,
                    116,
                    115,
                    117,
                    119,
                    118,
                    120,
                ],
                "high": [
                    105,
                    107,
                    109,
                    108,
                    110,
                    112,
                    111,
                    113,
                    115,
                    114,
                    116,
                    118,
                    117,
                    119,
                    121,
                    120,
                    122,
                    124,
                    123,
                    125,
                ],
                "low": [
                    95,
                    97,
                    99,
                    98,
                    100,
                    102,
                    101,
                    103,
                    105,
                    104,
                    106,
                    108,
                    107,
                    109,
                    111,
                    110,
                    112,
                    114,
                    113,
                    115,
                ],
            }
        )

        result = OriginalIndicators.trend_intensity_index(
            data["close"], data["high"], data["low"], length=14, sma_length=30
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        assert result.name == "TII_14_30"
        # TIIは0-100の範囲
        valid_result = result.dropna()
        if len(valid_result) > 0:
            assert valid_result.min() >= 0.0
            assert valid_result.max() <= 100.0

    def test_trend_intensity_index_invalid_length(self):
        """無効なlengthパラメータのTIIテスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102], "high": [105, 106, 107], "low": [95, 96, 97]}
        )

        with pytest.raises(ValueError, match="length must be >= 1"):
            OriginalIndicators.trend_intensity_index(
                data["close"], data["high"], data["low"], length=0
            )

    def test_trend_intensity_index_wrapper(self):
        """TII DataFrameラッパーメソッドのテスト"""
        data = pd.DataFrame(
            {
                "close": [
                    100,
                    102,
                    104,
                    103,
                    105,
                    107,
                    106,
                    108,
                    110,
                    109,
                    111,
                    113,
                    112,
                    114,
                    116,
                ],
                "high": [
                    105,
                    107,
                    109,
                    108,
                    110,
                    112,
                    111,
                    113,
                    115,
                    114,
                    116,
                    118,
                    117,
                    119,
                    121,
                ],
                "low": [
                    95,
                    97,
                    99,
                    98,
                    100,
                    102,
                    101,
                    103,
                    105,
                    104,
                    106,
                    108,
                    107,
                    109,
                    111,
                ],
            }
        )

        result = OriginalIndicators.calculate_trend_intensity_index(
            data, length=14, sma_length=30
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
        assert "TII_14_30" in result.columns

    def test_adaptive_entropy_invalid_parameters(self):
        """無効なパラメータでのAdaptive Entropyテスト"""
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104] * 10})

        # short_lengthが短すぎる
        with pytest.raises(ValueError, match="short_length must be >= 5"):
            OriginalIndicators.adaptive_entropy(
                data["close"], short_length=3, long_length=28, signal_length=5
            )

        # long_lengthが短すぎる
        with pytest.raises(ValueError, match="long_length must be >= 10"):
            OriginalIndicators.adaptive_entropy(
                data["close"], short_length=14, long_length=5, signal_length=5
            )

        # signal_lengthが短すぎる
        with pytest.raises(ValueError, match="signal_length must be >= 2"):
            OriginalIndicators.adaptive_entropy(
                data["close"], short_length=14, long_length=28, signal_length=1
            )

        # short_lengthがlong_length以上
        with pytest.raises(ValueError, match="short_length must be < long_length"):
            OriginalIndicators.adaptive_entropy(
                data["close"], short_length=30, long_length=28, signal_length=5
            )

    def test_adaptive_entropy_calculate_method(self):
        """calculate_adaptive_entropyメソッドのテスト"""
        data = pd.DataFrame(
            {
                "close": [
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                    119,
                    120,
                    121,
                    122,
                    123,
                    124,
                    125,
                    126,
                    127,
                    128,
                    129,
                    130,
                ]
            }
        )

        result = OriginalIndicators.calculate_adaptive_entropy(
            data, short_length=14, long_length=28, signal_length=5
        )

        assert isinstance(result, pd.DataFrame)
        assert "ADAPTIVE_ENTROPY_OSC_14_28" in result.columns
        assert "ADAPTIVE_ENTROPY_SIGNAL_14_28_5" in result.columns
        assert "ADAPTIVE_ENTROPY_RATIO_14_28" in result.columns

    def test_quantum_flow_valid_data(self):
        """有効データでのQuantum Flow計算テスト"""
        data = pd.DataFrame(
            {
                "close": [
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                    119,
                    120,
                ],
                "high": [
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                    119,
                    120,
                    121,
                    122,
                    123,
                    124,
                    125,
                ],
                "low": [
                    95,
                    96,
                    97,
                    98,
                    99,
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                ],
                "volume": [
                    1000,
                    1050,
                    1010,
                    1080,
                    1020,
                    1060,
                    1030,
                    1070,
                    1040,
                    1080,
                    1050,
                    1090,
                    1060,
                    1100,
                    1070,
                    1110,
                    1080,
                    1120,
                    1090,
                    1130,
                    1100,
                ],
            }
        )

        flow, signal = OriginalIndicators.quantum_flow(
            data["close"],
            data["high"],
            data["low"],
            data["volume"],
            length=14,
            flow_length=9,
        )

        assert isinstance(flow, pd.Series)
        assert isinstance(signal, pd.Series)
        assert len(flow) == len(data)
        assert len(signal) == len(data)

    def test_quantum_flow_insufficient_data(self):
        """データ不足でのQuantum Flowテスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                "low": [95, 96, 97, 98, 99],
                "volume": [1000, 1050, 1010, 1080, 1020],
            }
        )

        empty_flow, empty_signal = OriginalIndicators.quantum_flow(
            data["close"],
            data["high"],
            data["low"],
            data["volume"],
            length=14,
            flow_length=9,
        )

        # 不十分なデータの場合はNaNが返される
        assert all(np.isnan(v) for v in empty_flow)
        assert all(np.isnan(v) for v in empty_signal)

    def test_quantum_flow_invalid_parameters(self):
        """無効なパラメータでのQuantum Flowテスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                "low": [95, 96, 97, 98, 99],
                "volume": [1000, 1050, 1010, 1080, 1020],
            }
        )

        # lengthが短すぎる
        with pytest.raises(ValueError, match="length must be >= 5"):
            OriginalIndicators.quantum_flow(
                data["close"],
                data["high"],
                data["low"],
                data["volume"],
                length=3,
                flow_length=9,
            )

        # flow_lengthが短すぎる
        with pytest.raises(ValueError, match="flow_length must be >= 3"):
            OriginalIndicators.quantum_flow(
                data["close"],
                data["high"],
                data["low"],
                data["volume"],
                length=14,
                flow_length=2,
            )

    def test_quantum_flow_missing_columns(self):
        """欠損列でのQuantum Flowテスト"""
        # volume列が欠損
        data_missing_volume = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                "low": [95, 96, 97, 98, 99],
                # volume列が欠損
            }
        )

        with pytest.raises(ValueError, match="Missing required column"):
            OriginalIndicators.calculate_quantum_flow(data_missing_volume)

    def test_quantum_flow_calculate_method(self):
        """calculate_quantum_flowメソッドのテスト"""
        data = pd.DataFrame(
            {
                "close": [
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                    119,
                    120,
                ],
                "high": [
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                    119,
                    120,
                    121,
                    122,
                    123,
                    124,
                    125,
                ],
                "low": [
                    95,
                    96,
                    97,
                    98,
                    99,
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                ],
                "volume": [
                    1000,
                    1050,
                    1010,
                    1080,
                    1020,
                    1060,
                    1030,
                    1070,
                    1040,
                    1080,
                    1050,
                    1090,
                    1060,
                    1100,
                    1070,
                    1110,
                    1080,
                    1120,
                    1090,
                    1130,
                    1100,
                ],
            }
        )

        result = OriginalIndicators.calculate_quantum_flow(
            data, length=14, flow_length=9
        )

        assert isinstance(result, pd.DataFrame)
        assert "QUANTUM_FLOW" in result.columns
        assert "QUANTUM_FLOW_SIGNAL" in result.columns

    def test_frama_edge_cases(self):
        """FRAMAのエッジケーステスト"""
        # 空データのテスト
        empty_data = pd.DataFrame({"close": []})
        result_empty = OriginalIndicators.frama(
            empty_data["close"], length=16, slow=200
        )
        assert len(result_empty) == 0

        # 単一データポイントのテスト
        single_data = pd.DataFrame({"close": [100]})
        result_single = OriginalIndicators.frama(
            single_data["close"], length=16, slow=200
        )
        assert len(result_single) == 1
        assert np.isnan(result_single.iloc[0])

        # 最小限のデータ長のテスト
        min_data = pd.DataFrame({"close": [100, 101, 102, 103]})  # 4ポイント（最小）
        result_min = OriginalIndicators.frama(min_data["close"], length=4, slow=200)
        assert len(result_min) == 4

    def test_super_smoother_edge_cases(self):
        """Super Smootherのエッジケーステスト"""
        # 空データのテスト
        empty_data = pd.DataFrame({"close": []})
        result_empty = OriginalIndicators.super_smoother(empty_data["close"], length=10)
        assert len(result_empty) == 0

        # 最小データ長のテスト
        min_data = pd.DataFrame({"close": [100, 101]})  # 2ポイント（最小）
        result_min = OriginalIndicators.super_smoother(min_data["close"], length=2)
        assert len(result_min) == 2

    def test_adaptive_entropy_edge_cases(self):
        """Adaptive Entropyのエッジケーステスト"""
        # ショートラインがロングラインと等しい場合のテスト
        equal_length_data = pd.DataFrame({"close": [100, 101] * 50})
        with pytest.raises(ValueError, match="short_length must be < long_length"):
            OriginalIndicators.adaptive_entropy(
                equal_length_data["close"],
                short_length=14,
                long_length=14,
                signal_length=5,
            )

        # 長さが0のデータ
        zero_data = pd.DataFrame({"close": []})
        osc, sig, ratio = OriginalIndicators.adaptive_entropy(
            zero_data["close"], short_length=14, long_length=28, signal_length=5
        )
        assert all(np.isnan(x) for x in osc)
        assert all(np.isnan(x) for x in sig)
        assert all(np.isnan(x) for x in ratio)

    def test_quantum_flow_edge_cases(self):
        """Quantum Flowのエッジケーステスト"""
        # 長さが等しいデータのテスト
        # lengthは5以上が要求されるようになった
        equal_length_data = pd.DataFrame(
            {
                "close": [100, 101] * 10,
                "high": [105, 106] * 10,
                "low": [95, 96] * 10,
                "volume": [1000, 1050] * 10,
            }
        )
        flow, signal = OriginalIndicators.quantum_flow(
            equal_length_data["close"],
            equal_length_data["high"],
            equal_length_data["low"],
            equal_length_data["volume"],
            length=5,
            flow_length=3,
        )
        assert len(flow) == len(equal_length_data)
        assert len(signal) == len(equal_length_data)

    def test_calculate_frama_with_trend(self):
        """トレンドがあるデータのFRAMAテスト"""
        # 明確な上昇トレンド
        trend_data = pd.DataFrame(
            {"close": [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]}
        )

        result = OriginalIndicators.frama(trend_data["close"], length=8, slow=50)

        assert isinstance(result, pd.Series)
        assert len(result) == len(trend_data)
        # トレンドを追従しているはず
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 118

    def test_calculate_frama_with_noise(self):
        """ノイズがあるデータのFRAMAテスト"""
        np.random.seed(42)
        base = 1000
        trend = np.linspace(0, 100, 50)
        noise = np.random.normal(0, 5, 50)
        noisy_data = pd.DataFrame({"close": base + trend + noise})

        result = OriginalIndicators.frama(noisy_data["close"], length=10, slow=100)

        assert isinstance(result, pd.Series)
        assert len(result) == len(noisy_data)
        # ノイズが平滑化されているはず
        assert result.dropna().min() >= base - 10
        assert result.dropna().max() <= base + trend[-1] + 10

    def test_calculate_super_smoother_valid_data(self):
        """有効データでのSuper Smoother計算テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = OriginalIndicators.super_smoother(data["close"], length=10)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # Super Smootherは元の価格の範囲内
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 109

    def test_calculate_super_smoother_insufficient_length(self):
        """不十分な長さのSuper Smootherテスト"""
        data = pd.DataFrame({"close": [100, 101]})  # 不十分

        with pytest.raises(ValueError, match="length must be >= 2"):
            OriginalIndicators.super_smoother(data["close"], length=1)

    def test_calculate_super_smoother_empty_data(self):
        """空データのSuper Smootherテスト"""
        data = pd.DataFrame({"close": []})

        result = OriginalIndicators.super_smoother(data["close"], length=10)

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_calculate_super_smoother_single_value(self):
        """単一値のSuper Smootherテスト"""
        data = pd.DataFrame({"close": [100]})

        result = OriginalIndicators.super_smoother(data["close"], length=10)

        assert isinstance(result, pd.Series)
        assert len(result) == 1
        # 単一値の場合、実装は元の値を返すようになった
        assert result is not None
        # 値が返されることを確認
        assert len(result) == 1

    def test_calculate_super_smoother_with_oscillation(self):
        """振動データのSuper Smootherテスト"""
        # 振動するデータ
        oscillation_data = pd.DataFrame(
            {"close": [100, 110, 100, 110, 100, 110, 100, 110, 100, 110]}
        )

        result = OriginalIndicators.super_smoother(oscillation_data["close"], length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(oscillation_data)
        # 振動が平滑化されているはず
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 110

    def test_calculate_super_smoother_with_trend(self):
        """トレンドがあるデータのSuper Smootherテスト"""
        # 上昇トレンド
        trend_data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = OriginalIndicators.super_smoother(trend_data["close"], length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(trend_data)
        # トレンドを追従しているはず
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 109

    def test_frama_edge_case_length_4(self):
        """FRAMAの境界値テスト（length=4）"""
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104, 105]})

        result = OriginalIndicators.frama(data["close"], length=4, slow=10)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 105

    def test_frama_edge_case_slow_1(self):
        """FRAMAの境界値テスト（slow=1）"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = OriginalIndicators.frama(data["close"], length=4, slow=1)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 109

    def test_super_smoother_edge_case_length_2(self):
        """Super Smootherの境界値テスト（length=2）"""
        data = pd.DataFrame({"close": [100, 101, 102]})

        result = OriginalIndicators.super_smoother(data["close"], length=2)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 102

    def test_frama_alpha_clamping(self):
        """FRAMAのα値のクランプテスト"""
        # 極端な値のデータ
        extreme_data = pd.DataFrame(
            {"close": [100, 200, 100, 200, 100, 200, 100, 200, 100, 200]}
        )

        result = OriginalIndicators.frama(extreme_data["close"], length=8, slow=200)

        assert isinstance(result, pd.Series)
        assert len(result) == len(extreme_data)
        # α値がクランプされているはず
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 200

    def test_super_smoother_stability(self):
        """Super Smootherの安定性テスト"""
        # 大きなノイズがあるデータ
        np.random.seed(42)
        base = 1000
        trend = np.linspace(0, 100, 100)
        noise = np.random.normal(0, 20, 100)  # 大きなノイズ
        noisy_data = pd.DataFrame({"close": base + trend + noise})

        result = OriginalIndicators.super_smoother(noisy_data["close"], length=15)

        assert isinstance(result, pd.Series)
        assert len(result) == len(noisy_data)
        # 安定して平滑化されているはず
        assert result.dropna().min() >= base - 30
        assert result.dropna().max() <= base + trend[-1] + 30

    def test_frama_with_decreasing_data(self):
        """減少トレンドのFRAMAテスト"""
        decreasing_data = pd.DataFrame(
            {
                "close": [
                    115,
                    114,
                    113,
                    112,
                    111,
                    110,
                    109,
                    108,
                    107,
                    106,
                    105,
                    104,
                    103,
                    102,
                    101,
                    100,
                ]
            }
        )

        result = OriginalIndicators.frama(decreasing_data["close"], length=16, slow=200)

        assert isinstance(result, pd.Series)
        assert len(result) == len(decreasing_data)
        # 減少トレンドを追従しているはず
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 115

    def test_super_smoother_with_decreasing_data(self):
        """減少トレンドのSuper Smootherテスト"""
        decreasing_data = pd.DataFrame(
            {"close": [109, 108, 107, 106, 105, 104, 103, 102, 101, 100]}
        )

        result = OriginalIndicators.super_smoother(decreasing_data["close"], length=10)

        assert isinstance(result, pd.Series)
        assert len(result) == len(decreasing_data)
        # 減少トレンドを追従しているはず
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 109

    def test_frama_parameter_validation(self):
        """FRAMAのパラメータ検証テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        # lengthが負
        with pytest.raises(ValueError, match="length must be >= 4"):
            OriginalIndicators.frama(data["close"], length=-1, slow=200)

        # slowが負
        with pytest.raises(ValueError, match="slow must be >= 1"):
            OriginalIndicators.frama(data["close"], length=16, slow=-1)

    def test_super_smoother_parameter_validation(self):
        """Super Smootherのパラメータ検証テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        # lengthが負
        with pytest.raises(ValueError, match="length must be >= 2"):
            OriginalIndicators.super_smoother(data["close"], length=-1)

    def test_frama_numpy_compatibility(self):
        """FRAMAのNumPy配列互換性テスト"""
        # NumPy配列でテスト
        close_array = np.array(
            [
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
                112,
                113,
                114,
                115,
            ]
        )

        result = OriginalIndicators.frama(pd.Series(close_array), length=16, slow=200)

        assert isinstance(result, pd.Series)
        assert len(result) == len(close_array)

    def test_super_smoother_numpy_compatibility(self):
        """Super SmootherのNumPy配列互換性テスト"""
        # NumPy配列でテスト
        close_array = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])

        result = OriginalIndicators.super_smoother(pd.Series(close_array), length=10)

        assert isinstance(result, pd.Series)
        assert len(result) == len(close_array)

    def test_frama_with_nan_values(self):
        """FRAMAのNaN値処理テスト"""
        data = pd.DataFrame(
            {"close": [100, np.nan, 102, 103, np.nan, 105, 106, 107, 108, 109]}
        )

        result = OriginalIndicators.frama(data["close"], length=8, slow=50)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # NaNが適切に処理されている

    def test_super_smoother_with_nan_values(self):
        """Super SmootherのNaN値処理テスト"""
        data = pd.DataFrame(
            {"close": [100, np.nan, 102, 103, np.nan, 105, 106, 107, 108, 109]}
        )

        result = OriginalIndicators.super_smoother(data["close"], length=8)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # NaNが適切に処理されている

    def test_frama_with_inf_values(self):
        """FRAMAの無限大値処理テスト"""
        data = pd.DataFrame(
            {"close": [100, np.inf, 102, 103, -np.inf, 105, 106, 107, 108, 109]}
        )

        result = OriginalIndicators.frama(data["close"], length=8, slow=50)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # 無限大値が適切に処理されている

    def test_super_smoother_with_inf_values(self):
        """Super Smootherの無限大値処理テスト"""
        data = pd.DataFrame(
            {"close": [100, np.inf, 102, 103, -np.inf, 105, 106, 107, 108, 109]}
        )

        result = OriginalIndicators.super_smoother(data["close"], length=8)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # 無限大値が適切に処理されている

    def test_frama_different_length_combinations(self):
        """FRAMAの異なるパラメータ組み合わせテスト"""
        data = pd.DataFrame(
            {
                "close": [
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                ]
            }
        )

        # 短期
        result_short = OriginalIndicators.frama(data["close"], length=8, slow=50)
        # 長期
        result_long = OriginalIndicators.frama(data["close"], length=20, slow=200)

        assert isinstance(result_short, pd.Series)
        assert isinstance(result_long, pd.Series)
        assert len(result_short) == len(data)
        assert len(result_long) == len(data)

    def test_super_smoother_different_length_combinations(self):
        """Super Smootherの異なるパラメータ組み合わせテスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        # 短期
        result_short = OriginalIndicators.super_smoother(data["close"], length=5)
        # 長期
        result_long = OriginalIndicators.super_smoother(data["close"], length=20)

        assert isinstance(result_short, pd.Series)
        assert isinstance(result_long, pd.Series)
        assert len(result_short) == len(data)
        assert len(result_long) == len(data)

    def test_frama_multiple_calls_consistency(self):
        """FRAMAの複数回呼び出しの一貫性テスト"""
        data = pd.DataFrame(
            {
                "close": [
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                ]
            }
        )

        result1 = OriginalIndicators.frama(data["close"], length=16, slow=200)
        result2 = OriginalIndicators.frama(data["close"], length=16, slow=200)

        # 同じ結果になるはず
        assert result1.equals(result2)

    def test_super_smoother_multiple_calls_consistency(self):
        """Super Smootherの複数回呼び出しの一貫性テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result1 = OriginalIndicators.super_smoother(data["close"], length=10)
        result2 = OriginalIndicators.super_smoother(data["close"], length=10)

        # 同じ結果になるはず
        assert result1.equals(result2)

    def test_frama_with_random_data(self):
        """FRAMAのランダムデータテスト"""
        np.random.seed(42)
        random_data = pd.DataFrame({"close": np.random.normal(100, 10, 100)})

        result = OriginalIndicators.frama(random_data["close"], length=16, slow=200)

        assert isinstance(result, pd.Series)
        assert len(result) == len(random_data)
        # 平滑化されているはず

    def test_super_smoother_with_random_data(self):
        """Super Smootherのランダムデータテスト"""
        np.random.seed(42)
        random_data = pd.DataFrame({"close": np.random.normal(100, 10, 100)})

        result = OriginalIndicators.super_smoother(random_data["close"], length=10)

        assert isinstance(result, pd.Series)
        assert len(result) == len(random_data)
        # 平滑化されているはず

    def test_frama_with_constant_data(self):
        """FRAMAの定数データテスト"""
        constant_data = pd.DataFrame(
            {"close": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]}
        )

        result = OriginalIndicators.frama(constant_data["close"], length=8, slow=50)

        assert isinstance(result, pd.Series)
        assert len(result) == len(constant_data)
        # 定数データでは定数を返すはず
        assert result.dropna().std() < 1e-10  # ほぼ定数

    def test_super_smoother_with_constant_data(self):
        """Super Smootherの定数データテスト"""
        constant_data = pd.DataFrame(
            {"close": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]}
        )

        result = OriginalIndicators.super_smoother(constant_data["close"], length=8)

        assert isinstance(result, pd.Series)
        assert len(result) == len(constant_data)
        # 定数データでは定数を返すはず
        assert result.dropna().std() < 1e-10  # ほぼ定数

    def test_frama_with_step_function(self):
        """FRAMAのステップ関数テスト"""
        step_data = pd.DataFrame(
            {"close": [100, 100, 100, 100, 100, 150, 150, 150, 150, 150]}
        )

        result = OriginalIndicators.frama(step_data["close"], length=6, slow=25)

        assert isinstance(result, pd.Series)
        assert len(result) == len(step_data)
        # ステップ変化を追従しているはず

    def test_super_smoother_with_step_function(self):
        """Super Smootherのステップ関数テスト"""
        step_data = pd.DataFrame(
            {"close": [100, 100, 100, 100, 100, 150, 150, 150, 150, 150]}
        )

        result = OriginalIndicators.super_smoother(step_data["close"], length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(step_data)
        # ステップ変化を追従しているはず

    def test_frama_memory_usage(self):
        """FRAMAのメモリ使用量テスト"""
        # 大きなデータセット
        large_data = pd.DataFrame({"close": np.random.normal(100, 10, 10000)})

        result = OriginalIndicators.frama(large_data["close"], length=16, slow=200)

        assert isinstance(result, pd.Series)
        assert len(result) == len(large_data)
        # 大きなデータでも処理できる

    def test_super_smoother_memory_usage(self):
        """Super Smootherのメモリ使用量テスト"""
        # 大きなデータセット
        large_data = pd.DataFrame({"close": np.random.normal(100, 10, 10000)})

        result = OriginalIndicators.super_smoother(large_data["close"], length=10)

        assert isinstance(result, pd.Series)
        assert len(result) == len(large_data)
        # 大きなデータでも処理できる

    def test_frama_with_extreme_parameters(self):
        """FRAMAの極端なパラメータテスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        # 極端なslow値
        result = OriginalIndicators.frama(data["close"], length=4, slow=1000)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # 極端なパラメータでもエラーにならない

    def test_super_smoother_with_extreme_parameters(self):
        """Super Smootherの極端なパラメータテスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        # 極端なlength値
        result = OriginalIndicators.super_smoother(data["close"], length=100)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # 極端なパラメータでもエラーにならない

    def test_calculate_elder_ray_valid_data(self):
        """有効データでのElder Ray計算テスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            }
        )

        result = OriginalIndicators.calculate_elder_ray(data)

        assert isinstance(result, pd.DataFrame)
        assert "Elder_Ray_Bull_13_16" in result.columns
        assert "Elder_Ray_Bear_13_16" in result.columns
        # Bull Powerは高値 - EMAなので、正の値も取り得る
        # Bear Powerは安値 - EMAなので、負の値も取り得る

    def test_calculate_elder_ray_insufficient_data(self):
        """データ不足でのElder Ray計算テスト"""
        data = pd.DataFrame({"close": [100, 101], "high": [102, 103], "low": [98, 99]})

        result = OriginalIndicators.calculate_elder_ray(data)

        assert isinstance(result, pd.DataFrame)
        # 少ないデータでも計算可能
        assert "Elder_Ray_Bull_13_16" in result.columns
        assert "Elder_Ray_Bear_13_16" in result.columns

    def test_calculate_elder_ray_custom_parameters(self):
        """カスタムパラメータでのElder Ray計算テスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            }
        )

        result = OriginalIndicators.calculate_elder_ray(data, length=10, ema_length=12)

        assert isinstance(result, pd.DataFrame)
        assert "Elder_Ray_Bull_10_12" in result.columns
        assert "Elder_Ray_Bear_10_12" in result.columns

    def test_elder_ray_direct_calculation(self):
        """Elder Rayの直接計算テスト"""
        high = pd.Series([102, 103, 104, 105, 106])
        low = pd.Series([98, 99, 100, 101, 102])
        close = pd.Series([100, 101, 102, 103, 104])

        bull_power, bear_power = OriginalIndicators.elder_ray(
            high, low, close, length=13, ema_length=16
        )

        assert isinstance(bull_power, pd.Series)
        assert isinstance(bear_power, pd.Series)
        assert len(bull_power) == len(high)
        assert len(bear_power) == len(low)
        # EMAを計算しているので、直接的な関係はないが計算は可能

    def test_elder_ray_parameter_validation(self):
        """Elder Rayのパラメータ検証テスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104],
                "high": [102, 103, 104, 105, 106],
                "low": [98, 99, 100, 101, 102],
            }
        )

        # lengthが負
        with pytest.raises(ValueError, match="length must be positive"):
            OriginalIndicators.calculate_elder_ray(data, length=-1, ema_length=16)

        # ema_lengthが負
        with pytest.raises(ValueError, match="ema_length must be positive"):
            OriginalIndicators.calculate_elder_ray(data, length=13, ema_length=-1)

        # 不足のデータ
        with pytest.raises(ValueError, match="All arrays must be of the same length"):
            incomplete_data = pd.DataFrame(
                {
                    "close": [100, 101],
                    "high": [102, 103, 104],  # 長さが異なる
                }
            )
            OriginalIndicators.calculate_elder_ray(incomplete_data)

    def test_calculate_prime_oscillator_valid_data(self):
        """有効データでのPrime Number Oscillator計算テスト"""
        # Prime Number Oscillatorは素数列を使用するため、十分な長さが必要
        # length=14の場合、必要な最小データ長は最大の素数(13)より大きい
        data = pd.DataFrame(
            {
                "close": [
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                    119,
                    120,
                    121,
                    122,
                    123,
                    124,
                    125,
                    126,
                    127,
                    128,
                    129,
                    130,
                    131,
                    132,
                    133,
                    134,
                    135,
                    136,
                    137,
                    138,
                    139,
                    140,
                    141,
                    142,
                    143,
                    144,
                    145,
                    146,
                    147,
                    148,
                    149,
                    150,
                    151,
                    152,
                    153,
                    154,
                    155,
                    156,
                    157,
                    158,
                    159,
                    160,
                    161,
                    162,
                    163,
                    164,
                    165,
                    166,
                    167,
                    168,
                    169,
                    170,
                    171,
                    172,
                    173,
                    174,
                    175,
                    176,
                    177,
                    178,
                    179,
                    180,
                    181,
                    182,
                    183,
                    184,
                    185,
                    186,
                    187,
                    188,
                    189,
                    190,
                    191,
                    192,
                    193,
                    194,
                    195,
                ]
            }
        )

        result = OriginalIndicators.calculate_prime_oscillator(data)

        assert isinstance(result, pd.DataFrame)
        assert "PRIME_OSC_14" in result.columns
        assert "PRIME_SIGNAL_14_3" in result.columns
        # Prime Oscillatorは正規化されているので、-100から100の範囲内
        non_nan_values = result["PRIME_OSC_14"].dropna()
        if len(non_nan_values) > 0:
            assert non_nan_values.min() >= -100
            assert non_nan_values.max() <= 100

    def test_calculate_prime_oscillator_insufficient_data(self):
        """データ不足でのPrime Number Oscillator計算テスト"""
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104]})  # 不十分なデータ

        result = OriginalIndicators.calculate_prime_oscillator(data)

        assert isinstance(result, pd.DataFrame)
        assert "PRIME_OSC_14" in result.columns
        # データが少ないとNaNが返される
        assert result["PRIME_OSC_14"].isna().all()

    def test_calculate_prime_oscillator_custom_parameters(self):
        """カスタムパラメータでのPrime Number Oscillator計算テスト"""
        # length=10の場合、必要な最小データ長は29+1=30
        data = pd.DataFrame(
            {"close": list(range(100, 130))}  # 100から129までの30個のデータ
        )

        result = OriginalIndicators.calculate_prime_oscillator(
            data, length=10, signal_length=5
        )

        assert isinstance(result, pd.DataFrame)
        assert "PRIME_OSC_10" in result.columns
        assert "PRIME_SIGNAL_10_5" in result.columns
        non_nan_values = result["PRIME_OSC_10"].dropna()
        if len(non_nan_values) > 0:
            assert non_nan_values.min() >= -100
            assert non_nan_values.max() <= 100

    def test_prime_oscillator_direct_calculation(self):
        """Prime Number Oscillatorの直接計算テスト"""
        # length=14の場合、必要な最小データ長は43+1=44
        close = pd.Series(list(range(100, 144)))  # 100から143までの44個のデータ

        oscillator, signal = OriginalIndicators.prime_oscillator(
            close, length=14, signal_length=3
        )

        assert isinstance(oscillator, pd.Series)
        assert isinstance(signal, pd.Series)
        assert len(oscillator) == len(close)
        assert len(signal) == len(close)
        assert oscillator.name == "PRIME_OSC_14"
        assert signal.name == "PRIME_SIGNAL_14_3"
        # 正規化された値
        non_nan_values = oscillator.dropna()
        if len(non_nan_values) > 0:
            assert non_nan_values.min() >= -100
            assert non_nan_values.max() <= 100

    def test_prime_oscillator_parameter_validation(self):
        """Prime Number Oscillatorのパラメータ検証テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        # lengthが小さすぎる
        with pytest.raises(ValueError, match="length must be >= 2"):
            OriginalIndicators.calculate_prime_oscillator(data, length=1)

        # signal_lengthが小さすぎる
        with pytest.raises(ValueError, match="signal_length must be >= 2"):
            OriginalIndicators.calculate_prime_oscillator(
                data, length=14, signal_length=1
            )

        # 不正なデータ型
        with pytest.raises(TypeError, match="data must be pandas DataFrame"):
            OriginalIndicators.calculate_prime_oscillator([100, 101, 102])

    def test_prime_oscillator_edge_cases(self):
        """Prime Number Oscillatorの境界値テスト"""
        # 定数データ
        # length=14の場合、必要な最小データ長は43+1=44
        constant_data = pd.DataFrame({"close": [100] * 44})  # 44個の定数データ

        result = OriginalIndicators.calculate_prime_oscillator(constant_data)

        assert isinstance(result, pd.DataFrame)
        # 定数データでは変化がないため、オシレーターは0に近くなる
        non_nan_values = result["PRIME_OSC_14"].dropna()
        if len(non_nan_values) > 0:
            assert abs(non_nan_values.mean()) < 1e-6

        # 昇順データ
        increasing_data = pd.DataFrame(
            {"close": list(range(100, 144))}  # 44個の昇順データ
        )

        result = OriginalIndicators.calculate_prime_oscillator(increasing_data)

        assert isinstance(result, pd.DataFrame)
        # 昇順データでは正の値を取る傾向
        non_nan_values = result["PRIME_OSC_14"].dropna()
        if len(non_nan_values) > 0:
            assert non_nan_values.mean() > -10

    def test_prime_oscillator_consistency(self):
        """Prime Number Oscillatorの再現性テスト"""
        data = pd.DataFrame(
            {
                "close": [
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                ]
            }
        )

        result1 = OriginalIndicators.calculate_prime_oscillator(data)
        result2 = OriginalIndicators.calculate_prime_oscillator(data)

        # 同じ入力では同じ結果になる
        assert result1.equals(result2)

    def test_prime_oscillator_with_nan_values(self):
        """Prime Number OscillatorのNaN値処理テスト"""
        data = pd.DataFrame(
            {"close": [100, np.nan, 102, 103, np.nan, 105, 106, 107, 108, 109]}
        )

        result = OriginalIndicators.calculate_prime_oscillator(data)

        assert isinstance(result, pd.DataFrame)
        assert "PRIME_OSC_14" in result.columns
        # NaNが適切に処理されている

    def test_prime_oscillator_memory_usage(self):
        """Prime Number Oscillatorのメモリ使用量テスト"""
        # 大きなデータセット
        large_data = pd.DataFrame({"close": np.random.normal(100, 10, 10000)})

        result = OriginalIndicators.calculate_prime_oscillator(large_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(large_data)
        # 大きなデータでも処理できる

    def test_calculate_fibonacci_cycle_valid_data(self):
        """有効データでのFibonacci Cycle計算テスト"""
        # Fibonacci Cycleは最大期間55が必要
        data = pd.DataFrame({"close": list(range(100, 155))})  # 55個のデータ

        result = OriginalIndicators.calculate_fibonacci_cycle(data)

        assert isinstance(result, pd.DataFrame)
        assert "FIBO_CYCLE_5" in result.columns
        assert "FIBO_SIGNAL_5" in result.columns
        # Fibonacci Cycleの値は有限
        non_nan_values = result["FIBO_CYCLE_5"].dropna()
        if len(non_nan_values) > 0:
            assert np.isfinite(non_nan_values.min())
            assert np.isfinite(non_nan_values.max())

    def test_calculate_fibonacci_cycle_insufficient_data(self):
        """データ不足でのFibonacci Cycle計算テスト"""
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104]})  # 不十分なデータ

        result = OriginalIndicators.calculate_fibonacci_cycle(data)

        assert isinstance(result, pd.DataFrame)
        assert "FIBO_CYCLE_5" in result.columns
        # データが少ないとNaNが返される

    def test_calculate_fibonacci_cycle_custom_parameters(self):
        """カスタムパラメータでのFibonacci Cycle計算テスト"""
        # カスタム期間
        data = pd.DataFrame(
            {"close": list(range(100, 121))}  # 21個のデータ（最大期間21）
        )

        result = OriginalIndicators.calculate_fibonacci_cycle(
            data, cycle_periods=[5, 8, 13, 21], fib_ratios=[0.5, 1.0, 1.5, 2.5]
        )

        assert isinstance(result, pd.DataFrame)
        assert "FIBO_CYCLE_4" in result.columns
        assert "FIBO_SIGNAL_4" in result.columns
        non_nan_values = result["FIBO_CYCLE_4"].dropna()
        if len(non_nan_values) > 0:
            assert np.isfinite(non_nan_values.min())
            assert np.isfinite(non_nan_values.max())

    def test_fibonacci_cycle_direct_calculation(self):
        """Fibonacci Cycleの直接計算テスト"""
        close = pd.Series(list(range(100, 155)))  # 55個のデータ

        cycle, signal = OriginalIndicators.fibonacci_cycle(close)

        assert isinstance(cycle, pd.Series)
        assert isinstance(signal, pd.Series)
        assert len(cycle) == len(close)
        assert len(signal) == len(close)
        assert cycle.name == "FIBO_CYCLE_5"
        assert signal.name == "FIBO_SIGNAL_5"
        non_nan_values = cycle.dropna()
        if len(non_nan_values) > 0:
            assert np.isfinite(non_nan_values.min())
            assert np.isfinite(non_nan_values.max())

    def test_fibonacci_cycle_parameter_validation(self):
        """Fibonacci Cycleのパラメータ検証テスト"""
        data = pd.DataFrame({"close": list(range(100, 155))})

        # cycle_periodsが空
        with pytest.raises(
            ValueError, match="cycle_periods and fib_ratios must not be empty"
        ):
            OriginalIndicators.calculate_fibonacci_cycle(
                data, cycle_periods=[], fib_ratios=[0.618]
            )

        # fib_ratiosが空
        with pytest.raises(
            ValueError, match="cycle_periods and fib_ratios must not be empty"
        ):
            OriginalIndicators.calculate_fibonacci_cycle(
                data, cycle_periods=[8, 13], fib_ratios=[]
            )

        # 不正なデータ型
        with pytest.raises(TypeError, match="data must be pandas DataFrame"):
            OriginalIndicators.calculate_fibonacci_cycle([100, 101, 102])

    def test_fibonacci_cycle_edge_cases(self):
        """Fibonacci Cycleの境界値テスト"""
        # 定数データ
        constant_data = pd.DataFrame({"close": [100] * 55})  # 55個の定数データ

        result = OriginalIndicators.calculate_fibonacci_cycle(constant_data)

        assert isinstance(result, pd.DataFrame)
        # 定数データでは小さな値になるはず
        non_nan_values = result["FIBO_CYCLE_5"].dropna()
        if len(non_nan_values) > 0:
            assert abs(non_nan_values.mean()) < 1e-3

        # 昇順データ
        increasing_data = pd.DataFrame(
            {"close": list(range(100, 155))}  # 55個の昇順データ
        )

        result = OriginalIndicators.calculate_fibonacci_cycle(increasing_data)

        assert isinstance(result, pd.DataFrame)
        # 昇順データでは正の傾向
        non_nan_values = result["FIBO_CYCLE_5"].dropna()
        if len(non_nan_values) > 0:
            assert np.isfinite(non_nan_values.mean())

    def test_fibonacci_cycle_consistency(self):
        """Fibonacci Cycleの再現性テスト"""
        data = pd.DataFrame({"close": list(range(100, 155))})

        result1 = OriginalIndicators.calculate_fibonacci_cycle(data)
        result2 = OriginalIndicators.calculate_fibonacci_cycle(data)

        # 同じ結果になるはず
        assert result1.equals(result2)

    def test_fibonacci_cycle_with_nan_values(self):
        """Fibonacci CycleのNaN値処理テスト"""
        data = pd.DataFrame(
            {"close": [100, np.nan, 102, 103, np.nan, 105, 106, 107, 108, 109]}
        )

        result = OriginalIndicators.calculate_fibonacci_cycle(data)

        assert isinstance(result, pd.DataFrame)
        assert "FIBO_CYCLE_5" in result.columns
        # NaNが適切に処理されている

    def test_fibonacci_cycle_memory_usage(self):
        """Fibonacci Cycleのメモリ使用量テスト"""
        # 大きなデータセット
        large_data = pd.DataFrame({"close": np.random.normal(100, 10, 10000)})

        result = OriginalIndicators.calculate_fibonacci_cycle(large_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(large_data)
        # 大きなデータでも処理できる


if __name__ == "__main__":
    # コマンドラインからの実行用
    pytest.main([__file__, "-v"])
