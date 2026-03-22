"""
OriginalIndicatorsのテスト

共通ヘルパーを使用して重複コードを削減
"""

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.technical_indicators.original import OriginalIndicators


class TestOriginalIndicators:
    """OriginalIndicatorsのテストクラス"""

    def test_init(self):
        """初期化のテスト"""
        assert OriginalIndicators is not None

    # FRAMA関連テスト
    def test_calculate_frama_valid_data(self, sample_data):
        """有効データでのFRAMA計算テスト"""
        result = OriginalIndicators.frama(sample_data, length=16, slow=200)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert result.dropna().min() >= sample_data.min()
        assert result.dropna().max() <= sample_data.max()

    def test_calculate_frama_insufficient_data(self):
        """データ不足でのFRAMA計算テスト"""
        data = pd.Series([100, 101, 102])
        result = OriginalIndicators.frama(data, length=2, slow=200)
        assert isinstance(result, pd.Series)
        assert result.isna().all()

    def test_calculate_frama_empty_data(self):
        """空データのFRAMAテスト"""
        data = pd.Series([])
        result = OriginalIndicators.frama(data, length=16, slow=200)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_calculate_frama_edge_cases(self):
        """FRAMAのエッジケーステスト"""
        # 単一データポイント
        single_data = pd.Series([100])
        result_single = OriginalIndicators.frama(single_data, length=16, slow=200)
        assert len(result_single) == 1
        assert np.isnan(result_single.iloc[0])

        # 最小限のデータ長
        min_data = pd.Series([100, 101, 102, 103])
        result_min = OriginalIndicators.frama(min_data, length=4, slow=200)
        assert len(result_min) == 4

    # Super Smoother関連テスト
    def test_calculate_super_smoother_valid_data(self, sample_data):
        """有効データでのSuper Smoother計算テスト"""
        result = OriginalIndicators.super_smoother(sample_data, length=10)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert result.index.equals(sample_data.index)
        assert result.name == "SUPER_SMOOTHER"
        assert result.notna().all()
        assert result.iloc[0] == sample_data.iloc[0]
        assert result.iloc[1] == sample_data.iloc[1]

    def test_calculate_super_smoother_insufficient_length(self):
        """不十分な長さのSuper Smootherテスト"""
        data = pd.Series([100, 101])
        with pytest.raises(ValueError, match="length must be >= 2"):
            OriginalIndicators.super_smoother(data, length=1)

    def test_calculate_super_smoother_empty_data(self):
        """空データのSuper Smootherテスト"""
        data = pd.Series([])
        result = OriginalIndicators.super_smoother(data, length=10)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_calculate_super_smoother_edge_cases(self):
        """Super Smootherのエッジケーステスト"""
        # 空データ
        empty_data = pd.Series([])
        result_empty = OriginalIndicators.super_smoother(empty_data, length=10)
        assert len(result_empty) == 0

        # 最小データ長
        min_data = pd.Series([100, 101])
        result_min = OriginalIndicators.super_smoother(min_data, length=2)
        assert len(result_min) == 2

    # Adaptive Entropy関連テスト
    def test_adaptive_entropy_valid_data(self, sample_data):
        """有効データでのAdaptive Entropy計算テスト"""
        oscillator, signal, ratio = OriginalIndicators.adaptive_entropy(
            sample_data, short_length=14, long_length=28, signal_length=5
        )
        assert isinstance(oscillator, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(ratio, pd.Series)
        assert len(oscillator) == len(sample_data)
        assert len(signal) == len(sample_data)
        assert len(ratio) == len(sample_data)

    def test_adaptive_entropy_insufficient_data(self):
        """データ不足でのAdaptive Entropyテスト"""
        data = pd.Series([100, 101, 102, 103, 104])
        oscillator, signal, ratio = OriginalIndicators.adaptive_entropy(
            data, short_length=14, long_length=28, signal_length=5
        )
        assert all(np.isnan(v) for v in oscillator)
        assert all(np.isnan(v) for v in signal)
        assert all(np.isnan(v) for v in ratio)

    def test_adaptive_entropy_invalid_parameters(self, sample_data):
        """無効なパラメータでのAdaptive Entropyテスト"""
        # short_lengthが短すぎる
        with pytest.raises(ValueError, match="short_length must be >= 5"):
            OriginalIndicators.adaptive_entropy(
                sample_data, short_length=3, long_length=28, signal_length=5
            )

        # long_lengthが短すぎる
        with pytest.raises(ValueError, match="long_length must be >= 10"):
            OriginalIndicators.adaptive_entropy(
                sample_data, short_length=14, long_length=5, signal_length=5
            )

        # signal_lengthが短すぎる
        with pytest.raises(ValueError, match="signal_length must be >= 2"):
            OriginalIndicators.adaptive_entropy(
                sample_data, short_length=14, long_length=28, signal_length=1
            )

        # short_lengthがlong_length以上
        with pytest.raises(ValueError, match="short_length must be < long_length"):
            OriginalIndicators.adaptive_entropy(
                sample_data, short_length=30, long_length=28, signal_length=5
            )

    # McGinley Dynamic関連テスト
    def test_mcginley_dynamic_valid_data(self, sample_data):
        """有効データでのMcGinley Dynamic計算テスト"""
        result = OriginalIndicators.mcginley_dynamic(sample_data, length=10, k=0.6)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert result.name == "MCGINLEY_10"

    def test_mcginley_dynamic_invalid_length(self, sample_data):
        """無効なlengthパラメータのMcGinley Dynamicテスト"""
        with pytest.raises(ValueError, match="length must be >= 1"):
            OriginalIndicators.mcginley_dynamic(sample_data, length=0)

    def test_mcginley_dynamic_invalid_k(self, sample_data):
        """無効なkパラメータのMcGinley Dynamicテスト"""
        with pytest.raises(ValueError, match="k must be > 0"):
            OriginalIndicators.mcginley_dynamic(sample_data, length=10, k=0)

    # Chande Kroll Stop関連テスト
    def test_chande_kroll_stop_valid_data(self, sample_df):
        """有効データでのChande Kroll Stop計算テスト"""
        long_stop, short_stop = OriginalIndicators.chande_kroll_stop(
            sample_df["high"], sample_df["low"], sample_df["close"], p=10, x=1, q=9
        )
        assert isinstance(long_stop, pd.Series)
        assert isinstance(short_stop, pd.Series)
        assert len(long_stop) == len(sample_df)
        assert len(short_stop) == len(sample_df)

    def test_chande_kroll_stop_invalid_params(self, sample_df):
        """無効なパラメータのChande Kroll Stopテスト"""
        with pytest.raises(ValueError, match="p must be >= 1"):
            OriginalIndicators.chande_kroll_stop(
                sample_df["high"], sample_df["low"], sample_df["close"], p=0
            )

    # Trend Intensity Index関連テスト
    def test_trend_intensity_index_valid_data(self, sample_df):
        """有効データでのTrend Intensity Index計算テスト"""
        result = OriginalIndicators.trend_intensity_index(
            sample_df["close"],
            sample_df["high"],
            sample_df["low"],
            length=14,
            sma_length=30,
        )
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_df)
        assert result.name == "TII_14_30"
        # TIIは0-100の範囲
        valid_result = result.dropna()
        if len(valid_result) > 0:
            assert valid_result.min() >= 0.0
            assert valid_result.max() <= 100.0

    def test_trend_intensity_index_invalid_length(self, sample_df):
        """無効なlengthパラメータのTIIテスト"""
        with pytest.raises(ValueError, match="length must be >= 1"):
            OriginalIndicators.trend_intensity_index(
                sample_df["close"], sample_df["high"], sample_df["low"], length=0
            )

    # Quantum Flow関連テスト
    def test_quantum_flow_valid_data(self, sample_df):
        """有効データでのQuantum Flow計算テスト"""
        flow, signal = OriginalIndicators.quantum_flow(
            sample_df["close"],
            sample_df["high"],
            sample_df["low"],
            sample_df["volume"],
            length=14,
            flow_length=9,
        )
        assert isinstance(flow, pd.Series)
        assert isinstance(signal, pd.Series)
        assert len(flow) == len(sample_df)
        assert len(signal) == len(sample_df)

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
        assert all(np.isnan(v) for v in empty_flow)
        assert all(np.isnan(v) for v in empty_signal)

    def test_quantum_flow_invalid_parameters(self, sample_df):
        """無効なパラメータでのQuantum Flowテスト"""
        # lengthが短すぎる
        with pytest.raises(ValueError):
            OriginalIndicators.quantum_flow(
                sample_df["close"],
                sample_df["high"],
                sample_df["low"],
                sample_df["volume"],
                length=3,
                flow_length=9,
            )

        # flow_lengthが短すぎる
        with pytest.raises(ValueError):
            OriginalIndicators.quantum_flow(
                sample_df["close"],
                sample_df["high"],
                sample_df["low"],
                sample_df["volume"],
                length=14,
                flow_length=2,
            )

    # Elder Ray関連テスト
    def test_calculate_elder_ray_valid_data(self, sample_df):
        """有効データでのElder Ray計算テスト"""
        result = OriginalIndicators.calculate_elder_ray(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert "Elder_Ray_Bull_13_16" in result.columns
        assert "Elder_Ray_Bear_13_16" in result.columns

    def test_calculate_elder_ray_custom_parameters(self, sample_df):
        """カスタムパラメータでのElder Ray計算テスト"""
        result = OriginalIndicators.calculate_elder_ray(
            sample_df, length=10, ema_length=12
        )
        assert isinstance(result, pd.DataFrame)
        assert "Elder_Ray_Bull_10_12" in result.columns
        assert "Elder_Ray_Bear_10_12" in result.columns

    def test_elder_ray_parameter_validation(self, sample_df):
        """Elder Rayのパラメータ検証テスト"""
        # lengthが負
        with pytest.raises(ValueError, match="length must be positive"):
            OriginalIndicators.calculate_elder_ray(sample_df, length=-1, ema_length=16)

        # ema_lengthが負
        with pytest.raises(ValueError, match="ema_length must be positive"):
            OriginalIndicators.calculate_elder_ray(sample_df, length=13, ema_length=-1)

    # Prime Oscillator関連テスト
    def test_calculate_prime_oscillator_valid_data(self, sample_data):
        """有効データでのPrime Number Oscillator計算テスト"""
        result = OriginalIndicators.calculate_prime_oscillator(
            pd.DataFrame({"close": sample_data})
        )
        assert isinstance(result, pd.DataFrame)
        assert "PRIME_OSC_14" in result.columns
        assert "PRIME_SIGNAL_14_3" in result.columns
        # Prime Oscillatorは正規化（Z-score * 100）されているが、
        # 統計的に-300から300の範囲内に収まることが多い（厳密な制限はない）
        non_nan_values = result["PRIME_OSC_14"].dropna()
        if len(non_nan_values) > 0:
            assert non_nan_values.min() >= -300
            assert non_nan_values.max() <= 300

    def test_calculate_prime_oscillator_insufficient_data(self):
        """データ不足でのPrime Number Oscillator計算テスト"""
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104]})
        result = OriginalIndicators.calculate_prime_oscillator(data)
        assert isinstance(result, pd.DataFrame)
        assert "PRIME_OSC_14" in result.columns
        assert result["PRIME_OSC_14"].isna().all()

    # Fibonacci Cycle関連テスト
    def test_calculate_fibonacci_cycle_valid_data(self, sample_data):
        """有効データでのFibonacci Cycle計算テスト"""
        result = OriginalIndicators.calculate_fibonacci_cycle(
            pd.DataFrame({"close": sample_data})
        )
        assert isinstance(result, pd.DataFrame)
        assert "FIBO_CYCLE_5" in result.columns
        assert "FIBO_SIGNAL_5" in result.columns
        non_nan_values = result["FIBO_CYCLE_5"].dropna()
        if len(non_nan_values) > 0:
            assert np.isfinite(non_nan_values.min())
            assert np.isfinite(non_nan_values.max())

    def test_calculate_fibonacci_cycle_insufficient_data(self):
        """データ不足でのFibonacci Cycle計算テスト"""
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104]})
        result = OriginalIndicators.calculate_fibonacci_cycle(data)
        assert isinstance(result, pd.DataFrame)
        assert "FIBO_CYCLE_5" in result.columns

    # GRI関連テスト
    def test_calculate_gri_valid_data(self, sample_df):
        """有効データでのGRI計算テスト"""
        result = OriginalIndicators.gri(
            sample_df["high"], sample_df["low"], sample_df["close"], length=5
        )
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_df)
        assert not result.isna().all()

    def test_calculate_gri_insufficient_data(self):
        """データ不足でのGRI計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103],
                "low": [98, 99],
                "close": [100, 101],
            }
        )
        result = OriginalIndicators.gri(
            data["high"], data["low"], data["close"], length=14
        )
        assert isinstance(result, pd.Series)
        assert len(result) == 2
        assert result.isna().all()

    def test_calculate_gri_invalid_length(self, sample_df):
        """無効な長さのGRIテスト"""
        with pytest.raises(ValueError, match="length must be positive"):
            OriginalIndicators.gri(
                sample_df["high"], sample_df["low"], sample_df["close"], length=-1
            )

    # ラッパーメソッドのテスト
    def test_calculate_mcginley_dynamic_wrapper(self, sample_data):
        """McGinley Dynamic DataFrameラッパーメソッドのテスト"""
        result = OriginalIndicators.calculate_mcginley_dynamic(
            pd.DataFrame({"close": sample_data}), length=10, k=0.6
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        assert "MCGINLEY_10" in result.columns

    def test_calculate_chande_kroll_stop_wrapper(self, sample_df):
        """Chande Kroll Stop DataFrameラッパーメソッドのテスト"""
        result = OriginalIndicators.calculate_chande_kroll_stop(
            sample_df, p=10, x=1, q=9
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)
        assert "CKS_LONG_10" in result.columns
        assert "CKS_SHORT_10" in result.columns

    def test_calculate_trend_intensity_index_wrapper(self, sample_df):
        """TII DataFrameラッパーメソッドのテスト"""
        result = OriginalIndicators.calculate_trend_intensity_index(
            sample_df, length=14, sma_length=30
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)
        assert "TII_14_30" in result.columns

    def test_calculate_adaptive_entropy_wrapper(self, sample_data):
        """calculate_adaptive_entropyメソッドのテスト"""
        result = OriginalIndicators.calculate_adaptive_entropy(
            pd.DataFrame({"close": sample_data}),
            short_length=14,
            long_length=28,
            signal_length=5,
        )
        assert isinstance(result, pd.DataFrame)
        assert "ADAPTIVE_ENTROPY_OSC_14_28" in result.columns
        assert "ADAPTIVE_ENTROPY_SIGNAL_14_28_5" in result.columns
        assert "ADAPTIVE_ENTROPY_RATIO_14_28" in result.columns

    def test_calculate_quantum_flow_wrapper(self, sample_df):
        """calculate_quantum_flowメソッドのテスト"""
        result = OriginalIndicators.calculate_quantum_flow(
            sample_df, length=14, flow_length=9
        )
        assert isinstance(result, pd.DataFrame)
        assert "QUANTUM_FLOW" in result.columns
        assert "QUANTUM_FLOW_SIGNAL" in result.columns

    def test_calculate_prime_oscillator_wrapper(self, sample_data):
        """calculate_prime_oscillatorメソッドのテスト"""
        result = OriginalIndicators.calculate_prime_oscillator(
            pd.DataFrame({"close": sample_data}), length=14, signal_length=3
        )
        assert isinstance(result, pd.DataFrame)
        assert "PRIME_OSC_14" in result.columns
        assert "PRIME_SIGNAL_14_3" in result.columns

    def test_calculate_fibonacci_cycle_wrapper(self, sample_data):
        """calculate_fibonacci_cycleメソッドのテスト"""
        result = OriginalIndicators.calculate_fibonacci_cycle(
            pd.DataFrame({"close": sample_data})
        )
        assert isinstance(result, pd.DataFrame)
        assert "FIBO_CYCLE_5" in result.columns
        assert "FIBO_SIGNAL_5" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
