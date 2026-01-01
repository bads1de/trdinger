import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from app.services.indicators.technical_indicators.original import OriginalIndicators


class TestOriginalUnitExtended:
    @pytest.fixture
    def sample_df(self):
        rows = 300
        return pd.DataFrame(
            {
                "open": np.random.normal(100, 5, rows),
                "high": np.random.normal(105, 5, rows),
                "low": np.random.normal(95, 5, rows),
                "close": np.random.normal(100, 5, rows),
                "volume": np.random.normal(1000, 100, rows),
            },
            index=pd.date_range("2023-01-01", periods=rows),
        )

    def test_original_all_methods_and_aliases(self, sample_df):
        h, l, c, v = (
            sample_df["high"],
            sample_df["low"],
            sample_df["close"],
            sample_df["volume"],
        )

        # 1. Elder Ray & alias
        assert len(OriginalIndicators.elder_ray(h, l, c)) == 2
        # calculate_系はDataFrameを返す
        assert isinstance(
            OriginalIndicators.calculate_elder_ray(sample_df), pd.DataFrame
        )

        # 2. Adaptive Entropy & alias
        assert len(OriginalIndicators.adaptive_entropy(c)) == 3
        assert isinstance(
            OriginalIndicators.calculate_adaptive_entropy(sample_df), pd.DataFrame
        )

        # 3. Quantum Flow & alias
        assert len(OriginalIndicators.quantum_flow(c, h, l, v)) == 2
        assert isinstance(
            OriginalIndicators.calculate_quantum_flow(sample_df), pd.DataFrame
        )

        # 4. Harmonic Resonance & alias
        assert len(OriginalIndicators.harmonic_resonance(c, h, l)) == 2
        assert isinstance(
            OriginalIndicators.calculate_harmonic_resonance(sample_df), pd.DataFrame
        )

        # 5. Chaos Fractal Dimension & alias
        assert len(OriginalIndicators.chaos_fractal_dimension(c, h, l, v)) == 2
        assert isinstance(
            OriginalIndicators.calculate_chaos_fractal_dimension(sample_df),
            pd.DataFrame,
        )

        # 6. Prime Oscillator & alias
        assert len(OriginalIndicators.prime_oscillator(c)) == 2
        assert isinstance(
            OriginalIndicators.calculate_prime_oscillator(sample_df), pd.DataFrame
        )

        # 7. Fibonacci Cycle & alias
        assert len(OriginalIndicators.fibonacci_cycle(c)) == 2
        assert isinstance(
            OriginalIndicators.calculate_fibonacci_cycle(sample_df), pd.DataFrame
        )

        # 8. McGinley Dynamic & alias
        assert isinstance(OriginalIndicators.mcginley_dynamic(c), pd.Series)
        # 内部で単一カラムのDataFrameとして返ってくる
        assert isinstance(
            OriginalIndicators.calculate_mcginley_dynamic(sample_df),
            (pd.Series, pd.DataFrame),
        )

        # 9. Chande Kroll Stop & alias
        assert len(OriginalIndicators.chande_kroll_stop(h, l, c)) == 2
        assert isinstance(
            OriginalIndicators.calculate_chande_kroll_stop(sample_df), pd.DataFrame
        )

        # 10. Trend Intensity Index & alias
        assert isinstance(OriginalIndicators.trend_intensity_index(c, h, l), pd.Series)
        assert isinstance(
            OriginalIndicators.calculate_trend_intensity_index(sample_df),
            (pd.Series, pd.DataFrame),
        )

        # 11. Connors RSI & alias
        assert isinstance(OriginalIndicators.connors_rsi(c), pd.Series)
        assert isinstance(
            OriginalIndicators.calculate_connors_rsi(sample_df),
            (pd.Series, pd.DataFrame),
        )

    def test_internal_helpers(self):
        # 内部ヘルパー関数の直接テスト
        assert OriginalIndicators._is_prime(7) is True
        assert OriginalIndicators._is_prime(10) is False
        assert len(OriginalIndicators._get_prime_sequence(5)) >= 5
        assert len(OriginalIndicators._generate_fibonacci_sequence(5)) >= 5

        # Entropy
        data = np.abs(np.random.normal(0, 1, 100)) + 0.1  # 正の値を保証
        entropy_val = OriginalIndicators._entropy(data, window=10)
        # NaNを除いて検証 (微分エントロピーは負になりうるので非NaNであることを確認)
        valid_mask = ~np.isnan(entropy_val)
        if np.any(valid_mask):
            assert isinstance(entropy_val[valid_mask][0], (float, np.float64))

    def test_frama_edge_cases(self, sample_df):
        c = sample_df["close"]
        # パラメータエラー（内部で調整されるためエラーにならない）
        assert not OriginalIndicators.frama(c, length=3).isna().all()
        assert not OriginalIndicators.frama(c, length=5).isna().all()
        assert not OriginalIndicators.frama(c, slow=0).isna().all()

        # データ不足
        assert np.isnan(OriginalIndicators.frama(c[:5], length=16)).all()

    def test_original_boundary_cases(self, sample_df):
        c = sample_df["close"]
        # 1. 全て同じ値 (分散0) での計算
        constant_data = pd.Series([100.0] * 100, index=c.index[:100])
        res = OriginalIndicators.adaptive_entropy(constant_data)
        assert len(res) == 3

        # 2. 極端な値 (inf, -inf)
        inf_data = c.copy()
        inf_data.iloc[10] = np.inf
        res = OriginalIndicators.super_smoother(inf_data)
        assert isinstance(res, pd.Series)

    def test_ichimoku_failure_handling(self, sample_df):
        """ichimoku 例外時は NaN を返す（フォールバック計算は廃止）"""
        from app.services.indicators.technical_indicators.overlap import (
            OverlapIndicators,
        )

        h, l, c = sample_df["high"], sample_df["low"], sample_df["close"]
        # pandas-taが例外を投げる状況をシミュレート
        with patch(
            "pandas_ta.ichimoku", side_effect=RuntimeError("Internal Ta-Lib Error")
        ):
            res = OverlapIndicators.ichimoku(h, l, c)
            assert "tenkan_sen" in res
            # フォールバックは廃止したので、NaNを返す
            assert res["tenkan_sen"].isna().all()

    def test_find_dominant_frequencies_coverage(self):
        # プライベートメソッドへの直接アクセス（カバレッジのため）
        data = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        freqs = OriginalIndicators._find_dominant_frequencies(data)
        assert isinstance(freqs, np.ndarray)
