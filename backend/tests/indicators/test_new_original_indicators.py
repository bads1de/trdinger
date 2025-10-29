"""
New OriginalIndicatorsのテスト
"""

import pytest
import pandas as pd
import numpy as np

from app.services.indicators.technical_indicators.original import OriginalIndicators


class TestNewOriginalIndicators:
    """新しいOriginalIndicatorsのテストクラス"""

    def test_harmonic_resonance_valid_data(self):
        """有効データでのHarmonic Resonance計算テスト"""
        data = pd.DataFrame({
            "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
            "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122],
            "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118]
        })

        hri, signal = OriginalIndicators.harmonic_resonance(
            data["close"], data["high"], data["low"], length=20, resonance_bands=5, signal_length=3
        )

        assert isinstance(hri, pd.Series)
        assert isinstance(signal, pd.Series)
        assert hri.name == "HARMONIC_RESONANCE"
        assert signal.name == "HRI_SIGNAL"
        assert len(hri) == len(data)
        assert len(signal) == len(data)

    def test_harmonic_resonance_calculation(self):
        """Harmonic Resonance計算の詳細テスト"""
        data = pd.DataFrame({
            "close": list(range(100, 150)),  # 20期間 + α (多めに)
            "high": list(range(102, 152)),
            "low": list(range(98, 148))
        })

        hri, signal = OriginalIndicators.harmonic_resonance(
            data["close"], data["high"], data["low"], length=20, resonance_bands=5, signal_length=3
        )

        # データが不足している期間はNaNであるべき
        assert np.isnan(hri.iloc[:30]).all()  # 最初の30期間はNaN
        
        # データが十分にある期間は計算されているべき（計算が実行されることの確認）
        assert isinstance(hri, pd.Series)
        assert len(hri) == len(data)

    def test_chaos_fractal_dimension_valid_data(self):
        """有効データでのChaos Fractal Dimension計算テスト"""
        data = pd.DataFrame({
            "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125],
            "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127],
            "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123],
            "volume": [1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250]
        })

        ctf, signal = OriginalIndicators.chaos_fractal_dimension(
            data["close"], data["high"], data["low"], data["volume"], length=25, embedding_dim=3, signal_length=4
        )

        assert isinstance(ctf, pd.Series)
        assert isinstance(signal, pd.Series)
        assert ctf.name == "CHAOS_FRACTAL_DIM"
        assert signal.name == "CTFD_SIGNAL"
        assert len(ctf) == len(data)
        assert len(signal) == len(data)

        # 正規化された値は[-1, 1]の範囲にあるべき
        non_nan_values = ctf.dropna()
        assert (non_nan_values >= -1.0).all()
        assert (non_nan_values <= 1.0).all()

    def test_chaos_fractal_dimension_edge_cases(self):
        """Chaos Fractal Dimensionのエッジケーステスト"""
        # 完全に定数のデータ
        data_constant = pd.DataFrame({
            "close": [100] * 30,
            "high": [102] * 30,
            "low": [98] * 30,
            "volume": [1000] * 30
        })

        ctf, signal = OriginalIndicators.chaos_fractal_dimension(
            data_constant["close"], data_constant["high"], data_constant["low"], data_constant["volume"], length=25, embedding_dim=3, signal_length=4
        )

        # 定数データでは予測可能性が高くなる（値が1に近くなる）
        non_nan_values = ctf.dropna()
        if len(non_nan_values) > 0:
            assert non_nan_values.mean() > 0.5

    def test_harmonic_resonance_invalid_parameters(self):
        """Harmonic Resonanceの無効パラメータテスト"""
        data = pd.DataFrame({
            "close": list(range(100, 120)),
            "high": list(range(102, 122)),
            "low": list(range(98, 118))
        })

        # 無効なlength
        with pytest.raises(ValueError, match="length must be >= 10"):
            OriginalIndicators.harmonic_resonance(
                data["close"], data["high"], data["low"], length=5, resonance_bands=5, signal_length=3
            )

        # 無効なresonance_bands
        with pytest.raises(ValueError, match="resonance_bands must be between 3 and 10"):
            OriginalIndicators.harmonic_resonance(
                data["close"], data["high"], data["low"], length=20, resonance_bands=15, signal_length=3
            )

    def test_chaos_fractal_dimension_invalid_parameters(self):
        """Chaos Fractal Dimensionの無効パラメータテスト"""
        data = pd.DataFrame({
            "close": list(range(100, 130)),
            "high": list(range(102, 132)),
            "low": list(range(98, 128)),
            "volume": list(range(1000, 1030))
        })

        # 無効なlength
        with pytest.raises(ValueError, match="length must be >= 15"):
            OriginalIndicators.chaos_fractal_dimension(
                data["close"], data["high"], data["low"], data["volume"], length=10, embedding_dim=3, signal_length=4
            )

        # 無効なembedding_dim
        with pytest.raises(ValueError, match="embedding_dim must be between 2 and 5"):
            OriginalIndicators.chaos_fractal_dimension(
                data["close"], data["high"], data["low"], data["volume"], length=25, embedding_dim=6, signal_length=4
            )

    def test_harmonic_resonance_wrapped_function(self):
        """Harmonic Resonanceラッパーメソッドのテスト"""
        data = pd.DataFrame({
            "close": list(range(100, 130)),
            "high": list(range(102, 132)),
            "low": list(range(98, 128))
        })

        result = OriginalIndicators.calculate_harmonic_resonance(data, length=20, resonance_bands=5, signal_length=3)

        assert isinstance(result, pd.DataFrame)
        assert "HARMONIC_RESONANCE" in result.columns
        assert "HRI_SIGNAL" in result.columns
        assert len(result) == len(data)

    def test_chaos_fractal_dimension_wrapped_function(self):
        """Chaos Fractal Dimensionラッパーメソッドのテスト"""
        data = pd.DataFrame({
            "close": list(range(100, 130)),
            "high": list(range(102, 132)),
            "low": list(range(98, 128)),
            "volume": list(range(1000, 1030))
        })

        result = OriginalIndicators.calculate_chaos_fractal_dimension(
            data, length=25, embedding_dim=3, signal_length=4
        )

        assert isinstance(result, pd.DataFrame)
        assert "CHAOS_FRACTAL_DIM" in result.columns
        assert "CTFD_SIGNAL" in result.columns
        assert len(result) == len(data)
