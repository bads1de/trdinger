"""独自テクニカル指標の互換パッケージ."""

from __future__ import annotations

from .frama import frama
from .quantum_flow import quantum_flow
from .harmonic_resonance import harmonic_resonance
from .chaos_fractal_dimension import chaos_fractal_dimension
from .prime_oscillator import prime_oscillator
from .fibonacci_cycle import fibonacci_cycle
from .adaptive_entropy import adaptive_entropy
from .connors_rsi import connors_rsi
from .damiani_volatmeter import damiani_volatmeter
from .kairi_relative_index import kairi_relative_index
from .entropy_volatility_index import entropy_volatility_index
from .trend_intensity_index import trend_intensity_index
from .gri import gri


class OriginalIndicators:
    """新規の独自指標を提供するクラス."""

    frama = staticmethod(frama)
    prime_oscillator = staticmethod(prime_oscillator)
    fibonacci_cycle = staticmethod(fibonacci_cycle)
    adaptive_entropy = staticmethod(adaptive_entropy)
    quantum_flow = staticmethod(quantum_flow)
    harmonic_resonance = staticmethod(harmonic_resonance)
    chaos_fractal_dimension = staticmethod(chaos_fractal_dimension)
    trend_intensity_index = staticmethod(trend_intensity_index)
    connors_rsi = staticmethod(connors_rsi)
    gri = staticmethod(gri)
    damiani_volatmeter = staticmethod(damiani_volatmeter)
    kairi_relative_index = staticmethod(kairi_relative_index)
    entropy_volatility_index = staticmethod(entropy_volatility_index)


__all__ = ["OriginalIndicators"]