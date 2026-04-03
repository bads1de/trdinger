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
from .demarker import demarker
from .kairi_relative_index import kairi_relative_index
from .entropy_volatility_index import entropy_volatility_index
from .trend_intensity_index import trend_intensity_index
from .gri import gri
from .rmi import rmi
from .polarized_fractal_efficiency import pfe
from .market_meanness_index import mmi
from .ehlers_cyber_cycle import ehlers_cyber_cycle
from .ehlers_instantaneous_trendline import ehlers_instantaneous_trendline
from .smoothed_adaptive_momentum import smoothed_adaptive_momentum
from .vortex_rsi import vortex_rsi
from .trend_trigger_factor import ttf
from .random_walk_index import rwi
from .hurst_exponent import hurst_exponent


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
    demarker = staticmethod(demarker)
    kairi_relative_index = staticmethod(kairi_relative_index)
    entropy_volatility_index = staticmethod(entropy_volatility_index)
    rmi = staticmethod(rmi)
    pfe = staticmethod(pfe)
    mmi = staticmethod(mmi)
    ehlers_cyber_cycle = staticmethod(ehlers_cyber_cycle)
    ehlers_instantaneous_trendline = staticmethod(ehlers_instantaneous_trendline)
    smoothed_adaptive_momentum = staticmethod(smoothed_adaptive_momentum)
    vortex_rsi = staticmethod(vortex_rsi)
    ttf = staticmethod(ttf)
    rwi = staticmethod(rwi)
    hurst_exponent = staticmethod(hurst_exponent)


__all__ = ["OriginalIndicators"]
