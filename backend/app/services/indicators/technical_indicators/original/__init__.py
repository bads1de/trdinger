"""独自テクニカル指標の互換パッケージ."""

from __future__ import annotations

from .filters import (
    frama,
)
from .flow import (
    calculate_chaos_fractal_dimension,
    calculate_harmonic_resonance,
    calculate_quantum_flow,
    chaos_fractal_dimension,
    harmonic_resonance,
    quantum_flow,
)
from .oscillators import (
    adaptive_entropy,
    calculate_adaptive_entropy,
    calculate_connors_rsi,
    calculate_damiani_volatmeter,
    calculate_entropy_volatility_index,
    calculate_fibonacci_cycle,
    calculate_kairi_relative_index,
    calculate_prime_oscillator,
    connors_rsi,
    damiani_volatmeter,
    entropy_volatility_index,
    fibonacci_cycle,
    kairi_relative_index,
    prime_oscillator,
)
from .trend import (
    calculate_direction_entropy,
    calculate_trend_intensity_index,
    direction_entropy,
    gri,
    trend_intensity_index,
)


class OriginalIndicators:
    """新規の独自指標を提供するクラス."""

    frama = staticmethod(frama)
    prime_oscillator = staticmethod(prime_oscillator)
    calculate_prime_oscillator = staticmethod(calculate_prime_oscillator)
    fibonacci_cycle = staticmethod(fibonacci_cycle)
    calculate_fibonacci_cycle = staticmethod(calculate_fibonacci_cycle)
    adaptive_entropy = staticmethod(adaptive_entropy)
    calculate_adaptive_entropy = staticmethod(calculate_adaptive_entropy)
    quantum_flow = staticmethod(quantum_flow)
    calculate_quantum_flow = staticmethod(calculate_quantum_flow)
    harmonic_resonance = staticmethod(harmonic_resonance)
    calculate_harmonic_resonance = staticmethod(calculate_harmonic_resonance)
    chaos_fractal_dimension = staticmethod(chaos_fractal_dimension)
    calculate_chaos_fractal_dimension = staticmethod(calculate_chaos_fractal_dimension)
    trend_intensity_index = staticmethod(trend_intensity_index)
    calculate_trend_intensity_index = staticmethod(calculate_trend_intensity_index)
    direction_entropy = staticmethod(direction_entropy)
    calculate_direction_entropy = staticmethod(calculate_direction_entropy)
    connors_rsi = staticmethod(connors_rsi)
    calculate_connors_rsi = staticmethod(calculate_connors_rsi)
    gri = staticmethod(gri)
    damiani_volatmeter = staticmethod(damiani_volatmeter)
    calculate_damiani_volatmeter = staticmethod(calculate_damiani_volatmeter)
    kairi_relative_index = staticmethod(kairi_relative_index)
    calculate_kairi_relative_index = staticmethod(calculate_kairi_relative_index)
    entropy_volatility_index = staticmethod(entropy_volatility_index)
    calculate_entropy_volatility_index = staticmethod(
        calculate_entropy_volatility_index
    )


__all__ = ["OriginalIndicators"]
