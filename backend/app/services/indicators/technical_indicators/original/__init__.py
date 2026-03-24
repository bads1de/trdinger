"""独自テクニカル指標の互換パッケージ."""

from __future__ import annotations

from .filters import (
    calculate_mcginley_dynamic,
    frama,
    mcginley_dynamic,
    super_smoother,
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
    calculate_elder_ray,
    calculate_fibonacci_cycle,
    calculate_prime_oscillator,
    connors_rsi,
    elder_ray,
    fibonacci_cycle,
    prime_oscillator,
)
from .trend import (
    calculate_chande_kroll_stop,
    calculate_trend_intensity_index,
    chande_kroll_stop,
    gri,
    trend_intensity_index,
)


class OriginalIndicators:
    """新規の独自指標を提供するクラス."""

    frama = staticmethod(frama)
    super_smoother = staticmethod(super_smoother)
    elder_ray = staticmethod(elder_ray)
    calculate_elder_ray = staticmethod(calculate_elder_ray)
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
    calculate_chaos_fractal_dimension = staticmethod(
        calculate_chaos_fractal_dimension
    )
    mcginley_dynamic = staticmethod(mcginley_dynamic)
    calculate_mcginley_dynamic = staticmethod(calculate_mcginley_dynamic)
    chande_kroll_stop = staticmethod(chande_kroll_stop)
    calculate_chande_kroll_stop = staticmethod(calculate_chande_kroll_stop)
    trend_intensity_index = staticmethod(trend_intensity_index)
    calculate_trend_intensity_index = staticmethod(
        calculate_trend_intensity_index
    )
    connors_rsi = staticmethod(connors_rsi)
    calculate_connors_rsi = staticmethod(calculate_connors_rsi)
    gri = staticmethod(gri)


__all__ = ["OriginalIndicators"]
