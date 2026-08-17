from __future__ import annotations

import math

CONSTRAINT_SCALING_REFERENCE_DAYS = 180
CONSTRAINT_SCALING_MAX_DRAWDOWN = 0.40
CONSTRAINT_SCALING_MIN_DAYS = 7


def effective_max_drawdown_limit(
    base_limit: float,
    window_days: int | None,
    *,
    reference_days: int = CONSTRAINT_SCALING_REFERENCE_DAYS,
    cap: float = CONSTRAINT_SCALING_MAX_DRAWDOWN,
) -> float:
    if window_days is None or window_days <= 0:
        return float(base_limit)
    if window_days <= reference_days:
        return float(base_limit)
    if reference_days <= 0:
        return float(base_limit)
    scale = math.sqrt(window_days / reference_days)
    scaled = float(base_limit) * scale
    return min(scaled, float(cap))


def effective_min_trades(
    base_min_trades: int,
    window_days: int | None,
    *,
    reference_days: int = CONSTRAINT_SCALING_REFERENCE_DAYS,
    min_days: int = CONSTRAINT_SCALING_MIN_DAYS,
) -> int:
    base = int(base_min_trades)
    if window_days is None or window_days <= 0:
        return base
    if reference_days <= 0:
        return base
    effective_window = max(window_days, int(min_days))
    effective_ref = max(int(min_days), int(reference_days))
    ratio = effective_window / effective_ref
    if ratio >= 1.0:
        return base
    # 切り捨てで下限を 1 まで緩和する（1トレードでも失格にしない）。
    # round を使うと短いフォールド（例: 30日）で 2 に丸め上がり、
    # 無取引側への選抜圧を強めるため floor に変更。
    return max(1, int(base * ratio))


def get_effective_fitness_constraints(
    base_constraints: dict,
    window_days: int | None,
    *,
    reference_days: int = CONSTRAINT_SCALING_REFERENCE_DAYS,
) -> dict:
    effective = dict(base_constraints) if base_constraints else {}
    if "max_drawdown_limit" in effective:
        raw = effective.get("max_drawdown_limit")
        if isinstance(raw, (int, float)):
            effective["max_drawdown_limit"] = effective_max_drawdown_limit(
                float(raw),
                window_days,
                reference_days=reference_days,
            )
    if "min_trades" in effective:
        raw = effective.get("min_trades")
        try:
            base_min = int(raw)  # type: ignore[arg-type]
        except Exception:
            base_min = 0
        effective["min_trades"] = effective_min_trades(
            base_min,
            window_days,
            reference_days=reference_days,
        )
    return effective
