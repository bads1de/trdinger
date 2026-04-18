"""
遺伝子パラメータ範囲定義

GA生成時の探索範囲（GENERATION）と検証用の許容範囲（VALIDATION）を定義します。
"""

from typing import Dict, Tuple

# ==================== PositionSizingGene ====================

POSITION_SIZING_GENERATION_RANGES: Dict[str, Tuple[float, float]] = {
    "lookback_period": (50, 200),
    "optimal_f_multiplier": (0.25, 0.75),
    "atr_period": (10, 30),
    "atr_multiplier": (1.0, 4.0),
    "risk_per_trade": (0.01, 0.05),
    "fixed_ratio": (0.05, 0.3),
    "fixed_quantity": (0.1, 10.0),
    "min_position_size": (0.01, 0.1),
    "max_position_size": (5.0, 50.0),
    "var_confidence": (0.8, 0.99),
    "max_var_ratio": (0.005, 0.05),
    "max_expected_shortfall_ratio": (0.01, 0.1),
    "var_lookback": (50, 500),
    "priority": (0.5, 1.5),
}

POSITION_SIZING_VALIDATION_RANGES: Dict[str, Tuple[float, float]] = {
    "lookback_period": (10, 500),
    "optimal_f_multiplier": (0.1, 1.0),
    "atr_period": (5, 50),
    "atr_multiplier": (0.5, 10.0),
    "risk_per_trade": (0.001, 0.1),
    "fixed_ratio": (0.01, 10.0),
    "fixed_quantity": (0.01, 1000.0),
    "min_position_size": (0.001, 1.0),
    "max_position_size": (0.001, 1000000000.0),
    "var_confidence": (0.8, 0.999),
    "max_var_ratio": (0.001, 0.1),
    "max_expected_shortfall_ratio": (0.001, 0.2),
    "var_lookback": (20, 1000),
    "priority": (0.5, 1.5),
}

# ==================== TPSLGene ====================

TPSL_GENERATION_RANGES: Dict[str, Tuple[float, float]] = {
    "stop_loss_pct": (0.01, 0.08),
    "take_profit_pct": (0.02, 0.15),
    "risk_reward_ratio": (1.2, 4.0),
    "base_stop_loss": (0.01, 0.06),
    "atr_multiplier_sl": (1.0, 3.0),
    "atr_multiplier_tp": (2.0, 5.0),
    "atr_period": (10, 30),
    "lookback_period": (50, 200),
    "confidence_threshold": (0.5, 0.9),
    "priority": (0.5, 1.5),
    "trailing_step_pct": (0.005, 0.05),
}

TPSL_VALIDATION_RANGES: Dict[str, Tuple[float, float]] = {
    "stop_loss_pct": (0.005, 0.15),  # 0.5%-15%
    "take_profit_pct": (0.01, 0.3),  # 1%-30%
    "risk_reward_ratio": (1.0, 10.0),  # 1:10まで
    "base_stop_loss": (0.01, 0.06),
    "atr_multiplier_sl": (0.5, 3.0),
    "atr_multiplier_tp": (1.0, 5.0),
    "confidence_threshold": (0.1, 0.9),
    "priority": (0.5, 1.5),
    "lookback_period": (50, 200),
    "atr_period": (10, 30),
    "trailing_step_pct": (0.005, 0.05),
}

# ==================== EntryGene ====================

ENTRY_GENERATION_RANGES: Dict[str, Tuple[float, float]] = {
    "limit_offset_pct": (0.005, 0.02),
    "stop_offset_pct": (0.005, 0.02),
    "order_validity_bars": (5, 20),
    "priority": (0.5, 1.5),
}

ENTRY_VALIDATION_RANGES: Dict[str, Tuple[float, float]] = {
    "limit_offset_pct": (0.0, 0.1),
    "stop_offset_pct": (0.0, 0.1),
    "order_validity_bars": (0, 100),
    "priority": (0.5, 1.5),
}

# ==================== ExitGene ====================

EXIT_GENERATION_RANGES: Dict[str, Tuple[float, float]] = {
    "partial_exit_pct": (0.5, 0.8),
    "priority": (0.5, 1.5),
}

EXIT_VALIDATION_RANGES: Dict[str, Tuple[float, float]] = {
    "partial_exit_pct": (0.1, 0.9),
    "priority": (0.5, 1.5),
}
