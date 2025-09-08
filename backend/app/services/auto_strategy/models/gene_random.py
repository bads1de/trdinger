"""
遺伝子のランダム生成ユーティリティ
"""
from __future__ import annotations

import random

from .enums import PositionSizingMethod, TPSLMethod
from .position_sizing_gene import PositionSizingGene
from .tpsl_gene import TPSLGene


def create_random_position_sizing_gene(config=None) -> PositionSizingGene:
    """ランダムなポジションサイジング遺伝子を生成"""
    method_choices = list(PositionSizingMethod)
    method = random.choice(method_choices)

    return PositionSizingGene(
        method=method,
        lookback_period=random.randint(50, 200),
        optimal_f_multiplier=random.uniform(0.25, 0.75),
        atr_period=random.randint(10, 30),
        atr_multiplier=random.uniform(1.0, 4.0),
        risk_per_trade=random.uniform(0.01, 0.05),
        fixed_ratio=random.uniform(0.05, 0.3),
        fixed_quantity=random.uniform(0.1, 10.0),
        min_position_size=random.uniform(0.01, 0.05),
        max_position_size=random.uniform(5.0, 50.0),
        enabled=True,
        priority=random.uniform(0.5, 1.5),
    )


def create_random_tpsl_gene() -> TPSLGene:
    """ランダムなTP/SL遺伝子を生成"""
    method = random.choice(list(TPSLMethod))

    return TPSLGene(
        method=method,
        stop_loss_pct=random.uniform(0.01, 0.08),
        take_profit_pct=random.uniform(0.02, 0.15),
        risk_reward_ratio=random.uniform(1.2, 4.0),
        base_stop_loss=random.uniform(0.01, 0.06),
        atr_multiplier_sl=random.uniform(1.0, 3.0),
        atr_multiplier_tp=random.uniform(2.0, 5.0),
        atr_period=random.randint(10, 30),
        lookback_period=random.randint(50, 200),
        confidence_threshold=random.uniform(0.5, 0.9),
        method_weights={
            "fixed": random.uniform(0.1, 0.4),
            "risk_reward": random.uniform(0.2, 0.5),
            "volatility": random.uniform(0.1, 0.4),
            "statistical": random.uniform(0.1, 0.3),
        },
        enabled=True,
        priority=random.uniform(0.5, 1.5),
    )
