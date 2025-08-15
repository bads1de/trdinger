"""
TP/SL遺伝子モデル

TP/SL設定をGA最適化対象として表現するための遺伝子モデルです。
テクニカル指標パラメータと同様に、GA操作（交叉、突然変異）の対象となります。
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TPSLMethod(Enum):
    """TP/SL決定方式"""

    FIXED_PERCENTAGE = "fixed_percentage"  # 固定パーセンテージ
    RISK_REWARD_RATIO = "risk_reward_ratio"  # リスクリワード比ベース
    VOLATILITY_BASED = "volatility_based"  # ボラティリティベース
    STATISTICAL = "statistical"  # 統計的優位性ベース
    ADAPTIVE = "adaptive"  # 適応的（複数手法の組み合わせ）


@dataclass
class TPSLGene:
    """
    TP/SL遺伝子

    GA最適化対象としてのTP/SL設定を表現します。
    テクニカル指標パラメータと同じレベルで進化します。
    """

    # TP/SL決定方式
    method: TPSLMethod = TPSLMethod.RISK_REWARD_RATIO

    # 固定パーセンテージ方式のパラメータ
    stop_loss_pct: float = 0.03  # SL固定値（3%）
    take_profit_pct: float = 0.06  # TP固定値（6%）

    # リスクリワード比方式のパラメータ
    risk_reward_ratio: float = 2.0  # リスクリワード比（1:2）
    base_stop_loss: float = 0.03  # ベースSL（3%）

    # ボラティリティベース方式のパラメータ
    atr_multiplier_sl: float = 2.0  # SL用ATR倍率
    atr_multiplier_tp: float = 3.0  # TP用ATR倍率
    atr_period: int = 14  # ATR計算期間

    # 統計的方式のパラメータ
    lookback_period: int = 100  # 統計計算期間
    confidence_threshold: float = 0.7  # 信頼度閾値

    # 適応的方式のパラメータ
    method_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "fixed": 0.25,
            "risk_reward": 0.35,
            "volatility": 0.25,
            "statistical": 0.15,
        }
    )

    # メタデータ
    enabled: bool = True
    priority: float = 1.0  # 他の要素との優先度

    def to_dict(self) -> Dict[str, Any]:
        """遺伝子を辞書形式に変換（数値を適切な桁数に丸める）"""
        return {
            "method": self.method.value,
            "stop_loss_pct": round(self.stop_loss_pct, 4),
            "take_profit_pct": round(self.take_profit_pct, 4),
            "risk_reward_ratio": round(self.risk_reward_ratio, 3),
            "base_stop_loss": round(self.base_stop_loss, 4),
            "atr_multiplier_sl": round(self.atr_multiplier_sl, 3),
            "atr_multiplier_tp": round(self.atr_multiplier_tp, 3),
            "atr_period": self.atr_period,
            "lookback_period": self.lookback_period,
            "confidence_threshold": round(self.confidence_threshold, 3),
            "method_weights": {k: round(v, 3) for k, v in self.method_weights.items()},
            "enabled": self.enabled,
            "priority": round(self.priority, 2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TPSLGene":
        """辞書形式から遺伝子を復元"""
        return cls(
            method=TPSLMethod(data.get("method", "risk_reward_ratio")),
            stop_loss_pct=data.get("stop_loss_pct", 0.03),
            take_profit_pct=data.get("take_profit_pct", 0.06),
            risk_reward_ratio=data.get("risk_reward_ratio", 2.0),
            base_stop_loss=data.get("base_stop_loss", 0.03),
            atr_multiplier_sl=data.get("atr_multiplier_sl", 2.0),
            atr_multiplier_tp=data.get("atr_multiplier_tp", 3.0),
            atr_period=data.get("atr_period", 14),
            lookback_period=data.get("lookback_period", 100),
            confidence_threshold=data.get("confidence_threshold", 0.7),
            method_weights=data.get(
                "method_weights",
                {
                    "fixed": 0.25,
                    "risk_reward": 0.35,
                    "volatility": 0.25,
                    "statistical": 0.15,
                },
            ),
            enabled=data.get("enabled", True),
            priority=data.get("priority", 1.0),
        )

    def validate(self) -> tuple[bool, List[str]]:
        """遺伝子の妥当性を検証"""
        errors = []

        from ..config.constants import TPSL_LIMITS

        # パーセンテージ・期間系は定数から範囲を取得
        sl_min, sl_max = TPSL_LIMITS["stop_loss_pct"]
        tp_min, tp_max = TPSL_LIMITS["take_profit_pct"]
        base_sl_min, base_sl_max = TPSL_LIMITS["base_stop_loss"]
        atr_sl_min, atr_sl_max = TPSL_LIMITS["atr_multiplier_sl"]
        atr_tp_min, atr_tp_max = TPSL_LIMITS["atr_multiplier_tp"]
        atr_p_min, atr_p_max = TPSL_LIMITS["atr_period"]
        lb_min, lb_max = TPSL_LIMITS["lookback_period"]
        conf_min, conf_max = TPSL_LIMITS["confidence_threshold"]

        if not (sl_min <= self.stop_loss_pct <= sl_max):
            errors.append(
                f"stop_loss_pct must be between {sl_min*100:.1f}% and {sl_max*100:.0f}%"
            )

        if not (tp_min <= self.take_profit_pct <= tp_max):
            errors.append(
                f"take_profit_pct must be between {tp_min*100:.0f}% and {tp_max*100:.0f}%"
            )

        if not (base_sl_min <= self.base_stop_loss <= base_sl_max):
            errors.append(
                f"base_stop_loss must be between {base_sl_min*100:.1f}% and {base_sl_max*100:.0f}%"
            )

        # リスクリワード比チェック（値は既定のまま）
        if not (0.5 <= self.risk_reward_ratio <= 10.0):
            errors.append("risk_reward_ratio must be between 0.5 and 10.0")

        # ATR関連
        if not (atr_sl_min <= self.atr_multiplier_sl <= atr_sl_max):
            errors.append(
                f"atr_multiplier_sl must be between {atr_sl_min} and {atr_sl_max}"
            )

        if not (atr_tp_min <= self.atr_multiplier_tp <= atr_tp_max):
            errors.append(
                f"atr_multiplier_tp must be between {atr_tp_min} and {atr_tp_max}"
            )

        if not (atr_p_min <= self.atr_period <= atr_p_max):
            errors.append(f"atr_period must be between {atr_p_min} and {atr_p_max}")

        # 統計系
        if not (lb_min <= self.lookback_period <= lb_max):
            errors.append(f"lookback_period must be between {lb_min} and {lb_max}")

        if not (conf_min <= self.confidence_threshold <= conf_max):
            errors.append(
                f"confidence_threshold must be between {conf_min} and {conf_max}"
            )

        # 重み合計チェック
        if self.method_weights:
            weight_sum = sum(self.method_weights.values())
            if not (0.8 <= weight_sum <= 1.2):
                errors.append("method_weights sum should be close to 1.0")

        return len(errors) == 0, errors

    


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


def crossover_tpsl_genes(
    parent1: TPSLGene, parent2: TPSLGene
) -> tuple[TPSLGene, TPSLGene]:
    """TP/SL遺伝子の交叉"""
    # 単純な一点交叉
    if random.random() < 0.5:
        # 方式を交換
        child1_method = parent2.method
        child2_method = parent1.method
    else:
        child1_method = parent1.method
        child2_method = parent2.method

    # パラメータの交叉
    child1 = TPSLGene(
        method=child1_method,
        stop_loss_pct=(parent1.stop_loss_pct + parent2.stop_loss_pct) / 2,
        take_profit_pct=(parent1.take_profit_pct + parent2.take_profit_pct) / 2,
        risk_reward_ratio=(parent1.risk_reward_ratio + parent2.risk_reward_ratio) / 2,
        base_stop_loss=(parent1.base_stop_loss + parent2.base_stop_loss) / 2,
        atr_multiplier_sl=(parent1.atr_multiplier_sl + parent2.atr_multiplier_sl) / 2,
        atr_multiplier_tp=(parent1.atr_multiplier_tp + parent2.atr_multiplier_tp) / 2,
        atr_period=random.choice([parent1.atr_period, parent2.atr_period]),
        lookback_period=random.choice(
            [parent1.lookback_period, parent2.lookback_period]
        ),
        confidence_threshold=(
            parent1.confidence_threshold + parent2.confidence_threshold
        )
        / 2,
    )

    child2 = TPSLGene(
        method=child2_method,
        stop_loss_pct=(parent2.stop_loss_pct + parent1.stop_loss_pct) / 2,
        take_profit_pct=(parent2.take_profit_pct + parent1.take_profit_pct) / 2,
        risk_reward_ratio=(parent2.risk_reward_ratio + parent1.risk_reward_ratio) / 2,
        base_stop_loss=(parent2.base_stop_loss + parent1.base_stop_loss) / 2,
        atr_multiplier_sl=(parent2.atr_multiplier_sl + parent1.atr_multiplier_sl) / 2,
        atr_multiplier_tp=(parent2.atr_multiplier_tp + parent1.atr_multiplier_tp) / 2,
        atr_period=random.choice([parent2.atr_period, parent1.atr_period]),
        lookback_period=random.choice(
            [parent2.lookback_period, parent1.lookback_period]
        ),
        confidence_threshold=(
            parent2.confidence_threshold + parent1.confidence_threshold
        )
        / 2,
    )

    return child1, child2


def mutate_tpsl_gene(gene: TPSLGene, mutation_rate: float = 0.1) -> TPSLGene:
    """TP/SL遺伝子の突然変異"""
    mutated = TPSLGene(**gene.to_dict())

    # 方式の突然変異
    if random.random() < mutation_rate:
        mutated.method = random.choice(list(TPSLMethod))

    # パラメータの突然変異
    if random.random() < mutation_rate:
        mutated.stop_loss_pct *= random.uniform(0.8, 1.2)
        mutated.stop_loss_pct = max(0.005, min(mutated.stop_loss_pct, 0.15))

    if random.random() < mutation_rate:
        mutated.take_profit_pct *= random.uniform(0.8, 1.2)
        mutated.take_profit_pct = max(0.01, min(mutated.take_profit_pct, 0.3))

    if random.random() < mutation_rate:
        mutated.risk_reward_ratio *= random.uniform(0.8, 1.2)
        mutated.risk_reward_ratio = max(0.5, min(mutated.risk_reward_ratio, 10.0))

    if random.random() < mutation_rate:
        mutated.base_stop_loss *= random.uniform(0.8, 1.2)
        mutated.base_stop_loss = max(0.005, min(mutated.base_stop_loss, 0.15))

    if random.random() < mutation_rate:
        mutated.atr_multiplier_sl *= random.uniform(0.8, 1.2)
        mutated.atr_multiplier_sl = max(0.5, min(mutated.atr_multiplier_sl, 5.0))

    if random.random() < mutation_rate:
        mutated.atr_multiplier_tp *= random.uniform(0.8, 1.2)
        mutated.atr_multiplier_tp = max(1.0, min(mutated.atr_multiplier_tp, 10.0))

    return mutated
