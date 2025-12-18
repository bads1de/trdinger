"""
TP/SL 遺伝子
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils.gene_utils import BaseGene
from ..config.constants import TPSLMethod

logger = logging.getLogger(__name__)


@dataclass
class TPSLGene(BaseGene):
    """
    TP/SL遺伝子

    GA最適化対象としてのTP/SL設定を表現します。
    BaseGeneを継承して共通機能を活用します。
    """

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TPSLGene:
        """辞書形式からTPSLGeneオブジェクトを復元

        BaseGeneのfrom_dictを安全に拡張し、すべてのフィールドを適切に初期化します。
        """
        init_params = {}

        # クラスアノテーションからパラメータ情報を取得
        if hasattr(cls, "__annotations__"):
            annotations = cls.__annotations__

            for param_name, param_type in annotations.items():
                if param_name in data:
                    value = data[param_name]

                    # Enum型への変換
                    if hasattr(param_type, "__members__"):
                        if isinstance(value, str):
                            try:
                                init_params[param_name] = param_type(value)
                            except ValueError:
                                logger.warning(f"無効なEnum値 {value} を無視")
                        else:
                            init_params[param_name] = value
                    else:
                        init_params[param_name] = value

        # TPSLGene固有のフィールドを明示的に処理
        for param_name in TPSLGene().__dict__.keys():
            if param_name not in init_params and param_name in data:
                init_params[param_name] = data[param_name]

        return cls(**init_params)

    method: TPSLMethod = TPSLMethod.RISK_REWARD_RATIO
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    risk_reward_ratio: float = 2.0
    base_stop_loss: float = 0.03
    atr_multiplier_sl: float = 2.0
    atr_multiplier_tp: float = 3.0
    atr_period: int = 14
    lookback_period: int = 100
    confidence_threshold: float = 0.7
    method_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "fixed": 0.25,
            "risk_reward": 0.35,
            "volatility": 0.25,
            "statistical": 0.15,
        }
    )
    enabled: bool = True
    priority: float = 1.0

    # トレーリングストップ関連パラメータ
    trailing_stop: bool = False  # トレーリングストップ有効化フラグ
    trailing_step_pct: float = 0.01  # トレーリング更新幅（1% = 0.01）
    trailing_take_profit: bool = False  # トレーリングTP有効化（TP到達後も利益を伸ばす）

    def _validate_parameters(self, errors: List[str]) -> None:
        """パラメータ固有の検証を実装"""
        try:
            from ..config.constants import TPSL_LIMITS

            sl_min, sl_max = TPSL_LIMITS["stop_loss_pct"]
            if not (sl_min <= self.stop_loss_pct <= sl_max):
                errors.append(
                    f"stop_loss_pct must be between {sl_min * 100:.1f}% and {sl_max * 100:.0f}%"
                )

            tp_min, tp_max = TPSL_LIMITS["take_profit_pct"]
            if not (tp_min <= self.take_profit_pct <= tp_max):
                errors.append(
                    f"take_profit_pct must be between {tp_min * 100:.1f}% and {tp_max * 100:.0f}%"
                )

            # 他のパラメータ検証
            self._validate_range(
                self.risk_reward_ratio, 1.0, 10.0, "risk_reward_ratio", errors
            )
            self._validate_range(
                self.confidence_threshold, 0.0, 1.0, "confidence_threshold", errors
            )
            self._validate_range(
                self.atr_multiplier_sl, 0.1, 5.0, "atr_multiplier_sl", errors
            )
            self._validate_range(
                self.atr_multiplier_tp, 0.1, 10.0, "atr_multiplier_tp", errors
            )

            # method_weightsの検証
            # 各値が0以上であることを確認
            for key, value in self.method_weights.items():
                if value < 0:
                    errors.append(f"method_weights[{key}]は0以上である必要があります")

            # 必要なキーがすべて含まれていることを確認
            required_keys = {"fixed", "risk_reward", "volatility", "statistical"}
            missing_keys = required_keys - set(self.method_weights.keys())
            if missing_keys:
                errors.append(f"method_weightsに不足しているキー: {missing_keys}")

            # 合計値が1.0であることを確認
            total_weight = sum(self.method_weights.values())
            if not (0.99 <= total_weight <= 1.01):  # 浮動小数点誤差考慮
                errors.append("method_weightsの合計は1.0である必要があります")

        except ImportError:
            # 定数が利用できない場合の基本検証
            if not (0.005 <= self.stop_loss_pct <= 0.15):
                errors.append("stop_loss_pct must be between 0.5% and 15%")

            if not (0.01 <= self.take_profit_pct <= 0.3):
                errors.append("take_profit_pct must be between 1% and 30%")


@dataclass
class TPSLResult:
    """TP/SL計算結果

    TPSLGeneratorとTPSLServiceでの計算結果を統一して表現します。
    """

    stop_loss_pct: float
    take_profit_pct: float
    method_used: str
    confidence_score: float = 0.0
    expected_performance: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """後処理"""
        if self.expected_performance is None:
            self.expected_performance = {}
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "method_used": self.method_used,
            "confidence_score": self.confidence_score,
            "expected_performance": self.expected_performance,
            "metadata": self.metadata,
        }


def create_random_tpsl_gene(config: Any = None) -> TPSLGene:
    """ランダムなTP/SL遺伝子を生成"""
    import random

    method = random.choice(list(TPSLMethod))

    method_weights = {
        "fixed": random.uniform(0.1, 0.4),
        "risk_reward": random.uniform(0.2, 0.5),
        "volatility": random.uniform(0.1, 0.4),
        "statistical": random.uniform(0.1, 0.3),
    }

    # Normalize method_weights to sum to 1.0
    total_weight = sum(method_weights.values())
    if total_weight > 0:
        for key in method_weights:
            method_weights[key] /= total_weight

    tpsl_gene = TPSLGene(
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
        method_weights=method_weights,
        enabled=True,
        priority=random.uniform(0.5, 1.5),
    )

    if config:
        # Anyの制約を適用（設定されている場合）
        if hasattr(config, "tpsl_method_constraints"):
            # 許可されたメソッドのみを使用
            allowed_methods = config.tpsl_method_constraints
            if allowed_methods:
                tpsl_gene.method = random.choice(
                    [TPSLMethod(m) for m in allowed_methods]
                )

        if hasattr(config, "tpsl_sl_range") and config.tpsl_sl_range is not None:
            sl_min, sl_max = config.tpsl_sl_range
            tpsl_gene.stop_loss_pct = random.uniform(sl_min, sl_max)
            tpsl_gene.base_stop_loss = random.uniform(sl_min, sl_max)

        if hasattr(config, "tpsl_tp_range") and config.tpsl_tp_range is not None:
            # TP範囲制約
            tp_min, tp_max = config.tpsl_tp_range
            tpsl_gene.take_profit_pct = random.uniform(tp_min, tp_max)

        if hasattr(config, "tpsl_rr_range") and config.tpsl_rr_range is not None:
            # リスクリワード比範囲制約
            rr_min, rr_max = config.tpsl_rr_range
            tpsl_gene.risk_reward_ratio = random.uniform(rr_min, rr_max)

        if (
            hasattr(config, "tpsl_atr_multiplier_range")
            and config.tpsl_atr_multiplier_range is not None
        ):
            # ATR倍率範囲制約
            atr_min, atr_max = config.tpsl_atr_multiplier_range
            tpsl_gene.atr_multiplier_sl = random.uniform(atr_min, atr_max)
            tpsl_gene.atr_multiplier_tp = random.uniform(atr_min * 1.5, atr_max * 2.0)

    return tpsl_gene


def crossover_tpsl_genes(
    parent1: TPSLGene, parent2: TPSLGene
) -> tuple[TPSLGene, TPSLGene]:
    """TP/SL遺伝子の交叉（ジェネリック関数使用）"""
    from ..utils.gene_utils import GeneticUtils

    # 基本フィールドのカテゴリ分け
    numeric_fields = [
        "stop_loss_pct",
        "take_profit_pct",
        "risk_reward_ratio",
        "base_stop_loss",
        "atr_multiplier_sl",
        "atr_multiplier_tp",
        "confidence_threshold",
        "priority",
        "lookback_period",
        "atr_period",
    ]
    enum_fields = ["method"]
    choice_fields = ["enabled"]

    # ジェネリック交叉を実行
    child1, child2 = GeneticUtils.crossover_generic_genes(
        parent1_gene=parent1,
        parent2_gene=parent2,
        gene_class=TPSLGene,
        numeric_fields=numeric_fields,
        enum_fields=enum_fields,
        choice_fields=choice_fields,
    )

    # 共有参照を防ぐため、method_weightsをコピー
    if hasattr(child1, "method_weights") and isinstance(child1.method_weights, dict):
        child1.method_weights = child1.method_weights.copy()
    if hasattr(child2, "method_weights") and isinstance(child2.method_weights, dict):
        child2.method_weights = child2.method_weights.copy()

    # method_weightsの特殊処理
    # 辞書の各キーにたいして比率の平均を取る
    all_keys = set(parent1.method_weights.keys()) | set(parent2.method_weights.keys())
    for key in all_keys:
        if key in parent1.method_weights and key in parent2.method_weights:
            # 両方にある場合、平均を取る
            child1.method_weights[key] = (
                parent1.method_weights[key] + parent2.method_weights[key]
            ) / 2
            child2.method_weights[key] = (
                parent1.method_weights[key] + parent2.method_weights[key]
            ) / 2
        else:
            # 片方しかない場合、そのまま継承
            if key in parent1.method_weights:
                child1.method_weights[key] = parent1.method_weights[key]
                child2.method_weights[key] = parent1.method_weights[key]
            else:
                child1.method_weights[key] = parent2.method_weights[key]
                child2.method_weights[key] = parent2.method_weights[key]

    return child1, child2


def mutate_tpsl_gene(gene: TPSLGene, mutation_rate: float = 0.1) -> TPSLGene:
    """TP/SL遺伝子の突然変異（ジェネリック関数使用）"""
    import random
    from ..utils.gene_utils import GeneticUtils

    # 基本フィールド
    numeric_fields: List[str] = [
        "stop_loss_pct",
        "take_profit_pct",
        "risk_reward_ratio",
        "base_stop_loss",
        "atr_multiplier_sl",
        "atr_multiplier_tp",
        "confidence_threshold",
        "priority",
        "lookback_period",
        "atr_period",
    ]

    enum_fields = ["method"]

    # 各フィールドの許容範囲
    numeric_ranges: Dict[str, tuple[float, float]] = {
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
    }

    # ジェネリック突然変異を実行
    mutated_gene = GeneticUtils.mutate_generic_gene(
        gene=gene,
        gene_class=TPSLGene,
        mutation_rate=mutation_rate,
        numeric_fields=numeric_fields,
        enum_fields=enum_fields,
        numeric_ranges=numeric_ranges,
    )

    # method_weightsの突然変異（辞書フィールドの特殊処理）
    if random.random() < mutation_rate:
        # method_weightsを乱数で調整
        for key in mutated_gene.method_weights:
            current_weight = mutated_gene.method_weights[key]
            # 現在の値を中心とした範囲で変動
            mutated_gene.method_weights[key] = current_weight * random.uniform(0.8, 1.2)

        # 合計が1.0になるよう正規化
        total_weight = sum(mutated_gene.method_weights.values())
        if total_weight > 0:
            for key in mutated_gene.method_weights:
                mutated_gene.method_weights[key] /= total_weight

    return mutated_gene
