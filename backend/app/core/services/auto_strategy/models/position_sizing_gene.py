"""
ポジションサイジング遺伝子モデル

ポジションサイジング設定をGA最適化対象として表現するための遺伝子モデルです。
TP/SL遺伝子と同様に、GA操作（交叉、突然変異）の対象となります。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import random
import logging

logger = logging.getLogger(__name__)


class PositionSizingMethod(Enum):
    """ポジションサイジング決定方式"""

    HALF_OPTIMAL_F = "half_optimal_f"  # ハーフオプティマルF
    VOLATILITY_BASED = "volatility_based"  # ボラティリティベース
    FIXED_RATIO = "fixed_ratio"  # 固定比率ベース
    FIXED_QUANTITY = "fixed_quantity"  # 枚数ベース


@dataclass
class PositionSizingGene:
    """
    ポジションサイジング遺伝子

    GA最適化対象としてのポジションサイジング設定を表現します。
    TP/SL遺伝子と同じレベルで進化します。
    """

    # ポジションサイジング決定方式
    method: PositionSizingMethod = PositionSizingMethod.FIXED_RATIO

    # ハーフオプティマルF方式のパラメータ
    lookback_period: int = 100  # 過去データ参照期間（50-200日）
    optimal_f_multiplier: float = 0.5  # オプティマルFの倍率（0.25-0.75）

    # ボラティリティベース方式のパラメータ
    atr_period: int = 14  # ATR計算期間（10-30日）
    atr_multiplier: float = 2.0  # ATRに対する倍率（1.0-4.0）
    risk_per_trade: float = 0.02  # 1取引あたりのリスク（1%-5%）

    # 固定比率ベース方式のパラメータ
    fixed_ratio: float = 0.1  # 口座残高に対する比率（5%-30%）

    # 枚数ベース方式のパラメータ
    fixed_quantity: float = 1.0  # 固定枚数（0.1-5.0単位）

    # 共通パラメータ
    min_position_size: float = 0.01  # 最小ポジションサイズ
    max_position_size: float = 10.0  # 最大ポジションサイズ（枚数単位）
    enabled: bool = True  # 有効フラグ
    priority: float = 1.0  # 優先度（0.5-1.5）

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "method": self.method.value,
            "lookback_period": self.lookback_period,
            "optimal_f_multiplier": self.optimal_f_multiplier,
            "atr_period": self.atr_period,
            "atr_multiplier": self.atr_multiplier,
            "risk_per_trade": self.risk_per_trade,
            "fixed_ratio": self.fixed_ratio,
            "fixed_quantity": self.fixed_quantity,
            "min_position_size": self.min_position_size,
            "max_position_size": self.max_position_size,
            "enabled": self.enabled,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PositionSizingGene":
        """辞書形式から遺伝子を復元"""
        return cls(
            method=PositionSizingMethod(data.get("method", "fixed_ratio")),
            lookback_period=data.get("lookback_period", 100),
            optimal_f_multiplier=data.get("optimal_f_multiplier", 0.5),
            atr_period=data.get("atr_period", 14),
            atr_multiplier=data.get("atr_multiplier", 2.0),
            risk_per_trade=data.get("risk_per_trade", 0.02),
            fixed_ratio=data.get("fixed_ratio", 0.1),
            fixed_quantity=data.get("fixed_quantity", 1.0),
            min_position_size=data.get("min_position_size", 0.01),
            max_position_size=data.get("max_position_size", 1.0),
            enabled=data.get("enabled", True),
            priority=data.get("priority", 1.0),
        )

    def validate(self) -> Tuple[bool, List[str]]:
        """
        遺伝子の妥当性を検証

        Returns:
            (is_valid, error_messages)のタプル
        """
        errors = []

        # 基本パラメータの検証
        if self.lookback_period < 10 or self.lookback_period > 500:
            errors.append("lookback_periodは10-500の範囲である必要があります")

        if self.optimal_f_multiplier < 0.1 or self.optimal_f_multiplier > 1.0:
            errors.append("optimal_f_multiplierは0.1-1.0の範囲である必要があります")

        if self.atr_period < 5 or self.atr_period > 50:
            errors.append("atr_periodは5-50の範囲である必要があります")

        if self.atr_multiplier < 0.5 or self.atr_multiplier > 10.0:
            errors.append("atr_multiplierは0.5-10.0の範囲である必要があります")

        if self.risk_per_trade < 0.001 or self.risk_per_trade > 0.1:
            errors.append("risk_per_tradeは0.1%-10%の範囲である必要があります")

        if self.fixed_ratio < 0.01 or self.fixed_ratio > 0.5:
            errors.append("fixed_ratioは1%-50%の範囲である必要があります")

        if (
            self.fixed_quantity < 0.01 or self.fixed_quantity > 100.0
        ):  # 上限を100.0に拡大
            errors.append("fixed_quantityは0.01-100.0の範囲である必要があります")

        if self.min_position_size < 0.001 or self.min_position_size > 1.0:
            errors.append("min_position_sizeは0.001-1.0の範囲である必要があります")

        if (
            self.max_position_size < self.min_position_size
            or self.max_position_size > 100.0  # 上限を100.0に拡大
        ):
            errors.append(
                "max_position_sizeはmin_position_size以上100.0以下である必要があります"
            )

        if self.priority < 0.1 or self.priority > 2.0:
            errors.append("priorityは0.1-2.0の範囲である必要があります")

        return len(errors) == 0, errors

    def calculate_position_size(
        self,
        account_balance: float,
        current_price: float,
        market_data: Optional[Dict[str, Any]] = None,
        trade_history: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """
        遺伝子設定に基づいてポジションサイズを計算

        Args:
            account_balance: 口座残高
            current_price: 現在価格
            market_data: 市場データ（ATR値など）
            trade_history: 取引履歴（ハーフオプティマルF用）

        Returns:
            計算されたポジションサイズ
        """
        try:
            if not self.enabled:
                return self.min_position_size

            if self.method == PositionSizingMethod.HALF_OPTIMAL_F:
                return self._calculate_half_optimal_f(account_balance, trade_history)

            elif self.method == PositionSizingMethod.VOLATILITY_BASED:
                return self._calculate_volatility_based(
                    account_balance, current_price, market_data
                )

            elif self.method == PositionSizingMethod.FIXED_RATIO:
                return self._calculate_fixed_ratio(account_balance, current_price)

            elif self.method == PositionSizingMethod.FIXED_QUANTITY:
                return self._calculate_fixed_quantity()

            else:
                # フォールバック: 固定比率
                return self._calculate_fixed_ratio(account_balance, current_price)

        except Exception as e:
            logger.error(f"ポジションサイズ計算エラー: {e}")
            # エラー時はmin_position_sizeを返す
            return self.min_position_size

    def _calculate_half_optimal_f(
        self, account_balance: float, trade_history: Optional[List[Dict[str, Any]]]
    ) -> float:
        """ハーフオプティマルF方式でポジションサイズを計算"""
        try:
            if not trade_history or len(trade_history) < 10:
                # データ不足時は固定比率にフォールバック
                return self._calculate_fixed_ratio(account_balance, 1.0)

            # 過去のlookback_period分の取引を分析
            recent_trades = trade_history[-self.lookback_period :]

            wins = [t for t in recent_trades if t.get("pnl", 0) > 0]
            losses = [t for t in recent_trades if t.get("pnl", 0) < 0]

            if len(recent_trades) == 0:
                return self.min_position_size

            win_rate = len(wins) / len(recent_trades)
            avg_win = sum(t.get("pnl", 0) for t in wins) / len(wins) if wins else 0
            avg_loss = (
                abs(sum(t.get("pnl", 0) for t in losses) / len(losses)) if losses else 0
            )

            if avg_win <= 0 or avg_loss <= 0:
                return self.min_position_size

            # オプティマルF計算
            optimal_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            half_optimal_f = max(0, optimal_f * self.optimal_f_multiplier)

            # 口座残高に対する比率として適用
            position_size = account_balance * half_optimal_f

            return self._apply_size_limits(position_size)

        except Exception as e:
            logger.error(f"ハーフオプティマルF計算エラー: {e}")
            return self.min_position_size

    def _calculate_volatility_based(
        self,
        account_balance: float,
        current_price: float,
        market_data: Optional[Dict[str, Any]],
    ) -> float:
        """ボラティリティベース方式でポジションサイズを計算"""
        try:
            # ATR値を取得（デフォルト値を設定）
            atr_value = 0.02 * current_price  # デフォルト2%
            if market_data and "atr" in market_data:
                atr_value = market_data["atr"]
            elif market_data and "atr_pct" in market_data:
                atr_value = market_data["atr_pct"] * current_price

            # リスク量を計算
            risk_amount = account_balance * self.risk_per_trade

            # ポジションサイズを計算
            position_size = risk_amount / (atr_value * self.atr_multiplier)

            return self._apply_size_limits(position_size)

        except Exception as e:
            logger.error(f"ボラティリティベース計算エラー: {e}")
            return self.min_position_size

    def _calculate_fixed_ratio(
        self, account_balance: float, current_price: float
    ) -> float:
        """固定比率方式でポジションサイズを計算"""
        try:
            position_size = account_balance * self.fixed_ratio
            return self._apply_size_limits(position_size)

        except Exception as e:
            logger.error(f"固定比率計算エラー: {e}")
            return self.min_position_size

    def _calculate_fixed_quantity(self) -> float:
        """固定枚数方式でポジションサイズを計算"""
        try:
            return self._apply_size_limits(self.fixed_quantity)

        except Exception as e:
            logger.error(f"固定枚数計算エラー: {e}")
            return self.min_position_size

    def _apply_size_limits(self, position_size: float) -> float:
        """ポジションサイズに制限を適用"""
        return max(self.min_position_size, min(position_size, self.max_position_size))


def create_random_position_sizing_gene() -> PositionSizingGene:
    """ランダムなポジションサイジング遺伝子を生成"""
    method = random.choice(list(PositionSizingMethod))

    return PositionSizingGene(
        method=method,
        lookback_period=random.randint(50, 200),
        optimal_f_multiplier=random.uniform(0.25, 0.75),
        atr_period=random.randint(10, 30),
        atr_multiplier=random.uniform(1.0, 4.0),
        risk_per_trade=random.uniform(0.01, 0.05),
        fixed_ratio=random.uniform(0.05, 0.3),
        fixed_quantity=random.uniform(0.1, 10.0),  # 上限を10.0に調整
        min_position_size=random.uniform(0.01, 0.05),
        max_position_size=random.uniform(5.0, 50.0),  # 適切な範囲に調整
        enabled=True,
        priority=random.uniform(0.5, 1.5),
    )


def crossover_position_sizing_genes(
    parent1: PositionSizingGene, parent2: PositionSizingGene
) -> Tuple[PositionSizingGene, PositionSizingGene]:
    """ポジションサイジング遺伝子の交叉"""
    # 方式の交叉（ランダム選択）
    child1_method = random.choice([parent1.method, parent2.method])
    child2_method = random.choice([parent1.method, parent2.method])

    # パラメータの交叉（平均値）
    child1 = PositionSizingGene(
        method=child1_method,
        lookback_period=random.choice(
            [parent1.lookback_period, parent2.lookback_period]
        ),
        optimal_f_multiplier=(
            parent1.optimal_f_multiplier + parent2.optimal_f_multiplier
        )
        / 2,
        atr_period=random.choice([parent1.atr_period, parent2.atr_period]),
        atr_multiplier=(parent1.atr_multiplier + parent2.atr_multiplier) / 2,
        risk_per_trade=(parent1.risk_per_trade + parent2.risk_per_trade) / 2,
        fixed_ratio=(parent1.fixed_ratio + parent2.fixed_ratio) / 2,
        fixed_quantity=(parent1.fixed_quantity + parent2.fixed_quantity) / 2,
        min_position_size=(parent1.min_position_size + parent2.min_position_size) / 2,
        max_position_size=(parent1.max_position_size + parent2.max_position_size) / 2,
        enabled=random.choice([parent1.enabled, parent2.enabled]),
        priority=(parent1.priority + parent2.priority) / 2,
    )

    child2 = PositionSizingGene(
        method=child2_method,
        lookback_period=random.choice(
            [parent2.lookback_period, parent1.lookback_period]
        ),
        optimal_f_multiplier=(
            parent2.optimal_f_multiplier + parent1.optimal_f_multiplier
        )
        / 2,
        atr_period=random.choice([parent2.atr_period, parent1.atr_period]),
        atr_multiplier=(parent2.atr_multiplier + parent1.atr_multiplier) / 2,
        risk_per_trade=(parent2.risk_per_trade + parent1.risk_per_trade) / 2,
        fixed_ratio=(parent2.fixed_ratio + parent1.fixed_ratio) / 2,
        fixed_quantity=(parent2.fixed_quantity + parent1.fixed_quantity) / 2,
        min_position_size=(parent2.min_position_size + parent1.min_position_size) / 2,
        max_position_size=(parent2.max_position_size + parent1.max_position_size) / 2,
        enabled=random.choice([parent2.enabled, parent1.enabled]),
        priority=(parent2.priority + parent1.priority) / 2,
    )

    return child1, child2


def mutate_position_sizing_gene(
    gene: PositionSizingGene, mutation_rate: float = 0.1
) -> PositionSizingGene:
    """ポジションサイジング遺伝子の突然変異"""
    mutated = PositionSizingGene(**gene.to_dict())

    # 方式の突然変異
    if random.random() < mutation_rate:
        mutated.method = random.choice(list(PositionSizingMethod))

    # パラメータの突然変異
    if random.random() < mutation_rate:
        mutated.lookback_period = max(
            50, min(200, int(mutated.lookback_period * random.uniform(0.8, 1.2)))
        )

    if random.random() < mutation_rate:
        mutated.optimal_f_multiplier = max(
            0.25, min(0.75, mutated.optimal_f_multiplier * random.uniform(0.8, 1.2))
        )

    if random.random() < mutation_rate:
        mutated.atr_period = max(
            10, min(30, int(mutated.atr_period * random.uniform(0.8, 1.2)))
        )

    if random.random() < mutation_rate:
        mutated.atr_multiplier = max(
            1.0, min(4.0, mutated.atr_multiplier * random.uniform(0.8, 1.2))
        )

    if random.random() < mutation_rate:
        mutated.risk_per_trade = max(
            0.01, min(0.05, mutated.risk_per_trade * random.uniform(0.8, 1.2))
        )

    if random.random() < mutation_rate:
        mutated.fixed_ratio = max(
            0.05, min(0.3, mutated.fixed_ratio * random.uniform(0.8, 1.2))
        )

    if random.random() < mutation_rate:
        mutated.fixed_quantity = max(
            0.1,
            min(
                100.0, mutated.fixed_quantity * random.uniform(0.8, 1.2)
            ),  # 上限を100.0に拡大
        )

    if random.random() < mutation_rate:
        mutated.min_position_size = max(
            0.01, min(0.1, mutated.min_position_size * random.uniform(0.8, 1.2))
        )

    if random.random() < mutation_rate:
        mutated.max_position_size = max(
            mutated.min_position_size,
            min(
                100.0, mutated.max_position_size * random.uniform(0.8, 1.2)
            ),  # 上限を100.0に拡大
        )

    if random.random() < mutation_rate:
        mutated.priority = max(
            0.5, min(1.5, mutated.priority * random.uniform(0.8, 1.2))
        )

    return mutated
