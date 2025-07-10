"""
SmartConditionGenerator

責務を集約したロング・ショート条件生成器
計画書に基づいて、異なる指標の組み合わせ戦略、時間軸分離戦略、
複合条件戦略、指標特性活用戦略を実装
"""

import logging
import random
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum

from ..models.gene_strategy import Condition, IndicatorGene

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """戦略タイプ"""
    DIFFERENT_INDICATORS = "different_indicators"  # 異なる指標の組み合わせ
    TIME_SEPARATION = "time_separation"  # 時間軸分離
    COMPLEX_CONDITIONS = "complex_conditions"  # 複合条件
    INDICATOR_CHARACTERISTICS = "indicator_characteristics"  # 指標特性活用


class IndicatorType(Enum):
    """指標分類"""
    MOMENTUM = "momentum"  # モメンタム系
    TREND = "trend"  # トレンド系
    VOLATILITY = "volatility"  # ボラティリティ系


# 指標特性データベース（計画書の設計に基づく）
INDICATOR_CHARACTERISTICS = {
    "RSI": {
        "type": IndicatorType.MOMENTUM,
        "range": (0, 100),
        "long_zones": [(0, 30), (40, 60)],
        "short_zones": [(40, 60), (70, 100)],
        "neutral_zone": (40, 60),
        "oversold_threshold": 30,
        "overbought_threshold": 70
    },
    "STOCH": {
        "type": IndicatorType.MOMENTUM,
        "range": (0, 100),
        "long_zones": [(0, 20), (40, 60)],
        "short_zones": [(40, 60), (80, 100)],
        "neutral_zone": (40, 60),
        "oversold_threshold": 20,
        "overbought_threshold": 80
    },
    "CCI": {
        "type": IndicatorType.MOMENTUM,
        "range": (-200, 200),
        "long_zones": [(-200, -100), (-50, 50)],
        "short_zones": [(-50, 50), (100, 200)],
        "neutral_zone": (-50, 50),
        "oversold_threshold": -100,
        "overbought_threshold": 100
    },
    "MACD": {
        "type": IndicatorType.MOMENTUM,
        "range": None,  # 価格依存
        "zero_cross": True,
        "signal_line": True
    },
    "SMA": {
        "type": IndicatorType.TREND,
        "price_comparison": True,
        "trend_following": True
    },
    "EMA": {
        "type": IndicatorType.TREND,
        "price_comparison": True,
        "trend_following": True
    },
    "MAMA": {
        "type": IndicatorType.TREND,
        "price_comparison": True,
        "adaptive": True
    },
    "ADX": {
        "type": IndicatorType.TREND,
        "range": (0, 100),
        "trend_strength": True,
        "no_direction": True,  # 方向性を示さない
        "strong_trend_threshold": 25
    },
    "BB": {
        "type": IndicatorType.VOLATILITY,
        "components": ["upper", "middle", "lower"],
        "mean_reversion": True,
        "breakout_strategy": True
    },
    "ATR": {
        "type": IndicatorType.VOLATILITY,
        "range": (0, None),
        "volatility_measure": True
    }
}

# 組み合わせルール（計画書の設計に基づく）
COMBINATION_RULES = {
    "trend_momentum": {
        "description": "トレンド系 + モメンタム系",
        "long_indicators": [IndicatorType.TREND, IndicatorType.MOMENTUM],
        "short_indicators": [IndicatorType.TREND, IndicatorType.MOMENTUM],
        "weight": 0.4
    },
    "volatility_trend": {
        "description": "ボラティリティ系 + トレンド系",
        "long_indicators": [IndicatorType.VOLATILITY, IndicatorType.TREND],
        "short_indicators": [IndicatorType.VOLATILITY, IndicatorType.TREND],
        "weight": 0.3
    },
    "momentum_volatility": {
        "description": "モメンタム系 + ボラティリティ系",
        "long_indicators": [IndicatorType.MOMENTUM, IndicatorType.VOLATILITY],
        "short_indicators": [IndicatorType.MOMENTUM, IndicatorType.VOLATILITY],
        "weight": 0.2
    },
    "single_indicator_multi_timeframe": {
        "description": "単一指標の複数時間軸",
        "long_indicators": [IndicatorType.MOMENTUM],
        "short_indicators": [IndicatorType.MOMENTUM],
        "weight": 0.1
    }
}


class SmartConditionGenerator:
    """
    責務を集約したロング・ショート条件生成器

    計画書に基づいて以下の戦略を実装：
    1. 異なる指標の組み合わせ戦略
    2. 時間軸分離戦略
    3. 複合条件戦略
    4. 指標特性活用戦略
    """

    def __init__(self, enable_smart_generation: bool = True):
        """
        初期化

        Args:
            enable_smart_generation: 新しいスマート生成を有効にするか
        """
        self.enable_smart_generation = enable_smart_generation
        self.logger = logger

    def generate_balanced_conditions(
        self,
        indicators: List[IndicatorGene]
    ) -> Tuple[List[Condition], List[Condition], List[Condition]]:
        """
        バランスの取れたロング・ショート条件を生成

        Args:
            indicators: 指標リスト

        Returns:
            (long_entry_conditions, short_entry_conditions, exit_conditions)のタプル
        """
        try:
            if not self.enable_smart_generation:
                return self._generate_fallback_conditions()

            if not indicators:
                return self._generate_fallback_conditions()

            # 戦略タイプを選択
            strategy_type = self._select_strategy_type(indicators)

            # 選択された戦略に基づいて条件を生成
            if strategy_type == StrategyType.DIFFERENT_INDICATORS:
                return self._generate_different_indicators_strategy(indicators)
            elif strategy_type == StrategyType.TIME_SEPARATION:
                return self._generate_time_separation_strategy(indicators)
            elif strategy_type == StrategyType.COMPLEX_CONDITIONS:
                return self._generate_complex_conditions_strategy(indicators)
            elif strategy_type == StrategyType.INDICATOR_CHARACTERISTICS:
                return self._generate_indicator_characteristics_strategy(indicators)
            else:
                return self._generate_fallback_conditions()

        except Exception as e:
            self.logger.error(f"スマート条件生成エラー: {e}")
            return self._generate_fallback_conditions()

    def _select_strategy_type(self, indicators: List[IndicatorGene]) -> StrategyType:
        """
        利用可能な指標に基づいて戦略タイプを選択

        Args:
            indicators: 指標リスト

        Returns:
            選択された戦略タイプ
        """
        try:
            # 指標の種類を分析
            indicator_types = set()
            for indicator in indicators:
                if indicator.enabled and indicator.type in INDICATOR_CHARACTERISTICS:
                    char = INDICATOR_CHARACTERISTICS[indicator.type]
                    indicator_types.add(char["type"])

            # 複数の指標タイプがある場合は異なる指標の組み合わせ戦略
            if len(indicator_types) >= 2:
                return StrategyType.DIFFERENT_INDICATORS

            # 同じ指標が複数ある場合は時間軸分離戦略
            indicator_counts = {}
            for indicator in indicators:
                if indicator.enabled:
                    indicator_counts[indicator.type] = indicator_counts.get(indicator.type, 0) + 1

            if any(count >= 2 for count in indicator_counts.values()):
                return StrategyType.TIME_SEPARATION

            # ボリンジャーバンドがある場合は指標特性活用戦略
            if any(indicator.type == "BB" and indicator.enabled for indicator in indicators):
                return StrategyType.INDICATOR_CHARACTERISTICS

            # デフォルトは複合条件戦略
            return StrategyType.COMPLEX_CONDITIONS

        except Exception as e:
            self.logger.error(f"戦略タイプ選択エラー: {e}")
            return StrategyType.DIFFERENT_INDICATORS

    def _generate_fallback_conditions(self) -> Tuple[List[Condition], List[Condition], List[Condition]]:
        """
        フォールバック条件を生成

        Returns:
            (long_entry_conditions, short_entry_conditions, exit_conditions)のタプル
        """
        long_conditions = [
            Condition(left_operand="close", operator=">", right_operand="open")
        ]
        short_conditions = [
            Condition(left_operand="close", operator="<", right_operand="open")
        ]
        exit_conditions = []

        return long_conditions, short_conditions, exit_conditions

    def _generate_different_indicators_strategy(
        self,
        indicators: List[IndicatorGene]
    ) -> Tuple[List[Condition], List[Condition], List[Condition]]:
        """
        異なる指標の組み合わせ戦略

        Args:
            indicators: 指標リスト

        Returns:
            (long_entry_conditions, short_entry_conditions, exit_conditions)のタプル
        """
        try:
            # 指標をタイプ別に分類
            indicators_by_type = {
                IndicatorType.MOMENTUM: [],
                IndicatorType.TREND: [],
                IndicatorType.VOLATILITY: []
            }

            for indicator in indicators:
                if indicator.enabled and indicator.type in INDICATOR_CHARACTERISTICS:
                    char = INDICATOR_CHARACTERISTICS[indicator.type]
                    indicators_by_type[char["type"]].append(indicator)

            # トレンド系 + モメンタム系の組み合わせを優先
            long_conditions = []
            short_conditions = []

            # ロング条件：トレンド系指標
            if indicators_by_type[IndicatorType.TREND]:
                trend_indicator = random.choice(indicators_by_type[IndicatorType.TREND])
                long_conditions.extend(self._create_trend_long_conditions(trend_indicator))

            # ロング条件：モメンタム系指標
            if indicators_by_type[IndicatorType.MOMENTUM]:
                momentum_indicator = random.choice(indicators_by_type[IndicatorType.MOMENTUM])
                long_conditions.extend(self._create_momentum_long_conditions(momentum_indicator))

            # ショート条件：異なる指標を使用
            if indicators_by_type[IndicatorType.TREND]:
                trend_indicator = random.choice(indicators_by_type[IndicatorType.TREND])
                short_conditions.extend(self._create_trend_short_conditions(trend_indicator))

            if indicators_by_type[IndicatorType.MOMENTUM]:
                # ロングとは異なるモメンタム指標を選択
                available_momentum = [
                    ind for ind in indicators_by_type[IndicatorType.MOMENTUM]
                    if ind.type != (long_conditions[1].left_operand.split('_')[0] if len(long_conditions) > 1 else "")
                ]
                if available_momentum:
                    momentum_indicator = random.choice(available_momentum)
                else:
                    momentum_indicator = random.choice(indicators_by_type[IndicatorType.MOMENTUM])
                short_conditions.extend(self._create_momentum_short_conditions(momentum_indicator))

            # 条件が空の場合はフォールバック
            if not long_conditions:
                long_conditions = [Condition(left_operand="close", operator=">", right_operand="open")]
            if not short_conditions:
                short_conditions = [Condition(left_operand="close", operator="<", right_operand="open")]

            return long_conditions, short_conditions, []

        except Exception as e:
            self.logger.error(f"異なる指標組み合わせ戦略エラー: {e}")
            return self._generate_fallback_conditions()

    def _create_trend_long_conditions(self, indicator: IndicatorGene) -> List[Condition]:
        """トレンド系指標のロング条件を生成"""
        indicator_name = f"{indicator.type}_{indicator.parameters.get('period', 14)}"

        if indicator.type in ["SMA", "EMA", "MAMA"]:
            return [Condition(left_operand="close", operator=">", right_operand=indicator_name)]
        else:
            return []

    def _create_trend_short_conditions(self, indicator: IndicatorGene) -> List[Condition]:
        """トレンド系指標のショート条件を生成"""
        indicator_name = f"{indicator.type}_{indicator.parameters.get('period', 14)}"

        if indicator.type in ["SMA", "EMA", "MAMA"]:
            return [Condition(left_operand="close", operator="<", right_operand=indicator_name)]
        else:
            return []

    def _create_momentum_long_conditions(self, indicator: IndicatorGene) -> List[Condition]:
        """モメンタム系指標のロング条件を生成"""
        indicator_name = f"{indicator.type}_{indicator.parameters.get('period', 14)}"

        if indicator.type == "RSI":
            # RSI: 売られすぎ領域でロング
            threshold = random.uniform(20, 35)
            return [Condition(left_operand=indicator_name, operator="<", right_operand=threshold)]
        elif indicator.type == "STOCH":
            # STOCH: 売られすぎ領域でロング
            threshold = random.uniform(15, 25)
            return [Condition(left_operand=indicator_name, operator="<", right_operand=threshold)]
        elif indicator.type == "CCI":
            # CCI: 売られすぎ領域でロング
            threshold = random.uniform(-150, -80)
            return [Condition(left_operand=indicator_name, operator="<", right_operand=threshold)]
        elif indicator.type == "MACD":
            # MACD: ゼロライン上抜けでロング
            return [Condition(left_operand=indicator_name, operator=">", right_operand=0)]
        else:
            return []

    def _create_momentum_short_conditions(self, indicator: IndicatorGene) -> List[Condition]:
        """モメンタム系指標のショート条件を生成"""
        indicator_name = f"{indicator.type}_{indicator.parameters.get('period', 14)}"

        if indicator.type == "RSI":
            # RSI: 買われすぎ領域でショート
            threshold = random.uniform(65, 85)
            return [Condition(left_operand=indicator_name, operator=">", right_operand=threshold)]
        elif indicator.type == "STOCH":
            # STOCH: 買われすぎ領域でショート
            threshold = random.uniform(75, 85)
            return [Condition(left_operand=indicator_name, operator=">", right_operand=threshold)]
        elif indicator.type == "CCI":
            # CCI: 買われすぎ領域でショート
            threshold = random.uniform(80, 150)
            return [Condition(left_operand=indicator_name, operator=">", right_operand=threshold)]
        elif indicator.type == "MACD":
            # MACD: ゼロライン下抜けでショート
            return [Condition(left_operand=indicator_name, operator="<", right_operand=0)]
        else:
            return []

    def _generate_time_separation_strategy(
        self,
        indicators: List[IndicatorGene]
    ) -> Tuple[List[Condition], List[Condition], List[Condition]]:
        """
        時間軸分離戦略（同じ指標の異なる期間を使用）

        Args:
            indicators: 指標リスト

        Returns:
            (long_entry_conditions, short_entry_conditions, exit_conditions)のタプル
        """
        try:
            # 同じタイプの指標を見つける
            indicator_groups = {}
            for indicator in indicators:
                if indicator.enabled:
                    if indicator.type not in indicator_groups:
                        indicator_groups[indicator.type] = []
                    indicator_groups[indicator.type].append(indicator)

            long_conditions = []
            short_conditions = []

            # 複数の期間を持つ指標を選択
            for indicator_type, indicator_list in indicator_groups.items():
                if len(indicator_list) >= 2 and indicator_type in INDICATOR_CHARACTERISTICS:
                    # 短期と長期の指標を選択
                    sorted_indicators = sorted(
                        indicator_list,
                        key=lambda x: x.parameters.get('period', 14)
                    )
                    short_term = sorted_indicators[0]
                    long_term = sorted_indicators[-1]

                    # 短期・長期組み合わせ条件を生成
                    if indicator_type == "RSI":
                        # 短期RSI売られすぎ + 長期RSI上昇トレンド
                        short_name = f"{short_term.type}_{short_term.parameters.get('period', 7)}"
                        long_name = f"{long_term.type}_{long_term.parameters.get('period', 21)}"

                        long_conditions.extend([
                            Condition(left_operand=short_name, operator="<", right_operand=30),
                            Condition(left_operand=long_name, operator=">", right_operand=50)
                        ])

                        short_conditions.extend([
                            Condition(left_operand=short_name, operator=">", right_operand=70),
                            Condition(left_operand=long_name, operator="<", right_operand=50)
                        ])
                    break

            # 条件が空の場合はフォールバック
            if not long_conditions:
                return self._generate_fallback_conditions()

            return long_conditions, short_conditions, []

        except Exception as e:
            self.logger.error(f"時間軸分離戦略エラー: {e}")
            return self._generate_fallback_conditions()

    def _generate_indicator_characteristics_strategy(
        self,
        indicators: List[IndicatorGene]
    ) -> Tuple[List[Condition], List[Condition], List[Condition]]:
        """
        指標特性活用戦略（ボリンジャーバンドの正しい実装など）

        Args:
            indicators: 指標リスト

        Returns:
            (long_entry_conditions, short_entry_conditions, exit_conditions)のタプル
        """
        try:
            long_conditions = []
            short_conditions = []

            # ボリンジャーバンドの特性を活用
            bb_indicators = [ind for ind in indicators if ind.type == "BB" and ind.enabled]
            if bb_indicators:
                bb_indicator = bb_indicators[0]
                period = bb_indicator.parameters.get('period', 20)

                # ボリンジャーバンドの3つの値を活用（計画書の設計通り）
                bb_upper = f"BB_Upper_{period}"
                bb_middle = f"BB_Middle_{period}"
                bb_lower = f"BB_Lower_{period}"

                # 逆張り戦略：バンド突破後の回帰を狙う
                long_conditions.extend([
                    Condition(left_operand="close", operator="<", right_operand=bb_lower),  # 下限突破
                    Condition(left_operand="close", operator=">", right_operand=bb_middle)  # 中央線回復
                ])

                short_conditions.extend([
                    Condition(left_operand="close", operator=">", right_operand=bb_upper),  # 上限突破
                    Condition(left_operand="close", operator="<", right_operand=bb_middle)  # 中央線割れ
                ])

            # ADXの正しい活用（方向性指標との組み合わせ）
            adx_indicators = [ind for ind in indicators if ind.type == "ADX" and ind.enabled]
            if adx_indicators and not bb_indicators:
                adx_indicator = adx_indicators[0]
                period = adx_indicator.parameters.get('period', 14)
                adx_name = f"ADX_{period}"

                # ADX + 価格方向の組み合わせ（計画書の設計通り）
                long_conditions.extend([
                    Condition(left_operand=adx_name, operator=">", right_operand=25),  # 強いトレンド
                    Condition(left_operand="close", operator=">", right_operand="open")  # 上昇方向
                ])

                short_conditions.extend([
                    Condition(left_operand=adx_name, operator=">", right_operand=25),  # 強いトレンド
                    Condition(left_operand="close", operator="<", right_operand="open")  # 下降方向
                ])

            # 条件が空の場合はフォールバック
            if not long_conditions:
                return self._generate_fallback_conditions()

            return long_conditions, short_conditions, []

        except Exception as e:
            self.logger.error(f"指標特性活用戦略エラー: {e}")
            return self._generate_fallback_conditions()

    def _generate_complex_conditions_strategy(
        self,
        indicators: List[IndicatorGene]
    ) -> Tuple[List[Condition], List[Condition], List[Condition]]:
        """
        複合条件戦略（複数の条件を組み合わせて確率を高める）

        Args:
            indicators: 指標リスト

        Returns:
            (long_entry_conditions, short_entry_conditions, exit_conditions)のタプル
        """
        try:
            # 利用可能な指標から複数の条件を組み合わせ
            long_conditions = []
            short_conditions = []

            for indicator in indicators[:2]:  # 最大2つの指標を使用
                if not indicator.enabled:
                    continue

                if indicator.type in INDICATOR_CHARACTERISTICS:
                    # 各指標の特性に基づいて条件を追加
                    long_conds = self._create_momentum_long_conditions(indicator)
                    short_conds = self._create_momentum_short_conditions(indicator)

                    if long_conds:
                        long_conditions.extend(long_conds)
                    if short_conds:
                        short_conditions.extend(short_conds)

            # 条件が空の場合はフォールバック
            if not long_conditions:
                return self._generate_fallback_conditions()

            return long_conditions, short_conditions, []

        except Exception as e:
            self.logger.error(f"複合条件戦略エラー: {e}")
            return self._generate_fallback_conditions()