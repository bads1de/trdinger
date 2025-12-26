"""
条件生成のためのComplexConditionsStrategy。

トレンド押し目買い、ゴールデンクロス、ブレイクアウトなどの
王道パターンに基づいた条件を生成します。
特定の指標名に依存せず、メタデータ（スケールタイプ、カテゴリ、戻り値定義）を活用して
汎用的に条件を構築します。
"""

import logging
import random
from typing import Any, List, Tuple, Union

from app.services.indicators.config import indicator_registry, IndicatorScaleType
from ..config.constants import IndicatorType
from ..genes import Condition, ConditionGroup, IndicatorGene

logger = logging.getLogger(__name__)


class ComplexConditionsStrategy:
    """
    複数の指標組み合わせで意味のある条件を生成する戦略。
    """

    def __init__(self, condition_generator: Any) -> None:
        self.gen = condition_generator

    def generate_conditions(self, indicators: List[IndicatorGene]) -> Tuple[
        List[Union[Condition, ConditionGroup]],
        List[Union[Condition, ConditionGroup]],
        List[Condition],
    ]:
        """
        複数の指標を組み合わせて王道パターンに基づいた条件を生成

        トレンドフォロー、移動平均クロス、ボラティリティブレイクアウトの
        3つの主要なパターンから取引条件を構築します。

        Args:
            indicators: 生成済みの指標遺伝子リスト

        Returns:
            (ロング条件, ショート条件, 予備)のタプル
        """
        long_conds, short_conds = [], []

        # 1. トレンドフォローパターン (Trend + Momentum 順張り)
        classified = self.gen._classify_indicators(indicators)
        tp_long, tp_short = self._create_trend_follow(classified)
        long_conds.extend(tp_long)
        short_conds.extend(tp_short)

        # 2. 移動平均クロス
        cross_long, cross_short = self._create_cross(indicators)
        long_conds.extend(cross_long)
        short_conds.extend(cross_short)

        # 3. ボラティリティブレイクアウト
        break_long, break_short = self._create_breakout(indicators)
        long_conds.extend(break_long)
        short_conds.extend(break_short)

        # 条件が全く生成されなかった場合のフォールバック
        if not long_conds and not short_conds:
            return self.gen.generate_fallback_conditions(indicators)

        return (
            self.gen._structure_conditions(long_conds),
            self.gen._structure_conditions(short_conds),
            [],
        )

    def _create_trend_follow(self, classified):
        """トレンドフォロー条件（順張り特化）を生成"""
        longs, shorts = [], []
        trends, momentums = (
            classified[IndicatorType.TREND],
            classified[IndicatorType.MOMENTUM],
        )
        if not trends or not momentums:
            return longs, shorts

        trend, momentum = random.choice(trends), random.choice(momentums)
        t_name, m_name = self.gen._get_indicator_name(
            trend
        ), self.gen._get_indicator_name(momentum)

        # 順張り特化: Close > Trend AND Momentum > Bullish
        cfg = indicator_registry.get_indicator_config(momentum.type)
        scale_type = cfg.scale_type if cfg else None

        th_long, th_short = 0, 0
        if scale_type == IndicatorScaleType.OSCILLATOR_0_100:
            th_long = random.choice([55, 60, 65])
            th_short = random.choice([45, 40, 35])
        elif scale_type == IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100:
            th_long = random.choice([10, 25, 50])
            th_short = random.choice([-10, -25, -50])
        elif scale_type == IndicatorScaleType.MOMENTUM_ZERO_CENTERED:
            th_long = 0
            th_short = 0
        elif scale_type == IndicatorScaleType.PRICE:
            # 価格スケールの場合は閾値ではなく、Close自体と比較させる
            # ただしここでは right_operand に数値を期待している箇所もあるため
            # 文字列 "close" を許容するようにシステム全体が作られている前提
            th_long = "close"
            th_short = "close"
        else:
            # 不明な場合やその他のスケール
            if scale_type == IndicatorScaleType.PRICE_RATIO:
                th_long = 1.01
                th_short = 0.99
            else:
                th_long = 0
                th_short = 0

        # ロング: Close > Trend AND Momentum > High
        long_conds = [
            Condition(left_operand="Close", operator=">", right_operand=t_name),
        ]
        # Momentumの比較対象が "close" の場合は、Momentum > Close (または < Close) になる
        # もしMomentumがRSI(0-100)なら、RSI > 60 となる
        
        # operatorの決定: 
        # オシレーターなら > th_long
        # もし th_long が "close" なら、Momentum > Close (トレンドフォローならこれで良いか？)
        # SUPERTREND > Close は上昇トレンドを示すのでOK
        op_long = ">"
        op_short = "<"
        
        long_conds.append(Condition(left_operand=m_name, operator=op_long, right_operand=th_long))

        longs.append(
            ConditionGroup(
                operator="AND",
                conditions=long_conds,
            )
        )
        
        # ショート: Close < Trend AND Momentum < Low
        short_conds = [
            Condition(left_operand="Close", operator="<", right_operand=t_name),
        ]
        short_conds.append(Condition(left_operand=m_name, operator=op_short, right_operand=th_short))

        shorts.append(
            ConditionGroup(
                operator="AND",
                conditions=short_conds,
            )
        )
        return longs, shorts

    def _create_cross(self, indicators):
        """指標クロス条件を生成"""
        longs, shorts = [], []
        price_inds = [
            i for i in indicators if i.enabled and self.gen._is_price_scale(i)
        ]
        if len(price_inds) < 2:
            return longs, shorts

        i1, i2 = random.sample(price_inds, 2)
        p1, p2 = i1.parameters.get("period", 0), i2.parameters.get("period", 0)
        if abs(p1 - p2) < 1:
            return longs, shorts

        short_ma, long_ma = (i1, i2) if p1 < p2 else (i2, i1)
        s_name, l_name = self.gen._get_indicator_name(
            short_ma
        ), self.gen._get_indicator_name(long_ma)

        longs.append(Condition(left_operand=s_name, operator=">", right_operand=l_name))
        shorts.append(
            Condition(left_operand=s_name, operator="<", right_operand=l_name)
        )
        return longs, shorts

    def _create_breakout(self, indicators):
        """ボラティリティブレイクアウト条件を生成"""
        longs, shorts = [], []
        candidates = [
            i for i in indicators if i.enabled and self.gen._is_band_indicator(i)
        ]
        if not candidates:
            return longs, shorts

        target = random.choice(candidates)
        up_name, low_name = self.gen._get_band_names(target)

        longs.append(
            Condition(left_operand="Close", operator=">", right_operand=up_name)
        )
        shorts.append(
            Condition(left_operand="Close", operator="<", right_operand=low_name)
        )
        return longs, shorts

    # テスト互換用エイリアス
    def _get_indicator_name(self, ind):
        return self.gen._get_indicator_name(ind)

    def _classify_indicators(self, inds):
        return self.gen._classify_indicators(inds)

    def _structure_conditions(self, conds):
        return self.gen._structure_conditions(conds)
