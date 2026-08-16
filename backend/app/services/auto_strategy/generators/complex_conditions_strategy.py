"""
条件生成のためのComplexConditionsStrategy。

トレンド押し目買い、ゴールデンクロス、ブレイクアウトなどの
王道パターンに基づいた条件を生成します。
特定の指標名に依存せず、メタデータ（スケールタイプ、カテゴリ、戻り値定義）を活用して
汎用的に条件を構築します。
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from ..config.constants import IndicatorType
from ..genes import Condition, ConditionGroup, IndicatorGene
from ..utils.scale_compat import default_threshold_for

if TYPE_CHECKING:
    from .condition_generator import ConditionGenerator

logger = logging.getLogger(__name__)


class ComplexConditionsStrategy:
    """
    複数の指標を組み合わせて意味のある条件を生成する戦略。
    """

    def __init__(self, condition_generator: ConditionGenerator) -> None:
        self.gen = condition_generator

    def generate_conditions(
        self, indicators: list[IndicatorGene]
    ) -> tuple[
        list[Condition | ConditionGroup],
        list[Condition | ConditionGroup],
        list[Condition],
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
        classified = self.gen._dynamic_classify(indicators)
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

    def _create_trend_follow(
        self, classified: dict[IndicatorType, list[IndicatorGene]]
    ) -> tuple[
        list[Condition | ConditionGroup],
        list[Condition | ConditionGroup],
    ]:
        """トレンドフォロー条件（順張り特化）を生成"""
        longs: list[Condition | ConditionGroup] = []
        shorts: list[Condition | ConditionGroup] = []
        trends, momentums = (
            classified[IndicatorType.TREND],
            classified[IndicatorType.MOMENTUM],
        )
        # Close と直接比較できるのは価格スケールの指標のみ。
        # 非価格指標（OI/FR/LSR由来等）と組ませると恒真/恒偽条件になる。
        price_trends = [t for t in trends if self.gen._is_price_scale(t)]
        if not price_trends or not momentums:
            return longs, shorts

        trend, momentum = random.choice(price_trends), random.choice(momentums)
        t_name, m_name = (
            self.gen._get_indicator_name(trend),
            self.gen._get_indicator_name(momentum),
        )

        # 順張り特化: Close > Trend AND Momentum > Bullish
        # 閾値はレジストリ定義（normal プロファイル）とスケール型から決定する。
        # スケール型が PRICE_* でも close と同尺度でない指標（ATR 等の比率系、
        # WHALE_DIVERGENCE 等の閾値付き比率系）を close と比較させると
        # 恒真/恒偽になるため、default_threshold_for 側で判定する。
        if self.gen._is_price_scale(momentum):
            th_long: int | float | str = "close"
            th_short: int | float | str = "close"
        else:
            th_long = default_threshold_for(momentum.type, bullish=True)
            th_short = default_threshold_for(momentum.type, bullish=False)

        # ロング: Close > Trend AND Momentum > High
        long_conds: list[Condition | ConditionGroup] = [
            Condition(left_operand="close", operator=">", right_operand=t_name),
        ]
        # Momentumの比較対象が "close" の場合は、Momentum > Close (または < Close) になる
        # もしMomentumがRSI(0-100)なら、RSI > 60 となる

        # operatorの決定:
        # オシレーターなら > th_long
        # もし th_long が "close" なら、Momentum > Close (トレンドフォローならこれで良いか？)
        # SUPERTREND > Close は上昇トレンドを示すのでOK
        op_long = ">"
        op_short = "<"
        long_conds.append(
            Condition(left_operand=m_name, operator=op_long, right_operand=th_long)
        )

        # ショート: Close < Trend AND Momentum < Low
        short_conds: list[Condition | ConditionGroup] = [
            Condition(left_operand="close", operator="<", right_operand=t_name),
        ]
        short_conds.append(
            Condition(left_operand=m_name, operator=op_short, right_operand=th_short)
        )

        build_and_groups = getattr(type(self.gen), "_build_and_groups", None)
        if build_and_groups is None:
            long_group = ConditionGroup(operator="AND", conditions=long_conds)
            short_group = ConditionGroup(operator="AND", conditions=short_conds)
        else:
            long_group, short_group = build_and_groups(
                self.gen,
                long_conds,
                short_conds,
            )
        longs.append(long_group)
        shorts.append(short_group)
        return longs, shorts

    def _create_cross(
        self, indicators: list[IndicatorGene]
    ) -> tuple[
        list[Condition | ConditionGroup],
        list[Condition | ConditionGroup],
    ]:
        """指標クロス条件を生成"""
        longs: list[Condition | ConditionGroup] = []
        shorts: list[Condition | ConditionGroup] = []
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
        s_name, l_name = (
            self.gen._get_indicator_name(short_ma),
            self.gen._get_indicator_name(long_ma),
        )

        longs.append(Condition(left_operand=s_name, operator=">", right_operand=l_name))
        shorts.append(
            Condition(left_operand=s_name, operator="<", right_operand=l_name)
        )
        return longs, shorts

    def _create_breakout(
        self, indicators: list[IndicatorGene]
    ) -> tuple[
        list[Condition | ConditionGroup],
        list[Condition | ConditionGroup],
    ]:
        """ボラティリティブレイクアウト条件を生成"""
        longs: list[Condition | ConditionGroup] = []
        shorts: list[Condition | ConditionGroup] = []
        candidates = [
            i for i in indicators if i.enabled and self.gen._is_band_indicator(i)
        ]
        if not candidates:
            return longs, shorts

        target = random.choice(candidates)
        up_name, low_name = self.gen._get_band_names(target)

        longs.append(
            Condition(left_operand="close", operator=">", right_operand=up_name)
        )
        shorts.append(
            Condition(left_operand="close", operator="<", right_operand=low_name)
        )
        return longs, shorts
