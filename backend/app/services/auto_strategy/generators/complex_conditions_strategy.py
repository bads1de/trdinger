"""
条件生成のためのComplexConditionsStrategy。

トレンド押し目買い、ゴールデンクロス、ブレイクアウトなどの
王道パターンに基づいた条件を生成します。
特定の指標名に依存せず、メタデータ（スケールタイプ、カテゴリ、戻り値定義）を活用して
汎用的に条件を構築します。
"""

import logging
import random
from typing import List, Union, Dict, Tuple, Any

from app.services.indicators.config import (
    IndicatorScaleType,
    indicator_registry,
)
from ..config.constants import IndicatorType
from ..genes import Condition, ConditionGroup, IndicatorGene

logger = logging.getLogger(__name__)


class ComplexConditionsStrategy:
    """
    複数の指標組み合わせで複雑かつ意味のある条件を生成する戦略。
    メタデータ駆動型のアプローチにより、未知の指標にも対応します。
    """

    def __init__(self, condition_generator: Any) -> None:
        """
        コンテキストで戦略を初期化。

        Args:
            condition_generator: 共有状態のためのConditionGeneratorへの参照
        """
        self.condition_generator = condition_generator

    @property
    def context(self) -> Dict[str, Any]:
        """
        生成コンテキストを取得。
        """
        ctx = getattr(self.condition_generator, "context", None)
        return ctx if ctx is not None else {}

    def generate_conditions(self, indicators: List[IndicatorGene]) -> Tuple[
        List[Union[Condition, ConditionGroup]],
        List[Union[Condition, ConditionGroup]],
        List[Condition],
    ]:
        """
        複数の指標を組み合わせて条件を生成。
        """
        long_conditions = []
        short_conditions = []
        exit_conditions = []

        # 指標をタイプ別に分類（既存のロジック）
        classified_indicators = self._classify_indicators(indicators)

        # 1. トレンド押し目買いパターン (Trend + Momentum)
        tp_long, tp_short = self._create_trend_pullback_conditions(
            classified_indicators
        )
        long_conditions.extend(tp_long)
        short_conditions.extend(tp_short)

        # 2. 移動平均クロス (Price Scale Indicators)
        # トレンド指標だけでなく、価格スケールの指標なら何でもクロス候補にする
        cross_long, cross_short = self._create_cross_conditions(indicators)
        long_conditions.extend(cross_long)
        short_conditions.extend(cross_short)

        # 3. ボラティリティブレイクアウト (Volatility / Band Indicators)
        break_long, break_short = self._create_breakout_conditions(indicators)
        long_conditions.extend(break_long)
        short_conditions.extend(break_short)

        # 条件が生成されなかった場合、従来（汎用）のロジックにフォールバック
        if not long_conditions and not short_conditions:
            return self._generate_fallback_conditions(indicators)

        # 条件の階層化（グループ化）
        long_result = self._structure_conditions(long_conditions)
        short_result = self._structure_conditions(short_conditions)

        return long_result, short_result, exit_conditions

    def _classify_indicators(
        self, indicators: List[IndicatorGene]
    ) -> Dict[IndicatorType, List[IndicatorGene]]:
        """指標をタイプ別に分類"""
        categorized: Dict[IndicatorType, List[IndicatorGene]] = {
            IndicatorType.MOMENTUM: [],
            IndicatorType.TREND: [],
            IndicatorType.VOLATILITY: [],
        }
        for ind in indicators:
            if not ind.enabled:
                continue
            try:
                ind_type = self.condition_generator._get_indicator_type(ind)
                categorized[ind_type].append(ind)
            except Exception:
                # 分類不能な場合はTREND扱い（安全策）
                categorized[IndicatorType.TREND].append(ind)
        return categorized

    def _get_indicator_name(self, indicator: IndicatorGene) -> str:
        """
        IndicatorCalculatorと一致する一意な指標名を取得。
        """
        if indicator.id:
            indicator_id_suffix = f"_{indicator.id[:8]}"
            return f"{indicator.type}{indicator_id_suffix}"
        return indicator.type

    def _structure_conditions(
        self, conditions: List[Union[Condition, ConditionGroup]]
    ) -> List[Union[Condition, ConditionGroup]]:
        """
        条件リストを確率的に階層化（グループ化）。
        """
        if len(conditions) < 2:
            return conditions

        structured: List[Union[Condition, ConditionGroup]] = []
        i = 0
        while i < len(conditions):
            # 残りが2つ以上あり、30%の確率でグループ化
            if i + 1 < len(conditions) and random.random() < 0.3:
                group = ConditionGroup(
                    operator="OR", conditions=[conditions[i], conditions[i + 1]]
                )
                structured.append(group)
                i += 2
            else:
                structured.append(conditions[i])
                i += 1

        return structured

    def _create_trend_pullback_conditions(
        self, classified: Dict[IndicatorType, List[IndicatorGene]]
    ) -> Tuple[
        List[Union[Condition, ConditionGroup]], List[Union[Condition, ConditionGroup]]
    ]:
        """トレンド押し目買い/戻り売り条件を生成"""
        longs = []
        shorts = []

        trends = classified[IndicatorType.TREND]
        momentums = classified[IndicatorType.MOMENTUM]

        if not trends or not momentums:
            return longs, shorts

        # 組み合わせ（最大1ペア）
        trend = random.choice(trends)
        momentum = random.choice(momentums)

        trend_name = self._get_indicator_name(trend)
        momentum_name = self._get_indicator_name(momentum)

        # メタデータチェック: トレンド指標が価格スケールであることを確認
        trend_config = indicator_registry.get_indicator_config(trend.type)
        if trend_config and trend_config.scale_type != IndicatorScaleType.PRICE_RATIO:
            # 価格スケールでないトレンド指標（ADXなど）は価格との比較に使えない
            pass
        else:
            # ロング用条件セット
            long_set = []
            long_set.append(
                Condition(left_operand="Close", operator=">", right_operand=trend_name)
            )

            momentum_conds = self.condition_generator._generic_short_conditions(
                momentum
            )
            if momentum_conds:
                cond = momentum_conds[0]
                cond.left_operand = momentum_name
                long_set.append(cond)

            if len(long_set) > 1:
                longs.append(ConditionGroup(operator="AND", conditions=long_set))
            elif long_set:
                longs.append(long_set[0])

            # ショート用条件セット
            short_set = []
            short_set.append(
                Condition(left_operand="Close", operator="<", right_operand=trend_name)
            )

            momentum_conds_long = self.condition_generator._generic_long_conditions(
                momentum
            )
            if momentum_conds_long:
                cond = momentum_conds_long[0]
                cond.left_operand = momentum_name
                short_set.append(cond)

            if len(short_set) > 1:
                shorts.append(ConditionGroup(operator="AND", conditions=short_set))
            elif short_set:
                shorts.append(short_set[0])

        return longs, shorts

    def _create_cross_conditions(
        self, indicators: List[IndicatorGene]
    ) -> Tuple[List[Condition], List[Condition]]:
        """指標クロス条件を生成（メタデータ駆動）"""
        longs = []
        shorts = []

        # 価格スケールの指標を抽出
        price_indicators = []
        for ind in indicators:
            if not ind.enabled:
                continue
            config = indicator_registry.get_indicator_config(ind.type)
            # 設定がない場合は安全のためスキップ、またはデフォルトで価格スケールとみなすか？
            # ここでは厳密にチェック
            if config and config.scale_type == IndicatorScaleType.PRICE_RATIO:
                price_indicators.append(ind)
            elif ind.type in [
                "SMA",
                "EMA",
                "WMA",
                "HMA",
                "KAMA",
                "TRIMA",
            ]:  # フォールバック
                price_indicators.append(ind)

        if len(price_indicators) < 2:
            return longs, shorts

        # 2つ選ぶ
        pair = random.sample(price_indicators, 2)
        ind1, ind2 = pair[0], pair[1]

        # 期間(period/length)で短期・長期を判別
        def get_period(ind):
            return ind.parameters.get("period", ind.parameters.get("length", 0))

        p1 = get_period(ind1)
        p2 = get_period(ind2)

        # 同じ期間の場合はクロス条件を作らない
        if abs(p1 - p2) < 1:
            return longs, shorts

        short_ma = ind1 if p1 < p2 else ind2
        long_ma = ind2 if p1 < p2 else ind1

        short_name = self._get_indicator_name(short_ma)
        long_name = self._get_indicator_name(long_ma)

        # ゴールデンクロス (短期 > 長期)
        longs.append(
            Condition(left_operand=short_name, operator=">", right_operand=long_name)
        )

        # デッドクロス (短期 < 長期)
        shorts.append(
            Condition(left_operand=short_name, operator="<", right_operand=long_name)
        )

        return longs, shorts

    def _create_breakout_conditions(
        self, indicators: List[IndicatorGene]
    ) -> Tuple[List[Condition], List[Condition]]:
        """ボラティリティブレイクアウト条件を生成（メタデータ駆動）"""
        longs = []
        shorts = []

        # バンド系指標（Upper/Lowerを持つ）またはボラティリティ指標を探す
        breakout_candidates = []

        for ind in indicators:
            if not ind.enabled:
                continue
            config = indicator_registry.get_indicator_config(ind.type)

            is_band = False
            if config:
                # return_cols に upper/lower が含まれるかチェック
                if config.return_cols:
                    lower_cols = [
                        c
                        for c in config.return_cols
                        if any(k in c.lower() for k in ["upper", "top", "high"])
                    ]
                    if lower_cols:
                        is_band = True

                # またはカテゴリがVolatilityで、かつ価格スケールであること
                if (
                    not is_band
                    and config.category == "volatility"
                    and config.scale_type == IndicatorScaleType.PRICE_RATIO
                ):
                    is_band = True  # 例: Keltner Channels, Donchian

            # フォールバック: 名前にBB, BANDなどが含まれる
            if not config and any(
                k in ind.type.upper() for k in ["BB", "BAND", "KELTNER", "DONCHIAN"]
            ):
                is_band = True

            if is_band:
                breakout_candidates.append(ind)

        if not breakout_candidates:
            return longs, shorts

        target_ind = random.choice(breakout_candidates)
        base_name = self._get_indicator_name(target_ind)
        config = indicator_registry.get_indicator_config(target_ind.type)

        # Upper/Lowerバンド名の解決
        upper_suffix = "_2"  # デフォルト（pandas-taの多くは [lower, mid, upper] の順だが、upperが最後の場合が多い）
        lower_suffix = "_0"

        if config and config.return_cols:
            # メタデータから正確な位置を特定
            for i, col in enumerate(config.return_cols):
                col_lower = col.lower()
                if "upper" in col_lower or "top" in col_lower or "high" in col_lower:
                    upper_suffix = f"_{i}"
                if "lower" in col_lower or "bottom" in col_lower or "low" in col_lower:
                    lower_suffix = f"_{i}"

        # 名前解決（IndicatorCalculatorの命名規則に合わせる）
        # IndicatorCalculatorは複数出力の場合、base_name_0, base_name_1... とする
        upper_name = f"{base_name}{upper_suffix}"
        lower_name = f"{base_name}{lower_suffix}"

        # ブレイクアウト条件生成
        # Upperブレイク (Close > UpperBand) -> ロング
        longs.append(
            Condition(left_operand="Close", operator=">", right_operand=upper_name)
        )

        # Lowerブレイク (Close < LowerBand) -> ショート
        shorts.append(
            Condition(left_operand="Close", operator="<", right_operand=lower_name)
        )

        return longs, shorts

    def _generate_fallback_conditions(
        self, indicators: List[IndicatorGene]
    ) -> Tuple[List, List, List]:
        """従来の生成ロジック（フォールバック）"""
        long_conditions = []
        short_conditions = []

        for indicator in indicators:
            if not indicator.enabled:
                continue

            ind_name = self._get_indicator_name(indicator)

            try:
                long_conds = self.condition_generator._generic_long_conditions(
                    indicator
                )
                short_conds = self.condition_generator._generic_short_conditions(
                    indicator
                )

                # オペランド名を修正
                for c in long_conds:
                    c.left_operand = ind_name
                for c in short_conds:
                    c.left_operand = ind_name

                long_conditions.extend(long_conds)
                short_conditions.extend(short_conds)
            except Exception:
                pass

        return (
            self._structure_conditions(long_conditions),
            self._structure_conditions(short_conditions),
            [],
        )
