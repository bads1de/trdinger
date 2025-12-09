"""
条件生成のためのComplexConditions戦略。
"""

import logging
import random
from typing import List, Union

from ...config.constants import IndicatorType
from ...models.strategy_models import Condition, ConditionGroup, IndicatorGene
from .base_strategy import ConditionStrategy

logger = logging.getLogger(__name__)


class ComplexConditionsStrategy(ConditionStrategy):
    """複数の指標組み合わせで複雑な条件を生成する戦略。"""

    def generate_conditions(self, indicators: List[IndicatorGene]):
        """
        複数の指標を組み合わせて条件を生成。

        より複雑な条件セットを作成するために最大3つの指標を使用。
        """
        long_conditions = []
        short_conditions = []

        # バランスの取れた条件を作成するために最大3つの指標を使用
        selected_indicators = indicators[:3]

        for indicator in selected_indicators:
            if not indicator.enabled:
                continue

            # 使用する条件作成メソッドを決定するために指標タイプを取得
            indicator_type = self.condition_generator._get_indicator_type(indicator)

            try:
                # 指標タイプに基づいて適切な条件作成メソッドを使用
                if indicator_type == IndicatorType.MOMENTUM:
                    long_conds = (
                        self.condition_generator._create_momentum_long_conditions(
                            indicator
                        )
                    )
                    short_conds = (
                        self.condition_generator._create_momentum_short_conditions(
                            indicator
                        )
                    )
                elif indicator_type == IndicatorType.TREND:
                    long_conds = self.condition_generator._create_trend_long_conditions(
                        indicator
                    )
                    short_conds = (
                        self.condition_generator._create_trend_short_conditions(
                            indicator
                        )
                    )
                else:
                    # 不明な指標タイプ - 汎用条件を使用
                    long_conds = self.condition_generator._generic_long_conditions(
                        indicator
                    )
                    short_conds = self.condition_generator._generic_short_conditions(
                        indicator
                    )

                if long_conds:
                    long_conditions.extend(long_conds)

                # 各指標に対してショート条件も生成することを確認
                if short_conds:
                    short_conditions.extend(short_conds)

            except Exception as e:
                self.condition_generator.logger.warning(
                    f"Error generating conditions for {indicator.type}: {e}"
                )
                # 汎用条件にフォールバック
                long_conditions.extend(
                    self.condition_generator._generic_long_conditions(indicator)
                )
                short_conditions.extend(
                    self.condition_generator._generic_short_conditions(indicator)
                )

        # 条件が生成されなかった場合、より多くの指標で試行
        if not long_conditions:
            for indicator in indicators[:2]:
                if not indicator.enabled:
                    continue
                long_conditions.extend(
                    self.condition_generator._generic_long_conditions(indicator)
                )
                short_conditions.extend(
                    self.condition_generator._generic_short_conditions(indicator)
                )

                # 条件が生成できなかった場合、例外を投げる
                if not long_conditions:
                    raise RuntimeError("条件生成に失敗しました")

        # 適切な戻り値タイプに変換
        # 階層構造化（ランダムに条件をグループ化）
        long_result = self._structure_conditions(long_conditions)
        short_result = self._structure_conditions(short_conditions)
        exit_result = []

        return long_result, short_result, exit_result

    def _structure_conditions(
        self, conditions: List[Condition]
    ) -> List[Union[Condition, ConditionGroup]]:
        """
        条件リストを確率的に階層化（グループ化）します。

        単純なフラットリスト [A, B, C] (A AND B AND C) から
        [ConditionGroup([A, B], OR), C] ((A OR B) AND C) のような
        構造的バリエーションを生成します。
        """
        if len(conditions) < 2:
            return conditions

        structured = []
        i = 0
        while i < len(conditions):
            # 残りが2つ以上あり、50%の確率でグループ化
            if i + 1 < len(conditions) and random.random() < 0.5:
                # 2つの条件をORグループ化
                group = ConditionGroup(
                    operator="OR", conditions=[conditions[i], conditions[i + 1]]
                )
                structured.append(group)
                i += 2
            else:
                structured.append(conditions[i])
                i += 1

        return structured
