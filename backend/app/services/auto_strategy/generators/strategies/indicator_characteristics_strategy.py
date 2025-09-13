"""
MLおよび専門指標のためのIndicatorCharacteristics戦略。
"""

import logging
from typing import List
from .base_strategy import ConditionStrategy
from ...models.strategy_models import IndicatorGene, Condition


logger = logging.getLogger(__name__)


class IndicatorCharacteristicsStrategy(ConditionStrategy):
    """指標特性に基づいて条件を生成する戦略（主にML）。"""

    def generate_conditions(self, indicators: List[IndicatorGene]):
        """
        指標特性に焦点を当てた条件を生成。

        主にML指標とその確率ベースの条件を扱う。
        """
        long_conditions = []
        short_conditions = []

        # この戦略ではML指標に焦点を当てる
        ml_indicators = [
            ind for ind in indicators if ind.enabled and ind.type.startswith("ML_")
        ]

        if ml_indicators:
            # ML特有のロング条件を生成
            ml_long_conds = self.condition_generator._create_ml_long_conditions(
                ml_indicators
            )
            if ml_long_conds:
                long_conditions.extend(ml_long_conds)

            # ML特有のショート条件を生成
            # ショートシグナルのための補完ML指標を探す
            if len(ml_indicators) >= 1:
                # ML確率に基づくショート条件を追加
                short_conditions.extend(self._create_ml_short_conditions(ml_indicators))

        # まだ条件がなく、通常の指標がある場合、制限付き汎用条件を使用
        if not long_conditions:
            # 非ML指標の場合、非常に基本的な条件を使用
            regular_indicators = [
                ind
                for ind in indicators
                if ind.enabled and not ind.type.startswith("ML_")
            ]

            if regular_indicators:
                # この戦略を焦点を保つために最初の指標のみを使用
                first_indicator = regular_indicators[0]
                long_conditions.extend(
                    self.condition_generator._generic_long_conditions(first_indicator)
                )
                short_conditions.extend(
                    self.condition_generator._generic_short_conditions(first_indicator)
                )

        # 少なくとも基本条件があることを確認
        if not long_conditions or not short_conditions:
            # 最後の手段としてフォールバック条件を使用
            try:
                fallback_result = (
                    self.condition_generator._generate_fallback_conditions()
                )
                # タプルの場合のみアンパック
                if isinstance(fallback_result, tuple) and len(fallback_result) == 3:
                    longfallback, shortfallback, _ = fallback_result
                    if not long_conditions:
                        long_conditions = longfallback
                    if not short_conditions:
                        short_conditions = shortfallback
                else:
                    # 適切なタプルでない場合、デフォルトフォールバックを使用
                    if not long_conditions:
                        long_conditions = [
                            Condition(
                                left_operand="close", operator=">", right_operand="open"
                            )
                        ]
                    if not short_conditions:
                        short_conditions = [
                            Condition(
                                left_operand="close", operator="<", right_operand="open"
                            )
                        ]
                    logger.warning(
                        "Invalid fallback result format, using default conditions"
                    )
            except Exception as e:
                logger.error(f"Error in fallback generation: {e}")
                # 絶対デフォルト条件を使用
                if not long_conditions:
                    long_conditions = [
                        Condition(
                            left_operand="close", operator=">", right_operand="open"
                        )
                    ]
                if not short_conditions:
                    short_conditions = [
                        Condition(
                            left_operand="close", operator="<", right_operand="open"
                        )
                    ]

        # 適切な戻り値タイプに変換
        long_result = list(long_conditions)
        short_result = list(short_conditions)
        exit_result = []

        return long_result, short_result, exit_result

    def _create_ml_short_conditions(
        self, ml_indicators: List[IndicatorGene]
    ) -> List[Condition]:
        """
        MLベースのショート条件を作成。

        Args:
            ml_indicators: ML指標のリスト

        Returns:
            ショートエントリー条件のリスト
        """
        short_conditions = []

        # 特定のML確率指標を確認
        for indicator in ml_indicators:
            if indicator.type == "ML_DOWN_PROB":
                short_conditions.append(
                    Condition(
                        left_operand="ML_DOWN_PROB",
                        operator=">",
                        right_operand=0.6,
                    )
                )
            elif indicator.type == "ML_UP_PROB":
                # 上昇確率の場合、ショートのために逆を使用可能
                short_conditions.append(
                    Condition(
                        left_operand="ML_UP_PROB",
                        operator="<",
                        right_operand=0.3,
                    )
                )

        # 特定の指標がない場合、レンジ確率を使用
        if not short_conditions:
            range_indicators = [
                ind for ind in ml_indicators if ind.type == "ML_RANGE_PROB"
            ]
            if range_indicators:
                short_conditions.append(
                    Condition(
                        left_operand="ML_RANGE_PROB",
                        operator=">",
                        right_operand=0.7,
                    )
                )

        return short_conditions
