"""
条件生成のためのDifferentIndicators戦略。
"""

import logging
import random
from typing import List
from .base_strategy import ConditionStrategy
from ...models.strategy_models import IndicatorGene, Condition
from ...constants import IndicatorType

logger = logging.getLogger(__name__)


class DifferentIndicatorsStrategy(ConditionStrategy):
    """異なるタイプの指標が利用可能な場合に条件を生成する戦略。"""

    def generate_conditions(self, indicators: List[IndicatorGene]):
        """
        異なる指標タイプを使用して条件を生成。

        トレンド + モメンタムまたはモメンタム + ボラティリティ指標を組み合わせ。
        """
        logger.debug(
            f"Generating conditions for {len(indicators)} indicators using DifferentIndicators strategy"
        )

        # タイプ別に指標をグループ化
        indicators_by_type = self._classify_indicators_by_type(indicators)
        logger.debug(
            f"Indicator classification: {[f'{k.name}:{len(v)}' for k, v in indicators_by_type.items() if v]}"
        )

        long_conditions = []
        short_conditions = []

        # ML指標が優先
        ml_indicators = [
            ind for ind in indicators if ind.enabled and ind.type.startswith("ML_")
        ]
        logger.debug(f"ML indicators count: {len(ml_indicators)}")

        # ロングのためのトレンドベースの条件を追加
        if indicators_by_type.get(IndicatorType.TREND) and len(indicators_by_type[IndicatorType.TREND]) > 0:
            selected_trend = self._create_trend_long_conditions(
                random.choice(indicators_by_type[IndicatorType.TREND])
            )
            long_conditions.extend(selected_trend)
            logger.debug(f"Added {len(selected_trend)} trend long conditions")

        # ロングのためのモメンタム条件を追加
        if indicators_by_type.get(IndicatorType.MOMENTUM) and len(indicators_by_type[IndicatorType.MOMENTUM]) > 0:
            selected_momentum = self._create_momentum_long_conditions(
                random.choice(indicators_by_type[IndicatorType.MOMENTUM])
            )
            long_conditions.extend(selected_momentum)
            logger.debug(f"Added {len(selected_momentum)} momentum long conditions")

        # ロングのためのML条件を追加
        if ml_indicators:
            ml_long_conditions = self.condition_generator._create_ml_long_conditions(
                ml_indicators
            )
            long_conditions.extend(ml_long_conditions)
            logger.debug(f"Added {len(ml_long_conditions)} ML long conditions")

        # ショート条件（逆方向）
        if indicators_by_type.get(IndicatorType.TREND) and len(indicators_by_type[IndicatorType.TREND]) > 0:
            selected_trend_short = self._create_trend_short_conditions(
                random.choice(indicators_by_type[IndicatorType.TREND])
            )
            short_conditions.extend(selected_trend_short)
            logger.debug(f"Added {len(selected_trend_short)} trend short conditions")

        if indicators_by_type.get(IndicatorType.MOMENTUM) and len(indicators_by_type[IndicatorType.MOMENTUM]) > 0:
            selected_momentum_short = self._create_momentum_short_conditions(
                random.choice(indicators_by_type[IndicatorType.MOMENTUM])
            )
            short_conditions.extend(selected_momentum_short)
            logger.debug(
                f"Added {len(selected_momentum_short)} momentum short conditions"
            )

        # ショートのためのML逆シグナル
        if ml_indicators and len(ml_indicators) >= 2:
            if any(ind.type == "ML_DOWN_PROB" for ind in ml_indicators):
                short_conditions.append(
                    Condition(
                        left_operand="ML_DOWN_PROB", operator=">", right_operand=0.6
                    )
                )

        # 少なくとも基本条件があることを確認
        if not long_conditions:
            long_conditions = [
                Condition(left_operand="close", operator=">", right_operand="open")
            ]

        if not short_conditions:
            short_conditions = [
                Condition(left_operand="close", operator="<", right_operand="open")
            ]

        # 適切なタイプに変換
        long_result = list(long_conditions)
        short_result = list(short_conditions)
        exit_result = []

        logger.debug(
            f"Generated {len(long_result)} long and {len(short_result)} short conditions"
        )
        return long_result, short_result, exit_result

    def _create_trend_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """トレンドベースのロング条件を作成。"""
        return self.condition_generator._create_trend_long_conditions(indicator)

    def _create_trend_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """トレンドベースのショート条件を作成。"""
        return self.condition_generator._create_trend_short_conditions(indicator)

    def _create_momentum_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """モメンタムベースのロング条件を作成。"""
        return self.condition_generator._create_momentum_long_conditions(indicator)

    def _create_momentum_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """モメンタムベースのショート条件を作成。"""
        return self.condition_generator._create_momentum_short_conditions(indicator)
