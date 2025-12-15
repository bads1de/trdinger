"""
条件生成のためのベース戦略クラス。
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, TypeAlias

from ...genes import Condition, IndicatorGene

ConditionList: TypeAlias = List[Condition]

logger = logging.getLogger(__name__)


class ConditionStrategy(ABC):
    """条件生成戦略のベースクラス。"""

    def __init__(self, condition_generator):
        """
        コンテキストで戦略を初期化。

        Args:
            condition_generator: 共有状態のためのConditionGeneratorへの参照
        """
        self.condition_generator = condition_generator

    @abstractmethod
    def generate_conditions(
        self, indicators: List[IndicatorGene]
    ) -> Tuple[List, List, List]:
        """
        戦略の条件を生成。

        Returns:
            (long_entry_conditions, short_entry_conditions, exit_conditions)
        """

    # サブクラスで使用できるヘルパーメソッド
    def _classify_indicators_by_type(self, indicators: List[IndicatorGene]) -> dict:
        """
        条件生成器の分類を使用して指標をタイプ別に分類。

        Args:
            indicators: 分類する指標のリスト

        Returns:
            IndicatorTypeを指標のリストにマッピングする辞書
        """
        return self.condition_generator._dynamic_classify(indicators)

    def _create_generic_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """指標の汎用ロング条件を作成。"""
        return self.condition_generator._generic_long_conditions(indicator)

    def _create_generic_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """指標の汎用ショート条件を作成。"""
        return self.condition_generator._generic_short_conditions(indicator)





