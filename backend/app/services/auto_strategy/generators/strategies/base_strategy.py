"""
条件生成のためのベース戦略クラス。

サブクラスで共通して使用されるヘルパーメソッドを提供し、
コードの重複を削減します。
"""

import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Union

from app.services.indicators.config import indicator_registry, IndicatorConfig

from ...genes import Condition, ConditionGroup, IndicatorGene
from ...config.constants import IndicatorType

ConditionList: TypeAlias = List[Condition]

logger = logging.getLogger(__name__)


class ConditionStrategy(ABC):
    """
    条件生成戦略のベースクラス。

    サブクラスに共通ヘルパーメソッドを提供:
    - _get_indicator_name: 一意な指標名の取得
    - _get_indicator_config: 指標設定の取得
    - _get_indicator_type: 指標タイプの取得
    - _classify_indicators_by_type: 指標のタイプ別分類
    - _structure_conditions: 条件の階層化
    - context: 生成コンテキストへのアクセス
    """

    def __init__(self, condition_generator: Any) -> None:
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

    # ===== プロパティ =====

    @property
    def context(self) -> Dict[str, Any]:
        """
        生成コンテキストを取得。

        Returns:
            タイムフレームやシンボルなどのコンテキスト情報
        """
        ctx = getattr(self.condition_generator, "context", None)
        return ctx if ctx is not None else {}

    # ===== 指標関連ヘルパー =====

    def _get_indicator_name(self, indicator: IndicatorGene) -> str:
        """
        IndicatorCalculatorと一致する一意な指標名を取得。

        Args:
            indicator: 指標遺伝子

        Returns:
            一意な指標名（例: "RSI_abc12345"）
        """
        if indicator.id:
            indicator_id_suffix = f"_{indicator.id[:8]}"
            return f"{indicator.type}{indicator_id_suffix}"
        return indicator.type

    def _get_indicator_config(self, indicator_type: str) -> Optional[IndicatorConfig]:
        """
        レジストリから指標設定を取得。

        Args:
            indicator_type: 指標タイプ名（例: "RSI", "SMA"）

        Returns:
            IndicatorConfig、見つからない場合はNone
        """
        return indicator_registry.get_indicator_config(indicator_type)

    def _get_indicator_type(self, indicator: IndicatorGene) -> IndicatorType:
        """
        指標のタイプ（TREND, MOMENTUM, VOLATILITY）を取得。

        Args:
            indicator: 指標遺伝子

        Returns:
            IndicatorType enum値
        """
        return self.condition_generator._get_indicator_type(indicator)

    def _classify_indicators_by_type(
        self, indicators: List[IndicatorGene]
    ) -> Dict[IndicatorType, List[IndicatorGene]]:
        """
        条件生成器の分類を使用して指標をタイプ別に分類。

        Args:
            indicators: 分類する指標のリスト

        Returns:
            IndicatorTypeを指標のリストにマッピングする辞書
        """
        return self.condition_generator._dynamic_classify(indicators)

    # ===== 条件生成ヘルパー =====

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

    def _structure_conditions(
        self, conditions: List[Union[Condition, ConditionGroup]]
    ) -> List[Union[Condition, ConditionGroup]]:
        """
        条件リストを確率的に階層化（グループ化）。

        複数の条件がある場合、30%の確率でOR演算子で
        グループ化することで、より柔軟な条件構造を生成します。

        Args:
            conditions: 条件のリスト

        Returns:
            階層化された条件のリスト
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
