"""
条件生成器

ランダム戦略の条件部分を生成する専門ジェネレーター
"""

import logging
import random
from typing import List

from ..config.constants import OPERATORS
from ..genes import Condition
from .random_operand_generator import OperandGenerator

logger = logging.getLogger(__name__)


class ConditionGenerator:
    """
    ランダム戦略の条件生成を担当するクラス
    """

    def __init__(self, config: any):
        """
        初期化

        Args:
            config: GA設定オブジェクト
        """
        self.config = config
        self.operand_generator = OperandGenerator(config)
        self.available_operators = OPERATORS

    def generate_random_conditions(
        self, indicators: List[any], condition_type: str
    ) -> List[Condition]:
        """ランダムな条件リストを生成"""
        # 条件数はプロファイルや生成器の方針により 1〜max_conditions に広げる
        # ここでは min_conditions〜max_conditions の範囲で選択（下限>上限にならないようにガード）
        low = int(self.config.min_conditions)
        high = int(self.config.max_conditions)
        if high < low:
            low, high = high, low
        num_conditions = random.randint(low, max(low, high))
        conditions = []

        for _ in range(num_conditions):
            condition = self._generate_single_condition(indicators, condition_type)
            if condition:
                conditions.append(condition)

        # 最低1つの条件は保証
        if not conditions:
            conditions.append(self._generate_fallback_condition(condition_type))

        return conditions

    def _generate_single_condition(
        self, indicators: List[any], condition_type: str
    ) -> Condition:
        """単一の条件を生成"""
        # 左オペランドの選択
        left_operand = self.operand_generator.choose_operand(indicators)

        # 演算子の選択
        operator = random.choice(self.available_operators)

        # 右オペランドの選択
        right_operand = self.operand_generator.choose_right_operand(
            left_operand, indicators, condition_type
        )

        return Condition(
            left_operand=left_operand, operator=operator, right_operand=right_operand
        )

    def _generate_fallback_condition(self, condition_type: str) -> Condition:
        """フォールバック用の基本条件を生成（JSON形式の指標名）"""
        if condition_type == "entry":
            return Condition(left_operand="close", operator=">", right_operand="SMA")
        else:
            return Condition(left_operand="close", operator="<", right_operand="SMA")





