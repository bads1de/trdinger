"""
遺伝子関連ユーティリティ関数

auto_strategy全体で使用される遺伝子関連の共通機能を提供します。
"""

import logging

from typing import Union


logger = logging.getLogger(__name__)


class GeneUtils:
    """遺伝子関連ユーティリティ"""

    @staticmethod
    def normalize_parameter(
        value: Union[int, float], min_val: int = 1, max_val: int = 200
    ) -> float:
        """
        パラメータ値を正規化（0-1の範囲に変換）

        Args:
            value: 正規化対象の値
            min_val: 最小値（デフォルト: 1）
            max_val: 最大値（デフォルト: 200）

        Returns:
            0-1の範囲に正規化した値
        """
        if not isinstance(value, (int, float)):
            logger.warning(
                f"数値でないパラメータを正規化: {value}, デフォルト値0.1を返却"
            )
            return 0.1

        # 範囲内に制限
        clamped_value = max(min_val, min(max_val, value))

        # 0-1の範囲に正規化
        normalized = (clamped_value - min_val) / (max_val - min_val)

        return float(normalized)

    @staticmethod
    def create_default_strategy_gene(strategy_gene_class):
        """デフォルトの戦略遺伝子を作成（StrategyGene.create_defaultに委譲）"""
        return strategy_gene_class.create_default()


# 外部で使用可能な便利関数
create_default_strategy_gene = GeneUtils.create_default_strategy_gene
normalize_parameter = GeneUtils.normalize_parameter
