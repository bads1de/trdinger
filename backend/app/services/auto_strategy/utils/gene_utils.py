"""
遺伝子関連ユーティリティ関数

auto_strategy全体で使用される遺伝子関連の共通機能を提供します。
"""

import logging
import uuid
from typing import Union

from .indicator_utils import get_all_indicators

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
        """デフォルトの戦略遺伝子を作成"""
        try:
            # 動的インポートを避けるため、引数として渡すか、呼び出し側でインポートする
            # ここでは基本的な構造のみを提供
            from ..genes import (
                Condition,
                IndicatorGene,
                PositionSizingGene,
                PositionSizingMethod,
                TPSLGene,
            )

            indicators = [
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ]

            # デフォルト条件
            long_entry_conditions = [
                Condition(left_operand="close", operator=">", right_operand="open")
            ]
            short_entry_conditions = [
                Condition(left_operand="close", operator="<", right_operand="open")
            ]

            # デフォルトリスク管理
            risk_management = {"position_size": 0.1}

            # デフォルトTP/SL遺伝子
            tpsl_gene = TPSLGene(
                take_profit_pct=0.01, stop_loss_pct=0.005, enabled=True
            )

            # デフォルトポジションサイジング遺伝子
            position_sizing_gene = PositionSizingGene(
                method=PositionSizingMethod.FIXED_QUANTITY,
                fixed_quantity=1000,
                enabled=True,
            )

            # メタデータ
            metadata = {
                "generated_by": "create_default_strategy_gene",
                "source": "fallback",
                "indicators_count": len(indicators),
                "tpsl_gene_included": tpsl_gene is not None,
                "position_sizing_gene_included": position_sizing_gene is not None,
            }

            return strategy_gene_class(
                id=str(uuid.uuid4()),  # 新しいIDを生成
                indicators=indicators,
                long_entry_conditions=long_entry_conditions,
                short_entry_conditions=short_entry_conditions,
                risk_management=risk_management,
                tpsl_gene=tpsl_gene,
                position_sizing_gene=position_sizing_gene,
                metadata=metadata,
            )
        except Exception as inner_e:
            logger.error(f"デフォルト戦略遺伝子作成エラー: {inner_e}")
            raise ValueError(f"デフォルト戦略遺伝子の作成に失敗: {inner_e}")


# 外部で使用可能な便利関数
create_default_strategy_gene = GeneUtils.create_default_strategy_gene
normalize_parameter = GeneUtils.normalize_parameter