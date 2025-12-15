"""
ポジションサイジング生成器

ランダム戦略のポジションサイジング遺伝子を生成する専門ジェネレーター
"""

import logging

from ...models import (
    PositionSizingGene,
    PositionSizingMethod,
    create_random_position_sizing_gene,
)

logger = logging.getLogger(__name__)


class PositionSizingGenerator:
    """
    ポジションサイジング遺伝子の生成と管理を担当するクラス
    """

    def __init__(self, config: any):
        """
        初期化

        Args:
            config: GA設定オブジェクト
        """
        self.config = config

    def generate_position_sizing_gene(self):
        """
        ポジションサイジング遺伝子を生成

        設定に基づいてランダムなポジションサイジング遺伝子を生成する。
        エラー発生時はデフォルトの遺伝子を返す。

        Returns:
            PositionSizingGene: 生成されたポジションサイジング遺伝子
        """
        try:
            return create_random_position_sizing_gene(self.config)
        except Exception as e:
            logger.error(f"ポジションサイジング遺伝子生成失敗: {e}")
            # フォールバック: デフォルト遺伝子を返す
            return PositionSizingGene(
                method=PositionSizingMethod.FIXED_RATIO,
                fixed_ratio=0.1,
                max_position_size=20.0,  # より大きなデフォルト値
                enabled=True,
            )


