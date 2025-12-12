"""
エントリー遺伝子生成器

ランダム戦略のエントリー遺伝子を生成する専門ジェネレーター
"""

import logging
import random

from ...models.entry_gene import EntryGene
from ...models.enums import EntryType

logger = logging.getLogger(__name__)


class EntryGenerator:
    """
    エントリー遺伝子の生成を担当するクラス
    """

    # エントリータイプの出現確率（初期段階では成行注文を多めに設定）
    DEFAULT_ENTRY_TYPE_WEIGHTS = {
        EntryType.MARKET: 0.6,
        EntryType.LIMIT: 0.2,
        EntryType.STOP: 0.15,
        EntryType.STOP_LIMIT: 0.05,
    }

    def __init__(self, config: any):
        """
        初期化

        Args:
            config: GA設定オブジェクト
        """
        self.config = config

    def generate_entry_gene(self) -> EntryGene:
        """
        ランダムなエントリー遺伝子を生成

        Returns:
            生成されたエントリー遺伝子
        """
        try:
            # エントリータイプを重み付きでランダム選択
            entry_type = self._select_entry_type()

            # オフセット値をランダム生成（0.1% ~ 2.0%）
            limit_offset_pct = random.uniform(0.001, 0.02)
            stop_offset_pct = random.uniform(0.001, 0.02)

            # 有効期限をランダム生成（1 ~ 20バー）
            order_validity_bars = random.randint(1, 20)

            return EntryGene(
                entry_type=entry_type,
                limit_offset_pct=limit_offset_pct,
                stop_offset_pct=stop_offset_pct,
                order_validity_bars=order_validity_bars,
                enabled=True,
                priority=random.uniform(0.5, 1.5),
            )

        except Exception as e:
            logger.error(f"エントリー遺伝子生成エラー: {e}")
            # フォールバック: 成行注文のデフォルト遺伝子
            return EntryGene(
                entry_type=EntryType.MARKET,
                limit_offset_pct=0.005,
                stop_offset_pct=0.005,
                order_validity_bars=5,
                enabled=True,
            )

    def _select_entry_type(self) -> EntryType:
        """
        重み付きでエントリータイプを選択

        Returns:
            選択されたエントリータイプ
        """
        # 設定から重みを取得、なければデフォルトを使用
        weights = self.DEFAULT_ENTRY_TYPE_WEIGHTS

        if hasattr(self.config, "entry_type_weights"):
            custom_weights = self.config.entry_type_weights
            if custom_weights:
                weights = {EntryType(k): v for k, v in custom_weights.items()}

        # 重み付きランダム選択
        types = list(weights.keys())
        type_weights = list(weights.values())

        return random.choices(types, weights=type_weights, k=1)[0]
