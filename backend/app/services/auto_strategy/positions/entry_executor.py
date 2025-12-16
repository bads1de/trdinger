"""
エントリー注文実行サービス

エントリー注文のタイプに基づいて、backtesting.py の buy()/sell() に渡すパラメータを計算します。
"""

import logging
from typing import Any, Dict, Optional

from ..genes.entry import EntryGene
from ..config.constants import EntryType

logger = logging.getLogger(__name__)


class EntryExecutor:
    """
    エントリー注文実行サービス

    EntryGene に基づいて、backtesting.py の buy()/sell() メソッドに渡す
    パラメータ（limit, stop など）を計算します。
    """

    def calculate_entry_params(
        self,
        entry_gene: Optional[EntryGene],
        current_price: float,
        direction: float,  # 1.0=Long, -1.0=Short
    ) -> Dict[str, Any]:
        """
        backtesting.py の buy/sell メソッドに渡すパラメータを計算

        Args:
            entry_gene: エントリー遺伝子（Noneの場合は成行注文）
            current_price: 現在価格
            direction: 取引方向 (1.0=Long, -1.0=Short)

        Returns:
            パラメータ辞書（例: {"limit": price} or {"stop": price} or {}）
        """
        # entry_gene が None または無効な場合は成行注文
        if entry_gene is None or not entry_gene.enabled:
            return {}

        entry_type = entry_gene.entry_type

        # 成行注文: パラメータなし
        if entry_type == EntryType.MARKET:
            return {}

        # 指値注文: 有利な価格で約定を狙う
        if entry_type == EntryType.LIMIT:
            limit_price = self._calculate_limit_price(
                current_price, direction, entry_gene.limit_offset_pct / 100.0
            )
            return {"limit": limit_price}

        # 逆指値注文: ブレイクアウトで約定
        if entry_type == EntryType.STOP:
            stop_price = self._calculate_stop_price(
                current_price, direction, entry_gene.stop_offset_pct / 100.0
            )
            return {"stop": stop_price}

        # 逆指値指値注文: ストップ発動後、指値で約定を狙う
        if entry_type == EntryType.STOP_LIMIT:
            stop_price = self._calculate_stop_price(
                current_price, direction, entry_gene.stop_offset_pct / 100.0
            )
            limit_price = self._calculate_limit_price(
                stop_price, direction, entry_gene.limit_offset_pct / 100.0
            )
            return {"stop": stop_price, "limit": limit_price}

        # フォールバック: 成行注文
        logger.warning(
            f"未知のエントリータイプ: {entry_type}、成行注文としてフォールバック"
        )
        return {}

    def _calculate_limit_price(
        self, base_price: float, direction: float, offset_pct: float
    ) -> float:
        """
        指値価格を計算

        Long: 現在価格より低い価格で買い指値（有利な価格）
        Short: 現在価格より高い価格で売り指値（有利な価格）

        Args:
            base_price: 基準価格
            direction: 取引方向 (1.0=Long, -1.0=Short)
            offset_pct: オフセット比率

        Returns:
            指値価格
        """
        if direction > 0:  # Long: 現在価格より低い価格
            return base_price * (1.0 - offset_pct)
        else:  # Short: 現在価格より高い価格
            return base_price * (1.0 + offset_pct)

    def _calculate_stop_price(
        self, base_price: float, direction: float, offset_pct: float
    ) -> float:
        """
        逆指値価格を計算

        Long: 現在価格より高い価格でブレイクアウト買い
        Short: 現在価格より低い価格でブレイクアウト売り

        Args:
            base_price: 基準価格
            direction: 取引方向 (1.0=Long, -1.0=Short)
            offset_pct: オフセット比率

        Returns:
            逆指値価格
        """
        if direction > 0:  # Long: 現在価格より高い価格でブレイクアウト
            return base_price * (1.0 + offset_pct)
        else:  # Short: 現在価格より低い価格でブレイクアウト
            return base_price * (1.0 - offset_pct)
