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

        注文タイプ（成行、指値、逆指値等）に応じた適切な
        価格パラメータを生成します。

        Args:
            entry_gene: エントリー遺伝子（Noneの場合は成行注文）
            current_price: 現在価格（基準価格）
            direction: 取引方向 (1.0=Long, -1.0=Short)

        Returns:
            buy()/sell() に渡すキーワード引数の辞書
            （例: {"limit": 50000.0}）
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

    def _calculate_offset_price(
        self, base_price: float, direction: float, offset_pct: float, is_limit: bool
    ) -> float:
        """
        オフセット価格を計算（共通ロジック）
        is_limit=True の場合、有利な価格（指値）を計算
        is_limit=False の場合、不利な価格（逆指値）を計算
        """
        # Longの場合: limitはマイナス(下落待ち), stopはプラス(ブレイク待ち)
        # Shortの場合: limitはプラス(上昇待ち), stopはマイナス(ブレイク待ち)
        multiplier = -1.0 if (direction > 0) == is_limit else 1.0
        return base_price * (1.0 + multiplier * offset_pct)

    def _calculate_limit_price(
        self, base_price: float, direction: float, offset_pct: float
    ) -> float:
        """指値価格を計算（有利な価格）"""
        return self._calculate_offset_price(base_price, direction, offset_pct, True)

    def _calculate_stop_price(
        self, base_price: float, direction: float, offset_pct: float
    ) -> float:
        """逆指値価格を計算（ブレイクアウト価格）"""
        return self._calculate_offset_price(base_price, direction, offset_pct, False)
