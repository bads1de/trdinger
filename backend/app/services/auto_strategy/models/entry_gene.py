"""
エントリー注文遺伝子

エントリー注文のタイプとパラメータを定義します。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ..config.enums import EntryType


@dataclass
class EntryGene:
    """
    エントリー注文遺伝子

    GAによって最適化されるエントリー注文のパラメータを定義します。
    """

    # エントリータイプ
    entry_type: EntryType = EntryType.MARKET

    # 指値のオフセット（現在価格からの乖離率）
    # Long: 現在価格 * (1 - limit_offset_pct) で買い指値
    # Short: 現在価格 * (1 + limit_offset_pct) で売り指値
    limit_offset_pct: float = 0.005  # 0.5%

    # 逆指値のオフセット（現在価格からの乖離率）
    # Long: 現在価格 * (1 + stop_offset_pct) でブレイクアウト買い
    # Short: 現在価格 * (1 - stop_offset_pct) でブレイクアウト売り
    stop_offset_pct: float = 0.005  # 0.5%

    # 注文の有効期限（バー数） - 0は無制限
    order_validity_bars: int = 5

    # 有効/無効フラグ
    enabled: bool = True

    # 優先度（将来の拡張用）
    priority: float = 1.0

    def validate(self) -> Tuple[bool, List[str]]:
        """
        エントリー遺伝子の妥当性を検証

        Returns:
            (is_valid, errors) のタプル
        """
        errors: List[str] = []

        # limit_offset_pct の範囲チェック (0% ~ 10%)
        if self.limit_offset_pct < 0 or self.limit_offset_pct > 0.1:
            errors.append(
                f"limit_offset_pct は 0~0.1 の範囲である必要があります: {self.limit_offset_pct}"
            )

        # stop_offset_pct の範囲チェック (0% ~ 10%)
        if self.stop_offset_pct < 0 or self.stop_offset_pct > 0.1:
            errors.append(
                f"stop_offset_pct は 0~0.1 の範囲である必要があります: {self.stop_offset_pct}"
            )

        # order_validity_bars の範囲チェック (0以上)
        if self.order_validity_bars < 0:
            errors.append(
                f"order_validity_bars は 0以上である必要があります: {self.order_validity_bars}"
            )

        # entry_type が有効な値かチェック
        if not isinstance(self.entry_type, EntryType):
            try:
                EntryType(self.entry_type)
            except ValueError:
                errors.append(f"無効な entry_type です: {self.entry_type}")

        return len(errors) == 0, errors

    def to_dict(self) -> dict:
        """
        辞書形式に変換

        Returns:
            辞書形式のデータ
        """
        return {
            "entry_type": (
                self.entry_type.value
                if isinstance(self.entry_type, EntryType)
                else self.entry_type
            ),
            "limit_offset_pct": self.limit_offset_pct,
            "stop_offset_pct": self.stop_offset_pct,
            "order_validity_bars": self.order_validity_bars,
            "enabled": self.enabled,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EntryGene":
        """
        辞書形式から復元

        Args:
            data: 辞書形式のデータ

        Returns:
            EntryGene オブジェクト
        """
        entry_type_value = data.get("entry_type", "market")
        if isinstance(entry_type_value, str):
            entry_type = EntryType(entry_type_value)
        elif isinstance(entry_type_value, EntryType):
            entry_type = entry_type_value
        else:
            entry_type = EntryType.MARKET

        return cls(
            entry_type=entry_type,
            limit_offset_pct=data.get("limit_offset_pct", 0.005),
            stop_offset_pct=data.get("stop_offset_pct", 0.005),
            order_validity_bars=data.get("order_validity_bars", 5),
            enabled=data.get("enabled", True),
            priority=data.get("priority", 1.0),
        )
