"""
ツール基底クラス

すべてのエントリーフィルターツールが継承する抽象クラスを定義します。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class ToolContext:
    """
    ツールに渡されるコンテキスト情報

    エントリー判定時に必要な市場データや状態を保持します。
    """

    # 現在のバーのタイムスタンプ
    timestamp: Optional[pd.Timestamp] = None

    # 現在の価格データ
    current_price: float = 0.0
    current_high: float = 0.0
    current_low: float = 0.0
    current_volume: float = 0.0

    # 追加の市場データ（OI, FR など）
    extra_data: Dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """
    エントリーフィルターツールの基底クラス

    すべてのツールはこのクラスを継承し、共通インターフェースを実装します。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        ツールの一意な識別名

        Returns:
            ツール名（例: 'weekend_filter'）
        """
        pass

    @property
    def description(self) -> str:
        """
        ツールの説明

        Returns:
            人間が読める説明文
        """
        return ""

    @abstractmethod
    def should_skip_entry(self, context: ToolContext, params: Dict[str, Any]) -> bool:
        """
        エントリーをスキップすべきか判定

        Args:
            context: 現在の市場状態を含むコンテキスト
            params: ツール固有のパラメータ

        Returns:
            True: エントリーをスキップ（ブロック）
            False: エントリーを許可
        """
        pass

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """
        デフォルトパラメータを取得

        Returns:
            パラメータ辞書
        """
        pass

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        パラメータの妥当性を検証

        Args:
            params: 検証するパラメータ

        Returns:
            True: 有効なパラメータ
        """
        return True

    def mutate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        パラメータを突然変異させる（GAで使用）

        Args:
            params: 元のパラメータ

        Returns:
            変異後のパラメータ
        """
        # デフォルトは変異なし
        return params.copy()


