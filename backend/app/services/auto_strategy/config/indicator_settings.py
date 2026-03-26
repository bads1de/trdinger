"""
テクニカル指標設定

IndicatorSettings クラスを提供します。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils.indicators import get_valid_indicator_types
from app.services.indicators.config.indicator_config import indicator_registry

from .base import BaseConfig
from .constants import (
    DATA_SOURCES,
    OPERATORS,
)


def _build_indicator_characteristics() -> Dict[str, Any]:
    """指標メタデータを遅延構築する。"""
    indicator_registry.ensure_initialized()
    return {
        name: {
            "type": cfg.category or "technical",
            "scale_type": cfg.scale_type.value,
        }
        for name, cfg in indicator_registry.get_all_indicators().items()
    }


@dataclass
class IndicatorSettings(BaseConfig):
    """テクニカル指標設定"""

    # 利用可能な指標
    valid_indicator_types: List[str] = field(default_factory=get_valid_indicator_types)

    # 指標特性データベース
    indicator_characteristics: Dict[str, Any] = field(
        default_factory=_build_indicator_characteristics
    )

    # 演算子とデータソース
    operators: List[str] = field(default_factory=lambda: OPERATORS.copy())
    data_sources: List[str] = field(default_factory=lambda: DATA_SOURCES.copy())

    def get_all_indicators(self) -> List[str]:
        """
        システムで利用可能な全テクニカル指標のリストを取得

        Returns:
            指標名のリスト（例: ['SMA', 'RSI', ...]）
        """
        return self.valid_indicator_types

    def get_indicator_characteristics(self, indicator: str) -> Optional[Dict[str, Any]]:
        """
        特定の指標の特性（スケールタイプ、デフォルト閾値等）を取得

        Args:
            indicator: 指標名

        Returns:
            特性情報を含む辞書。見つからない場合はNone。
        """
        return self.indicator_characteristics.get(indicator)
