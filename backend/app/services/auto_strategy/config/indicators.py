"""
IndicatorSettingsクラス

テクニカル指標設定を管理します。
"""

from typing import Any, Dict, List, Optional

from dataclasses import dataclass, field

from .base import BaseConfig
from ..constants import (
    OPERATORS,
    DATA_SOURCES,
)
from ..utils.indicator_characteristics import INDICATOR_CHARACTERISTICS
from ..utils.indicator_utils import get_valid_indicator_types


@dataclass
class IndicatorSettings(BaseConfig):
    """テクニカル指標設定"""

    # 利用可能な指標
    valid_indicator_types: List[str] = field(default_factory=get_valid_indicator_types)

    # 指標特性データベース
    indicator_characteristics: Dict[str, Any] = field(
        default_factory=lambda: INDICATOR_CHARACTERISTICS.copy()
    )

    # 演算子とデータソース
    operators: List[str] = field(default_factory=lambda: OPERATORS.copy())
    data_sources: List[str] = field(default_factory=lambda: DATA_SOURCES.copy())

    def get_default_values(self) -> Dict[str, Any]:
        """デフォルト値を取得（自動生成を利用）"""
        # フィールドから自動生成したデフォルト値を取得
        defaults = self.get_default_values_from_fields()
        # 必要に応じてカスタマイズ（外部定数など）
        return defaults

    def get_all_indicators(self) -> List[str]:
        """全指標タイプを取得"""
        return self.valid_indicator_types

    def get_indicator_characteristics(self, indicator: str) -> Optional[Dict[str, Any]]:
        """特定の指標の特性を取得"""
        return self.indicator_characteristics.get(indicator)
