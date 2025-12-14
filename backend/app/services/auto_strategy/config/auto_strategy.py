"""
AutoStrategyConfigクラス

オートストラテジー統合設定クラスを提供します。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .ga import GA_THRESHOLD_RANGES, GASettings
from .indicators import IndicatorSettings
from .position_sizing import PositionSizingSettings
from .tpsl import TPSLSettings
from .trading import TradingSettings

logger = logging.getLogger(__name__)


@dataclass
class AutoStrategyConfig:
    """オートストラテジー統合設定

    このクラスはオートストラテジーの全ての設定を一元管理します。
    """

    # 設定グループ
    trading: TradingSettings = field(default_factory=TradingSettings)
    indicators: IndicatorSettings = field(default_factory=IndicatorSettings)
    ga: GASettings = field(default_factory=GASettings)
    tpsl: TPSLSettings = field(default_factory=TPSLSettings)
    position_sizing: PositionSizingSettings = field(
        default_factory=PositionSizingSettings
    )

    # 共通設定
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    enable_async_processing: bool = False
    log_level: str = "WARNING"

    threshold_ranges: Dict[str, List] = field(
        default_factory=lambda: GA_THRESHOLD_RANGES.copy()
    )

    # 設定検証ルール
    validation_rules: Dict[str, Any] = field(
        default_factory=lambda: {
            "required_fields": [],
            "ranges": {
                "cache_ttl_hours": [1, 168],  # 1時間から1週間
            },
            "types": {
                "enable_caching": bool,
                "enable_async_processing": bool,
                "log_level": str,
            },
        }
    )

    def get_default_values(self) -> Dict[str, Any]:
        """デフォルト値を取得（サブコンポーネント付き）"""
        # 自動生成したデフォルト値にサブコンポーネントを統合
        return {
            "enable_caching": True,
            "cache_ttl_hours": 24,
            "enable_async_processing": False,
            "log_level": "WARNING",
            "threshold_ranges": GA_THRESHOLD_RANGES.copy(),
            "trading": TradingSettings().get_default_values(),
            "indicators": IndicatorSettings().get_default_values(),
            "ga": GASettings().get_default_values(),
            "tpsl": TPSLSettings().get_default_values(),
            "position_sizing": PositionSizingSettings().get_default_values(),
        }
