"""
自動戦略ツール

エントリーフィルターなどのツールを提供します。
"""

from .base import BaseTool, ToolContext, ToolDefinition
from .beginning_of_month_filter import (
    BeginningOfMonthFilter,
    beginning_of_month_filter,
)
from .end_of_month_filter import EndOfMonthFilter, end_of_month_filter
from .funding_rate_filter import FundingRateFilter, funding_rate_filter
from .london_fix_filter import LondonFixFilter, london_fix_filter
from .monday_reversal_filter import (
    MondayReversalFilter,
    monday_reversal_filter,
)
from .registry import ToolRegistry, register_tool, tool_registry
from .specific_day_filter import SpecificDayFilter, specific_day_filter
from .trend_filter import TrendFilter, trend_filter
from .us_lunch_filter import USLunchFilter, us_lunch_filter
from .us_market_open_filter import USMarketOpenFilter, us_market_open_filter
from .volatility_filter import VolatilityFilter, volatility_filter
from .volume_filter import VolumeFilter, volume_filter

# 利用可能なツールをインポート（自動登録される）
from .weekend_filter import WeekendFilter, weekend_filter

__all__ = [
    # 基底クラス
    "BaseTool",
    "ToolContext",
    "ToolDefinition",
    # レジストリ
    "ToolRegistry",
    "tool_registry",
    "register_tool",
    # ツール
    "WeekendFilter",
    "weekend_filter",
    "LondonFixFilter",
    "london_fix_filter",
    "MondayReversalFilter",
    "monday_reversal_filter",
    "USLunchFilter",
    "us_lunch_filter",
    "USMarketOpenFilter",
    "us_market_open_filter",
    "EndOfMonthFilter",
    "end_of_month_filter",
    "SpecificDayFilter",
    "specific_day_filter",
    "BeginningOfMonthFilter",
    "beginning_of_month_filter",
    # 新規フィルター
    "VolatilityFilter",
    "volatility_filter",
    "TrendFilter",
    "trend_filter",
    "VolumeFilter",
    "volume_filter",
    "FundingRateFilter",
    "funding_rate_filter",
]
