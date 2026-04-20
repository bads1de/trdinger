"""
トレンドフィルター

ADX（Average Directional Index）等でトレンド強度を測定し、
レンジ相場（トレンドが弱い状態）でのエントリーを回避します。
"""

import logging
from typing import Any, Dict

from .base import BaseTool, ToolContext, ToolDefinition
from .registry import register_tool

logger = logging.getLogger(__name__)


class TrendFilter(BaseTool):
    """
    トレンドフィルター

    ADX（Average Directional Index）が閾値以下の場合、レンジ相場と判断して
    エントリーをスキップします。トレンドフォロー戦略などで有効です。

    ADXの一般的な解釈:
        - 0-25: レンジ相場（トレンドなし）
        - 25-50: トレンド発生
        - 50-75: 強いトレンド
        - 75-100: 非常に強いトレンド

    パラメータ:
        min_adx: 最低ADX値。デフォルト25
        adx_period: ADX計算期間。デフォルト14
        enabled: 有効フラグ
    """

    tool_definition = ToolDefinition(
        name="trend_filter",
        description="ADXでトレンド強度を判定し、レンジ相場でのエントリーを回避します",
        default_params={"enabled": True, "min_adx": 25.0, "adx_period": 14},
        priority="optional",
    )

    def should_skip_entry(self, context: ToolContext, params: Dict[str, Any]) -> bool:
        """
        トレンドが弱ければエントリーをスキップ

        Args:
            context: ツールコンテキスト
            params: パラメータ
                min_adx (float): 最低ADX値
                enabled (bool): 有効かどうか

        Returns:
            bool: スキップすべきならTrue
        """
        if not params.get("enabled", True):
            return False

        min_adx = params.get("min_adx", 25.0)

        # extra_data にADX値が渡されているかチェック
        adx_value = context.extra_data.get("adx")
        if adx_value is not None:
            return adx_value < min_adx

        # データがない場合はスキップしない
        return False

    def mutate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """パラメータの突然変異"""
        import random

        new_params = super().mutate_params(params)

        if random.random() < 0.3:
            new_params["min_adx"] = max(
                10.0,
                min(50.0, new_params.get("min_adx", 25.0) * random.uniform(0.8, 1.2)),
            )
        if random.random() < 0.2:
            new_params["adx_period"] = max(
                7,
                min(
                    30, int(new_params.get("adx_period", 14) * random.uniform(0.8, 1.2))
                ),
            )

        return new_params


trend_filter = TrendFilter()
register_tool(trend_filter)
