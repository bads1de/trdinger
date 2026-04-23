"""
ファンディングレートフィルター

ファンディングレート（資金調達料金）が極端に高い/低い場合、
市場が歪んでいると判断してエントリーを回避します。
"""

import logging
from typing import Any, Dict

from .base import BaseTool, ToolContext, ToolDefinition
from .registry import register_tool

logger = logging.getLogger(__name__)


class FundingRateFilter(BaseTool):
    """
    ファンディングレートフィルター

    ファンディングレート（資金調達料金）が閾値を超える極端な値の場合、
    市場が歪んでいると判断してエントリーをスキップします。

    仮想通貨のパーペチュアルスワップでは、資金調達料金が定期的に発生します。
    極端な資金調達料金は、市場のアンバランス（ロング/ショートの偏り）を
    示しており、通常の戦略が機能しない可能性があります。

    パラメータ:
        max_funding_rate: 最大許容資金調達レート（絶対値）。デフォルト0.001（0.1%）
        enabled: 有効フラグ
    """

    tool_definition = ToolDefinition(
        name="funding_rate_filter",
        description="ファンディングレートが高い場合のエントリーを回避します",
        default_params={"enabled": True, "max_funding_rate": 0.001},
        priority="disabled",
    )

    def should_skip_entry(
        self, context: ToolContext, params: Dict[str, Any]
    ) -> bool:
        """
        ファンディングレートが極端ならエントリーをスキップ

        Args:
            context: ツールコンテキスト
            params: パラメータ
                max_funding_rate (float): 最大許容資金調達レート
                enabled (bool): 有効かどうか

        Returns:
            bool: スキップすべきならTrue
        """
        if not params.get("enabled", True):
            return False

        max_funding_rate = params.get("max_funding_rate", 0.001)

        # extra_data にファンディングレートが渡されているかチェック
        funding_rate = context.extra_data.get("funding_rate")
        if funding_rate is not None:
            return abs(funding_rate) > max_funding_rate

        # データがない場合はスキップしない
        return False

    def mutate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """パラメータの突然変異"""
        import random

        new_params = super().mutate_params(params)

        if random.random() < 0.3:
            new_params["max_funding_rate"] = max(
                0.0001,
                min(
                    0.01,
                    new_params.get("max_funding_rate", 0.001)
                    * random.uniform(0.5, 2.0),
                ),
            )

        return new_params


funding_rate_filter = FundingRateFilter()
register_tool(funding_rate_filter)
