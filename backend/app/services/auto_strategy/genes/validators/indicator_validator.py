"""
指標遺伝子バリデーター
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from enum import Enum
from typing import Any, List, Optional, Union

from app.utils.error_handler import safe_operation

from ...utils.indicators import get_all_indicators

logger = logging.getLogger(__name__)


class IndicatorValidator:
    """指標遺伝子の妥当性検証を担当します。"""

    def __init__(self) -> None:
        """初期化"""
        self.valid_indicator_types = get_all_indicators()

    @safe_operation(
        context="指標遺伝子バリデーション",
        is_api_call=False,
        default_return=False,
    )
    def validate_indicator_gene(self, indicator_gene) -> bool:
        """指標遺伝子の妥当性を検証"""
        if not indicator_gene.type or not isinstance(indicator_gene.type, str):
            logger.warning(f"指標タイプが無効: {indicator_gene.type}")
            return False

        if not isinstance(indicator_gene.parameters, dict):
            logger.warning(
                f"指標パラメータが無効: {indicator_gene.parameters}"
            )
            return False

        # ログ: 指標タイプが有効リストに含まれているかを確認
        logger.debug(
            f"指標タイプ {indicator_gene.type} が valid_indicator_types に含まれているか: {indicator_gene.type in self.valid_indicator_types}"
        )

        if indicator_gene.type not in self.valid_indicator_types:
            logger.warning(
                f"無効な指標タイプ: {indicator_gene.type}, 有効なタイプ: {self.valid_indicator_types[:10]}..."
            )  # 先頭10個のみ表示
            return False

        if "period" in indicator_gene.parameters:
            period = indicator_gene.parameters["period"]
            if not isinstance(period, (int, float)) or period <= 0:
                logger.warning(f"無効な期間パラメータ: {period}")
                return False

        # タイムフレームのバリデーション（設定されている場合のみ）
        timeframe = getattr(indicator_gene, "timeframe", None)
        if timeframe is not None:
            from app.config.constants import SUPPORTED_TIMEFRAMES

            if timeframe not in SUPPORTED_TIMEFRAMES:
                logger.warning(
                    f"無効なタイムフレーム: {timeframe}, "
                    f"サポートされるタイムフレーム: {SUPPORTED_TIMEFRAMES}"
                )
                return False

        # パラメータ依存関係制約のバリデーション
        from app.services.indicators.config import indicator_registry

        config = indicator_registry.get_indicator_config(indicator_gene.type)
        if config and config.parameter_constraints:
            is_valid, errors = config.validate_constraints(
                indicator_gene.parameters
            )
            if not is_valid:
                logger.warning(
                    f"パラメータ制約違反 ({indicator_gene.type}): {errors}"
                )
                return False

        logger.debug(f"指標タイプ {indicator_gene.type} は有効です")
        return True

    @safe_operation(
        context="生成用指標遺伝子バリデーション",
        is_api_call=False,
        default_return=False,
    )
    def validate_indicator_gene_for_generation(
        self,
        indicator_gene,
        indicator_universe_mode: Any = "curated",
        allowed_indicators: Optional[Collection[str]] = None,
    ) -> bool:
        """GA 生成・変異で使う指標遺伝子をユニバース込みで検証する。"""
        if not self.validate_indicator_gene(indicator_gene):
            return False

        if allowed_indicators is not None:
            try:
                normalized_allowed = {
                    str(name).upper() for name in allowed_indicators
                }
            except TypeError:
                return False
            return str(indicator_gene.type).upper() in normalized_allowed

        from ...config.indicator_universe import is_indicator_in_universe

        return is_indicator_in_universe(
            indicator_gene.type, indicator_universe_mode
        )
