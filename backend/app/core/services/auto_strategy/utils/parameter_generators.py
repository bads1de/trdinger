"""
パラメータ生成ユーティリティ

IndicatorParameterManagerを使用した統一されたパラメータ生成システム。
旧システムのParameterGeneratorクラスは廃止され、全てParameterManagerに統合されました。
"""

import logging
from typing import Dict, Any

from app.core.services.indicators.parameter_manager import IndicatorParameterManager
from app.core.services.indicators.config.indicator_config import indicator_registry

logger = logging.getLogger(__name__)


def generate_indicator_parameters(indicator_type: str) -> Dict[str, Any]:
    """
    指標タイプに応じたパラメータを生成

    IndicatorParameterManagerシステムを使用した統一されたパラメータ生成。
    """
    try:
        config = indicator_registry.get_indicator_config(indicator_type)
        if config:
            manager = IndicatorParameterManager()
            return manager.generate_parameters(indicator_type, config)
        else:
            logger.warning(f"指標 {indicator_type} の設定が見つかりません")
            return {}
    except Exception as e:
        logger.error(f"指標 {indicator_type} のパラメータ生成に失敗: {e}")
        return {}
