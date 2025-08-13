"""
遺伝子エンコーディングに関するユーティリティ関数
"""

import logging
from typing import Dict

from app.services.indicators.indicator_orchestrator import (
    TechnicalIndicatorService,
)

from .gene_tpsl import TPSLMethod

logger = logging.getLogger(__name__)


def get_indicator_ids() -> Dict[str, int]:
    """指標IDマッピングを取得（shared_constants.pyに統合済み）"""
    from ..config.constants import get_all_indicator_ids

    return get_all_indicator_ids()


def get_id_to_indicator(indicator_ids: Dict[str, int]) -> Dict[int, str]:
    """ID→指標の逆引きマッピングを取得（shared_constants.pyに統合済み）"""
    from ..config.constants import get_id_to_indicator_mapping

    return get_id_to_indicator_mapping(indicator_ids)


def normalize_parameter(
    value: float, min_val: float = 1, max_val: float = 200
) -> float:
    """パラメータを0-1の範囲に正規化（auto_strategy_utils.pyに統合済み）"""
    from ..utils.auto_strategy_utils import AutoStrategyUtils

    return AutoStrategyUtils.normalize_parameter(value, min_val, max_val)


def denormalize_parameter(
    normalized_val: float, min_val: float = 1, max_val: float = 200
) -> int:
    """正規化されたパラメータを元の範囲に戻す（auto_strategy_utils.pyに統合済み）"""
    from ..utils.auto_strategy_utils import AutoStrategyUtils

    return AutoStrategyUtils.denormalize_parameter(normalized_val, min_val, max_val)


def get_encoding_info(indicator_ids: Dict[str, int]) -> Dict:
    """エンコーディング情報を取得（auto_strategy_utils.pyに統合済み）"""
    from ..utils.auto_strategy_utils import AutoStrategyUtils

    return AutoStrategyUtils.get_encoding_info(indicator_ids)
