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
    """指標IDマッピングを取得（テクニカル指標 + ML指標）"""
    try:
        # テクニカル指標を取得
        indicator_service = TechnicalIndicatorService()
        technical_indicators = list(indicator_service.get_supported_indicators().keys())

        # ML指標を追加
        ml_indicators = ['ML_UP_PROB', 'ML_DOWN_PROB', 'ML_RANGE_PROB']

        # 全指標を結合
        all_indicators = technical_indicators + ml_indicators

        indicator_ids = {"": 0}  # 未使用
        for i, indicator in enumerate(all_indicators, 1):
            indicator_ids[indicator] = i

        return indicator_ids
    except Exception as e:
        logger.error(f"指標IDの取得に失敗しました: {e}")
        return {"": 0}


def get_id_to_indicator(indicator_ids: Dict[str, int]) -> Dict[int, str]:
    """ID→指標の逆引きマッピングを取得"""
    return {v: k for k, v in indicator_ids.items()}


def normalize_parameter(
    value: float, min_val: float = 1, max_val: float = 200
) -> float:
    """パラメータを0-1の範囲に正規化"""
    try:
        return (value - min_val) / (max_val - min_val)
    except ZeroDivisionError:
        return 0.0


def denormalize_parameter(
    normalized_val: float, min_val: float = 1, max_val: float = 200
) -> int:
    """正規化されたパラメータを元の範囲に戻す"""
    try:
        value = min_val + normalized_val * (max_val - min_val)
        return int(max(min_val, min(max_val, int(value))))
    except Exception:
        return int(min_val)


def get_encoding_info(indicator_ids: Dict[str, int]) -> Dict:
    """エンコーディング情報を取得"""
    return {
        "indicator_count": len(indicator_ids) - 1,
        "max_indicators": 5,
        "encoding_length": 32,  # 5指標×2 + 条件×6 + TP/SL×8 + ポジションサイジング×8
        "tpsl_encoding_length": 8,
        "position_sizing_encoding_length": 8,
        "supported_indicators": list(indicator_ids.keys())[1:],
        "supported_tpsl_methods": [method.value for method in TPSLMethod],
    }
