"""
指標関連ユーティリティ関数

"""

from typing import Dict, List

from ..config.constants import VALID_INDICATOR_TYPES, ML_INDICATOR_TYPES, SUPPORTED_SYMBOLS, SUPPORTED_TIMEFRAMES
from app.services.indicators import TechnicalIndicatorService
import logging


def get_all_indicators() -> List[str]:
    """全指標タイプを取得"""
    return VALID_INDICATOR_TYPES + ML_INDICATOR_TYPES


def validate_symbol(symbol: str) -> bool:
    """シンボルの妥当性を検証"""
    return symbol in SUPPORTED_SYMBOLS


def validate_timeframe(timeframe: str) -> bool:
    """時間軸の妥当性を検証"""
    return timeframe in SUPPORTED_TIMEFRAMES


def get_all_indicator_ids() -> Dict[str, int]:
    """
    全指標のIDマッピングを取得（統合版）

    テクニカル指標とML指標を統合したIDマッピングを提供します。
    gene_utils.py との重複機能を統合しています。
    """
    try:
        indicator_service = TechnicalIndicatorService()
        technical_indicators = list(indicator_service.get_supported_indicators().keys())

        # 全指標を結合
        all_indicators = technical_indicators + ML_INDICATOR_TYPES

        # IDマッピングを作成（空文字列は0、その他は1から開始）
        return {"": 0, **{ind: i + 1 for i, ind in enumerate(all_indicators)}}
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"指標ID取得エラー: {e}")
        return {"": 0}


def get_id_to_indicator_mapping(indicator_ids: Dict[str, int]) -> Dict[int, str]:
    """ID→指標の逆引きマッピングを取得"""
    return {v: k for k, v in indicator_ids.items()}