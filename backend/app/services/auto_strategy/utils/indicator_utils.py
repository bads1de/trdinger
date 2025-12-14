"""
指標関連ユーティリティ関数

"""

import logging
from typing import Dict, List

from app.services.indicators import TechnicalIndicatorService


def _load_indicator_registry():
    """indicator_registry を取得"""
    from app.services.indicators.config.indicator_config import (
        indicator_registry,
    )

    return indicator_registry


def indicators_by_category(category: str) -> List[str]:
    """レジストリに登録済みのインジケーターからカテゴリ別に主名称のみ抽出"""
    registry = _load_indicator_registry()
    seen = set()
    results: List[str] = []
    for name, cfg in registry._configs.items():  # type: ignore[attr-defined]
        try:
            if cfg and getattr(cfg, "category", None) == category:
                primary = getattr(cfg, "indicator_name", name)
                if primary not in seen:
                    seen.add(primary)
                    results.append(primary)
        except Exception:
            continue
    results.sort()
    return results


def get_volume_indicators() -> List[str]:
    return indicators_by_category("volume")


def get_momentum_indicators() -> List[str]:
    return indicators_by_category("momentum")


def get_trend_indicators() -> List[str]:
    return indicators_by_category("trend")


def get_volatility_indicators() -> List[str]:
    return indicators_by_category("volatility")


def get_original_indicators() -> List[str]:
    return indicators_by_category("original")


def get_all_indicators() -> List[str]:
    """全指標タイプを取得（テクニカル + 複合指標）"""
    # 遅延インポートで循環依存を回避
    from ..config.constants import COMPOSITE_INDICATORS

    technical = (
        get_volume_indicators()
        + get_momentum_indicators()
        + get_trend_indicators()
        + get_volatility_indicators()
    )
    # 重複除去して順序維持
    seen = set()
    ordered: List[str] = []
    for n in technical + COMPOSITE_INDICATORS:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def get_all_indicator_ids() -> Dict[str, int]:
    """
    全指標のIDマッピングを取得（統合版）

    テクニカル指標のIDマッピングを提供します。
    gene_utils.py との重複機能を統合しています。
    """
    try:
        indicator_service = TechnicalIndicatorService()
        technical_indicators = list(indicator_service.get_supported_indicators().keys())

        # IDマッピングを作成（空文字列は0、その他は1から開始）
        return {"": 0, **{ind: i + 1 for i, ind in enumerate(technical_indicators)}}
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"指標ID取得エラー: {e}")
        return {"": 0}


def get_valid_indicator_types() -> List[str]:
    """有効な指標タイプを取得（VALID_INDICATOR_TYPESの実装）"""
    # 重複除去して順序維持
    technical = (
        get_volume_indicators()
        + get_momentum_indicators()
        + get_trend_indicators()
        + get_volatility_indicators()
        + get_original_indicators()
    )
    from ..config.constants import COMPOSITE_INDICATORS

    all_indicators = technical + COMPOSITE_INDICATORS

    seen = set()
    valid_types: List[str] = []
    for name in all_indicators:
        if name not in seen:
            seen.add(name)
            valid_types.append(name)

    return valid_types
