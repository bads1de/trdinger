"""
指標関連ユーティリティ関数

指標のリスト取得、カテゴリ分類、設定の読み込みなどの機能を提供します。
"""

import logging
from typing import Dict, List

from app.services.indicators import TechnicalIndicatorService
from app.services.indicators.config.indicator_config import indicator_registry


# =============================================================================
# 指標リスト取得関連
# =============================================================================


def indicators_by_category(category: str) -> List[str]:
    """レジストリに登録済みのインジケーターからカテゴリ別に主名称とエイリアスを抽出"""
    indicator_registry.ensure_initialized()
    seen = set()
    results: List[str] = []

    for name, cfg in indicator_registry.get_all_indicators().items():
        try:
            if cfg and getattr(cfg, "category", None) == category:
                # 主名称を追加
                primary = cfg.indicator_name
                if primary not in seen:
                    seen.add(primary)
                    results.append(primary)
                # エイリアスも追加
                if cfg.aliases:
                    for alias in cfg.aliases:
                        if alias not in seen:
                            seen.add(alias)
                            results.append(alias)
        except Exception:
            continue
    results.sort()
    return results


def get_all_indicators(include_composite: bool = True) -> List[str]:
    """
    全指標タイプを取得

    Args:
        include_composite: 複合指標（COMPOSITE_INDICATORS）を含めるか
    """
    indicator_registry.ensure_initialized()
    all_types = indicator_registry.list_indicators()

    if include_composite:
        from ..config.constants import COMPOSITE_INDICATORS

        all_types.extend(COMPOSITE_INDICATORS)

    # 重複除去して順序維持
    seen = set()
    return [x for x in all_types if not (x in seen or seen.add(x))]


def get_volume_indicators() -> List[str]:
    return indicators_by_category("volume")


def get_momentum_indicators() -> List[str]:
    return indicators_by_category("momentum")


def get_trend_indicators() -> List[str]:
    return indicators_by_category("trend")


def get_volatility_indicators() -> List[str]:
    return indicators_by_category("volatility")


def get_valid_indicator_types() -> List[str]:
    """有効な指標タイプを取得"""
    return get_all_indicators(include_composite=True)


def get_all_indicator_ids() -> Dict[str, int]:
    """
    全指標のIDマッピングを取得

    テクニカル指標のIDマッピングを提供します。
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


# =============================================================================
# 設定・特性関連ユーティリティ
# =============================================================================
