"""
指標関連ユーティリティ関数

指標のリスト取得、カテゴリ分類、設定の読み込みなどの機能を提供します。
"""

from app.services.indicators.config.indicator_config import (
    get_cached_indicators,
    indicator_registry,
)

from ..config.constants import COMPOSITE_INDICATORS


def get_all_indicators(include_composite: bool = True) -> list[str]:
    """
    全指標タイプを取得

    Args:
        include_composite: 複合指標（COMPOSITE_INDICATORS）を含めるか
    """
    indicator_registry.ensure_initialized()
    # キャッシュを使用
    all_types = get_cached_indicators()

    if include_composite:
        return all_types
    else:
        return [t for t in all_types if t not in COMPOSITE_INDICATORS]
