"""
指標関連ユーティリティ関数

"""

from typing import Dict, List
import logging
from app.services.indicators import TechnicalIndicatorService


def _load_indicator_registry():
    """indicator_registry を初期化して返す（副作用目的の import を含む）"""
    # setup_* の実行を保証するための side-effect import
    from app.services.indicators.config import indicator_definitions  # noqa: F401
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


def get_all_indicators() -> List[str]:
    """全指標タイプを取得（テクニカル + ML）"""
    # 遅延インポートで循環依存を回避
    from ..config.constants import ML_INDICATOR_TYPES

    technical = (
        get_volume_indicators()
        + get_momentum_indicators()
        + get_trend_indicators()
        + get_volatility_indicators()
    )
    # 重複除去して順序維持
    seen = set()
    ordered: List[str] = []
    for n in technical + ML_INDICATOR_TYPES:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def validate_symbol(symbol: str) -> bool:
    """シンボルの妥当性を検証"""
    # 遅延インポートで循環依存を回避
    from ..config.constants import SUPPORTED_SYMBOLS

    return symbol in SUPPORTED_SYMBOLS


def validate_timeframe(timeframe: str) -> bool:
    """時間軸の妥当性を検証"""
    # 遅延インポートで循環依存を回避
    from ..config.constants import SUPPORTED_TIMEFRAMES

    return timeframe in SUPPORTED_TIMEFRAMES


def get_all_indicator_ids() -> Dict[str, int]:
    """
    全指標のIDマッピングを取得（統合版）

    テクニカル指標とML指標を統合したIDマッピングを提供します。
    gene_utils.py との重複機能を統合しています。
    """
    try:
        indicator_service = TechnicalIndicatorService()
        technical_indicators = list(
            indicator_service.get_supported_indicators().keys()
        )

        # 遅延インポートで循環依存を回避
        from ..config.constants import ML_INDICATOR_TYPES

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