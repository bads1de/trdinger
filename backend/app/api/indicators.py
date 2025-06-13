"""
指標関連のAPIエンドポイント
"""

from fastapi import APIRouter
from typing import List, Dict, Any

from app.core.services.indicators.constants import (
    ALL_INDICATORS,
    TREND_INDICATORS,
    MOMENTUM_INDICATORS,
    VOLATILITY_INDICATORS,
    VOLUME_INDICATORS,
    PRICE_TRANSFORM_INDICATORS,
    OTHER_INDICATORS,
    INDICATOR_INFO,
    TOTAL_INDICATORS,
)

router = APIRouter()


@router.get("/indicators", response_model=List[str])
async def get_all_indicators():
    """全指標リストを取得"""
    return ALL_INDICATORS


@router.get("/indicators/categories", response_model=Dict[str, List[str]])
async def get_indicators_by_category():
    """カテゴリ別指標リストを取得"""
    return {
        "trend": TREND_INDICATORS,
        "momentum": MOMENTUM_INDICATORS,
        "volatility": VOLATILITY_INDICATORS,
        "volume": VOLUME_INDICATORS,
        "price_transform": PRICE_TRANSFORM_INDICATORS,
        "other": OTHER_INDICATORS,
    }


@router.get("/indicators/info", response_model=Dict[str, Dict[str, Any]])
async def get_indicators_info():
    """指標情報辞書を取得"""
    return INDICATOR_INFO


@router.get("/indicators/count", response_model=Dict[str, int])
async def get_indicators_count():
    """指標数統計を取得"""
    return {
        "total": TOTAL_INDICATORS,
        "trend": len(TREND_INDICATORS),
        "momentum": len(MOMENTUM_INDICATORS),
        "volatility": len(VOLATILITY_INDICATORS),
        "volume": len(VOLUME_INDICATORS),
        "price_transform": len(PRICE_TRANSFORM_INDICATORS),
        "other": len(OTHER_INDICATORS),
    }
