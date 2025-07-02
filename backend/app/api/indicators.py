"""
指標関連のAPIエンドポイント
"""

from fastapi import APIRouter, Depends
from typing import List, Dict, Any
from collections import defaultdict

from app.core.services.indicators import TechnicalIndicatorService

router = APIRouter()


# 依存性注入
def get_indicator_service():
    return TechnicalIndicatorService()


@router.get("/indicators", response_model=List[str])
async def get_all_indicators(service: TechnicalIndicatorService = Depends(get_indicator_service)):
    """サポートされている全指標のリストを取得"""
    return list(service.get_supported_indicators().keys())


@router.get("/indicators/categories", response_model=Dict[str, List[str]])
async def get_indicators_by_category(
    service: TechnicalIndicatorService = Depends(get_indicator_service),
):
    """カテゴリ別の指標リストを取得"""
    supported_indicators = service.get_supported_indicators()
    categories = defaultdict(list)
    for name, config in service.registry._configs.items():
        if name in supported_indicators:
            category = config.category or "unknown"
            categories[category].append(name)
    return categories


@router.get("/indicators/info", response_model=Dict[str, Dict[str, Any]])
async def get_indicators_info(service: TechnicalIndicatorService = Depends(get_indicator_service)):
    """全指標の詳細情報を取得"""
    return service.get_supported_indicators()


@router.get("/indicators/count", response_model=Dict[str, int])
async def get_indicators_count(service: TechnicalIndicatorService = Depends(get_indicator_service)):
    """指標数の統計を取得"""
    categories = await get_indicators_by_category(service)
    count = {category: len(indicators) for category, indicators in categories.items()}
    count["total"] = len(await get_all_indicators(service))
    return count