"""
市場データ処理ハンドラ

市場データの準備、キャッシュ管理を提供します。
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from app.config.unified_config import unified_config


@dataclass
class MarketDataCache:
    """市場データキャッシュ"""

    atr_values: Dict[str, float]
    volatility_metrics: Dict[str, float]
    price_data: Optional[pd.DataFrame]
    last_updated: datetime

    def is_expired(self, max_age_minutes: int = 5) -> bool:
        """キャッシュが期限切れかチェック"""
        return (
            datetime.now() - self.last_updated
        ).total_seconds() > max_age_minutes * 60


logger = logging.getLogger(__name__)


class MarketDataHandler:
    """市場データ処理ハンドラ"""

    def __init__(self):
        self._cache: Optional[MarketDataCache] = None

    def prepare_market_data(
        self,
        symbol: str,
        current_price: float,
        market_data: Optional[Dict[str, Any]],
        use_cache: bool,
    ) -> Dict[str, Any]:
        """市場データの準備と拡張"""
        enhanced_data = market_data.copy() if market_data else {}

        # キャッシュチェック
        if use_cache and self._cache and not self._cache.is_expired():
            enhanced_data.update(self._cache.atr_values)
            enhanced_data.update(self._cache.volatility_metrics)

        # ATR値の確保
        if "atr" not in enhanced_data and "atr_pct" not in enhanced_data:
            # デフォルトATR値を設定（現在価格の設定値%）
            default_atr_pct = unified_config.auto_strategy.default_atr_multiplier
            enhanced_data["atr"] = current_price * default_atr_pct
            enhanced_data["atr_pct"] = default_atr_pct
            enhanced_data["atr_source"] = "default"

        # ボラティリティメトリクスの追加
        if "volatility" not in enhanced_data:
            enhanced_data["volatility"] = enhanced_data.get("atr_pct", 0.02)

        return enhanced_data

    def update_cache(
        self,
        atr_values: Dict[str, float],
        volatility_metrics: Dict[str, float],
        price_data: Optional[pd.DataFrame] = None,
    ):
        """キャッシュの更新"""
        self._cache = MarketDataCache(
            atr_values=atr_values,
            volatility_metrics=volatility_metrics,
            price_data=price_data,
            last_updated=datetime.now(),
        )

    def get_cache(self) -> Optional[MarketDataCache]:
        """キャッシュの取得"""
        return self._cache

    def clear_cache(self):
        """キャッシュのクリア"""
        self._cache = None

    def is_cache_valid(self) -> bool:
        """キャッシュが有効かチェック"""
        return self._cache is not None and not self._cache.is_expired()





