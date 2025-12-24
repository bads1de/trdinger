"""
市場データ処理ハンドラ

市場データの準備、キャッシュ管理を提供します。
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from ..config.constants import AUTO_STRATEGY_DEFAULTS


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
        """
        市場データを準備し、デフォルト値やキャッシュで情報を拡張します。

        ATR値やボラティリティ指標がmarket_dataに含まれていない場合、
        キャッシュまたは設定ファイルからのデフォルト値（例: 2%）で補完します。

        Args:
            symbol: 通貨ペア
            current_price: 現在価格
            market_data: 外部から提供された市場データ（任意）
            use_cache: キャッシュを利用するかどうか

        Returns:
            拡張された市場データの辞書
        """
        enhanced = market_data.copy() if market_data else {}

        # 1. キャッシュの統合
        if use_cache and self._cache and not self._cache.is_expired():
            enhanced.update(self._cache.atr_values)
            enhanced.update(self._cache.volatility_metrics)

        # 2. ATR値の保証
        if "atr" not in enhanced and "atr_pct" not in enhanced:
            default_atr_pct = AUTO_STRATEGY_DEFAULTS["default_atr_multiplier"]
            enhanced.update(
                {
                    "atr": current_price * default_atr_pct,
                    "atr_pct": default_atr_pct,
                    "atr_source": "default",
                }
            )

        # 3. ボラティリティメトリクスの正規化
        # デフォルト値は 0.02 (2%)
        enhanced.setdefault("volatility", enhanced.get("atr_pct", 0.02))

        return enhanced

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
