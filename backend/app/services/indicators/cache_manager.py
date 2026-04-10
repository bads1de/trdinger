"""
インジケーターキャッシュ管理モジュール

計算結果のキャッシュ管理を担当します。
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
from cachetools import LRUCache

logger = logging.getLogger(__name__)


class IndicatorCacheManager:
    """
    インジケーター計算結果のキャッシュ管理クラス

    LRUキャッシュを使用して計算結果をキャッシュし、
    同じ計算を繰り返し実行するのを防ぎます。
    """

    def __init__(self, maxsize: int = 5000):
        """
        初期化

        Args:
            maxsize: キャッシュの最大サイズ
        """
        self._calculation_cache: LRUCache = LRUCache(maxsize=maxsize)

    def clear_cache(self) -> None:
        """計算キャッシュをクリアする"""
        self._calculation_cache.clear()
        logger.info("Indicator calculation cache cleared.")

    def make_cache_key(
        self, indicator_type: str, params: Dict[str, Any], df: pd.DataFrame
    ) -> Optional[tuple]:
        """
        キャッシュキーを生成（データの内容に基づいた一意なキー）
        DataFrame はミュータブルなので、属性にハッシュを保持しない。
        in-place 更新でも必ず再計算されるよう、毎回内容からハッシュを作る。
        """
        try:
            # パラメータをソートされた不変セットに変換
            cache_params = frozenset(sorted([(k, str(v)) for k, v in params.items()]))

            # データのメタデータを抽出
            data_meta: tuple
            if not df.empty:
                # 全列のハッシュの合計を使うことで、どの列が変わっても検知できるようにする
                data_hash = pd.util.hash_pandas_object(df, index=True).sum()  # type: ignore[reportAttributeAccessIssue]

                data_meta = (
                    df.index[0],  # 開始日
                    df.index[-1],  # 終了日
                    len(df),  # データ長
                    data_hash,  # データ内容のハッシュ
                )
            else:
                data_meta = ("empty",)

            return (indicator_type, cache_params, data_meta)
        except Exception:
            return None

    def get_cached_result(self, cache_key: Optional[tuple]) -> Optional[Any]:
        """
        キャッシュから結果を取得

        Args:
            cache_key: キャッシュキー

        Returns:
            キャッシュされた結果、またはNone
        """
        if cache_key and cache_key in self._calculation_cache:
            return self._calculation_cache[cache_key]
        return None

    def cache_result(self, cache_key: Optional[tuple], result: Any) -> None:
        """
        結果をキャッシュに保存

        Args:
            cache_key: キャッシュキー
            result: キャッシュする結果
        """
        if result is not None and cache_key:
            self._calculation_cache[cache_key] = result
