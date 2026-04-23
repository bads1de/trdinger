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
    DataFrameの内容に基づいたハッシュキーを使用するため、
    in-place更新でも必ず再計算されます。
    """

    def __init__(self, maxsize: int = 10000):
        """
        IndicatorCacheManagerを初期化します。

        Args:
            maxsize: キャッシュの最大サイズ（デフォルト: 10000）
        """
        self._calculation_cache: LRUCache = LRUCache(maxsize=maxsize)
        self._cache_hits = 0
        self._cache_misses = 0

    def clear_cache(self) -> None:
        """
        計算キャッシュをクリアする

        すべてのキャッシュされた計算結果を削除します。
        """
        self._calculation_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Indicator calculation cache cleared.")

    def make_cache_key(
        self, indicator_type: str, params: Dict[str, Any], df: pd.DataFrame
    ) -> Optional[tuple]:
        """キャッシュキーを生成（データの内容に基づいた一意なキー）

        DataFrame はミュータブルなので、属性にハッシュを保持しません。
        in-place 更新でも必ず再計算されるよう、毎回内容からハッシュを作ります。
        キーにはインジケータータイプ、パラメータ、DataFrameのメタ情報が含まれます。

        Args:
            indicator_type: インジケータータイプ（例: 'sma', 'rsi'）
            params: インジケーターパラメータの辞書
            df: データフレーム（OHLCVデータ）

        Returns:
            Optional[tuple]: キャッシュキー（タプル）、失敗時はNone。
                ハッシュ衝突の確率は極めて低く（64ビットハッシュ）、
                実用上問題ないと考えられます。

        Note:
            - DataFrameの内容が完全に同一でも、インデックス型が異なる場合は
              別のキーとして扱われる可能性があります。
            - 巨大なDataFrameの場合、ハッシュ計算に時間がかかることがあります。
        """
        try:
            # パラメータをソートされた不変セットに変換
            cache_params = frozenset(
                sorted([(k, str(v)) for k, v in params.items()])
            )

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

        指定されたキーに対応するキャッシュされた計算結果を返します。

        Args:
            cache_key: キャッシュキー（make_cache_keyで生成）

        Returns:
            Optional[Any]: キャッシュされた結果、またはNone
        """
        if cache_key is None:
            return None
        result = self._calculation_cache.get(cache_key)
        if result is not None:
            self._cache_hits += 1
        else:
            self._cache_misses += 1
        return result

    def cache_result(self, cache_key: Optional[tuple], result: Any) -> None:
        """
        結果をキャッシュに保存

        計算結果をキャッシュに保存します。結果がNoneの場合は保存しません。

        Args:
            cache_key: キャッシュキー（make_cache_keyで生成）
            result: キャッシュする計算結果
        """
        if cache_key is not None and result is not None:
            self._calculation_cache[cache_key] = result

    def get_cache_statistics(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (
            self._cache_hits / total_requests if total_requests > 0 else 0.0
        )
        return {
            "cache_size": len(self._calculation_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
        }
