"""
汎用レジストリ基底クラス

辞書ベースの登録・検索・削除という共通ロジックを提供します。
スレッドセーフ（RLock）で、各ドメインのレジストリはこの基底を継承します。
"""

import threading
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class Registry(Generic[K, V]):
    """
    キー・バリュー形式の汎用レジストリ基底

    内部に ``dict`` と ``RLock`` を持ち、登録・検索・削除・クリアの
    共通操作をスレッドセーフに提供します。
    """

    def __init__(self) -> None:
        self._items: dict[K, V] = {}
        self._lock = threading.RLock()

    def set(self, key: K, item: V) -> None:
        """
        キーを指定してアイテムを登録する

        各サブクラスの ``register`` はキー導出ロジックが異なるため、
        このプリミティブに委譲します。

        Args:
            key: 登録キー
            item: 登録するアイテム
        """
        with self._lock:
            self._items[key] = item

    def get(self, key: K) -> V | None:
        """
        キーに対応するアイテムを取得する

        Args:
            key: 検索キー

        Returns:
            Optional[V]: アイテム、見つからない場合は None
        """
        with self._lock:
            return self._items.get(key)

    def get_all(self) -> list[V]:
        """
        登録済みの全アイテムをリストで返す

        Returns:
            List[V]: アイテムのリスト
        """
        with self._lock:
            return list(self._items.values())

    def remove(self, key: K, expected: V | None = None) -> None:
        """
        キーに対応するアイテムを削除する

        expected が指定された場合は、そのアイテムと同一オブジェクトのときのみ削除します。

        Args:
            key: 削除キー
            expected: 削除対象として期待するアイテム（オプション）
        """
        with self._lock:
            current = self._items.get(key)
            if current is None:
                return
            if expected is None or current is expected:
                self._items.pop(key, None)

    def clear(self) -> None:
        """登録をすべてクリアする。"""
        with self._lock:
            self._items.clear()
