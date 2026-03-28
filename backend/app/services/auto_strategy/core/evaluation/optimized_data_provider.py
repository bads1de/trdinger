"""
最適化されたバックテストデータプロバイダー

パフォーマンス最適化版のデータ取得を提供します。
"""

import logging
import threading
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

logger = logging.getLogger(__name__)


class OptimizedBacktestDataProvider:
    """
    最適化されたバックテストデータプロバイダー

    主な最適化ポイント:
    1. 読み取り専用ロックの使用（ReadWriteLock）
    2. データプリフェッチ
    3. キャッシュの効率化
    4. 並列データ読み込み
    """

    def __init__(
        self,
        backtest_service,
        data_cache: Dict,
        lock: threading.Lock,
        prefetch_enabled: bool = True,
        max_prefetch_workers: int = 2,
    ):
        self.backtest_service = backtest_service
        self._data_cache = data_cache
        self._lock = lock

        # 読み取りロック（ReadWriteLockの簡易実装）
        self._read_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._readers = 0

        # プリフェッチ設定
        self._prefetch_enabled = prefetch_enabled
        self._prefetch_executor = ThreadPoolExecutor(max_workers=max_prefetch_workers)
        self._prefetch_cache: Dict[str, Any] = {}

    def _acquire_read_lock(self):
        """読み取りロックを取得"""
        with self._write_lock:
            self._readers += 1
            if self._readers == 1:
                self._read_lock.acquire()

    def _release_read_lock(self):
        """読み取りロックを解放"""
        with self._write_lock:
            self._readers -= 1
            if self._readers == 0:
                self._read_lock.release()

    def _acquire_write_lock(self):
        """書き込みロックを取得"""
        self._write_lock.acquire()
        self._read_lock.acquire()

    def _release_write_lock(self):
        """書き込みロックを解放"""
        self._read_lock.release()
        self._write_lock.release()

    def get_cached_backtest_data(self, backtest_config: Dict[str, Any]) -> Any:
        """
        メイン時間軸のバックテストデータをキャッシュ付きで取得（最適化版）。

        最適化:
        - 読み取り専用ロックの使用
        - キャッシュヒットの高速化
        - ワーカーデータの優先チェック
        """
        try:
            from .parallel_evaluator import get_worker_data

            worker_data = get_worker_data("main_data")
            if worker_data is not None:
                return worker_data
        except ImportError:
            pass

        symbol = backtest_config.get("symbol")
        timeframe = backtest_config.get("timeframe")
        start_date = backtest_config.get("start_date")
        end_date = backtest_config.get("end_date")
        key = (symbol, timeframe, str(start_date), str(end_date))

        # 読み取りロックでキャッシュチェック
        self._acquire_read_lock()
        try:
            if key in self._data_cache:
                return self._data_cache[key]
        finally:
            self._release_read_lock()

        # キャッシュミス - 書き込みロックでデータ取得
        self._acquire_write_lock()
        try:
            # 二重チェック（他のスレッドが既にキャッシュした可能性）
            if key in self._data_cache:
                return self._data_cache[key]

            self.backtest_service.ensure_data_service_initialized()
            data = self.backtest_service.data_service.get_data_for_backtest(
                symbol=symbol,
                timeframe=timeframe,
                start_date=pd.to_datetime(start_date),
                end_date=pd.to_datetime(end_date),
            )
            self._data_cache[key] = data
            logger.debug(f"バックテストデータをキャッシュしました: {key}")

            return data
        finally:
            self._release_write_lock()

    def get_cached_minute_data(self, backtest_config: Dict[str, Any]) -> Any:
        """
        1分足データをキャッシュ付きで取得（最適化版）。

        最適化:
        - 読み取り専用ロックの使用
        - キャッシュヒットの高速化
        """
        try:
            from .parallel_evaluator import get_worker_data

            worker_data = get_worker_data("minute_data")
            if worker_data is not None:
                return worker_data
        except ImportError:
            pass

        symbol = backtest_config.get("symbol")
        start_date = backtest_config.get("start_date")
        end_date = backtest_config.get("end_date")
        key = ("minute", symbol, "1m", str(start_date), str(end_date))

        # 読み取りロックでキャッシュチェック
        self._acquire_read_lock()
        try:
            if key in self._data_cache:
                return self._data_cache[key]
        finally:
            self._release_read_lock()

        # キャッシュミス - 書き込みロックでデータ取得
        self._acquire_write_lock()
        try:
            # 二重チェック
            if key in self._data_cache:
                return self._data_cache[key]

            try:
                self.backtest_service.ensure_data_service_initialized()
                data = self.backtest_service.data_service.get_data_for_backtest(
                    symbol=symbol,
                    timeframe="1m",
                    start_date=pd.to_datetime(start_date),
                    end_date=pd.to_datetime(end_date),
                )
                if not data.empty:
                    self._data_cache[key] = data
                    logger.debug(f"1分足データをキャッシュしました: {key}")
                    return data
                else:
                    logger.debug(f"1分足データが空です: {key}")
                    return None
            except Exception as e:
                logger.warning(f"1分足データ取得エラー: {e}")
                return None
        finally:
            self._release_write_lock()

    def get_cached_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Any,
        end_date: Any,
        cache_prefix: str = "ohlcv",
    ) -> Any:
        """
        OHLCV データを汎用キャッシュ経由で取得（最適化版）。

        最適化:
        - 読み取り専用ロックの使用
        - キャッシュヒットの高速化
        - プリフェッチデータの活用
        """
        if not all([symbol, timeframe, start_date, end_date]):
            logger.warning(
                "OHLCVデータ取得: 必須パラメータが不足しています "
                f"(symbol={symbol}, timeframe={timeframe}, "
                f"start_date={start_date}, end_date={end_date})"
            )
            return None

        cache_key = (cache_prefix, symbol, timeframe, str(start_date), str(end_date))

        # 読み取りロックでキャッシュチェック
        self._acquire_read_lock()
        try:
            if cache_key in self._data_cache:
                cached_data = self._data_cache[cache_key]
                if hasattr(cached_data, "empty") and not cached_data.empty:
                    logger.debug(f"OHLCVデータ: キャッシュヒット (key={cache_key})")
                    return cached_data
        finally:
            self._release_read_lock()

        # プリフェッチデータをチェック
        if self._prefetch_enabled and cache_key in self._prefetch_cache:
            prefetch_data = self._prefetch_cache[cache_key]
            if prefetch_data is not None and hasattr(prefetch_data, "empty") and not prefetch_data.empty:
                # プリフェッチデータをキャッシュに移動
                self._acquire_write_lock()
                try:
                    self._data_cache[cache_key] = prefetch_data
                    del self._prefetch_cache[cache_key]
                finally:
                    self._release_write_lock()
                logger.debug(f"OHLCVデータ: プリフェッチヒット (key={cache_key})")
                return prefetch_data

        data_service = getattr(self.backtest_service, "data_service", None)
        if data_service is None:
            logger.warning("data_service が利用できません。")
            return None

        try:
            if hasattr(self.backtest_service, "ensure_data_service_initialized"):
                self.backtest_service.ensure_data_service_initialized()

            ohlcv_data = data_service.get_ohlcv_data(
                symbol, timeframe, start_date, end_date
            )

            if isinstance(ohlcv_data, pd.DataFrame) and not ohlcv_data.empty:
                self._acquire_write_lock()
                try:
                    self._data_cache[cache_key] = ohlcv_data
                finally:
                    self._release_write_lock()
                logger.debug(
                    f"OHLCVデータ: DBから取得・キャッシュ保存 (key={cache_key})"
                )
                return ohlcv_data

            logger.warning(
                f"OHLCVデータが空または無効です: symbol={symbol}, timeframe={timeframe}"
            )
            return None

        except Exception as exc:
            logger.warning(f"OHLCVデータ取得エラー: {exc}")
            return None

    def prefetch_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Any,
        end_date: Any,
        cache_prefix: str = "ohlcv",
    ):
        """
        データをプリフェッチ（バックグラウンド読み込み）。

        Args:
            symbol: シンボル
            timeframe: タイムフレーム
            start_date: 開始日
            end_date: 終了日
            cache_prefix: キャッシュプレフィックス
        """
        if not self._prefetch_enabled:
            return

        cache_key = (cache_prefix, symbol, timeframe, str(start_date), str(end_date))

        # 既にキャッシュにある場合はスキップ
        self._acquire_read_lock()
        try:
            if cache_key in self._data_cache:
                return
        finally:
            self._release_read_lock()

        # バックグラウンドでプリフェッチ
        def _prefetch_task():
            try:
                data_service = getattr(self.backtest_service, "data_service", None)
                if data_service is None:
                    return

                if hasattr(self.backtest_service, "ensure_data_service_initialized"):
                    self.backtest_service.ensure_data_service_initialized()

                ohlcv_data = data_service.get_ohlcv_data(
                    symbol, timeframe, start_date, end_date
                )

                if isinstance(ohlcv_data, pd.DataFrame) and not ohlcv_data.empty:
                    self._prefetch_cache[cache_key] = ohlcv_data
                    logger.debug(f"プリフェッチ完了: {cache_key}")
            except Exception as e:
                logger.debug(f"プリフェッチエラー: {e}")

        self._prefetch_executor.submit(_prefetch_task)

    def clear_cache(self):
        """キャッシュをクリア"""
        self._acquire_write_lock()
        try:
            self._data_cache.clear()
            self._prefetch_cache.clear()
        finally:
            self._release_write_lock()

    def get_cache_statistics(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        self._acquire_read_lock()
        try:
            return {
                "cache_size": len(self._data_cache),
                "prefetch_size": len(self._prefetch_cache),
                "readers": self._readers,
            }
        finally:
            self._release_read_lock()
