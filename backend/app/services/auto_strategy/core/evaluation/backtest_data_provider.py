"""IndividualEvaluator 用のバックテストデータ取得ヘルパー。"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Any, cast


import pandas as pd

from app.types import SerializableValue
from app.utils.datetime_utils import parse_datetime_range_optional

from .time_alignment import align_timestamp_to_index

logger = logging.getLogger(__name__)


class BacktestDataProvider:
    """キャッシュ付きのバックテスト用データ取得を担当する。"""

    def __init__(
        self,
        backtest_service,
        data_cache,
        lock,
        prefetch_enabled: bool = True,
        max_prefetch_workers: int = 2,
    ):
        self.backtest_service = backtest_service
        self._data_cache = data_cache
        self._lock = lock or threading.RLock()
        self._prefetch_enabled = prefetch_enabled
        self._prefetch_executor = ThreadPoolExecutor(max_workers=max_prefetch_workers)
        self._prefetch_cache: Dict[tuple, Any] = {}

    @staticmethod
    def _parse_key_date_range(
        key: tuple[Any, ...],
    ) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
        """キャッシュキー末尾の日時範囲を pandas.Timestamp に正規化する。"""
        if len(key) < 2:
            return None

        parsed_range = parse_datetime_range_optional(key[-2], key[-1])
        if parsed_range is None:
            return None

        return cast(pd.Timestamp, pd.Timestamp(parsed_range[0])), cast(pd.Timestamp, pd.Timestamp(parsed_range[1]))

    @staticmethod
    def _extract_worker_data(
        worker_payload: object, expected_key: tuple[object, ...]
    ) -> pd.DataFrame | None:
        """共有ワーカーデータが現在の要求期間と一致する場合のみ返す。"""
        if not isinstance(worker_payload, dict):
            return None
        worker_key = worker_payload.get("key")
        worker_data = worker_payload.get("data")
        if worker_key == expected_key:
            return worker_data if isinstance(worker_data, pd.DataFrame) else None
        if not BacktestDataProvider._is_compatible_worker_key(worker_key, expected_key):
            return None
        if not isinstance(worker_data, pd.DataFrame) or worker_data.empty:
            return None

        try:
            worker_index = pd.DatetimeIndex(worker_data.index)
            expected_range = BacktestDataProvider._parse_key_date_range(expected_key)
            if expected_range is None:
                return None

            expected_start, expected_end = expected_range
            if len(worker_index) > 0:
                expected_start = align_timestamp_to_index(expected_start, worker_index)
                expected_end = align_timestamp_to_index(expected_end, worker_index)
        except Exception as e:
            logger.debug(f"Workerデータのインデックス処理エラー: {e}")
            return None

        sliced = worker_data.loc[
            (worker_data.index >= expected_start) & (worker_data.index <= expected_end)
        ]
        return sliced if not sliced.empty else None

    @staticmethod
    def _is_compatible_worker_key(
        worker_key: Any,
        expected_key: tuple[Any, ...],
    ) -> bool:
        """共有データが要求期間を内包しているかを返す。"""
        if not isinstance(worker_key, tuple) or len(worker_key) != len(expected_key):
            return False
        if len(expected_key) < 4 or worker_key[:-2] != expected_key[:-2]:
            return False

        worker_range = BacktestDataProvider._parse_key_date_range(worker_key)
        expected_range = BacktestDataProvider._parse_key_date_range(expected_key)
        if worker_range is None or expected_range is None:
            return False

        worker_start, worker_end = worker_range
        expected_start, expected_end = expected_range
        return worker_start <= expected_start <= expected_end <= worker_end

    def get_cached_backtest_data(
        self, backtest_config: Dict[str, SerializableValue]
    ) -> pd.DataFrame | None:
        """メイン時間軸のバックテストデータをキャッシュ付きで取得する。"""
        symbol = backtest_config.get("symbol")
        timeframe = backtest_config.get("timeframe")
        start_date = backtest_config.get("start_date")
        end_date = backtest_config.get("end_date")
        key = (symbol, timeframe, str(start_date), str(end_date))

        try:
            from .parallel_evaluator import get_worker_data

            worker_data = self._extract_worker_data(get_worker_data("main_data"), key)
            if worker_data is not None:
                with self._lock:
                    self._data_cache[key] = worker_data
                return worker_data
        except ImportError:
            pass

        with self._lock:
            if key in self._data_cache:
                return self._data_cache[key]

        self.backtest_service.ensure_data_service_initialized()
        data = self.backtest_service.data_service.get_data_for_backtest(
            symbol=symbol,
            timeframe=timeframe,
            start_date=pd.to_datetime(start_date),  # type: ignore[reportArgumentType]
            end_date=pd.to_datetime(end_date),  # type: ignore[reportArgumentType]
        )
        with self._lock:
            if key in self._data_cache:
                return self._data_cache[key]
            self._data_cache[key] = data
        logger.debug(f"バックテストデータをキャッシュしました: {key}")
        return data

    def get_cached_minute_data(
        self, backtest_config: Dict[str, SerializableValue]
    ) -> pd.DataFrame | None:
        """1分足データをキャッシュ付きで取得する。"""
        symbol = backtest_config.get("symbol")
        start_date = backtest_config.get("start_date")
        end_date = backtest_config.get("end_date")
        key = ("minute", symbol, "1m", str(start_date), str(end_date))

        try:
            from .parallel_evaluator import get_worker_data

            worker_data = self._extract_worker_data(get_worker_data("minute_data"), key)
            if worker_data is not None:
                with self._lock:
                    self._data_cache[key] = worker_data
                return worker_data
        except ImportError:
            pass

        with self._lock:
            if key in self._data_cache:
                return self._data_cache[key]

        try:
            self.backtest_service.ensure_data_service_initialized()
            data = self.backtest_service.data_service.get_data_for_backtest(
                symbol=symbol,
                timeframe="1m",
                start_date=pd.to_datetime(cast(Any, start_date)),
                end_date=pd.to_datetime(cast(Any, end_date)),
            )
            if not data.empty:
                with self._lock:
                    if key in self._data_cache:
                        return self._data_cache[key]
                    self._data_cache[key] = data
                logger.debug(f"1分足データをキャッシュしました: {key}")
                return data
            logger.debug(f"1分足データが空です: {key}")
            return None
        except Exception as e:
            logger.warning(f"1分足データ取得エラー: {e}")
            return None

    def get_cached_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: object,
        end_date: object,
        cache_prefix: str = "ohlcv",
    ) -> pd.DataFrame | None:
        """OHLCV データを汎用キャッシュ経由で取得する。"""
        if not all([symbol, timeframe, start_date, end_date]):
            logger.warning(
                "OHLCVデータ取得: 必須パラメータが不足しています "
                f"(symbol={symbol}, timeframe={timeframe}, "
                f"start_date={start_date}, end_date={end_date})"
            )
            return None

        cache_key = (cache_prefix, symbol, timeframe, str(start_date), str(end_date))

        with self._lock:
            if cache_key in self._data_cache:
                cached_data = self._data_cache[cache_key]
                if hasattr(cached_data, "empty") and not cached_data.empty:
                    logger.debug(f"OHLCVデータ: キャッシュヒット (key={cache_key})")
                    return cached_data

        if self._prefetch_enabled:
            with self._lock:
                prefetch_data = self._prefetch_cache.get(cache_key)
                if (
                    prefetch_data is not None
                    and hasattr(prefetch_data, "empty")
                    and not prefetch_data.empty
                ):
                    self._data_cache[cache_key] = prefetch_data
                    self._prefetch_cache.pop(cache_key, None)
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
                with self._lock:
                    if cache_key not in self._data_cache:
                        self._data_cache[cache_key] = ohlcv_data
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
    ) -> None:
        """データをプリフェッチする。"""
        if not self._prefetch_enabled:
            return

        cache_key = (cache_prefix, symbol, timeframe, str(start_date), str(end_date))

        with self._lock:
            if cache_key in self._data_cache:
                return

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
                    with self._lock:
                        self._prefetch_cache[cache_key] = ohlcv_data
                    logger.debug(f"プリフェッチ完了: {cache_key}")
            except Exception as e:
                logger.debug(f"プリフェッチエラー: {e}")

        self._prefetch_executor.submit(_prefetch_task)

    def clear_cache(self) -> None:
        """キャッシュをクリアする。"""
        with self._lock:
            self._data_cache.clear()
            self._prefetch_cache.clear()

    def get_cache_statistics(self) -> Dict[str, Any]:
        """キャッシュ統計を取得する。"""
        with self._lock:
            return {
                "cache_size": len(self._data_cache),
                "prefetch_size": len(self._prefetch_cache),
                "readers": 0,
            }
