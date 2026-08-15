"""IndividualEvaluator 用のバックテストデータ取得ヘルパー。"""

from __future__ import annotations

import logging
import threading
from typing import Any, cast

import pandas as pd

from app.types import SerializableValue
from app.utils.datetime_utils import parse_datetime_range_optional

from .time_alignment import align_timestamp_to_index

logger = logging.getLogger(__name__)


class BacktestDataProvider:
    """キャッシュ付きのバックテスト用データ取得を担当する。"""

    def __init__(
        self,
        backtest_service: Any,
        data_cache: Any,
        lock: Any,
    ) -> None:
        self.backtest_service = backtest_service
        self._data_cache = data_cache
        self._lock = lock or threading.RLock()

    @staticmethod
    def _normalize_cache_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
        """キャッシュキーを正規化して一貫性を保つ。"""
        if len(key) >= 4:
            # 日時範囲を正規化
            parsed_range = parse_datetime_range_optional(key[-2], key[-1])
            if parsed_range is not None:
                normalized_start = str(pd.Timestamp(parsed_range[0]))
                normalized_end = str(pd.Timestamp(parsed_range[1]))
                return key[:-2] + (normalized_start, normalized_end)
        return key

    @staticmethod
    def _parse_key_date_range(
        key: tuple[Any, ...],
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """キャッシュキー末尾の日時範囲を pandas.Timestamp に正規化する。"""
        if len(key) < 2:
            return None

        parsed_range = parse_datetime_range_optional(key[-2], key[-1])
        if parsed_range is None:
            return None

        return cast(pd.Timestamp, pd.Timestamp(parsed_range[0])), cast(
            pd.Timestamp, pd.Timestamp(parsed_range[1])
        )

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
        worker_key: object,
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
        return bool(worker_start <= expected_start <= expected_end <= worker_end)

    def _find_encompassing_cached_data(
        self, key: tuple[Any, ...]
    ) -> pd.DataFrame | None:
        """ローカルキャッシュから要求期間を内包するデータを探してスライスする。

        WFA のフォールド評価はフォールドごとに異なる期間を要求するが、
        事前に全期間データがキャッシュされていれば（GA ワーカー用 prefetch など）、
        DB アクセスなしでスライスして返せる。キャッシュエントリ数は
        高々 100 程度なので全走査でも十分軽い。
        """
        if len(key) < 4:
            return None
        expected_range = self._parse_key_date_range(key)
        if expected_range is None:
            return None
        expected_start, expected_end = expected_range

        try:
            cache_items = list(self._data_cache.items())
        except Exception:
            return None

        for cached_key, cached_data in cache_items:
            if not isinstance(cached_data, pd.DataFrame) or cached_data.empty:
                continue
            if not self._is_compatible_worker_key(cached_key, key):
                continue
            try:
                cached_index = pd.DatetimeIndex(cached_data.index)
                if len(cached_index) == 0:
                    continue
                slice_start = align_timestamp_to_index(expected_start, cached_index)
                slice_end = align_timestamp_to_index(expected_end, cached_index)
                sliced = cached_data.loc[
                    (cached_data.index >= slice_start)
                    & (cached_data.index <= slice_end)
                ]
            except Exception:
                continue
            if not sliced.empty:
                return sliced
        return None

    def _find_extendable_cached_data(
        self, key: tuple[Any, ...]
    ) -> tuple[pd.DataFrame, pd.Timestamp] | None:
        """終端はカバーするが開始が不足するキャッシュデータを探す。

        warmup シフトにより要求開始がキャッシュ開始より過去にずれた場合、
        キャッシュデータの終端が要求終端をカバーしていれば、
        不足分（要求開始〜キャッシュ開始）だけを DB から取得して
        連結すればよい。キャッシュ側のスライスと不足開始点を返す。
        """
        if len(key) < 4:
            return None
        expected_range = self._parse_key_date_range(key)
        if expected_range is None:
            return None
        expected_start, expected_end = expected_range

        try:
            cache_items = list(self._data_cache.items())
        except Exception:
            return None

        best: tuple[pd.DataFrame, pd.Timestamp] | None = None
        for cached_key, cached_data in cache_items:
            if not isinstance(cached_data, pd.DataFrame) or cached_data.empty:
                continue
            if not isinstance(cached_key, tuple) or len(cached_key) != len(key):
                continue
            # シンボル・タイムフレーム等のプレフィックスが一致するものだけ対象。
            # 内包は要求しない（warmup シフトで開始が不足するケースを扱うため）。
            if cached_key[:-2] != key[:-2]:
                continue
            cached_range = self._parse_key_date_range(cached_key)
            if cached_range is None:
                continue
            cached_start, cached_end = cached_range
            if not (cached_end >= expected_end and cached_start > expected_start):
                continue
            try:
                cached_index = pd.DatetimeIndex(cached_data.index)
                if len(cached_index) == 0:
                    continue
                slice_end = align_timestamp_to_index(expected_end, cached_index)
                tail = cached_data.loc[cached_data.index <= slice_end]
                if tail.empty:
                    continue
                actual_start = pd.Timestamp(cached_data.index[0])
            except Exception:
                continue
            # 最も開始が早い（＝不足が少ない）エントリを優先
            if best is None or actual_start < best[1]:
                best = (tail, actual_start)
        return best

    @staticmethod
    def _ensure_utc(value: Any) -> pd.Timestamp:
        """naive なら UTC を付与し、aware なら UTC へ変換する。"""
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return cast(pd.Timestamp, ts.tz_localize("UTC"))
        return cast(pd.Timestamp, ts.tz_convert("UTC"))

    def _combine_with_gap(
        self,
        tail: pd.DataFrame,
        missing: pd.DataFrame,
        actual_start: pd.Timestamp,
    ) -> pd.DataFrame:
        """不足分（要求開始〜キャッシュ開始）を DB 取得結果で連結する。"""
        if missing is None or getattr(missing, "empty", True):
            return tail
        if tail is None or getattr(tail, "empty", True):
            return missing
        try:
            aligned_start = align_timestamp_to_index(actual_start, missing.index)
        except Exception:
            aligned_start = actual_start
        gap_part = missing.loc[missing.index < aligned_start]
        if gap_part.empty:
            return tail
        combined = pd.concat([gap_part, tail])
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined.sort_index()

    def get_cached_backtest_data(
        self, backtest_config: dict[str, SerializableValue]
    ) -> pd.DataFrame | None:
        """メイン時間軸のバックテストデータをキャッシュ付きで取得する。"""
        symbol = backtest_config.get("symbol")
        timeframe = backtest_config.get("timeframe")
        start_date = backtest_config.get("start_date")
        end_date = backtest_config.get("end_date")
        key = self._normalize_cache_key(
            (symbol, timeframe, str(start_date), str(end_date))
        )

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

            # 内包スライスを優先し、開始不足なら終端一致の拡張を試みる
            cached = self._find_encompassing_cached_data(key)
            if cached is not None:
                return cached

            extendable = self._find_extendable_cached_data(key)
            if extendable is not None:
                tail, actual_start = extendable
            else:
                tail, actual_start = None, None

        # DB アクセスは BacktestService の共有セッションを使うため、
        # 並列スレッドから同時に実行すると SQLite の
        # "concurrent operations are not permitted" が発生する。
        # キャッシュヒットしない場合のみロックを保持して直列化する。
        with self._lock:
            if key in self._data_cache:
                return self._data_cache[key]

            self.backtest_service.ensure_data_service_initialized()
            start_dt = self._ensure_utc(start_date)
            end_dt = self._ensure_utc(end_date)
            if tail is not None and actual_start is not None:
                # 不足分（要求開始〜キャッシュ開始）だけを DB から取得して連結
                missing = self.backtest_service.data_service.get_data_for_backtest(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_dt,
                    end_date=actual_start,
                )
                data = self._combine_with_gap(tail, missing, actual_start)
            else:
                data = self.backtest_service.data_service.get_data_for_backtest(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_dt,
                    end_date=end_dt,
                )
            self._data_cache[key] = data
        logger.debug(f"バックテストデータをキャッシュしました: {key}")
        return data

    def get_cached_minute_data(
        self, backtest_config: dict[str, SerializableValue]
    ) -> pd.DataFrame | None:
        """1分足データをキャッシュ付きで取得する。"""
        symbol = backtest_config.get("symbol")
        start_date = backtest_config.get("start_date")
        end_date = backtest_config.get("end_date")
        key = self._normalize_cache_key(
            ("minute", symbol, "1m", str(start_date), str(end_date))
        )

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

            # 要求期間を内包するキャッシュ済みデータがあればスライスして返す
            cached = self._find_encompassing_cached_data(key)
            if cached is not None:
                return cached

            extendable = self._find_extendable_cached_data(key)
            if extendable is not None:
                tail, actual_start = extendable
            else:
                tail, actual_start = None, None

        try:
            with self._lock:
                if key in self._data_cache:
                    return self._data_cache[key]
                self.backtest_service.ensure_data_service_initialized()
                if tail is not None and actual_start is not None:
                    # 不足分（要求開始〜キャッシュ開始）だけを DB から取得して連結
                    missing = self.backtest_service.data_service.get_data_for_backtest(
                        symbol=symbol,
                        timeframe="1m",
                        start_date=pd.to_datetime(cast(Any, start_date)),
                        end_date=actual_start,
                    )
                    data = self._combine_with_gap(tail, missing, actual_start)
                else:
                    data = self.backtest_service.data_service.get_data_for_backtest(
                        symbol=symbol,
                        timeframe="1m",
                        start_date=pd.to_datetime(cast(Any, start_date)),
                        end_date=pd.to_datetime(cast(Any, end_date)),
                    )
                if not data.empty:
                    self._data_cache[key] = data
                    logger.debug(f"1分足データをキャッシュしました: {key}")
                    return data
                logger.debug(f"1分足データが空です: {key}")
                return None
        except Exception as e:
            logger.warning(f"1分足データ取得エラー: {e}")
            return None
