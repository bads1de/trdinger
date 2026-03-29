"""
IndividualEvaluator 用のバックテストデータ取得ヘルパー。
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class BacktestDataProvider:
    """キャッシュ付きのバックテスト用データ取得を担当する。"""

    def __init__(self, backtest_service, data_cache, lock):
        self.backtest_service = backtest_service
        self._data_cache = data_cache
        self._lock = lock

    @staticmethod
    def _extract_worker_data(worker_payload: Any, expected_key: tuple[Any, ...]) -> Any:
        """共有ワーカーデータが現在の要求期間と一致する場合のみ返す。"""
        if not isinstance(worker_payload, dict):
            return None
        if worker_payload.get("key") != expected_key:
            return None
        return worker_payload.get("data")

    def get_cached_backtest_data(self, backtest_config: Dict[str, Any]) -> Any:
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
                return worker_data
        except ImportError:
            pass

        with self._lock:
            if key not in self._data_cache:
                import pandas as pd

                self.backtest_service.ensure_data_service_initialized()
                data = self.backtest_service.data_service.get_data_for_backtest(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=pd.to_datetime(start_date),
                    end_date=pd.to_datetime(end_date),
                )
                self._data_cache[key] = data
                logger.debug(f"バックテストデータをキャッシュしました: {key}")

            return self._data_cache[key]

    def get_cached_minute_data(self, backtest_config: Dict[str, Any]) -> Any:
        """1分足データをキャッシュ付きで取得する。"""
        symbol = backtest_config.get("symbol")
        start_date = backtest_config.get("start_date")
        end_date = backtest_config.get("end_date")
        key = ("minute", symbol, "1m", str(start_date), str(end_date))

        try:
            from .parallel_evaluator import get_worker_data

            worker_data = self._extract_worker_data(get_worker_data("minute_data"), key)
            if worker_data is not None:
                return worker_data
        except ImportError:
            pass

        with self._lock:
            if key not in self._data_cache:
                try:
                    import pandas as pd

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
                    else:
                        logger.debug(f"1分足データが空です: {key}")
                        return None
                except Exception as e:
                    logger.warning(f"1分足データ取得エラー: {e}")
                    return None

            return self._data_cache.get(key)

    def get_cached_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Any,
        end_date: Any,
        cache_prefix: str = "ohlcv",
    ) -> Any:
        """OHLCV データを汎用キャッシュ経由で取得する。"""
        import pandas as pd

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
                    self._data_cache[cache_key] = ohlcv_data
                logger.debug(
                    f"OHLCVデータ: DB から取得・キャッシュ保存 (key={cache_key})"
                )
                return ohlcv_data

            logger.warning(
                f"OHLCVデータが空または無効です: symbol={symbol}, timeframe={timeframe}"
            )
            return None

        except Exception as exc:
            logger.warning(f"OHLCVデータ取得エラー: {exc}")
            return None
