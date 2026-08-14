"""
BacktestDataProvider の拡張テスト

既存テスト (``test_backtest_data_provider.py``, ``test_backtest_data_provider_cache.py``) が
カバーしていない静的ヘルパー、``get_cached_minute_data`` の各分岐を検証します。
"""

from __future__ import annotations

import threading
from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd

from app.services.auto_strategy.core.evaluation.backtest_data_provider import (
    BacktestDataProvider,
)


def _make_provider(
    *,
    cache: dict | None = None,
) -> BacktestDataProvider:
    """シンプルな provider を構築するヘルパー"""
    service = Mock()
    service.ensure_data_service_initialized = Mock()
    service.data_service = Mock()
    return BacktestDataProvider(
        backtest_service=service,
        data_cache=cache if cache is not None else {},
        lock=threading.RLock(),
    )


class TestInit:
    """``__init__`` のテスト"""

    def test_default_settings(self) -> None:
        provider = _make_provider()
        assert isinstance(provider._data_cache, dict)

    def test_lock_defaults_to_rlock(self) -> None:
        provider = BacktestDataProvider(Mock(), {}, lock=None)
        # デフォルトで RLock が入る
        assert provider._lock is not None


class TestNormalizeCacheKey:
    """``_normalize_cache_key`` のテスト"""

    def test_short_key_unchanged(self) -> None:
        key = ("a", "b")
        assert BacktestDataProvider._normalize_cache_key(key) == ("a", "b")

    def test_long_key_with_dates_normalized(self) -> None:
        key = ("sym", "tf", "2024-01-01", "2024-01-02")
        normalized = BacktestDataProvider._normalize_cache_key(key)
        # 末尾 2 要素が Timestamp 文字列に
        assert normalized[0] == "sym"
        assert normalized[1] == "tf"
        assert "2024-01-01" in normalized[2]
        assert "2024-01-02" in normalized[3]

    def test_long_key_with_datetime_objects(self) -> None:
        key = ("sym", "tf", datetime(2024, 1, 1), datetime(2024, 1, 2))
        normalized = BacktestDataProvider._normalize_cache_key(key)
        assert normalized[0] == "sym"
        assert normalized[1] == "tf"
        assert isinstance(normalized[2], str)


class TestParseKeyDateRange:
    """``_parse_key_date_range`` のテスト"""

    def test_returns_none_for_short_key(self) -> None:
        assert BacktestDataProvider._parse_key_date_range(("a",)) is None
        assert BacktestDataProvider._parse_key_date_range(("a", "b")) is None

    def test_returns_none_for_unparseable_dates(self) -> None:
        assert (
            BacktestDataProvider._parse_key_date_range(
                ("a", "b", "not-a-date", "also-not")
            )
            is None
        )

    def test_returns_timestamps_for_valid_dates(self) -> None:
        result = BacktestDataProvider._parse_key_date_range(
            ("a", "b", "2024-01-01", "2024-01-02")
        )
        assert result is not None
        start, end = result
        assert isinstance(start, pd.Timestamp)
        assert isinstance(end, pd.Timestamp)
        assert start < end


class TestExtractWorkerData:
    """``_extract_worker_data`` のテスト"""

    def test_returns_none_when_payload_not_dict(self) -> None:
        result = BacktestDataProvider._extract_worker_data("not a dict", ("a", "b"))
        assert result is None

    def test_returns_data_when_keys_match(self) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        key = ("BTC", "1h", "2024-01-01", "2024-01-02")
        result = BacktestDataProvider._extract_worker_data(
            {"key": key, "data": df}, key
        )
        assert result is df

    def test_returns_none_when_data_not_dataframe(self) -> None:
        key = ("a",)
        result = BacktestDataProvider._extract_worker_data(
            {"key": key, "data": "not-a-df"}, key
        )
        assert result is None

    def test_returns_sliced_data_for_compatible_range(self) -> None:
        worker_df = pd.DataFrame(
            {"close": [1, 2, 3, 4, 5]},
            index=pd.date_range("2024-01-01", periods=5, freq="D"),
        )
        worker_key = ("BTC", "1h", "2024-01-01", "2024-01-05")
        expected_key = ("BTC", "1h", "2024-01-02", "2024-01-04")
        result = BacktestDataProvider._extract_worker_data(
            {"key": worker_key, "data": worker_df}, expected_key
        )
        assert result is not None
        assert len(result) == 3

    def test_returns_none_for_empty_sliced_data(self) -> None:
        worker_df = pd.DataFrame(
            {"close": [1, 2]},
            index=pd.date_range("2024-01-01", periods=2, freq="D"),
        )
        worker_key = ("BTC", "1h", "2024-01-01", "2024-01-02")
        expected_key = ("BTC", "1h", "2024-12-01", "2024-12-02")
        result = BacktestDataProvider._extract_worker_data(
            {"key": worker_key, "data": worker_df}, expected_key
        )
        # worker は 1月のみ、要求は 12月 → 空
        assert result is None

    def test_returns_none_when_worker_data_empty(self) -> None:
        empty_df = pd.DataFrame({"close": []})
        worker_key = ("BTC", "1h", "2024-01-01", "2024-01-05")
        expected_key = ("BTC", "1h", "2024-01-02", "2024-01-04")
        result = BacktestDataProvider._extract_worker_data(
            {"key": worker_key, "data": empty_df}, expected_key
        )
        assert result is None


class TestIsCompatibleWorkerKey:
    """``_is_compatible_worker_key`` のテスト"""

    def test_returns_false_when_keys_length_differ(self) -> None:
        result = BacktestDataProvider._is_compatible_worker_key(
            ("a", "b"), ("a", "b", "c")
        )
        assert result is False

    def test_returns_false_when_prefix_differs(self) -> None:
        worker = ("BTC", "1h", "2024-01-01", "2024-01-05")
        expected = ("ETH", "1h", "2024-01-02", "2024-01-04")
        assert BacktestDataProvider._is_compatible_worker_key(worker, expected) is False

    def test_returns_true_when_worker_envelops_expected(self) -> None:
        worker = ("BTC", "1h", "2024-01-01", "2024-01-10")
        expected = ("BTC", "1h", "2024-01-02", "2024-01-05")
        assert BacktestDataProvider._is_compatible_worker_key(worker, expected) is True

    def test_returns_false_when_worker_does_not_envelop(self) -> None:
        # worker の末尾が期待より早い
        worker = ("BTC", "1h", "2024-01-01", "2024-01-03")
        expected = ("BTC", "1h", "2024-01-02", "2024-01-05")
        assert BacktestDataProvider._is_compatible_worker_key(worker, expected) is False

    def test_returns_false_when_dates_unparseable(self) -> None:
        worker = ("BTC", "1h", "invalid", "invalid")
        expected = ("BTC", "1h", "invalid", "invalid")
        assert BacktestDataProvider._is_compatible_worker_key(worker, expected) is False

    def test_returns_false_when_worker_key_not_tuple(self) -> None:
        assert (
            BacktestDataProvider._is_compatible_worker_key("not a tuple", ("a",))
            is False
        )


class TestGetCachedBacktestData:
    """``get_cached_backtest_data`` の追加テスト"""

    def test_uses_worker_data_for_matching_key(self) -> None:
        provider = _make_provider()
        df = pd.DataFrame({"close": [1, 2, 3]})
        config = {
            "symbol": "BTC",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
        }
        key = (
            "BTC",
            "1h",
            str(pd.Timestamp("2024-01-01")),
            str(pd.Timestamp("2024-01-02")),
        )

        with patch(
            "app.services.auto_strategy.core.evaluation.parallel_evaluator.get_worker_data",
            return_value={"key": key, "data": df},
        ):
            result = provider.get_cached_backtest_data(config)

        assert result is df

    def test_handles_import_error_for_parallel_evaluator(self) -> None:
        """ImportError 時は worker data を使わず通常パスへ"""
        provider = _make_provider()
        df = pd.DataFrame({"close": [1, 2, 3]})
        provider.backtest_service.data_service.get_data_for_backtest.return_value = df

        config = {
            "symbol": "BTC",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
        }

        # get_worker_data が ImportError を投げるケース
        with patch(
            "app.services.auto_strategy.core.evaluation.parallel_evaluator.get_worker_data",
            side_effect=ImportError("no module"),
        ):
            result = provider.get_cached_backtest_data(config)

        assert result is df

    def test_tz_localize_conversion(self) -> None:
        """tzinfo なしの日付は UTC に localize される"""
        provider = _make_provider()
        df = pd.DataFrame({"close": [1, 2, 3]})
        provider.backtest_service.data_service.get_data_for_backtest.return_value = df

        with patch(
            "app.services.auto_strategy.core.evaluation.parallel_evaluator.get_worker_data",
            return_value=None,
        ):
            result = provider.get_cached_backtest_data(
                {
                    "symbol": "BTC",
                    "timeframe": "1h",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-02",
                }
            )

        assert result is df
        # get_data_for_backtest の呼び出し引数で tz_localize が効く
        call_kwargs = (
            provider.backtest_service.data_service.get_data_for_backtest.call_args[1]
        )
        assert call_kwargs["start_date"].tzinfo is not None


class TestGetCachedMinuteData:
    """``get_cached_minute_data`` のテスト"""

    def test_uses_cache_when_already_populated(self) -> None:
        provider = _make_provider()
        df = pd.DataFrame({"close": [1, 2, 3]})
        key = (
            "minute",
            "BTC",
            "1m",
            str(pd.Timestamp("2024-01-01")),
            str(pd.Timestamp("2024-01-02")),
        )
        provider._data_cache[key] = df

        result = provider.get_cached_minute_data(
            {
                "symbol": "BTC",
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
            }
        )

        assert result is df

    def test_returns_none_when_data_empty(self) -> None:
        provider = _make_provider()
        provider.backtest_service.data_service.get_data_for_backtest.return_value = (
            pd.DataFrame()
        )

        with patch(
            "app.services.auto_strategy.core.evaluation.parallel_evaluator.get_worker_data",
            return_value=None,
        ):
            result = provider.get_cached_minute_data(
                {
                    "symbol": "BTC",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-02",
                }
            )

        assert result is None

    def test_handles_exception(self) -> None:
        provider = _make_provider()
        provider.backtest_service.data_service.get_data_for_backtest.side_effect = (
            RuntimeError("api fail")
        )

        with patch(
            "app.services.auto_strategy.core.evaluation.parallel_evaluator.get_worker_data",
            return_value=None,
        ):
            result = provider.get_cached_minute_data(
                {
                    "symbol": "BTC",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-02",
                }
            )

        assert result is None

    def test_fetches_and_caches_data(self) -> None:
        provider = _make_provider()
        df = pd.DataFrame({"close": [1, 2, 3]})
        provider.backtest_service.data_service.get_data_for_backtest.return_value = df

        with patch(
            "app.services.auto_strategy.core.evaluation.parallel_evaluator.get_worker_data",
            return_value=None,
        ):
            result = provider.get_cached_minute_data(
                {
                    "symbol": "BTC",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-02",
                }
            )

        assert result is df
        # キャッシュに保存されている
        assert len(provider._data_cache) == 1
