"""collect_all_ls_ratio.py のシンボル解決テスト."""

import asyncio

import pytest

from scripts.data_ingestion.collect_all_ls_ratio import get_linear_symbols


def _run(coro):
    return asyncio.run(coro)


class TestGetLinearSymbols:
    """LSR収集対象シンボルの解決テスト"""

    def test_default_is_btc_only(self):
        assert _run(get_linear_symbols()) == ["BTC/USDT:USDT"]

    def test_single_symbol(self):
        assert _run(get_linear_symbols("ETH/USDT:USDT")) == ["ETH/USDT:USDT"]

    def test_comma_separated_multiple_symbols(self):
        result = _run(get_linear_symbols("BTC/USDT:USDT, ETH/USDT:USDT"))
        assert result == ["BTC/USDT:USDT", "ETH/USDT:USDT"]

    def test_unknown_symbol_raises_value_error(self):
        with pytest.raises(ValueError, match="未対応のシンボル"):
            _run(get_linear_symbols("PEPE/USDT:USDT"))

    def test_mixed_known_and_unknown_raises(self):
        with pytest.raises(ValueError, match="未対応のシンボル"):
            _run(get_linear_symbols("BTC/USDT:USDT,PEPE/USDT:USDT"))


class TestResolveTimeframes:
    """update_all_timeframes.resolve_timeframes のテスト"""

    def test_none_returns_all_supported(self):
        from scripts.data_ingestion.update_all_timeframes import resolve_timeframes

        assert resolve_timeframes(None, ["1h", "4h", "1d"]) == ["1h", "4h", "1d"]
        assert resolve_timeframes([], ["1h"]) == ["1h"]

    def test_valid_subset_is_returned_as_list(self):
        from scripts.data_ingestion.update_all_timeframes import resolve_timeframes

        assert resolve_timeframes(["4h"], ["1h", "4h", "1d"]) == ["4h"]

    def test_unknown_timeframe_raises_value_error(self):
        import pytest as _pytest

        from scripts.data_ingestion.update_all_timeframes import resolve_timeframes

        with _pytest.raises(ValueError, match="未対応の時間足"):
            resolve_timeframes(["2h"], ["1h", "4h"])
