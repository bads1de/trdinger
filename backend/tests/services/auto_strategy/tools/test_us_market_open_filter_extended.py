"""
USMarketOpenFilter の拡張テスト

既存テスト (``test_us_market_open_filter.py``) がカバーしていない分岐を検証します:
- ``enabled=False`` 時の早期 return
- ``context.timestamp is None`` 時の早期 return
- ``to_timezone_minutes`` 失敗時のサマータイムフォールバック
- ``mutate_params`` の window 変化なしケース
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from app.services.auto_strategy.tools.base import ToolContext
from app.services.auto_strategy.tools.us_market_open_filter import (
    USMarketOpenFilter,
)


def _make_context(timestamp=None) -> ToolContext:
    if timestamp is not None:
        return ToolContext(timestamp=timestamp)
    return ToolContext()


class TestUSMarketOpenFilterExtended:
    """``USMarketOpenFilter`` の追加テスト"""

    def test_returns_false_when_disabled(self) -> None:
        f = USMarketOpenFilter()
        ctx = _make_context(pd.Timestamp("2023-01-04 14:30:00", tz="UTC"))
        # enabled=False なら対象時間でもスキップしない
        assert f.should_skip_entry(ctx, {"enabled": False}) is False

    def test_returns_false_when_no_timestamp(self) -> None:
        f = USMarketOpenFilter()
        ctx = _make_context()
        assert f.should_skip_entry(ctx, {"enabled": True}) is False

    def test_summer_fallback_when_conversion_fails_in_window(self) -> None:
        """夏時間フォールバック: target=13:30 ± window 内でスキップ"""
        f = USMarketOpenFilter()
        # 7月、夏時間 → ターゲット 13:30
        ctx = _make_context(pd.Timestamp("2023-07-04 13:30:00"))

        with patch(
            "app.services.auto_strategy.tools.us_market_open_filter.to_timezone_minutes",
            side_effect=RuntimeError("convert fail"),
        ):
            # hour=13, min=30 → ターゲットと一致
            assert (
                f.should_skip_entry(ctx, {"enabled": True, "window_minutes": 30})
                is True
            )

    def test_summer_fallback_when_conversion_fails_out_of_window(self) -> None:
        """夏時間フォールバック: window 外ならスキップしない"""
        f = USMarketOpenFilter()
        # 7月、hour=20 → ターゲット 13:30 ± 30 分外
        ctx = _make_context(pd.Timestamp("2023-07-04 20:00:00"))

        with patch(
            "app.services.auto_strategy.tools.us_market_open_filter.to_timezone_minutes",
            side_effect=RuntimeError("convert fail"),
        ):
            assert (
                f.should_skip_entry(ctx, {"enabled": True, "window_minutes": 30})
                is False
            )

    def test_winter_fallback_when_conversion_fails_in_window(self) -> None:
        """冬時間フォールバック: target=14:30 ± window 内でスキップ"""
        f = USMarketOpenFilter()
        # 1月、冬時間 → ターゲット 14:30
        ctx = _make_context(pd.Timestamp("2023-01-04 14:30:00"))

        with patch(
            "app.services.auto_strategy.tools.us_market_open_filter.to_timezone_minutes",
            side_effect=RuntimeError("convert fail"),
        ):
            # hour=14, min=30 → ターゲットと一致
            assert (
                f.should_skip_entry(ctx, {"enabled": True, "window_minutes": 30})
                is True
            )

    def test_mutate_params_skips_window_change(self) -> None:
        """random >= 0.2 のとき window は変異しない"""
        f = USMarketOpenFilter()
        with patch(
            "app.services.auto_strategy.tools.us_market_open_filter.random.random",
            return_value=0.5,  # 20% 未満なのでスキップ
        ):
            params = {"enabled": True, "window_minutes": 30}
            new_params = f.mutate_params(params)
            # window は元のまま
            assert new_params["window_minutes"] == 30

    def test_mutate_params_with_no_window(self) -> None:
        """window_minutes がない場合のデフォルト"""
        f = USMarketOpenFilter()
        with patch(
            "app.services.auto_strategy.tools.us_market_open_filter.random.random",
            return_value=0.5,
        ):
            new_params = f.mutate_params({"enabled": True})
            # window キーは追加されない
            assert "window_minutes" not in new_params
