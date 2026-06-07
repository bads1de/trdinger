"""
USLunchFilter の拡張テスト

既存テスト (``test_us_lunch_filter.py``) がカバーしていない分岐を検証します:
- ``enabled=False`` 時の早期 return
- ``context.timestamp is None`` 時の早期 return
- ``to_timezone_minutes`` 失敗時のサマータイムフォールバック
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from app.services.auto_strategy.tools.base import ToolContext
from app.services.auto_strategy.tools.us_lunch_filter import USLunchFilter


def _make_context(timestamp=None) -> ToolContext:
    if timestamp is not None:
        return ToolContext(timestamp=timestamp)
    return ToolContext()


class TestUSLunchFilterExtended:
    """``USLunchFilter`` の追加テスト"""

    def test_returns_false_when_disabled(self) -> None:
        f = USLunchFilter()
        ctx = _make_context(pd.Timestamp("2023-01-04 17:30:00", tz="UTC"))
        # enabled=False ならランチタイムでもスキップしない
        assert f.should_skip_entry(ctx, {"enabled": False}) is False

    def test_returns_false_when_no_timestamp(self) -> None:
        f = USLunchFilter()
        ctx = _make_context()  # timestamp=None
        assert f.should_skip_entry(ctx, {"enabled": True}) is False

    def test_returns_false_when_enabled_key_missing(self) -> None:
        f = USLunchFilter()
        ctx = _make_context(pd.Timestamp("2023-01-04 17:30:00", tz="UTC"))
        # enabled キーがない → デフォルト True
        assert f.should_skip_entry(ctx, {}) is True

    def test_summer_fallback_when_conversion_fails_summer(self) -> None:
        """to_timezone_minutes 例外時、is_summer=True なら hour==16 で判定"""
        f = USLunchFilter()
        # 7月、夏時間
        ctx = _make_context(pd.Timestamp("2023-07-04 16:30:00"))

        with patch(
            "app.services.auto_strategy.tools.us_lunch_filter.to_timezone_minutes",
            side_effect=RuntimeError("convert fail"),
        ):
            # 夏時間で hour==16 → スキップ
            assert f.should_skip_entry(ctx, {"enabled": True}) is True

    def test_summer_fallback_when_conversion_fails_winter(self) -> None:
        """to_timezone_minutes 例外時、is_summer=False なら hour==17 で判定"""
        f = USLunchFilter()
        # 1月、冬時間
        ctx = _make_context(pd.Timestamp("2023-01-04 17:30:00"))

        with patch(
            "app.services.auto_strategy.tools.us_lunch_filter.to_timezone_minutes",
            side_effect=RuntimeError("convert fail"),
        ):
            # 冬時間で hour==17 → スキップ
            assert f.should_skip_entry(ctx, {"enabled": True}) is True

    def test_summer_fallback_skips_non_target_hour(self) -> None:
        """フォールバックで hour が対象外のときはスキップしない"""
        f = USLunchFilter()
        # 7月、hour=10 → 夏時間でも 16 ではない
        ctx = _make_context(pd.Timestamp("2023-07-04 10:30:00"))

        with patch(
            "app.services.auto_strategy.tools.us_lunch_filter.to_timezone_minutes",
            side_effect=RuntimeError("convert fail"),
        ):
            assert f.should_skip_entry(ctx, {"enabled": True}) is False

    def test_returns_none_minutes_does_not_skip(self) -> None:
        """to_timezone_minutes が None を返した場合はスキップしない"""
        f = USLunchFilter()
        ctx = _make_context(pd.Timestamp("2023-01-04 17:30:00", tz="UTC"))

        with patch(
            "app.services.auto_strategy.tools.us_lunch_filter.to_timezone_minutes",
            return_value=None,
        ):
            assert f.should_skip_entry(ctx, {"enabled": True}) is False
