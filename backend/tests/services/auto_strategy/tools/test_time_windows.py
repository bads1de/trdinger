"""
time_windows.py のテスト

app/services/auto_strategy/tools/time_windows.py のテストモジュール
"""

import pandas as pd
import pytest

from app.services.auto_strategy.tools.time_windows import (
    is_within_any_window,
    is_within_window,
    mutate_window_minutes,
    normalize_timestamp,
    to_timezone_minutes,
    to_utc_minutes,
)


class TestNormalizeTimestamp:
    """normalize_timestamp 関数のテスト"""

    def test_normalize_none(self):
        """Noneを渡した場合"""
        result = normalize_timestamp(None)
        assert result is None

    def test_normalize_naive_timestamp(self):
        """タイムゾーンなしのタイムスタンプ"""
        timestamp = pd.Timestamp("2023-01-01 12:00:00")
        result = normalize_timestamp(timestamp)
        assert result is not None
        assert result.tz is not None
        assert str(result.tz) == "UTC"

    def test_normalize_aware_timestamp(self):
        """タイムゾーン付きのタイムスタンプ"""
        timestamp = pd.Timestamp("2023-01-01 12:00:00", tz="UTC")
        result = normalize_timestamp(timestamp)
        assert result is not None
        assert str(result.tz) == "UTC"

    def test_normalize_with_custom_timezone(self):
        """カスタムタイムゾーンを指定"""
        timestamp = pd.Timestamp("2023-01-01 12:00:00")
        result = normalize_timestamp(timestamp, assume_timezone="Asia/Tokyo")
        assert result is not None
        assert str(result.tz) == "Asia/Tokyo"


class TestToTimezoneMinutes:
    """to_timezone_minutes 関数のテスト"""

    def test_to_timezone_minutes_none(self):
        """Noneを渡した場合"""
        result = to_timezone_minutes(None, "UTC")
        assert result is None

    def test_to_timezone_minutes_utc(self):
        """UTCでの分単位時刻"""
        timestamp = pd.Timestamp("2023-01-01 12:30:00", tz="UTC")
        result = to_timezone_minutes(timestamp, "UTC")
        assert result == 12 * 60 + 30  # 750分

    def test_to_timezone_minutes_different_timezone(self):
        """異なるタイムゾーンでの分単位時刻"""
        timestamp = pd.Timestamp("2023-01-01 12:30:00", tz="UTC")
        result = to_timezone_minutes(timestamp, "Asia/Tokyo")
        # UTC 12:30 は JST 21:30
        assert result == 21 * 60 + 30  # 1290分

    def test_to_timezone_minutes_naive_timestamp(self):
        """タイムゾーンなしのタイムスタンプ"""
        timestamp = pd.Timestamp("2023-01-01 12:30:00")
        result = to_timezone_minutes(timestamp, "UTC", assume_timezone="UTC")
        assert result == 12 * 60 + 30


class TestToUtcMinutes:
    """to_utc_minutes 関数のテスト"""

    def test_to_utc_minutes_none(self):
        """Noneを渡した場合"""
        result = to_utc_minutes(None)
        assert result is None

    def test_to_utc_minutes_from_utc(self):
        """UTCタイムスタンプからUTC分単位時刻"""
        timestamp = pd.Timestamp("2023-01-01 15:45:00", tz="UTC")
        result = to_utc_minutes(timestamp)
        assert result == 15 * 60 + 45  # 945分

    def test_to_utc_minutes_from_different_timezone(self):
        """異なるタイムゾーンからUTC分単位時刻"""
        timestamp = pd.Timestamp("2023-01-01 00:00:00", tz="Asia/Tokyo")
        # JST 00:00 は UTC 15:00（前日）
        result = to_utc_minutes(timestamp)
        assert result == 15 * 60  # 900分

    def test_to_utc_minutes_naive_timestamp(self):
        """タイムゾーンなしのタイムスタンプ"""
        timestamp = pd.Timestamp("2023-01-01 15:45:00")
        result = to_utc_minutes(timestamp, assume_timezone="UTC")
        assert result == 15 * 60 + 45


class TestIsWithinWindow:
    """is_within_window 関数のテスト"""

    def test_is_within_window_none_current(self):
        """current_minutesがNoneの場合"""
        result = is_within_window(None, 720, 30)
        assert result is False

    def test_is_within_window_exact_match(self):
        """正確に一致する場合"""
        result = is_within_window(720, 720, 30)
        assert result is True

    def test_is_within_window_within_range(self):
        """範囲内の場合"""
        result = is_within_window(730, 720, 30)
        assert result is True

    def test_is_within_window_at_boundary(self):
        """境界値の場合"""
        result = is_within_window(750, 720, 30)
        assert result is True
        result = is_within_window(751, 720, 30)
        assert result is False

    def test_is_within_window_outside_range(self):
        """範囲外の場合"""
        result = is_within_window(800, 720, 30)
        assert result is False

    def test_is_within_window_with_float_target(self):
        """floatターゲットの場合"""
        result = is_within_window(725, 720.5, 30)
        assert result is True


class TestIsWithinAnyWindow:
    """is_within_any_window 関数のテスト"""

    def test_is_within_any_window_none_current(self):
        """current_minutesがNoneの場合"""
        result = is_within_any_window(None, [720, 1080], 30)
        assert result is False

    def test_is_within_any_window_single_target(self):
        """単一ターゲット"""
        result = is_within_any_window(730, [720], 30)
        assert result is True

    def test_is_within_any_window_multiple_targets(self):
        """複数ターゲットのいずれかに一致"""
        result = is_within_any_window(1090, [720, 1080], 30)
        assert result is True

    def test_is_within_any_window_none_match(self):
        """どのターゲットにも一致しない"""
        result = is_within_any_window(900, [720, 1080], 30)
        assert result is False

    def test_is_within_any_window_empty_targets(self):
        """空のターゲットリスト"""
        result = is_within_any_window(720, [], 30)
        assert result is False

    def test_is_within_any_window_iterable_targets(self):
        """イテラブルなターゲット"""
        result = is_within_any_window(730, (720, 1080), 30)
        assert result is True


class TestMutateWindowMinutes:
    """mutate_window_minutes 関数のテスト"""

    def test_mutate_window_minutes_basic(self):
        """基本的な突然変異"""
        params = {"window_minutes": 30}
        result = mutate_window_minutes(
            params,
            key="window_minutes",
            default=30,
            minimum=10,
            maximum=120,
            delta_low=-10,
            delta_high=10,
        )
        assert "window_minutes" in result
        assert 10 <= result["window_minutes"] <= 120

    def test_mutate_window_minutes_missing_key(self):
        """キーが存在しない場合（デフォルト値使用）"""
        params = {}
        result = mutate_window_minutes(
            params,
            key="window_minutes",
            default=30,
            minimum=10,
            maximum=120,
            delta_low=-10,
            delta_high=10,
        )
        assert "window_minutes" in result
        assert 10 <= result["window_minutes"] <= 120

    def test_mutate_window_minutes_clamp_minimum(self):
        """最小値でクランプ"""
        params = {"window_minutes": 15}
        result = mutate_window_minutes(
            params,
            key="window_minutes",
            default=30,
            minimum=10,
            maximum=120,
            delta_low=-20,  # 大きく減らす
            delta_high=-5,
        )
        assert result["window_minutes"] >= 10

    def test_mutate_window_minutes_clamp_maximum(self):
        """最大値でクランプ"""
        params = {"window_minutes": 115}
        result = mutate_window_minutes(
            params,
            key="window_minutes",
            default=30,
            minimum=10,
            maximum=120,
            delta_low=5,
            delta_high=20,  # 大きく増やす
        )
        assert result["window_minutes"] <= 120

    def test_mutate_window_minutes_custom_key(self):
        """カスタムキーを使用"""
        params = {"custom_window": 45}
        result = mutate_window_minutes(
            params,
            key="custom_window",
            default=30,
            minimum=10,
            maximum=120,
            delta_low=-5,
            delta_high=5,
        )
        assert "custom_window" in result
        assert 10 <= result["custom_window"] <= 120

    def test_mutate_window_minutes_preserves_other_params(self):
        """他のパラメータを保持"""
        params = {"window_minutes": 30, "other_param": "value"}
        result = mutate_window_minutes(
            params,
            key="window_minutes",
            default=30,
            minimum=10,
            maximum=120,
            delta_low=-5,
            delta_high=5,
        )
        assert "other_param" in result
        assert result["other_param"] == "value"

    def test_mutate_window_minutes_zero_delta(self):
        """デルタが0の場合"""
        params = {"window_minutes": 30}
        result = mutate_window_minutes(
            params,
            key="window_minutes",
            default=30,
            minimum=10,
            maximum=120,
            delta_low=0,
            delta_high=0,
        )
        assert result["window_minutes"] == 30
