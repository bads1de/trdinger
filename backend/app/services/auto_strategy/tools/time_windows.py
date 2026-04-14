"""
時間帯フィルターで共通利用するヘルパー。
"""

from __future__ import annotations

import random
from typing import Iterable, Optional

import pandas as pd


def normalize_timestamp(
    timestamp: Optional[pd.Timestamp],
    *,
    assume_timezone: str = "UTC",
) -> Optional[pd.Timestamp]:
    """naive timestamp を指定タイムゾーンとして正規化する。"""
    if timestamp is None:
        return None

    normalized = timestamp
    if normalized.tz is None:
        normalized = normalized.tz_localize(assume_timezone)
    return normalized


def to_timezone_minutes(
    timestamp: Optional[pd.Timestamp],
    timezone_name: str,
    *,
    assume_timezone: str = "UTC",
) -> Optional[int]:
    """指定タイムゾーンでの分単位時刻を返す。"""
    normalized = normalize_timestamp(timestamp, assume_timezone=assume_timezone)
    if normalized is None:
        return None
    localized = normalized.tz_convert(timezone_name)
    return localized.hour * 60 + localized.minute


def to_utc_minutes(
    timestamp: Optional[pd.Timestamp],
    *,
    assume_timezone: str = "UTC",
) -> Optional[int]:
    """UTC での分単位時刻を返す。"""
    return to_timezone_minutes(timestamp, "UTC", assume_timezone=assume_timezone)


def is_within_window(
    current_minutes: Optional[int],
    target_minutes: int,
    window_minutes: int,
) -> bool:
    """分単位時刻が target の前後 window に入るか判定する。"""
    if current_minutes is None:
        return False
    return abs(current_minutes - int(target_minutes)) <= int(window_minutes)


def is_within_any_window(
    current_minutes: Optional[int],
    targets: Iterable[int],
    window_minutes: int,
) -> bool:
    """複数ターゲットのいずれかの前後 window に入るか判定する。"""
    if current_minutes is None:
        return False
    return any(
        is_within_window(current_minutes, target_minutes, window_minutes)
        for target_minutes in targets
    )


def mutate_window_minutes(
    params: dict,
    *,
    key: str = "window_minutes",
    default: int,
    minimum: int,
    maximum: int,
    delta_low: int,
    delta_high: int,
) -> dict:
    """window_minutes 系パラメータの突然変異を適用する。"""
    current = int(params.get(key, default))
    delta = random.randint(delta_low, delta_high)
    params[key] = max(minimum, min(maximum, current + delta))
    return params


def is_summer_time_by_month(timestamp: pd.Timestamp) -> bool:
    """
    月ベースの簡易夏時間判定（UTC timestamp用）。

    3月〜11月を夏時間（DST）として扱います。
    タイムゾーン変換に失敗した際のフォールバック判定に使用します。

    Args:
        timestamp: UTCのタイムスタンプ。

    Returns:
        夏時間期間内であればTrue。
    """
    return 3 <= timestamp.month <= 11
