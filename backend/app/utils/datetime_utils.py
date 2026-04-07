"""日時のパースと正規化を扱う共通ユーティリティ。"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from numbers import Real
from typing import Any, Optional, Tuple

import pandas as pd


def _is_missing_value(value: Any) -> bool:
    """pandas の missing 値判定を安全に行う。"""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def parse_datetime_value(value: Any) -> datetime:
    """datetime や ISO8601 文字列を datetime に変換する。"""
    if value is None or _is_missing_value(value):
        raise ValueError(f"サポートされていない日付形式: {type(value)}")

    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return pd.to_datetime(value).to_pydatetime()

    raise ValueError(f"サポートされていない日付形式: {type(value)}")


def parse_datetime_optional(value: Any) -> Optional[datetime]:
    """値を datetime に変換し、失敗時は None を返す。"""
    if value is None:
        return None

    if isinstance(value, str) and not value.strip():
        return None

    try:
        return parse_datetime_value(value)
    except Exception:
        return None


def parse_timestamp_safe(value: Any) -> Optional[datetime]:
    """ミリ秒 timestamp や日時表現を安全に datetime へ変換する。"""
    if value is None or _is_missing_value(value):
        return None

    try:
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()

        if isinstance(value, datetime):
            return value

        if isinstance(value, str):
            return parse_datetime_optional(value)

        if isinstance(value, bool):
            return None

        if isinstance(value, Real):
            numeric_value = float(value)
            if math.isnan(numeric_value) or numeric_value < 0:
                return None
            return datetime.fromtimestamp(numeric_value / 1000.0, tz=timezone.utc)

        if pd.isna(value):
            return None
    except (TypeError, ValueError, OSError):
        return None

    return None


def normalize_datetimes_for_comparison(
    start_date: datetime, end_date: datetime
) -> Tuple[datetime, datetime]:
    """aware / naive が混在しても比較できるように正規化する。"""
    if start_date.tzinfo is None and end_date.tzinfo is None:
        return start_date, end_date

    def _to_utc_aware(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    return _to_utc_aware(start_date), _to_utc_aware(end_date)


def current_datetime_like(reference: datetime) -> datetime:
    """reference の timezone に合わせた現在時刻を返す。"""
    if reference.tzinfo is None:
        return datetime.now()
    return datetime.now(reference.tzinfo)
