"""評価系で共有する Timestamp / DatetimeIndex の timezone 整合ヘルパー。"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _get_index_timezone(index: pd.Index) -> Any | None:
    """Index から timezone を取得する。"""
    if len(index) == 0:
        return None

    index_tz = getattr(index, "tz", None)
    if index_tz is not None:
        return index_tz

    try:
        first_value = pd.Timestamp(index[0])
    except Exception:
        return None
    return first_value.tzinfo


def align_timestamp_to_tz(value: Any, target_tz: Any | None) -> pd.Timestamp:
    """Timestamp を指定 timezone に揃える。"""
    timestamp = pd.Timestamp(value)

    if target_tz is None:
        if timestamp.tzinfo is not None:
            return timestamp.tz_localize(None)
        return timestamp

    if timestamp.tzinfo is None:
        return timestamp.tz_localize(target_tz)
    if timestamp.tzinfo != target_tz:
        return timestamp.tz_convert(target_tz)
    return timestamp


def align_timestamp_to_reference(value: Any, reference: Any) -> pd.Timestamp:
    """Timestamp を参照 Timestamp の timezone に揃える。"""
    reference_timestamp = pd.Timestamp(reference)
    return align_timestamp_to_tz(value, reference_timestamp.tzinfo)


def align_timestamp_to_index(value: Any, index: pd.Index) -> pd.Timestamp:
    """Timestamp を Index の timezone に揃える。"""
    return align_timestamp_to_tz(value, _get_index_timezone(index))
