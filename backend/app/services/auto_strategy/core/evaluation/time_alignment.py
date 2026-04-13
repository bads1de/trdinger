"""評価系で共有する Timestamp / DatetimeIndex の timezone 整合ヘルパー。"""

from __future__ import annotations

import logging
from typing import Any, cast
import pandas as pd

logger = logging.getLogger(__name__)


def _is_nat(value: Any) -> bool:
    """Check if a Timestamp is NaT."""
    return bool(pd.isna(value))


def _get_index_timezone(index: pd.Index) -> object | None:
    """
    Index から timezone を取得する

    Args:
        index: pandas Index

    Returns:
        timezone オブジェクトまたは None

    Note:
        Index が空の場合は None を返します。
        Index に timezone が設定されていない場合は None を返します。
    """
    if len(index) == 0:
        return None

    index_tz = getattr(index, "tz", None)
    if index_tz is not None:
        return index_tz

    try:
        first_value = pd.Timestamp(index[0])
    except Exception as e:
        logger.debug(f"Indexの先頭値からTimestampの生成に失敗しました: {e}")
        return None
    return first_value.tzinfo


def align_timestamp_to_tz(value: Any, target_tz: Any | None) -> pd.Timestamp:
    """
    Timestamp を指定 timezone に揃える

    Args:
        value: Timestamp または変換可能な値
        target_tz: 目標 timezone

    Returns:
        pd.Timestamp: 指定 timezone に揃えられた Timestamp

    Note:
        target_tz が None の場合は timezone を削除します。
        target_tz が存在し、value が timezone を持たない場合は local timezone で
        timezone を付与します。
    """
    timestamp = pd.Timestamp(value)

    # NaT check
    if _is_nat(timestamp):
        return cast(pd.Timestamp, timestamp)  # type: ignore[return-value]

    if target_tz is None:
        if timestamp.tzinfo is not None:
            return timestamp.tz_localize(None)  # type: ignore[return-value]
        return timestamp  # type: ignore[return-value]

    if timestamp.tzinfo is None:
        return timestamp.tz_localize(target_tz)  # type: ignore[return-value]
    if timestamp.tzinfo != target_tz:
        return timestamp.tz_convert(target_tz)  # type: ignore[return-value]
    return timestamp  # type: ignore[return-value]


def align_timestamp_to_reference(value: Any, reference: Any) -> pd.Timestamp:
    """
    Timestamp を参照 Timestamp の timezone に揃える

    Args:
        value: Timestamp または変換可能な値
        reference: 参照 Timestamp または変換可能な値

    Returns:
        pd.Timestamp: 参照 Timestamp の timezone に揃えられた Timestamp

    Note:
        参照値の timezone を使用して、値の timezone を変換します。
    """
    reference_timestamp = pd.Timestamp(reference)

    # NaT check
    if _is_nat(reference_timestamp):
        return align_timestamp_to_tz(value, None)

    return align_timestamp_to_tz(value, reference_timestamp.tzinfo)


def align_timestamp_to_index(value: Any, index: pd.Index) -> pd.Timestamp:
    """
    Timestamp を Index の timezone に揃える

    Args:
        value: Timestamp または変換可能な値
        index: pandas Index

    Returns:
        pd.Timestamp: Index の timezone に揃えられた Timestamp

    Note:
        Index の timezone を使用して、値の timezone を変換します。
    """
    return align_timestamp_to_tz(value, _get_index_timezone(index))
