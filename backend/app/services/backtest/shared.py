"""
バックテスト共通ユーティリティ

設定バリデーション、統計変換、結果変換で共通する処理を集約します。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Tuple

import pandas as pd

from app.utils.datetime_utils import current_datetime_like as _current_datetime_like
from app.utils.datetime_utils import (
    normalize_datetimes_for_comparison as _normalize_datetimes_for_comparison,
)
from app.utils.datetime_utils import parse_datetime_value as _parse_datetime_value
from app.utils.datetime_utils import parse_timestamp_safe as _parse_timestamp_safe


def resolve_stats_object(stats: Any, warning_logger: Any = None) -> Any:
    """statsオブジェクトの実体を取得する。callableなら呼び出す。"""
    if hasattr(stats, "__call__"):
        try:
            return stats()
        except Exception as exc:
            if warning_logger is not None:
                warning_logger.warning(f"statsの呼び出しに失敗: {exc}")
            return stats
    return stats


def safe_float_conversion(value: Any) -> float:
    """安全にfloatへ変換する。"""
    try:
        if value is None or pd.isna(value):
            return 0.0
    except (TypeError, ValueError):
        pass

    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def safe_int_conversion(value: Any) -> int:
    """安全にintへ変換する。"""
    try:
        if value is None or pd.isna(value):
            return 0
    except (TypeError, ValueError):
        pass

    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def parse_datetime_value(value: Any) -> datetime:
    """互換ラッパー。"""
    return _parse_datetime_value(value)


def safe_timestamp_conversion(value: Any) -> Optional[datetime]:
    """timestamp値を安全にdatetimeへ変換する。"""
    return _parse_timestamp_safe(value)


def normalize_datetimes_for_comparison(
    start_date: datetime, end_date: datetime
) -> Tuple[datetime, datetime]:
    """互換ラッパー。"""
    return _normalize_datetimes_for_comparison(start_date, end_date)


def current_datetime_like(reference: datetime) -> datetime:
    """互換ラッパー。"""
    return _current_datetime_like(reference)
