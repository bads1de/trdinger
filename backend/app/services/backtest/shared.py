"""
バックテスト共通ユーティリティ

設定バリデーション、統計変換、結果変換で共通する処理を集約します。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Sequence, Tuple

import pandas as pd

from app.utils.datetime_utils import current_datetime_like as _current_datetime_like
from app.utils.datetime_utils import (
    normalize_datetimes_for_comparison as _normalize_datetimes_for_comparison,
)
from app.utils.datetime_utils import parse_datetime_value as _parse_datetime_value
from app.utils.datetime_utils import parse_timestamp_safe as _parse_timestamp_safe

TRADE_PNL_COLUMNS: tuple[str, ...] = ("PnL", "Pnl", "Profit", "ProfitLoss")
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


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


def resolve_trade_pnl_column(
    trades_df: Any,
    preferred_columns: Sequence[str] = TRADE_PNL_COLUMNS,
) -> Optional[str]:
    """取引データから損益列名を解決する。"""
    if trades_df is None or not hasattr(trades_df, "columns"):
        return None

    columns = getattr(trades_df, "columns", ())
    for column_name in preferred_columns:
        if column_name in columns:
            return column_name
    return None


def normalize_ohlcv_columns(
    data_frame: pd.DataFrame,
    *,
    lowercase: bool = False,
    ensure_volume: bool = False,
    volume_default: float = 0.0,
) -> pd.DataFrame:
    """OHLCV列だけを正規化し、その他の列名はそのまま保持する。"""
    if not isinstance(data_frame, pd.DataFrame):
        return data_frame

    rename_map: dict[str, str] = {}
    for column in data_frame.columns:
        if not isinstance(column, str):
            continue
        normalized_name = column.lower()
        if normalized_name not in OHLCV_COLUMNS:
            continue
        target_name = normalized_name if lowercase else normalized_name.capitalize()
        if column != target_name:
            rename_map[column] = target_name

    normalized = data_frame.rename(columns=rename_map) if rename_map else data_frame
    volume_column = "volume" if lowercase else "Volume"
    if ensure_volume and volume_column not in normalized.columns:
        normalized = normalized.copy()
        normalized[volume_column] = volume_default
    return normalized


def normalize_datetimes_for_comparison(
    start_date: datetime, end_date: datetime
) -> Tuple[datetime, datetime]:
    """互換ラッパー。"""
    return _normalize_datetimes_for_comparison(start_date, end_date)


def current_datetime_like(reference: datetime) -> datetime:
    """互換ラッパー。"""
    return _current_datetime_like(reference)
