"""日時のパースと正規化を扱う共通ユーティリティ。"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from numbers import Real
from typing import Any, Optional, Tuple

import pandas as pd


def _is_missing_value(value: Any) -> bool:
    """
    pandas の missing 値判定を安全に行う。

    pd.isna() を使用して値が欠損値かどうかを判定します。
    型エラーが発生した場合は False を返します。

    Args:
        value: 判定対象の値（任意の型）

    Returns:
        bool: 値が欠損値（NaN、Noneなど）の場合はTrue、そうでない場合はFalse
    """
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def parse_datetime_value(value: Any) -> datetime:
    """
    datetime や ISO8601 文字列を datetime に変換する。

    pandas.Timestamp、datetime、ISO8601形式の文字列を
    Pythonのdatetimeオブジェクトに変換します。

    Args:
        value: 変換対象の値（pd.Timestamp、datetime、ISO8601文字列）

    Returns:
        datetime: 変換されたdatetimeオブジェクト

    Raises:
        ValueError: 変換できない形式の値が渡された場合、または値がNone/欠損値の場合
    """
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
    """
    値を datetime に変換し、失敗時は None を返す。

    parse_datetime_value の例外をキャッチする安全なバージョンです。
    変換に失敗した場合や値がNone/空文字の場合はNoneを返します。

    Args:
        value: 変換対象の値（任意の型）

    Returns:
        Optional[datetime]: 変換されたdatetimeオブジェクト、失敗時はNone
    """
    if value is None:
        return None

    if isinstance(value, str) and not value.strip():
        return None

    try:
        return parse_datetime_value(value)
    except Exception:
        return None


def parse_datetime_range_optional(
    start_date: Any,
    end_date: Any,
) -> Optional[Tuple[datetime, datetime]]:
    """
    開始・終了日時をまとめて変換し、比較可能な形で返す。

    両方の日時を変換し、タイムゾーンの正規化を行った上で
    開始日時 < 終了日時であることを確認します。

    Args:
        start_date: 開始日時（datetime、ISO8601文字列など）
        end_date: 終了日時（datetime、ISO8601文字列など）

    Returns:
        Optional[Tuple[datetime, datetime]]: 正規化された(開始日時, 終了日時)のタプル。
                                             変換失敗時または開始>=終了の場合はNone
    """
    parsed_start = parse_datetime_optional(start_date)
    parsed_end = parse_datetime_optional(end_date)

    if parsed_start is None or parsed_end is None:
        return None

    normalized_start, normalized_end = normalize_datetimes_for_comparison(
        parsed_start,
        parsed_end,
    )
    if normalized_start >= normalized_end:
        return None

    return normalized_start, normalized_end


def parse_timestamp_safe(value: Any) -> Optional[datetime]:
    """
    ミリ秒 timestamp や日時表現を安全に datetime へ変換する。

    以下の形式をサポートします：
    - pandas.Timestamp
    - datetime
    - ISO8601形式の文字列
    - ミリ秒単位の数値（UNIXタイムスタンプ）

    不正な値（None、NaN、負の値など）はNoneを返します。

    Args:
        value: 変換対象の値（任意の型）

    Returns:
        Optional[datetime]: 変換されたdatetimeオブジェクト（UTCタイムゾーン）、失敗時はNone
    """
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
    """
    aware / naive が混在しても比較できるように正規化する。

    タイムゾーン情報（aware）とタイムゾーン情報なし（naive）の
    datetimeオブジェクトが混在している場合、全てUTCのawareに変換して
    比較可能な状態にします。

    Args:
        start_date: 開始日時
        end_date: 終了日時

    Returns:
        Tuple[datetime, datetime]: UTCタイムゾーンに正規化された(開始日時, 終了日時)のタプル
    """
    if start_date.tzinfo is None and end_date.tzinfo is None:
        return start_date, end_date

    def _to_utc_aware(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    return _to_utc_aware(start_date), _to_utc_aware(end_date)


def current_datetime_like(reference: datetime) -> datetime:
    """
    reference の timezone に合わせた現在時刻を返す。

    参照用のdatetimeオブジェクトと同じタイムゾーン設定で
    現在時刻を取得します。referenceがnaiveの場合はnaiveな現在時刻を返します。

    Args:
        reference: タイムゾーン設定の参照元datetimeオブジェクト

    Returns:
        datetime: referenceと同じタイムゾーン設定の現在時刻
    """
    if reference.tzinfo is None:
        return datetime.now()
    return datetime.now(reference.tzinfo)


def ensure_utc_timezone(dt: datetime | None) -> datetime | None:
    """
    datetimeオブジェクトのタイムゾーンがUTCであることを保証する。

    naive（タイムゾーン情報なし）の場合はUTCを設定し、
    他のタイムゾーンがある場合はUTCに変換します。

    Args:
        dt: 変換対象のdatetimeオブジェクト（Noneの場合はNoneを返す）

    Returns:
        datetime | None: UTCタイムゾーンに正規化されたdatetimeオブジェクト
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
