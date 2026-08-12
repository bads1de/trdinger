"""
バックテスト共通ユーティリティ

設定バリデーション、統計変換、結果変換で共通する処理を集約します。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import pandas as pd

# 以下の datetime_utils の再エクスポートは backtest 系モジュールから参照されるため意図的
# （F401 対策として各行に noqa を付与）
from app.utils.data_processing.data_validator import OHLCV_COLUMNS
from app.utils.datetime_utils import (
    current_datetime_like,  # noqa: F401
    normalize_datetimes_for_comparison,  # noqa: F401
    parse_datetime_value,  # noqa: F401
)
from app.utils.datetime_utils import (
    parse_timestamp_safe as safe_timestamp_conversion,  # noqa: F401
)

TRADE_PNL_COLUMNS: tuple[str, ...] = ("PnL", "Pnl", "Profit", "ProfitLoss")

# backtesting.lib.FractionalBacktest のデフォルト fractional_unit（= 1 satoshi 相当）
# バックテスト実行中、戦略に渡される価格データはこの係数でスケーリングされる
# （例: 42,735 USDT → 0.00042735）。ポジションサイズ計算は実価格（USDT）で行い、
# 注文発注時にこの係数でフレーム単位へ変換する必要がある。
FRACTIONAL_UNIT: float = 1e-8

logger = logging.getLogger(__name__)


def resolve_stats_object(
    stats: object, warning_logger: logging.Logger | None = None
) -> object:
    """
    statsオブジェクトの実体を取得する。callableなら呼び出す。

    backtesting.pyのstatsオブジェクトは遅延評価される場合があり、
    実際の統計情報を取得するために呼び出しが必要です。

    Args:
        stats: statsオブジェクト（callableまたはdict）
        warning_logger: 警告ログ出力用ロガー（オプション）

    Returns:
        Any: statsオブジェクトの実体（呼び出し可能な場合は呼び出し結果、そうでない場合はそのまま）
    """
    if callable(stats):
        try:
            return stats()
        except Exception as exc:
            if warning_logger is not None:
                warning_logger.warning(f"statsの呼び出しに失敗: {exc}")
            return stats
    return stats


def safe_float_conversion(value: Any) -> float:
    """
    安全にfloatへ変換する。

    None、NaN、変換不可能な値を0.0に変換します。

    Args:
        value: 変換対象の値（任意の型）

    Returns:
        float: 変換されたfloat値（失敗時は0.0）
    """
    if value is None:
        return 0.0

    try:
        if pd.isna(value):
            return 0.0
    except (TypeError, ValueError):
        pass

    try:
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"float変換に失敗しました: {value!r}")
        return 0.0


def safe_duration_conversion(value: Any) -> float:
    """
    期間値を日数ベースのfloatへ安全に変換する。

    pd.Timedelta / datetime.timedelta / "5 days" のような文字列を
    日数単位のfloatに正規化します。変換できない場合は0.0を返します。
    """
    if value is None:
        return 0.0

    try:
        if pd.isna(value):
            return 0.0
    except (TypeError, ValueError):
        pass

    if isinstance(value, (pd.Timedelta, timedelta)):
        return value.total_seconds() / 86400.0

    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            try:
                parsed = pd.to_timedelta(value)
            except (TypeError, ValueError, OverflowError):
                logger.warning(f"duration変換に失敗しました: {value!r}")
                return 0.0
            if pd.isna(parsed):
                logger.warning(f"duration変換に失敗しました: {value!r}")
                return 0.0
            return float(parsed.total_seconds()) / 86400.0

    try:
        return float(value)
    except (ValueError, TypeError):
        try:
            parsed = pd.to_timedelta(value)
        except (TypeError, ValueError, OverflowError):
            logger.warning(f"duration変換に失敗しました: {value!r}")
            return 0.0
        if pd.isna(parsed):
            logger.warning(f"duration変換に失敗しました: {value!r}")
            return 0.0
        return float(parsed.total_seconds()) / 86400.0


def safe_int_conversion(value: Any) -> int:
    """
    安全にintへ変換する。

    None、NaN、変換不可能な値を0に変換します。

    Args:
        value: 変換対象の値（任意の型）

    Returns:
        int: 変換されたint値（失敗時は0）
    """
    if value is None:
        return 0

    try:
        if pd.isna(value):
            return 0
    except (TypeError, ValueError):
        pass

    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def resolve_trade_pnl_column(
    trades_df: Any,
    preferred_columns: Sequence[str] = TRADE_PNL_COLUMNS,
) -> str | None:
    """
    取引データから損益列名を解決する。

    バックテストライブラリや戦略によって損益列の名前が異なるため、
    優先順位付きで列名を探索します。

    Args:
        trades_df: 取引データを含むDataFrame
        preferred_columns: 優先する列名のシーケンス（デフォルト: TRADE_PNL_COLUMNS）

    Returns:
        Optional[str]: 見つかった損益列名、見つからない場合はNone
    """
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
    """
    OHLCV列だけを正規化し、その他の列名はそのまま保持する。

    Open、High、Low、Close、Volume列の大文字小文字を統一します。
    lowercase=Trueの場合は小文字、Falseの場合は先頭大文字に変換します。

    Args:
        data_frame: 正規化対象のDataFrame
        lowercase: 小文字に変換するか（デフォルト: False）
        ensure_volume: Volume列がない場合に追加するか（デフォルト: False）
        volume_default: Volume列を追加する際のデフォルト値（デフォルト: 0.0）

    Returns:
        pd.DataFrame: OHLCV列が正規化されたDataFrame
    """
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
