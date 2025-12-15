"""
時系列データ処理ユーティリティ

時系列データの処理に関する共通関数を提供します。
時間足の推定やPurgedKFold用のt1計算などを含みます。
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def infer_timeframe(index: pd.DatetimeIndex) -> str:
    """
    DatetimeIndexから時間足を推定する

    Args:
        index: 時系列インデックス

    Returns:
        推定された時間足文字列 (例: '1h', '4h', '1d', '15m')
        推定できない場合はデフォルトとして '1h' を返す
    """
    if len(index) < 2:
        return "1h"

    # インデックス間の差分を計算
    diffs = index.to_series().diff().dropna()

    # 最頻値を計算 (mode)
    if diffs.empty:
        return "1h"

    mode_diff = diffs.mode().iloc[0]

    # Timedeltaを文字列に変換
    seconds = mode_diff.total_seconds()

    if seconds == 900:  # 15 * 60
        return "15m"
    elif seconds == 1800:  # 30 * 60
        return "30m"
    elif seconds == 3600:  # 60 * 60
        return "1h"
    elif seconds == 14400:  # 4 * 60 * 60
        return "4h"
    elif seconds == 86400:  # 24 * 60 * 60
        return "1d"
    else:
        # その他の場合は時間単位で近似
        hours = seconds / 3600
        if hours >= 1 and hours.is_integer():
            return f"{int(hours)}h"

        minutes = seconds / 60
        if minutes >= 1 and minutes.is_integer():
            return f"{int(minutes)}m"

        return "1h"  # デフォルト


def get_t1_series(
    indices: pd.DatetimeIndex, horizon_n: int, timeframe: Optional[str] = None
) -> pd.Series:
    """
    PurgedKFold用のt1（ラベル終了時刻）シリーズを計算する

    Args:
        indices: 時系列インデックス
        horizon_n: 予測ホライゾン（バー数）
        timeframe: 時間足（指定がない場合はインデックスから推定）

    Returns:
        t1シリーズ（各バーの予測対象期間の終了時刻）
    """
    if timeframe is None:
        timeframe = infer_timeframe(indices)

    # 時間足文字列からTimedeltaを生成
    if timeframe.endswith("m"):
        minutes = int(timeframe[:-1])
        delta = pd.Timedelta(minutes=minutes * horizon_n)
    elif timeframe.endswith("h"):
        hours = int(timeframe[:-1])
        delta = pd.Timedelta(hours=hours * horizon_n)
    elif timeframe.endswith("d"):
        days = int(timeframe[:-1])
        delta = pd.Timedelta(days=days * horizon_n)
    else:
        # デフォルトは1時間と仮定
        logger.warning(f"不明な時間足形式: {timeframe}。1hとして扱います。")
        delta = pd.Timedelta(hours=horizon_n)

    t1 = pd.Series(indices + delta, index=indices)
    return t1


