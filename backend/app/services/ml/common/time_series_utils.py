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
    """DatetimeIndexから時間足を推定する"""
    if len(index) < 2:
        return "1h"
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return "1h"

    sec = diffs.mode().iloc[0].total_seconds()
    mapping = {900: "15m", 1800: "30m", 3600: "1h", 14400: "4h", 86400: "1d"}
    if sec in mapping:
        return mapping[sec]

    # 近似
    if sec >= 3600 and (sec / 3600).is_integer():
        return f"{int(sec / 3600)}h"
    if sec >= 60 and (sec / 60).is_integer():
        return f"{int(sec / 60)}m"
    return "1h"


def get_t1_series(
    indices: pd.DatetimeIndex, horizon_n: int, timeframe: Optional[str] = None
) -> pd.Series:
    """PurgedKFold用のt1（ラベル終了時刻）シリーズを計算"""
    tf = timeframe or infer_timeframe(indices)
    
    if tf.endswith("m"):
        delta = pd.Timedelta(minutes=int(tf[:-1]) * horizon_n)
    elif tf.endswith("h"):
        delta = pd.Timedelta(hours=int(tf[:-1]) * horizon_n)
    elif tf.endswith("d"):
        delta = pd.Timedelta(days=int(tf[:-1]) * horizon_n)
    else:
        logger.warning(f"Unknown tf format: {tf}, default to 1h")
        delta = pd.Timedelta(hours=horizon_n)

    return pd.Series(indices + delta, index=indices)



