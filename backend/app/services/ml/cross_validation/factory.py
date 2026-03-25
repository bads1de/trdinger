"""
CV スプリッター ファクトリー

時系列向けのクロスバリデーション スプリッターを生成するためのユーティリティ関数を提供します。
"""

import logging
from typing import Optional

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from .purged_kfold import PurgedKFold

logger = logging.getLogger(__name__)


def infer_timeframe(index: pd.DatetimeIndex) -> str:
    """DatetimeIndex の時間間隔から時間足を自動推定"""
    if len(index) < 2:
        return "1h"
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return "1h"

    sec = diffs.mode().iloc[0].total_seconds()
    mapping = {900: "15m", 1800: "30m", 3600: "1h", 14400: "4h", 86400: "1d"}
    if sec in mapping:
        return mapping[sec]

    if sec >= 3600 and (sec / 3600).is_integer():
        return f"{int(sec / 3600)}h"
    if sec >= 60 and (sec / 60).is_integer():
        return f"{int(sec / 60)}m"
    return "1h"


def get_t1_series(
    indices: pd.DatetimeIndex, horizon_n: int, timeframe: Optional[str] = None
) -> pd.Series:
    """PurgedKFold 用の t1（ラベリング終了時刻）を計算"""
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


def create_temporal_cv_splitter(
    cv_strategy: str,
    n_splits: int,
    index: pd.DatetimeIndex,
    *,
    t1: Optional[pd.Series] = None,
    pct_embargo: float = 0.01,
    horizon_n: Optional[int] = None,
    timeframe: Optional[str] = None,
):
    """時系列向けの CV splitter を一元生成する"""
    strategy = (cv_strategy or "purged_kfold").lower()

    if strategy == "kfold":
        return KFold(n_splits=n_splits, shuffle=False)

    if strategy == "stratified_kfold":
        return StratifiedKFold(n_splits=n_splits, shuffle=False)

    if strategy != "purged_kfold":
        raise ValueError(f"Unsupported cv_strategy: {cv_strategy}")

    if t1 is None:
        if horizon_n is None:
            raise ValueError("horizon_n is required when t1 is not provided")
        t1 = get_t1_series(index, horizon_n, timeframe=timeframe)

    return PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)
