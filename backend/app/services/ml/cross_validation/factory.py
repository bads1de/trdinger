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
    """DatetimeIndexの時間間隔から時間足を自動推定する。

    インデックスの差分の最頻値を計算し、対応する時間足文字列を返します。
    標準的な時間足（15m, 30m, 1h, 4h, 1d）に加えて、
    カスタムな時間間隔も推定します。

    Args:
        index: 推定対象のDatetimeIndex。2つ以上の要素が必要。

    Returns:
        str: 推定された時間足文字列（例: "1h", "4h", "1d"）。
            推定できない場合はデフォルトで"1h"を返します。
    """
    if len(index) < 2:
        return "1h"
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return "1h"

    sec = int(diffs.mode().iloc[0].total_seconds())
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
    """PurgedKFold 用の t1（ラベリング終了時刻）を計算する。

    各インデックス時刻に horizon_n 本の時間幅を加算して、
    Triple Barrier Method などのラベリング終了時刻を算出します。

    Args:
        indices: 計算対象のDatetimeIndex。
        horizon_n: ラベリング期間（バーの本数）。
            例: horizon_n=4, timeframe="4h" の場合は16時間後。
        timeframe: 時間足文字列（例: "1h", "4h"）。
            指定しない場合はindicesから自動推定します。

    Returns:
        pd.Series: 各インデックスに対応するt1（ラベリング終了時刻）のSeries。
            インデックスは元のindicesと同じ。
    """
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
    """時系列向けのCV splitterを一元生成する。

    指定された戦略に応じて、適切なクロスバリデーション分割器を
    生成して返します。時系列データの特性を考慮した分割が可能です。

    対応戦略:
        - "kfold": 標準的なK-Fold（シャッフルなし）
        - "stratified_kfold": 層化K-Fold（シャッフルなし）
        - "purged_kfold": Purged K-Fold（時系列特有のデータリーク防止）

    Args:
        cv_strategy: 分割戦略（"kfold", "stratified_kfold", "purged_kfold"）。
        n_splits: 分割数。
        index: 時系列インデックス。
        t1: 各サンプルのラベリング終了時刻（PurgedKFOLD用）。
            指定しない場合はhorizon_nとtimeframeから自動計算します。
        pct_embargo: Embargo期間の割合（PurgedKFOLD用）。デフォルト: 0.01（1%）。
        horizon_n: ラベリング期間（バーの本数）。t1未指定時に使用。
        timeframe: 時間足文字列。t1未指定時に使用。

    Returns:
        BaseCrossValidator: 指定された戦略のCV splitterインスタンス。
            KFold、StratifiedKFold、またはPurgedKFold。

    Raises:
        ValueError: サポートされていないcv_strategyが指定された場合、
            またはt1未指定時にhorizon_nが指定されていない場合。
    """
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
