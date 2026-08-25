"""市場レジーム検出器.

検証ハーネス（scripts/validate_pipeline.py）の研究と同じ定義を使用する:

    直近 lookback_days 日の B&H リターンが
      +bull_threshold_pp 超 -> bull
      -bull_threshold_pp 未満 -> bear
      その間               -> sideways

ウォークフォワード安全: 各時点のラベルはその時点までの終値のみで決まり、
未来データを参照しない（rolling のみで構成）。
"""

from __future__ import annotations

import pandas as pd

REGIME_BULL = "bull"
REGIME_SIDEWAYS = "sideways"
REGIME_BEAR = "bear"

BULL_THRESHOLD_PP = 20.0
DEFAULT_LOOKBACK_DAYS = 182
# ラベル計算に必要な最小ヒストリ（窓の9割）。欠損多い期間は None 扱い。
MIN_HISTORY_RATIO = 0.9


def classify_return(
    bh_return_pp: float, threshold_pp: float = BULL_THRESHOLD_PP
) -> str:
    """完了済み窓のB&Hリターン(pp)をレジームへ分類する。"""
    if bh_return_pp > threshold_pp:
        return REGIME_BULL
    if bh_return_pp < -threshold_pp:
        return REGIME_BEAR
    return REGIME_SIDEWAYS


class RegimeDetector:
    """終値系列からウォークフォワード安全なレジームラベル列を作る。"""

    def __init__(
        self,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        threshold_pp: float = BULL_THRESHOLD_PP,
        min_history_ratio: float = MIN_HISTORY_RATIO,
    ) -> None:
        if lookback_days < 2:
            raise ValueError("lookback_days は 2 以上 required")
        self.lookback_days = int(lookback_days)
        self.threshold_pp = float(threshold_pp)
        self.min_history_ratio = float(min_history_ratio)

    def detect_series(self, daily_closes: pd.Series) -> pd.Series:
        """日次終値（昇順 DatetimeIndex）から日次レジームラベル列を返す。

        履歴が窓の min_history_ratio に満たない区間は None を返す。
        """
        if not isinstance(daily_closes.index, pd.DatetimeIndex):
            raise TypeError("daily_closes の index は DatetimeIndex required")
        closes = daily_closes.sort_index().astype(float)
        window = self.lookback_days
        min_obs = max(2, int(window * self.min_history_ratio))
        # 欠損日の少ない連続データ前提だが、実観測行数も併せて要求する
        obs_count = closes.rolling(window, min_periods=min_obs).count()
        trailing_ret_pp = (closes / closes.shift(window) - 1.0) * 100.0
        labels = pd.Series(pd.NA, index=closes.index, dtype="object")
        valid = (obs_count >= min_obs) & trailing_ret_pp.notna()
        labels[valid] = trailing_ret_pp[valid].map(
            lambda v: classify_return(float(v), self.threshold_pp)
        )
        return labels

    def label_at(self, daily_closes: pd.Series, asof: pd.Timestamp) -> str | None:
        """asof 時点（その日を含む）でのレジームラベルを返す。判定不能なら None。"""
        labels = self.detect_series(daily_closes[daily_closes.index <= asof])
        if labels.empty:
            return None
        value = labels.iloc[-1]
        return None if pd.isna(value) else str(value)

    def monthly_labels(
        self, daily_closes: pd.Series, month_starts: list[pd.Timestamp]
    ) -> dict[pd.Timestamp, str | None]:
        """各月開始日の直前終値ベースのレジームラベルを返す（先読みなし）。

        month_start の「前日以前」のデータのみ使用する。
        """
        result: dict[pd.Timestamp, str | None] = {}
        sorted_closes = daily_closes.sort_index()
        for month_start in month_starts:
            prior = sorted_closes[sorted_closes.index < month_start]
            result[month_start] = self.label_at(
                prior, prior.index[-1] if len(prior) else month_start
            )
        return result


def forward_return_pp(
    closes: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> float | None:
    """[start, end] 区間の B&H リターン(pp)。データ不足時は None。"""
    c = closes.sort_index()
    s = c[c.index >= start]
    e = c[c.index <= end]
    if s.empty or e.empty:
        return None
    first, last = float(s.iloc[0]), float(e.iloc[-1])
    if first == 0:
        return None
    return (last / first - 1.0) * 100.0
