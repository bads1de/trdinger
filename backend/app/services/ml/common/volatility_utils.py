"""
ボラティリティ計算ユーティリティ

価格ボラティリティの計算を統一的に行うためのユーティリティ関数群。
これまで複数箇所で重複していたボラティリティ計算ロジックを集約。
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_volatility_std(
    returns: pd.Series,
    window: int = 24,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    標準偏差ベースのボラティリティ計算

    リターンのローリング標準偏差を計算します。

    Args:
        returns: リターン系列（pct_change()の結果など）
        window: ローリングウィンドウサイズ（デフォルト: 24時間）
        min_periods: 最小期間数（デフォルト: windowと同じ）

    Returns:
        ボラティリティ系列（標準偏差）

    Examples:
        >>> returns = df['close'].pct_change()
        >>> volatility = calculate_volatility_std(returns, window=24)
    """
    if len(returns) == 0:
        return pd.Series([], dtype=float)

    if min_periods is None:
        min_periods = window

    volatility = returns.rolling(window=window, min_periods=min_periods).std()

    return volatility


def calculate_volatility_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    as_percentage: bool = False,
) -> pd.Series:
    """ATRベースのボラティリティ計算"""
    if len(high) == 0:
        return pd.Series([], dtype=float)

    # True Range = max(h-l, |h-pc|, |l-pc|)
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()

    return atr / close if as_percentage else atr


def calculate_historical_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    年率換算のヒストリカルボラティリティ計算

    Args:
        returns: リターン系列
        window: 計算期間
        annualize: 年率換算するか（デフォルト: True）
        periods_per_year: 年間の期間数（デフォルト: 252営業日）

    Returns:
        ヒストリカルボラティリティ系列

    Examples:
        >>> log_returns = np.log(df['close'] / df['close'].shift(1))
        >>> hist_vol = calculate_historical_volatility(log_returns, window=20)
    """
    vol = returns.rolling(window=window).std()

    if annualize:
        vol = vol * np.sqrt(periods_per_year)

    return vol


def calculate_realized_volatility(
    returns: pd.Series,
    window: int = 24,
    periods_per_day: int = 24,
) -> pd.Series:
    """
    実現ボラティリティ（Realized Volatility）計算

    高頻度データから日次ボラティリティを推定します。

    Args:
        returns: 高頻度リターン系列（例: 1時間ごと）
        window: 集計期間（例: 24時間 = 1日）
        periods_per_day: 1日あたりの期間数（例: 24時間足なら24）

    Returns:
        実現ボラティリティ系列

    Examples:
        >>> hourly_returns = df['close'].pct_change()
        >>> realized_vol = calculate_realized_volatility(hourly_returns, window=24)
    """
    vol = returns.rolling(window=window).std()

    # 日次換算
    vol = vol * np.sqrt(periods_per_day)

    return vol



