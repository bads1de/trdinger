"""
過学習メトリクス（PBO / DSR）

GA のように多数の戦略を探索する場合、単純な OOS 指標は
「試行回数分の偶然の最大値」を見誤る恐れがある。
ここでは多重検定を意識した以下の指標を計算する。

- PBO (Probability of Backtest Overfitting) 風の負けフォールド比率
  : 検証フォールドのうち損失を出した割合。過半数が負けなら過学習と疑う。
- DSR (Deflated Sharpe Ratio, Bailey & Lopez de Prado 2014)
  : 観測シャープレシオが、N 回の探索を経た帰無分布の最大値と
    統計的に区別できる確率。探索試行数で割り引いた有意性指標。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import NormalDist

import pandas as pd

_STANDARD_NORMAL = NormalDist()

DAYS_PER_YEAR = 365.0


def probability_of_backtest_overfitting(
    fold_returns: Iterable[float],
) -> float | None:
    """負けフォールドの比率（PBO の単純推定）を返す。

    全フォールドの総リターン（total_return）を入力とし、
    値が負のフォールドの割合を返す。空リストの場合は None。
    """
    returns = [float(value) for value in fold_returns]
    if not returns:
        return None
    return sum(1.0 for value in returns if value < 0.0) / len(returns)


def expected_max_sharpe_ratio(
    n_trials: int,
    sigma_sharpe: float = 1.0,
    skewness: float = 0.0,
) -> float:
    """帰無仮説下で N 回独立試行した際の最大期待シャープレシオ。

    Bailey & Lopez de Prado (2012) の式に基づく:

        E[maxSR_N] = sigma_SR * [ (1 - gamma3) * Phi^-1(1 - 1/N)
                                  + gamma3 * Phi^-1(1 - 1/(N*e)) ]

    Args:
        n_trials: 独立試行数（GA の個体数 x 世代数など）
        sigma_sharpe: 帰無分布での年率シャープレシオの標準偏差
        skewness: 帰無分布の歪度（通常 0 を仮定）

    Returns:
        最大期待シャープレシオ。試行数が 1 以下の場合は 0.0。
    """
    if n_trials < 2:
        return 0.0
    inv_survive = _STANDARD_NORMAL.inv_cdf(1.0 - 1.0 / float(n_trials))
    inv_survive_e = _STANDARD_NORMAL.inv_cdf(1.0 - 1.0 / (float(n_trials) * math.e))
    return float(sigma_sharpe) * (
        (1.0 - skewness) * inv_survive + skewness * inv_survive_e
    )


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_observations: float,
    n_trials: int,
    sigma_sharpe: float = 1.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)。

    観測シャープレシオが、N 回の探索を経た帰無分布の最大値と
    統計的に区別できる確率（0.0-1.0）を返す。

        DSR = Phi( (SR - E[maxSR_N]) * sqrt(N_obs - 1) / sigma_SR_hat )
        sigma_SR_hat = sqrt(1 - skew * SR + (kurt - 1)/4 * SR^2)

    Args:
        observed_sharpe: 観測された年率換算シャープレシオ
        n_observations: シャープレシオ推定に使われた観測期間（年）
        n_trials: 探索試行数（GA の個体数 x 世代数など）
        sigma_sharpe: 帰無分布での年率シャープレシオの標準偏差
        skewness: リターン分布の歪度（不明時は 0 を仮定）
        kurtosis: リターン分布の尖度（不明時は正規分布の 3 を仮定）

    Returns:
        DSR 確率。情報不足（観測期間・試行数が 1 未満など）の場合は 0.5。
    """
    if not math.isfinite(observed_sharpe):
        return 0.5
    if n_observations < 2.0 or n_trials < 2:
        return 0.5
    if sigma_sharpe <= 0.0:
        return 0.5

    expected_max_sr = expected_max_sharpe_ratio(n_trials, sigma_sharpe, skewness)
    sr_variance = (
        1.0 - skewness * observed_sharpe + (kurtosis - 1.0) / 4.0 * observed_sharpe**2
    )
    sr_std = math.sqrt(max(sr_variance, 1e-6))
    z_score = (
        (observed_sharpe - expected_max_sr) * math.sqrt(n_observations - 1.0) / sr_std
    )
    return _STANDARD_NORMAL.cdf(z_score)


def estimate_test_window_years(
    start_date: str,
    end_date: str,
    n_folds: int,
    train_ratio: float,
) -> float | None:
    """WFA の単一テスト窓の期間（年）を推定する。

    WFA では全体期間の (1 - train_ratio) / n_folds が各テスト窓の長さに
    なるため、そこから年数（= 年率シャープレシオの観測期間）を概算する。

    Args:
        start_date: バックテスト開始日
        end_date: バックテスト終了日
        n_folds: WFA フォールド数
        train_ratio: WFA トレーニング比率（0.0-1.0）

    Returns:
        テスト窓の年数。期間が不正な場合は None。
    """
    try:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
    except Exception:
        return None
    if pd.isna(start) or pd.isna(end) or end <= start:
        return None
    if n_folds <= 0 or not 0.0 < float(train_ratio) < 1.0:
        return None

    total_days = float((end - start).total_seconds() / 86400.0)
    test_ratio = 1.0 - float(train_ratio)
    fold_years = total_days * test_ratio / int(n_folds) / DAYS_PER_YEAR
    return max(fold_years, 0.0)
