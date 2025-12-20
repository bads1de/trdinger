"""
暗号資産のための高度な特徴量エンジニアリング
戦略ドキュメント (features_strategy.pdf) からの「強力な特徴量」を実装します。
"""

import logging

import numpy as np
import pandas as pd

from ..data_validation import (
    handle_pandas_ta_errors,
    validate_multi_series_params,
    validate_series_params,
)

logger = logging.getLogger(__name__)


class AdvancedFeatures:
    """
    分数差分、清算カスケードなどを含む高度な特徴量。
    """

    @staticmethod
    def get_weights_ffd(d: float, thres: float, lim: int) -> np.ndarray:
        """
        分数微分（固定ウィンドウ）の重みを計算します。
        """
        w, k = [1.0], 1
        while True:
            w_k = -w[-1] / k * (d - k + 1)
            if abs(w_k) < thres or len(w) >= lim:
                break
            w.append(w_k)
            k += 1
        return np.array(w[::-1]).reshape(-1, 1)

    @staticmethod
    @handle_pandas_ta_errors
    def frac_diff_ffd(
        series: pd.Series,
        d: float = 0.4,
        thres: float = 1e-5,
        window: int = 2000,
    ) -> pd.Series:
        """
        分数微分（固定ウィンドウFFD）。
        """
        validation = validate_series_params(series, window)
        if validation is not None:
            return validation

        # 1. 重みの計算
        w = AdvancedFeatures.get_weights_ffd(d, thres, window)
        width = len(w)
        w = w.flatten()

        # 2. 重みの適用
        series_filled = series.ffill()
        series_vals = series_filled.values

        result = np.full(len(series), np.nan)

        if len(series) >= width:
            for i in range(width - 1, len(series)):
                window_val = series_vals[i - width + 1 : i + 1]
                result[i] = np.dot(window_val, w)

        return pd.Series(result, index=series.index)

    @staticmethod
    @handle_pandas_ta_errors
    def liquidation_cascade_score(
        close: pd.Series,
        open_interest: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        清算カスケードスコア
        """
        validation = validate_multi_series_params(
            {"close": close, "open_interest": open_interest, "volume": volume}
        )
        if validation is not None:
            return validation

        # 変化量の計算
        delta_p = close.diff()
        delta_oi = open_interest.diff()

        # 価格変化の符号 (+1, 0, -1)
        sign_p = np.sign(delta_p)

        # 計算式
        score = -1 * sign_p * delta_oi * volume

        return score.fillna(0.0)

    @staticmethod
    @handle_pandas_ta_errors
    def squeeze_probability(
        close: pd.Series,
        funding_rate: pd.Series,
        open_interest: pd.Series,
        low: pd.Series,
        lookback: int = 20,
    ) -> pd.Series:
        """
        ショートスクイーズ確率指数
        """
        validation = validate_multi_series_params(
            {
                "close": close,
                "funding_rate": funding_rate,
                "open_interest": open_interest,
                "low": low,
            },
            lookback,
        )
        if validation is not None:
            return validation

        # 1. 負のファンディングレート要因: I(FR < 0) * |FR|
        neg_fr_factor = np.where(funding_rate < 0, funding_rate.abs(), 0)

        # 2. OI変化要因: Delta OI
        delta_oi = open_interest.diff()
        delta_oi_factor = delta_oi.clip(lower=0)

        # 3. 価格位置要因: Price - Low_n
        low_n = low.rolling(window=lookback).min()
        price_location = close - low_n

        # 総合スコア（確率プロキシ）
        prob = (
            pd.Series(neg_fr_factor, index=close.index)
            * delta_oi_factor
            * price_location
        )

        return prob.fillna(0.0)

    @staticmethod
    @handle_pandas_ta_errors
    def trend_quality(
        close: pd.Series,
        open_interest: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """
        トレンド品質（VWAP/OIダイバージェンスの代替指標）
        """
        validation = validate_multi_series_params(
            {"close": close, "open_interest": open_interest}, window
        )
        if validation is not None:
            return validation

        delta_p = close.diff()
        delta_oi = open_interest.diff()

        correlation = delta_p.rolling(window=window).corr(delta_oi)

        return correlation.fillna(0.0)

    @staticmethod
    @handle_pandas_ta_errors
    def oi_weighted_funding_rate(
        funding_rate: pd.Series,
        open_interest: pd.Series,
    ) -> pd.Series:
        """
        OI加重ファンディングレート
        """
        validation = validate_multi_series_params(
            {"funding_rate": funding_rate, "open_interest": open_interest}
        )
        if validation is not None:
            return validation

        return funding_rate * open_interest

    @staticmethod
    @handle_pandas_ta_errors
    def liquidity_efficiency(open_interest: pd.Series, volume: pd.Series) -> pd.Series:
        """
        流動性効率（Open Interest / Volume）
        """
        validation = validate_multi_series_params(
            {"open_interest": open_interest, "volume": volume}
        )
        if validation is not None:
            return validation

        return (open_interest / volume).replace([np.inf, -np.inf], 0).fillna(0)

    @staticmethod
    @handle_pandas_ta_errors
    def leverage_ratio(open_interest: pd.Series, market_cap: pd.Series) -> pd.Series:
        """
        推定レバレッジ比率（Total OI / Estimated Market Cap）
        """
        validation = validate_multi_series_params(
            {"open_interest": open_interest, "market_cap": market_cap}
        )
        if validation is not None:
            return validation

        return (open_interest / market_cap).replace([np.inf, -np.inf], 0).fillna(0)
