"""
Advanced Feature Engineering for Crypto Assets
Implements "Strong Features" from the strategy document (features_strategy.pdf).
"""

import logging

import numpy as np
import pandas as pd

from ..utils import handle_pandas_ta_errors

logger = logging.getLogger(__name__)


class AdvancedFeatures:
    """
    Advanced features including Fractional Differencing, Liquidation Cascade, etc.
    """

    @staticmethod
    def get_weights_ffd(d: float, thres: float, lim: int) -> np.ndarray:
        """
        Calculate weights for Fractional Differentiation (Fixed Window).

        Args:
            d (float): Differencing order (e.g. 0.4)
            thres (float): Threshold for weight cutoff (e.g. 1e-5)
            lim (int): Maximum length of weights

        Returns:
            np.ndarray: Array of weights (reversed order: w[-1] is w_0)
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
        Fractional Differentiation (Fixed Window).
        Preserves memory while achieving stationarity.

        Args:
            series (pd.Series): Input price series (usually log-prices)
            d (float): Order of differencing (0 < d < 1)
            thres (float): Weight cutoff threshold
            window (int): Max window size for weights
        """
        if not isinstance(series, pd.Series):
            raise TypeError("series must be pandas Series")

        # 1. Compute weights
        w = AdvancedFeatures.get_weights_ffd(d, thres, window)
        width = len(w)
        w = w.flatten()

        # 2. Apply weights
        # Using a loop for safety and clarity, though vectorization is possible via striding
        series_filled = series.ffill()
        series_vals = series_filled.values

        result = np.full(len(series), np.nan)

        if len(series) >= width:
            for i in range(width - 1, len(series)):
                # Get window of data: X_{t-width+1} ... X_{t}
                # window_val needs to be matched with weights w which are [w_K, ..., w_0]
                # where w_0 corresponds to X_t, w_1 to X_{t-1}
                # get_weights_ffd returns [w_K, ..., w_0]
                # So simple dot product works if window_val is [X_{t-K}, ..., X_{t}]

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
        Liquidation Cascade Score

        Detects forced liquidations (longs puking or shorts covering) which often signal reversals.

        Formula: -1 * sign(Delta P) * Delta OI * Volume
        """
        # Calculate changes
        delta_p = close.diff()
        delta_oi = open_interest.diff()

        # Sign of price change (+1, 0, -1)
        sign_p = np.sign(delta_p)

        # Formula
        # Note: delta_oi is negative during liquidation
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
        Short Squeeze Probability Index

        Detects setup for short squeeze:
        - Negative Funding (Shorts paying Longs)
        - Increasing OI (Shorts adding)
        - Price holding above recent lows (Absorption)
        """
        # 1. I(FR < 0) * |FR|
        neg_fr_factor = np.where(funding_rate < 0, funding_rate.abs(), 0)

        # 2. Delta OI
        delta_oi = open_interest.diff()
        # We focus on INCREASING OI (Shorts adding)
        delta_oi_factor = delta_oi.clip(lower=0)

        # 3. Price - Low_n
        # "Price holding" - if Price is far above Low, pressure is released?
        # Or is it "Price - Low_n" is small?
        # PDF: (Price_t - Low_n)
        # If Price is AT Low_n, term is 0.
        # If Price is bouncing, term is positive.
        # The formula seems to imply: The HIGHER the price is above the low (while shorts are adding), the MORE PAIN for shorts?
        # Yes, if they shorted at Low and price moved up, they are underwater.

        low_n = low.rolling(window=lookback).min()
        price_location = close - low_n

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
        Trend Quality (VWAP/OI Divergence proxy)

        Correlation between Price Change and OI Change.
        Positive correlation = Healthy trend (New money driving price)
        Negative correlation = Weak trend (Liquidation/Covering driving price)
        """
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
        OI Weighted Funding Rate

        FR * OI
        Represents the total dollar value of funding being paid/received.
        """
        return funding_rate * open_interest

    @staticmethod
    @handle_pandas_ta_errors
    def leverage_ratio(
        open_interest: pd.Series,
        market_cap: pd.Series,
    ) -> pd.Series:
        """
        Leverage Ratio = Total OI / Market Cap

        Note: Market Cap data might not be available in OHLCV.
        If not available, this returns NaN or requires external data injection.
        """
        if market_cap is None or market_cap.empty:
            return pd.Series(
                np.full(len(open_interest), np.nan), index=open_interest.index
            )

        return open_interest / market_cap

    @staticmethod
    @handle_pandas_ta_errors
    def liquidity_efficiency(
        open_interest: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        Liquidity Efficiency (OI / Volume Ratio)

        High Ratio = High OI but low Volume (Positions stuck/illiquid).
        """
        # Avoid division by zero
        vol_clean = volume.replace(0, 1e-9)
        return open_interest / vol_clean
