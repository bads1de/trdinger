"""
出来高系テクニカル指標

登録してあるテクニカルの一覧:
- OBV (On-Balance Volume)
- AD (A/Dライン)
- ADOSC (A/Dオシレーター)
- CMF (マネーフロー指標)
- EFI (Elder Force Index)
- MFI (Money Flow Index)
- VWAP (出来高加重平均価格)
- PVO (Percentage Volume Oscillator)
- PVT (Price Volume Trend)
- NVI (Negative Volume Index)
- EOM (Ease of Movement)
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..data_validation import (
    handle_pandas_ta_errors,
    validate_multi_series_params,
)

logger = logging.getLogger(__name__)


class VolumeIndicators:
    """
    出来高系指標クラス

    OBV, Chaikin A/Dラインなどの出来高系テクニカル指標を提供。
    出来高と価格の関係性分析に使用します。
    """

    @staticmethod
    @handle_pandas_ta_errors
    def ad(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """チャイキンA/Dライン"""
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close, "volume": volume}
        )
        if validation is not None:
            return validation

        result = ta.ad(
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
        return result if result is not None else pd.Series([], dtype=float)

    @staticmethod
    @handle_pandas_ta_errors
    def adosc(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        fast: int = 3,
        slow: int = 10,
    ) -> pd.Series:
        """チャイキンA/Dオシレーター"""
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close, "volume": volume}
        )
        if validation is not None:
            return validation

        result = ta.adosc(
            high=high,
            low=low,
            close=close,
            volume=volume,
            fast=fast,
            slow=slow,
        )
        return result if result is not None else pd.Series([], dtype=float)

    @staticmethod
    @handle_pandas_ta_errors
    def obv(close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """オンバランスボリューム"""
        validation = validate_multi_series_params(
            {"close": close, "volume": volume}, length=period
        )
        if validation is not None:
            return validation

        # ゼロボリュームの処理: ゼロボリュームをNaNに変換
        volume_clean = volume.replace(0, np.nan)

        result = ta.obv(close=close, volume=volume_clean, length=period)
        return result if result is not None else pd.Series([], dtype=float)

    @staticmethod
    @handle_pandas_ta_errors
    def eom(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        length: int = 14,
        divisor: float = 100000000.0,
        drift: int = 1,
        offset: int = 0,
    ) -> pd.Series:
        """
        Ease of Movement（イーズ・オブ・ムーブメント）

        Args:
            high: 高値
            low: 安値
            close: 終値
            volume: 出来高
            length: 期間（デフォルト: 14）

        Returns:
            EOM の値
        """
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close, "volume": volume}, length
        )
        if validation is not None:
            return validation

        result = ta.eom(
            high=high,
            low=low,
            close=close,
            volume=volume,
            length=length,
            divisor=divisor,
            drift=drift,
            offset=offset,
        )

        if result is None or (hasattr(result, "empty") and result.empty):
            return pd.Series(np.full(len(high), np.nan), index=high.index)

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 10,
        anchor: str | None = None,
    ) -> pd.Series:
        """Volume Weighted Average Price"""
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close, "volume": volume}
        )
        if validation is not None:
            return validation

        # pandas-taのVWAP関数は時系列インデックスが必要なため、独自実装を使用
        try:
            # pandas-taのVWAPを試行（時系列インデックスがある場合）
            if hasattr(high.index, "to_period"):
                df = ta.vwap(
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    length=period,
                    anchor=anchor,
                )
                return df if df is not None else pd.Series([], dtype=float)
        except Exception:
            pass

        # フォールバック: 独自VWAP実装
        # 典型価格 = (H + L + C) / 3
        typical_price = (high + low + close) / 3

        # VWAP = Σ(典型価格 × 出来高) / Σ(出来高)
        # 累積で計算
        cumulative_pv = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()

        # ゼロ除算を避ける - pandas使用
        vwap = np.where(
            cumulative_volume != 0, cumulative_pv / cumulative_volume, typical_price
        )

        return pd.Series(vwap, index=high.index if hasattr(high, "index") else None)

    @staticmethod
    @handle_pandas_ta_errors
    def cmf(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        length: int = 20,
    ) -> pd.Series:
        """
        Chaikin Money Flow（チャイキン・マネーフロー）

        Args:
            high: 高値
            low: 安値
            close: 終値
            volume: 出来高
            length: 期間（デフォルト: 20）

        Returns:
            CMF の値
        """
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close, "volume": volume}, length
        )
        if validation is not None:
            return validation

        result = ta.cmf(high=high, low=low, close=close, volume=volume, length=length)

        if result is None or (hasattr(result, "empty") and result.empty):
            return pd.Series(np.full(len(high), np.nan), index=high.index)

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def efi(
        close: pd.Series,
        volume: pd.Series,
        period: int = 13,
        mamode: str = "ema",
        drift: int = 1,
    ) -> pd.Series:
        """
        Elder's Force Index（エルダーの勢力指数）

        Args:
            close: 終値
            volume: 出来高
            period: 期間（デフォルト: 13）
            mamode: 移動平均タイプ（デフォルト: "ema"）
            drift: 差分期間（デフォルト: 1）

        Returns:
            Elder's Force Index の値
        """
        validation = validate_multi_series_params(
            {"close": close, "volume": volume}, period
        )
        if validation is not None:
            return validation

        result = ta.efi(
            close=close,
            volume=volume,
            length=period,
            mamode=mamode,
            drift=drift,
        )

        if result is None or (hasattr(result, "empty") and result.empty):
            return pd.Series(np.full(len(close), np.nan), index=close.index)

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """
        Money Flow Index（マネーフローインデックス）

        Args:
            high: 高値
            low: 安値
            close: 終値
            volume: 出来高
            length: 期間（デフォルト: 14）

        Returns:
            MFI の値（0-100 の範囲）
        """
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close, "volume": volume}, length
        )
        if validation is not None:
            return validation

        result = ta.mfi(high=high, low=low, close=close, volume=volume, length=length)

        if result is None or (hasattr(result, "empty") and result.empty):
            return pd.Series(np.full(len(high), np.nan), index=high.index)

        return result

    @staticmethod
    @handle_pandas_ta_errors
    @staticmethod
    def pvo(
        volume: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        scalar: float = 100.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Percentage Volume Oscillator"""
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")
        if fast <= 0 or slow <= 0 or signal <= 0:
            raise ValueError("fast, slow, signal must be positive")

        result = ta.pvo(
            volume=volume,
            fast=fast,
            slow=slow,
            signal=signal,
            scalar=scalar,
        )

        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(volume), np.nan), index=volume.index)
            return nan_series, nan_series, nan_series

        result = result.bfill().fillna(0)
        return (
            result.iloc[:, 0].to_numpy(),
            result.iloc[:, 1].to_numpy(),
            result.iloc[:, 2].to_numpy(),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def pvt(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Price Volume Trend"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

        result = ta.pvt(close=close, volume=volume)
        if result is None:
            return np.full(len(close), np.nan)
        return result.bfill().fillna(0).to_numpy()

    @staticmethod
    @handle_pandas_ta_errors
    def kvo(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        fast: int = 34,
        slow: int = 55,
        signal: int = 13,
        scalar: float = 100.0,
        mamode: str = "ema",
        drift: int = 1,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Klinger Volume Oscillator（クリンガー出来高オシレーター）

        Args:
            high: 高値
            low: 安値
            close: 終値
            volume: 出来高
            fast: 短期EMA期間（デフォルト: 34）
            slow: 長期EMA期間（デフォルト: 55）
            signal: シグナル期間（デフォルト: 13）

        Returns:
            Tuple[KVO, Signal]
        """
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close, "volume": volume},
            max(fast, slow),
        )
        if validation is not None:
            return validation, validation

        result = ta.kvo(
            high=high,
            low=low,
            close=close,
            volume=volume,
            fast=fast,
            slow=slow,
            signal=signal,
            scalar=scalar,
            mamode=mamode,
            drift=drift,
        )

        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series

        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    @handle_pandas_ta_errors
    def nvi(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Negative Volume Index"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

        result = ta.nvi(close=close, volume=volume)
        if result is None or result.empty:
            return np.full(len(close), np.nan)
        return result.bfill().ffill().to_numpy()

    @staticmethod
    @handle_pandas_ta_errors
    def vwap_z_score(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 20,
    ) -> pd.Series:
        """
        VWAP Z-Score (VWAP Divergence)

        Z = (Price - VWAP) / sigma_VWAP
        where sigma_VWAP is the standard deviation of price relative to VWAP.
        """
        # Calculate VWAP first
        vwap_series = VolumeIndicators.vwap(high, low, close, volume, period=period)

        # Calculate deviation
        deviation = close - vwap_series

        # Calculate standard deviation of the deviation
        # Note: The PDF implies sigma is based on VWAP.
        # Standard interpretation: Standard Deviation of (Price - VWAP) or just Price std dev?
        # "sigma_VWAP is standard deviation of price based on VWAP"
        # Usually this means the standard deviation of the price distribution around the weighted average.
        # Calculating rolling std of (Close - VWAP) is a reasonable approximation for "divergence volatility".
        sigma = deviation.rolling(window=period).std()

        # Z-score
        z_score = deviation / sigma

        return z_score.fillna(0.0)

    @staticmethod
    @handle_pandas_ta_errors
    def rvol(
        volume: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """
        Relative Volume (RVOL)

        Ratio of current volume to average volume at the same time of day.
        If index is not DatetimeIndex, falls back to simple Volume / SMA(Volume).
        """
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

        # Check for DatetimeIndex
        if isinstance(volume.index, pd.DatetimeIndex):
            try:
                # Group by time of day and calculate rolling mean for each time bucket
                # Note: This might be slow for very large datasets
                # Transform applies the function to each group
                avg_vol = volume.groupby(volume.index.time).transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )

                # If grouping failed or returned empty (e.g. unique times), fallback
                if avg_vol.isna().all():
                    avg_vol = volume.rolling(window=window).mean()
            except Exception:
                # Fallback on error
                avg_vol = volume.rolling(window=window).mean()
        else:
            # Fallback
            avg_vol = volume.rolling(window=window).mean()

        rvol = volume / avg_vol
        return rvol.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    @handle_pandas_ta_errors
    def absorption_score(
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """
        Absorption Score = RVOL / Range

        High score indicates high volume with low price movement (absorption).
        """
        # Calculate RVOL
        rvol_series = VolumeIndicators.rvol(volume, window=window)

        # Calculate Range
        price_range = high - low

        # Avoid division by zero (if range is 0, absorption is theoretically infinite or max)
        # We replace 0 with a very small number or handle it
        price_range = price_range.replace(0, 1e-9)  # Epsilon

        score = rvol_series / price_range

        return score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
