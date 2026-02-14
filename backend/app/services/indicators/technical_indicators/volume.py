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
from numba import njit
import pandas_ta_classic as ta

from ..data_validation import (
    handle_pandas_ta_errors,
    validate_multi_series_params,
    validate_series_params,
)


logger = logging.getLogger(__name__)


@njit(cache=True)
def _njit_rvol_loop(
    volumes: np.ndarray, time_indices: np.ndarray, window: int
) -> np.ndarray:
    """
    Numba-optimized single-pass calculation of Relative Volume (RVOL).
    Uses running sums and circular buffers for each time-of-day bucket.
    """
    n = len(volumes)
    result = np.full(n, np.nan, dtype=np.float64)

    # Max seconds in a day = 86400. This covers up to 1-second resolution.
    max_buckets = 86401
    running_sums = np.zeros(max_buckets, dtype=np.float64)
    buffers = np.zeros((max_buckets, window), dtype=np.float64)
    counts = np.zeros(max_buckets, dtype=np.int32)
    write_indices = np.zeros(max_buckets, dtype=np.int32)

    for i in range(n):
        m = time_indices[i]
        vol = volumes[i]

        if np.isnan(vol) or m < 0 or m >= max_buckets:
            continue

        # Update circular buffer and running sum
        idx = write_indices[m]
        old_val = buffers[m, idx]
        running_sums[m] = running_sums[m] - old_val + vol
        buffers[m, idx] = vol

        # Update pointers and counts
        write_indices[m] = (idx + 1) % window
        counts[m] = min(counts[m] + 1, window)

        # Calculate average and RVOL
        if counts[m] > 0:
            avg = running_sums[m] / counts[m]
            if avg > 0:
                result[i] = vol / avg

    return result


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
        anchor: str | None = None,  # 互換性のため残すが未使用
    ) -> pd.Series:
        """Volume Weighted Average Price

        pandas-taのVWAPは時系列インデックスを必須とするため、
        独自の累積VWAP実装を使用。

        Args:
            high: 高値
            low: 安値
            close: 終値
            volume: 出来高
            period: 未使用（互換性のため残す）
            anchor: 未使用（互換性のため残す）

        Returns:
            VWAP (累積ベース)
        """
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close, "volume": volume}
        )
        if validation is not None:
            return validation

        # 典型価格 = (H + L + C) / 3
        typical_price = (high + low + close) / 3

        # VWAP = Σ(典型価格 × 出来高) / Σ(出来高)
        cumulative_pv = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()

        # ゼロ除算を避ける
        vwap = cumulative_pv / cumulative_volume.replace(0, np.nan)

        return vwap.fillna(typical_price)

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

        try:
            result = ta.mfi(
                high=high, low=low, close=close, volume=volume, length=length
            )
        except TypeError:
            # Fallback for strict environments where volume must be int
            result = ta.mfi(
                high=high,
                low=low,
                close=close,
                volume=volume.astype(int),
                length=length,
            )

        if result is None or (hasattr(result, "empty") and result.empty):
            return pd.Series(np.full(len(high), np.nan), index=high.index)

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def pvo(
        volume: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        scalar: float = 100.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Percentage Volume Oscillator"""
        validation = validate_series_params(volume, slow)
        if validation is not None:
            nan_series = pd.Series(np.full(len(volume), np.nan), index=volume.index)
            return nan_series, nan_series, nan_series

        if fast <= 0 or signal <= 0:
            raise ValueError("fast and signal must be positive")

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

        return (
            result.iloc[:, 0].to_numpy(),
            result.iloc[:, 1].to_numpy(),
            result.iloc[:, 2].to_numpy(),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def pvt(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Price Volume Trend"""
        validation = validate_multi_series_params({"close": close, "volume": volume})
        if validation is not None:
            return validation

        result = ta.pvt(close=close, volume=volume)
        if result is None:
            return None
        return result.to_numpy()

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
        """
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close, "volume": volume},
            max(fast, slow),
        )
        if validation is not None:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series

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
        validation = validate_multi_series_params({"close": close, "volume": volume})
        if validation is not None:
            return validation

        result = ta.nvi(close=close, volume=volume)
        if result is None or result.empty:
            return None
        return result.to_numpy()

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
        """
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close, "volume": volume}, period
        )
        if validation is not None:
            return validation

        # Calculate VWAP first
        vwap_series = VolumeIndicators.vwap(high, low, close, volume, period=period)

        # Calculate deviation
        deviation = close - vwap_series

        # Calculate standard deviation of the deviation
        sigma = deviation.rolling(window=period).std()

        # Z-score
        z_score = deviation / sigma

        return z_score

    @staticmethod
    @handle_pandas_ta_errors
    def rvol(
        volume: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """
        Relative Volume (RVOL) - Numba Optimized Version
        """
        validation = validate_series_params(volume, window)
        if validation is not None:
            return validation

        # DatetimeIndexがある場合、時間帯別の平均出来高を考慮したRVOLを計算
        if isinstance(volume.index, pd.DatetimeIndex):
            try:
                # 時間情報を秒単位のインデックスに変換 (0 - 86400)
                idx = volume.index
                time_indices = (
                    idx.hour * 3600 + idx.minute * 60 + idx.second
                ).values.astype(np.int32)
                vol_arr = volume.values.astype(np.float64)

                res_arr = _njit_rvol_loop(vol_arr, time_indices, window)
                rvol_series = pd.Series(res_arr, index=volume.index)

                if not rvol_series.isna().all():
                    return rvol_series.replace([np.inf, -np.inf], np.nan)
            except Exception as e:
                logger.warning(f"RVOL Numba optimization failed: {e}. Falling back...")

        # フォールバック: 標準的なローリング平均
        avg_vol = volume.rolling(window=window, min_periods=1).mean()
        rvol = volume / avg_vol
        return rvol.replace([np.inf, -np.inf], np.nan)

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
        """
        validation = validate_multi_series_params(
            {"high": high, "low": low, "volume": volume}, window
        )
        if validation is not None:
            return validation

        # Calculate RVOL
        rvol_series = VolumeIndicators.rvol(volume, window=window)

        # Calculate Range
        price_range = high - low

        # Avoid division by zero
        price_range = price_range.replace(0, 1e-9)  # Epsilon

        score = rvol_series / price_range

        return score.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    @handle_pandas_ta_errors
    def aobv(
        close: pd.Series,
        volume: pd.Series,
        fast: int = 4,
        slow: int = 12,
        max_lookback: int = 2,
        min_lookback: int = 5,
        mamode: str = "ema",
        scalar: float = 100.0,
    ) -> Tuple[
        pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series
    ]:
        """Archer On-Balance Volume"""
        validation = validate_multi_series_params(
            {"close": close, "volume": volume}, slow
        )
        if validation is not None:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            # AOBV returns 7 columns typically: OBV, MA_OBV, OBV_min_2, OBV_max_2, OBV_min_5, OBV_max_5, AOBV_LR_2
            # We conform to return all if possible, or key ones.
            # Let's return tuple of 7 NaNs to be safe as per type hint or simplified representation.
            return (nan_series,) * 7

        result = ta.aobv(
            close=close,
            volume=volume,
            fast=fast,
            slow=slow,
            max_lookback=max_lookback,
            min_lookback=min_lookback,
            mamode=mamode,
            scalar=scalar,
        )
        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return (nan_series,) * 7

        # Ensure we return valid series tuple
        return tuple(result.iloc[:, i] for i in range(result.shape[1]))

    @staticmethod
    @handle_pandas_ta_errors
    def pvi(close: pd.Series, volume: pd.Series, length: int = 13) -> pd.Series:
        """Positive Volume Index"""
        validation = validate_multi_series_params({"close": close, "volume": volume})
        if validation is not None:
            return validation

        result = ta.pvi(close=close, volume=volume, length=length)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def pvol(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Price-Volume"""
        validation = validate_multi_series_params({"close": close, "volume": volume})
        if validation is not None:
            return validation

        result = ta.pvol(close=close, volume=volume)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def pvr(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Price Volume Rank"""
        validation = validate_multi_series_params({"close": close, "volume": volume})
        if validation is not None:
            return validation

        result = ta.pvr(close=close, volume=volume)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result
