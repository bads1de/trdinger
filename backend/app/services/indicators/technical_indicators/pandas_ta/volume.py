"""
出来高系テクニカル指標 (Volume Indicators)

pandas-ta の volume カテゴリに対応。
出来高と価格の関係性分析に使用する指標群。

登録してあるテクニカルの一覧:
- AD (A/Dライン)
- ADOSC (A/Dオシレーター)
- OBV (On-Balance Volume)
- EOM (Ease of Movement)
- VWAP (Volume Weighted Average Price)
- CMF (Chaikin Money Flow)
- EFI (Elder Force Index)
- MFI (Money Flow Index)
- PVO (Percentage Volume Oscillator)
- PVT (Price Volume Trend)
- KVO (Klinger Volume Oscillator)
- NVI (Negative Volume Index)
- VWAP Z-Score
- RVOL (Relative Volume)
- Absorption Score
- AOBV (Archer OBV)
- PVI (Positive Volume Index)
- PVOL (Price-Volume)
- PVR (Price Volume Rank)
- VFI (Volume Flow Indicator)
- VP (Volume Profile)
"""

import logging
from typing import Tuple, cast

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from numba import njit

from ...data_validation import (
    create_nan_series_bundle,
    handle_pandas_ta_errors,
    normalize_non_finite,
    run_multi_series_indicator,
    run_series_indicator,
)

logger = logging.getLogger(__name__)


def _dataframe_to_series_tuple(
    result: pd.DataFrame | None,
) -> Tuple[pd.Series, ...] | None:
    """pandas-ta の DataFrame 結果を Series のタプルに変換する。"""
    if result is None or result.empty:
        return None
    return tuple(result.iloc[:, i] for i in range(result.shape[1]))


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
        return run_multi_series_indicator(
            {"high": high, "low": low, "close": close, "volume": volume},
            None,
            lambda: ta.ad(
                high=high,
                low=low,
                close=close,
                volume=volume,
            ),
        )

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
        return run_multi_series_indicator(
            {"high": high, "low": low, "close": close, "volume": volume},
            slow,
            lambda: ta.adosc(
                high=high,
                low=low,
                close=close,
                volume=volume,
                fast=fast,
                slow=slow,
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def obv(close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """オンバランスボリューム"""
        # ゼロボリュームの処理: ゼロボリュームをNaNに変換
        volume_clean = volume.replace(0, np.nan)

        return run_multi_series_indicator(
            {"close": close, "volume": volume},
            period,
            lambda: ta.obv(close=close, volume=volume_clean, length=period),
        )

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
        return run_multi_series_indicator(
            {"high": high, "low": low, "close": close, "volume": volume},
            length,
            lambda: ta.eom(
                high=high,
                low=low,
                close=close,
                volume=volume,
                length=length,
                divisor=divisor,
                drift=drift,
                offset=offset,
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 10,
    ) -> pd.Series:
        """Volume Weighted Average Price

        pandas-taのVWAPは時系列インデックスを必須とするため、
        独自の累積VWAP実装を使用。

        Args:
            high: 高値
            low: 安値
            close: 終値
            volume: 出来高
            period: 使用されない（累積VWAPを計算）

        Returns:
            VWAP (累積ベース)
        """

        def _calculate_vwap() -> pd.Series:
            typical_price = (high + low + close) / 3
            cumulative_pv = (typical_price * volume).cumsum()
            cumulative_volume = volume.cumsum()
            return (cumulative_pv / cumulative_volume.replace(0, np.nan)).fillna(
                typical_price
            )

        return run_multi_series_indicator(
            {"high": high, "low": low, "close": close, "volume": volume},
            None,
            _calculate_vwap,
        )

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
        return run_multi_series_indicator(
            {"high": high, "low": low, "close": close, "volume": volume},
            length,
            lambda: ta.cmf(
                high=high, low=low, close=close, volume=volume, length=length
            ),
        )

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
        return run_multi_series_indicator(
            {"close": close, "volume": volume},
            period,
            lambda: ta.efi(
                close=close,
                volume=volume,
                length=period,
                mamode=mamode,
                drift=drift,
            ),
        )

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

        def _calculate_mfi() -> pd.Series:
            try:
                res = ta.mfi(
                    high=high, low=low, close=close, volume=volume, length=length
                )
            except TypeError:
                # Fallback for strict environments where volume must be int
                res = ta.mfi(
                    high=high,
                    low=low,
                    close=close,
                    volume=volume.astype(int),
                    length=length,
                )

            if res is None:
                return pd.Series(np.nan, index=close.index, dtype=float)
            return cast(pd.Series, res)

        return run_multi_series_indicator(
            {"high": high, "low": low, "close": close, "volume": volume},
            length,
            _calculate_mfi,
        )

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
        if fast <= 0 or signal <= 0:
            raise ValueError("fast and signal must be positive")

        return run_series_indicator(
            volume,
            slow,
            lambda: _dataframe_to_series_tuple(
                ta.pvo(
                    volume=volume,
                    fast=fast,
                    slow=slow,
                    signal=signal,
                    scalar=scalar,
                )
            ),
            fallback_factory=lambda: create_nan_series_bundle(volume, 3),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def pvt(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Price Volume Trend"""
        return run_multi_series_indicator(
            {"close": close, "volume": volume},
            None,
            lambda: ta.pvt(close=close, volume=volume),
        )

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
        return run_multi_series_indicator(
            {"high": high, "low": low, "close": close, "volume": volume},
            max(fast, slow),
            lambda: _dataframe_to_series_tuple(
                ta.kvo(
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
            ),
            fallback_factory=lambda: create_nan_series_bundle(high, 2),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def nvi(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Negative Volume Index"""
        return run_multi_series_indicator(
            {"close": close, "volume": volume},
            None,
            lambda: ta.nvi(close=close, volume=volume),
        )

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

        def _calculate_vwap_z_score() -> pd.Series:
            vwap_series = VolumeIndicators.vwap(high, low, close, volume, period=period)
            deviation = close - vwap_series
            sigma = deviation.rolling(window=period).std()
            return deviation / sigma

        return run_multi_series_indicator(
            {"high": high, "low": low, "close": close, "volume": volume},
            period,
            _calculate_vwap_z_score,
        )

    @staticmethod
    @handle_pandas_ta_errors
    def rvol(
        volume: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """
        Relative Volume (RVOL) - Numba Optimized Version
        """

        def _calculate_rvol() -> pd.Series:
            # DatetimeIndexがある場合、時間帯別の平均出来高を考慮したRVOLを計算
            if isinstance(volume.index, pd.DatetimeIndex):
                try:
                    # 時間情報を秒単位のインデックスに変換 (0 - 86400)
                    idx = volume.index
                    _h = getattr(idx, "hour")
                    _m = getattr(idx, "minute")
                    _s = getattr(idx, "second")
                    time_indices = (_h * 3600 + _m * 60 + _s).values.astype(np.int32)
                    vol_arr = volume.values.astype(np.float64)

                    res_arr = _njit_rvol_loop(vol_arr, time_indices, window)
                    rvol_series = pd.Series(res_arr, index=volume.index)

                    if not rvol_series.isna().all():
                        return normalize_non_finite(rvol_series)
                except Exception as e:
                    logger.warning(
                        f"RVOL Numba optimization failed: {e}. Falling back..."
                    )

            # フォールバック: 標準的なローリング平均
            avg_vol = volume.rolling(window=window, min_periods=1).mean()
            rvol = volume / avg_vol
            return normalize_non_finite(rvol)

        return run_series_indicator(volume, window, _calculate_rvol)

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
        return run_multi_series_indicator(
            {"high": high, "low": low, "volume": volume},
            window,
            lambda: normalize_non_finite(
                VolumeIndicators.rvol(volume, window=window)
                / (high - low).replace(0, 1e-9)
            ),
        )

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
        return run_multi_series_indicator(
            {"close": close, "volume": volume},
            slow,
            lambda: _dataframe_to_series_tuple(
                ta.aobv(
                    close=close,
                    volume=volume,
                    fast=fast,
                    slow=slow,
                    max_lookback=max_lookback,
                    min_lookback=min_lookback,
                    mamode=mamode,
                    scalar=scalar,
                )
            ),
            fallback_factory=lambda: create_nan_series_bundle(close, 7),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def pvi(close: pd.Series, volume: pd.Series, length: int = 13) -> pd.Series:
        """Positive Volume Index"""
        return run_multi_series_indicator(
            {"close": close, "volume": volume},
            None,
            lambda: ta.pvi(close=close, volume=volume, length=length),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def pvol(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Price-Volume"""
        return run_multi_series_indicator(
            {"close": close, "volume": volume},
            None,
            lambda: ta.pvol(close=close, volume=volume),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def pvr(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Price Volume Rank"""
        return run_multi_series_indicator(
            {"close": close, "volume": volume},
            None,
            lambda: ta.pvr(close=close, volume=volume),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def vfi(
        close: pd.Series,
        volume: pd.Series,
        length: int = 130,
        coef: float = 0.2,
        vfactor: float = 0.2,
        anchored: bool = False,
    ) -> pd.Series:
        """
        Volume Flow Indicator (VFI)

        マネーフローに基づく出来高指標。価格変動と出来高の関係を分析し、
        買い圧力と売り圧力を検出する。

        Args:
            close: 終値
            volume: 出来高
            length: 期間（デフォルト: 130）
            coef: 係数（デフォルト: 0.2）
            vfactor: ボリュームファクター（デフォルト: 0.2）
            anchored: アンカー版を使用するか（デフォルト: False）

        Returns:
            VFI の値
        """
        return run_multi_series_indicator(
            {"close": close, "volume": volume},
            length,
            lambda: ta.vfi(
                close=close,
                volume=volume,
                length=length,
                coef=coef,
                vfactor=vfactor,
                anchored=anchored,
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def vp(
        close: pd.Series,
        volume: pd.Series,
        bins: int = 10,
        width: float | None = None,
    ) -> pd.DataFrame:
        """
        Volume Profile (VP)

        価格レベルごとの出来高分布を計算する。
        各ビン（価格レンジ）に対する出来高の合計を返す。

        Args:
            close: 終値
            volume: 出来高
            bins: 分割数（デフォルト: 10）
            width: ビンの幅（指定時はbinsより優先）

        Returns:
            Volume Profile の DataFrame（Price Level と Volume の列）
        """

        def _calculate_vp() -> pd.DataFrame | None:
            result = ta.vp(close=close, volume=volume, bins=bins, width=width)
            return result

        return run_multi_series_indicator(
            {"close": close, "volume": volume},
            None,
            _calculate_vp,
        )
