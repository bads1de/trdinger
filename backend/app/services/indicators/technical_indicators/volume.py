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

from ..utils import handle_pandas_ta_errors

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
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

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
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

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
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

        # OBVデータの長さチェック
        if len(close) != len(volume):
            raise ValueError(
                f"OBV requires close and volume series to have the same length. Got close={len(close)}, volume={len(volume)}"
            )

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
        """Ease of Movement"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")
        if drift <= 0:
            raise ValueError(f"drift must be positive: {drift}")
        if len({len(high), len(low), len(close), len(volume)}) != 1:
            raise ValueError("high, low, close, volume must have the same length")

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
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

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
        """Chaikin Money Flow"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

        # CMF特有のデータ検証: 全てのSeriesの長さが一致するか確認
        series_lengths = [len(high), len(low), len(close), len(volume)]
        if not all(length == series_lengths[0] for length in series_lengths):
            raise ValueError(
                f"CMF requires all input series to have the same length. Got lengths: high={len(high)}, low={len(low)}, close={len(close)}, volume={len(volume)}"
            )

        # 最小データ長チェック（pandas-taの要求）
        min_length = length + 1
        if series_lengths[0] < min_length:
            raise ValueError(
                f"Insufficient data for CMF calculation. Need at least {min_length} points, got {series_lengths[0]}"
            )

        # データ型チェック: 数値のみ許可
        for series_name, series in [
            ("high", high),
            ("low", low),
            ("close", close),
            ("volume", volume),
        ]:
            if series.dtype not in ["float64", "int64", "float32", "int32"]:
                try:
                    # 数値変換を試行
                    converted = pd.to_numeric(series, errors="coerce")
                    # NaNが発生した場合、無効なデータとしてエラー
                    if converted.isna().any():
                        raise ValueError(
                            f"CMF {series_name} series must contain valid numeric values"
                        )
                    if series_name == "high":
                        high = converted
                    elif series_name == "low":
                        low = converted
                    elif series_name == "close":
                        close = converted
                    elif series_name == "volume":
                        volume = converted
                except (ValueError, TypeError):
                    raise ValueError(
                        f"CMF {series_name} series must contain valid numeric values"
                    )

        df = ta.cmf(high=high, low=low, close=close, volume=volume, window=length)
        return df if df is not None else pd.Series([], dtype=float)

    @staticmethod
    @handle_pandas_ta_errors
    def efi(
        close: pd.Series,
        volume: pd.Series,
        period: int = 13,
        mamode: str = "ema",
        drift: int = 1,
    ) -> pd.Series:
        """Elder's Force Index"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

        # EFI特有のデータ検証: 不正値チェック
        if len(close) != len(volume):
            raise ValueError(
                "EFI requires close and volume series to have the same length"
            )

        # 最小データ長チェック
        length = period  # lengthパラメータを定義
        min_length = length + drift
        if len(close) < min_length:
            raise ValueError(
                f"Insufficient data for EFI calculation. Need at least {min_length} points, got {len(close)}"
            )

        # データの健全性チェックとクリーニング
        close_clean = close.copy()
        volume_clean = volume.copy()

        # 負の値や不正な値を処理
        if (close_clean < 0).any():
            # 警告を発行しながら負の価格を処理（絶対値に変換するか、無効として扱う）
            negative_indices = close_clean < 0
            close_clean.loc[negative_indices] = np.nan  # 無効化

        if (volume_clean < 0).any():
            # 負の出来高を無効化
            negative_indices = volume_clean < 0
            volume_clean.loc[negative_indices] = np.nan

        # inf/NaNの処理
        close_clean = close_clean.replace([np.inf, -np.inf], np.nan)
        volume_clean = volume_clean.replace([np.inf, -np.inf], np.nan)

        # 極端な値をクリッピング（価格が現実的でない場合）
        if close_clean.dropna().max() > close_clean.dropna().mean() * 100:
            # 平均の100倍以上の値をクリッピング
            max_reasonable = close_clean.dropna().mean() * 100
            extreme_indices = close_clean > max_reasonable
            close_clean.loc[extreme_indices] = max_reasonable

        # 計算実行
        df = ta.efi(
            close=close_clean,
            volume=volume_clean,
            length=period,
            mamode=mamode,
            drift=drift,
        )

        # 結果の後処理
        if df is not None:
            # 結果がinfやNaNを含む場合のクリーンアップ
            df = df.replace([np.inf, -np.inf], np.nan)
        else:
            df = pd.Series([], dtype=float)

        return df

    @staticmethod
    @handle_pandas_ta_errors
    def mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """マネーフローインデックス"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

        result = ta.mfi(
            high=high.astype(float),
            low=low.astype(float),
            close=close.astype(float),
            volume=volume.astype(float),
            length=length,
        )
        if result is None:
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
        """Klinger Volume Oscillator"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

        # KVO特有のデータ検証
        series_lengths = [len(high), len(low), len(close), len(volume)]
        if not all(length == series_lengths[0] for length in series_lengths):
            raise ValueError(
                f"KVO requires all input series to have the same length. Got lengths: high={len(high)}, low={len(low)}, close={len(close)}, volume={len(volume)}"
            )

        # 最小データ長チェック
        min_length = max(fast, slow) + drift + 5
        if series_lengths[0] < min_length:
            raise ValueError(
                f"Insufficient data for KVO calculation. Need at least {min_length} points, got {series_lengths[0]}"
            )

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
            nan_series = pd.Series(np.full(series_lengths[0], np.nan))
            return nan_series, nan_series

        # 結果が複数列の場合の処理
        if isinstance(result, pd.DataFrame):
            if len(result.columns) >= 2:
                return result.iloc[:, 0], result.iloc[:, 1]
            elif len(result.columns) == 1:
                return result.iloc[:, 0], pd.Series(np.full(len(result), np.nan))

        return result, pd.Series(np.full(len(result), np.nan))

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
        price_range = price_range.replace(0, 1e-9) # Epsilon
        
        score = rvol_series / price_range
        
        return score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
