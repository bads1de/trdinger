"""
出来高系テクニカル指標

登録してあるテクニカルの一覧:
- AD (Accumulation/Distribution Line)
- OBV (On-Balance Volume)
- VWAP (Volume Weighted Average Price)
- MFI (Money Flow Index)
- EOM (Ease of Movement)
- NVI (Negative Volume Index)
- PVI (Positive Volume Index)
- AOBV (Archer On-Balance Volume)
- EFI (Elder's Force Index)
- PVOL (Price-Volume)
- PVR (Price Volume Rank)
- VP (Volume Price)
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import logging

from ..utils import handle_pandas_ta_errors

logger = logging.getLogger(__name__)


class VolumeIndicators:
    """
    出来高系指標クラス
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
    def nvi(
             close: pd.Series, volume: pd.Series, period: int = 13, initial: int = 1000
         ) -> pd.Series:
         """Negative Volume Index"""
         if not isinstance(close, pd.Series):
             raise TypeError("close must be pandas Series")
         if not isinstance(volume, pd.Series):
             raise TypeError("volume must be pandas Series")
         df = ta.nvi(close=close, volume=volume, length=period, initial=initial)
         return df if df is not None else pd.Series([], dtype=float)

    @staticmethod
    @handle_pandas_ta_errors
    def pvi(
             close: pd.Series, volume: pd.Series, period: int = 13, initial: int = 1000
         ) -> pd.Series:
         """Positive Volume Index"""
         if not isinstance(close, pd.Series):
             raise TypeError("close must be pandas Series")
         if not isinstance(volume, pd.Series):
             raise TypeError("volume must be pandas Series")
         df = ta.pvi(close=close, volume=volume, length=period, initial=initial)
         return df if df is not None else pd.Series([], dtype=float)

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
                    high=high, low=low, close=close, volume=volume, length=period, anchor=anchor
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
    def eom(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        length: int = 14,
        divisor: int = 100000000,
        drift: int = 1,
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
        df = ta.eom(
            high=high,
            low=low,
            close=close,
            volume=volume,
            window=length,
            divisor=divisor,
            drift=drift,
        )
        return df if df is not None else pd.Series([], dtype=float)

    @staticmethod
    @handle_pandas_ta_errors
    def kvo(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        fast: int = 34,
        slow: int = 55,
    ) -> tuple[pd.Series, pd.Series]:
        """Klinger Volume Oscillator with enhanced validation and fallback"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

        # Enhanced data length validation
        series_lengths = [len(high), len(low), len(close), len(volume)]
        max_length = max(series_lengths)
        if not all(length == series_lengths[0] for length in series_lengths):
            # Auto-alignment by trimming to minimum length if slight mismatches
            min_length = min(series_lengths)
            if max_length - min_length <= 5:  # Small difference tolerance
                logger.warning(f"KVO auto-aligning series lengths: {series_lengths}")
                high = high.iloc[-min_length:]
                low = low.iloc[-min_length:]
                close = close.iloc[-min_length:]
                volume = volume.iloc[-min_length:]
            else:
                raise ValueError(
                    f"KVO requires all input series to have similar lengths. Got lengths: high={len(high)}, low={len(low)}, close={len(close)}, volume={len(volume)}"
                )

        data_length = len(high)
        # Minimum data length check - KVO needs sufficient data for calculation
        min_length = max(slow + 20, 100)  # Ensure adequate data for proper calculation
        if data_length < min_length:
            logger.warning(
                f"KVO: Insufficient data length {data_length}, minimum required {min_length}"
            )
            # Return NaN series instead of raising error
            nan_series = pd.Series(np.full(data_length, np.nan), index=high.index)
            return nan_series, nan_series

        # Parameter validation
        if fast <= 0 or slow <= 0 or fast >= slow:
            raise ValueError(
                f"invalid parameters: fast={fast}, slow={slow}. fast must be > 0 and < slow"
            )

        # Enhanced data cleaning and validation
        high_clean = (
            high.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0)
        )
        low_clean = (
            low.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0)
        )
        close_clean = (
            close.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0)
        )
        volume_clean = volume.replace([np.inf, -np.inf], np.nan).fillna(
            0
        )  # Volume 0 is meaningful

        # Ensure volume is non-negative (KVO requires positive volume)
        volume_clean = volume_clean.abs()

        try:
            # Try pandas-ta with enhanced error handling
            logger.warning(
                f"KVO: Calling ta.kvo with fast={fast}, slow={slow}, data_length={data_length}"
            )
            df = ta.kvo(
                high=high_clean,
                low=low_clean,
                close=close_clean,
                volume=volume_clean,
                fast=fast,
                slow=slow,
            )

            if (
                df is not None
                and isinstance(df, pd.DataFrame)
                and not df.empty
                and not df.isna().all().all()
            ):
                # Validate result structure
                if df.shape[1] >= 2:
                    kvo = df.iloc[:, 0]
                    kvos = df.iloc[:, 1]
                    # Post-process to handle any remaining NaN
                    kvo = kvo.fillna(method="bfill").fillna(0)
                    kvos = kvos.fillna(method="bfill").fillna(0)
                    return kvo, kvos
                else:
                    logger.warning(
                        f"KVO: pandas-ta returned insufficient columns: {df.shape[1]}"
                    )
            else:
                logger.warning(
                    f"KVO: pandas-ta returned invalid result: type={type(df)}, empty={df.empty if hasattr(df, 'empty') else 'N/A'}"
                )

        except Exception as e:
            logger.warning(f"KVO pandas-ta call failed: {e}")

        # Enhanced fallback implementation
        logger.warning("KVO: Using enhanced manual fallback implementation")
        try:
            # KVO calculation: volume force based on trend
            # VF = VF + ((2 * ((DM / CM) - 1)) * Volume * (2 / (fast + slow)) * 100)
            # where DM = close - previous close, CM = high - low (money range)

            vf_values = np.zeros(data_length)
            dm_values = close_clean.diff()  # Price direction
            cm_values = high_clean - low_clean  # Money range

            # Handle initial values
            dm_values.iloc[0] = 0
            cm_values.iloc[0] = (
                cm_values.iloc[1] if len(cm_values) > 1 else cm_values.iloc[0]
            )

            # Volume force calculation
            scaling = 2.0 / (fast + slow)

            for i in range(1, data_length):
                if cm_values.iloc[i] != 0:
                    trend_strength = 2 * ((dm_values.iloc[i] / cm_values.iloc[i]) - 1)
                else:
                    trend_strength = 0

                volume_force = trend_strength * volume_clean.iloc[i] * scaling * 100
                vf_values[i] = vf_values[i - 1] + volume_force

            # Create oscillator components
            kvo_raw = pd.Series(vf_values, index=high.index)
            signal_length = min(slow, data_length // 3)  # Adaptive signal length
            kvos = kvo_raw.ewm(span=signal_length, adjust=False).mean()

            # Enhanced NaN handling
            kvo_raw = kvo_raw.fillna(method="bfill").fillna(0)
            kvos = kvos.fillna(method="bfill").fillna(0)

            # Final validation
            if kvo_raw.isna().all() or kvos.isna().all():
                logger.warning("KVO fallback: resulting series are all NaN")
                nan_series = pd.Series(np.full(data_length, np.nan), index=high.index)
                return nan_series, nan_series

            return kvo_raw, kvos

        except Exception as e:
            logger.warning(f"KVO enhanced fallback calculation failed: {e}")
            # Final fallback: return NaN series
            nan_series = pd.Series(np.full(data_length, np.nan), index=high.index)
            return nan_series, nan_series

    @staticmethod
    @handle_pandas_ta_errors
    def pvt(close: pd.Series, volume: pd.Series, period: int = 10) -> pd.Series:
        """Price Volume Trend"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")
        df = ta.pvt(close=close, volume=volume, length=period)
        return df if df is not None else pd.Series([], dtype=float)

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
    def aobv(
        close: pd.Series,
        volume: pd.Series,
        fast: int = 5,
        slow: int = 10,
        max_lookback: int = 2,
        min_lookback: int = 2,
        mamode: str = "ema",
        period: int = 10,
    ) -> tuple[pd.Series, ...]:
        """Archer On-Balance Volume"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")
        df = ta.aobv(
            close=close,
            volume=volume,
            fast=fast,
            slow=slow,
            max_lookback=max_lookback,
            min_lookback=min_lookback,
            mamode=mamode,
            length=period,
        )
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            # 7つの空のシリーズを返す
            empty = pd.Series([], dtype=float)
            return empty, empty, empty, empty, empty, empty, empty
        # DataFrameの各列をシリーズとして返す
        return tuple(df.iloc[:, i] for i in range(df.shape[1]))

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
    def pvol(close: pd.Series, volume: pd.Series, signed: bool = True) -> pd.Series:
        """Price-Volume"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

        # Volumeデータの長さチェック
        if len(close) != len(volume):
            raise ValueError(
                f"PVol requires close and volume series to have the same length. Got close={len(close)}, volume={len(volume)}"
            )

        # ゼロボリュームの処理: ゼロボリュームをNaNに変換して処理
        volume_clean = volume.replace(0, np.nan)

        df = ta.pvol(close=close, volume=volume_clean, signed=signed)
        return df if df is not None else pd.Series([], dtype=float)

    @staticmethod
    @handle_pandas_ta_errors
    def pvr(close: pd.Series, volume: pd.Series, period: int = 10, length: int | None = None) -> pd.Series:
        """Price Volume Rank"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

        # ta.pvrはlengthパラメータを受け付けないので、periodパラメータのみ使用
        # lengthパラメータは互換性のため受け付けるが、無視する
        df = ta.pvr(close=close, volume=volume)
        return df if df is not None else pd.Series([], dtype=float)

    @staticmethod
    @handle_pandas_ta_errors
    def vp(
        close: pd.Series, volume: pd.Series, width: int = 10
    ) -> tuple[pd.Series, ...]:
        """Volume Price Confirmation (Volume Profile)"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

        # Try pandas-ta first
        try:
            df = ta.vp(close=close, volume=volume, width=width)
            if (
                df is not None
                and isinstance(df, pd.DataFrame)
                and not df.empty
                and not df.isna().all().all()
            ):
                # DataFrameであることを確認し、各列を個別のpandas Seriesとして返す
                return (
                    df.iloc[:, 0],  # low_0 (価格範囲下限) - pd.Series
                    df.iloc[:, 1],  # mean_0 (価格範囲平均) - pd.Series
                    df.iloc[:, 2],  # high_0 (価格範囲上限) - pd.Series
                    df.iloc[:, 3],  # pos_1 (陽線出来高) - pd.Series
                    df.iloc[:, 4],  # neg_1 (陰線出来高) - pd.Series
                    df.iloc[:, 5],  # total_1 (総出来高) - pd.Series
                )
        except Exception:
            pass

        # Enhanced Fallback: Simple volume profile implementation with better error handling
        try:
            if len(close) < width:
                logger.warning(
                    f"VP fallback: insufficient data length {len(close)}, required {width}"
                )
                # Insufficient data, return empty series
                empty = pd.Series([], dtype=float)
                return empty, empty, empty, empty, empty, empty

            # Data validation
            if close.isna().all() or volume.isna().all():
                logger.warning("VP fallback: all input data is NaN")
                empty = pd.Series([], dtype=float)
                return empty, empty, empty, empty, empty, empty

            # Simple volume profile calculation
            # Get price range
            min_price = close.min()
            max_price = close.max()
            price_range = max_price - min_price

            if price_range == 0 or np.isnan(price_range):
                # All prices the same or price range calculation failed, return single level
                try:
                    total_vol = volume.sum()
                    if np.isnan(total_vol):
                        total_vol = 0.0
                    empty = pd.Series([], dtype=float)
                    single_price = min_price if not np.isnan(min_price) else 0.0
                    return (
                        pd.Series(
                            [single_price] * len(close), index=close.index
                        ),  # low
                        pd.Series(
                            [single_price] * len(close), index=close.index
                        ),  # mean
                        pd.Series(
                            [single_price] * len(close), index=close.index
                        ),  # high
                        empty,  # pos vol
                        empty,  # neg vol
                        empty,  # total vol
                    )
                except Exception as e:
                    logger.warning(f"VP fallback single price calculation failed: {e}")
                    empty = pd.Series([], dtype=float)
                    return empty, empty, empty, empty, empty, empty

            # Create price bins with error handling
            try:
                n_bins = min(
                    width,
                    max(
                        1,
                        int(price_range / (close.std() / 4)) if close.std() > 0 else 1,
                    ),
                )
                price_bins = pd.cut(close, bins=n_bins, labels=False)
            except Exception as e:
                logger.warning(
                    f"VP fallback bin calculation failed: {e}, using simple binning"
                )
                n_bins = min(width, 10)  # Fallback to 10 bins
                price_bins = pd.qcut(
                    close.rank(method="first"),
                    q=n_bins,
                    labels=False,
                    duplicates="drop",
                )

            # Aggregate volume per price level with better error handling
            vol_profile = {}
            pos_vol_profile = {}
            neg_vol_profile = {}

            prev_close = close.iloc[0] if not close.empty else 0.0

            for i in range(len(close)):
                try:
                    level = price_bins[i]
                    if pd.isna(level):
                        continue

                    level_key = int(level)
                    if level_key not in vol_profile:
                        vol_profile[level_key] = 0.0
                        pos_vol_profile[level_key] = 0.0
                        neg_vol_profile[level_key] = 0.0

                    vol_profile[level_key] += (
                        volume.iloc[i] if not pd.isna(volume.iloc[i]) else 0.0
                    )
                    if close.iloc[i] >= prev_close:
                        pos_vol_profile[level_key] += (
                            volume.iloc[i] if not pd.isna(volume.iloc[i]) else 0.0
                        )
                    else:
                        neg_vol_profile[level_key] += (
                            volume.iloc[i] if not pd.isna(volume.iloc[i]) else 0.0
                        )
                    prev_close = close.iloc[i]
                except Exception as e:
                    logger.debug(f"VP fallback error processing index {i}: {e}")
                    continue

            # Create output series with error handling
            try:
                if not vol_profile:
                    logger.warning("VP fallback: no valid price levels found")
                    empty = pd.Series([], dtype=float)
                    return empty, empty, empty, empty, empty, empty

                # Validate profile data
                for profile_dict in [vol_profile, pos_vol_profile, neg_vol_profile]:
                    # Remove NaN keys and ensure values are numeric
                    keys_to_remove = [
                        k
                        for k in profile_dict.keys()
                        if pd.isna(k) or not isinstance(k, (int, str))
                    ]
                    for k in keys_to_remove:
                        del profile_dict[k]
                    # Ensure all values are numeric
                    for k in profile_dict.keys():
                        if pd.isna(profile_dict[k]):
                            profile_dict[k] = 0.0

                # Create output series
                total_vol_series = pd.Series(
                    list(vol_profile.values()), index=list(vol_profile.keys())
                )
                pos_vol_series = pd.Series(
                    list(pos_vol_profile.values()), index=list(pos_vol_profile.keys())
                )
                neg_vol_series = pd.Series(
                    list(neg_vol_profile.values()), index=list(neg_vol_profile.keys())
                )

                # Ensure series are not empty and have valid indices
                if total_vol_series.empty:
                    logger.warning("VP fallback: total volume series is empty")
                    empty = pd.Series([], dtype=float)
                    return empty, empty, empty, empty, empty, empty

                # Return as 6-tuple for compatibility
                return (
                    pd.Series(
                        [min_price] * len(total_vol_series),
                        index=total_vol_series.index,
                    ),  # low
                    pd.Series(
                        (min_price + max_price) / 2, index=total_vol_series.index
                    ),  # mean
                    pd.Series(
                        [max_price] * len(total_vol_series),
                        index=total_vol_series.index,
                    ),  # high
                    pos_vol_series,  # positive volume
                    neg_vol_series,  # negative volume
                    total_vol_series,  # total volume
                )
            except Exception as e:
                logger.warning(f"VP fallback output creation failed: {e}")
                empty = pd.Series([], dtype=float)
                return empty, empty, empty, empty, empty, empty

        except Exception as e:
            logger.warning(f"VP fallback failed completely: {e}")
            empty = pd.Series([], dtype=float)
            return empty, empty, empty, empty, empty, empty

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
            high=high,
            low=low,
            close=close,
            volume=volume,
            length=length,
        )
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result
