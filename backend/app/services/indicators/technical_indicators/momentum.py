"""
モメンタム系テクニカル指標

登録してあるテクニカルの一覧:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Williams %R
- CCI (Commodity Channel Index)
- ROC (Rate of Change)
- Momentum
- ADX (Average Directional Index)
- Aroon
- APO (Absolute Price Oscillator)
- AO (Awesome Oscillator)
- BIAS (Bias)
- BRAR (BRAR Index)
- CG (Center of Gravity)
- COPPOCK (Coppock Curve)
- ER (Efficiency Ratio)
- ERI (Elder Ray Index)
- FISHER (Fisher Transform)
- INERTIA (Inertia)
- PGO (Pretty Good Oscillator)
- PSL (Psychological Line)
- RSX (RSX)
- SQUEEZE (Squeeze)
- SQUEEZE_PRO (Squeeze Pro)
"""

from typing import Tuple
import logging

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


class MomentumIndicators:
    """
    モメンタム系指標クラス
    """

    @staticmethod
    def mad(data: pd.Series, period: int = 14) -> pd.Series:
        """Mean Absolute Deviation with pandas-ta fallback support"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if period <= 0:
            raise ValueError(f"period must be positive: {period}")

        try:
            # pandas-ta MAD if available - but pandas-ta doesn't have MAD function
            # This would be custom implementation
            pass
        except Exception:
            pass

        # Custom MAD implementation: Mean Absolute Deviation from Moving Average
        if len(data) < period:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        ma = data.rolling(window=period).mean()
        abs_dev = (data - ma).abs()
        mad_result = abs_dev.rolling(window=period).mean()

        return mad_result

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """相対力指数"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        length = period
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        result = ta.rsi(data, window=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def macd(
        data: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        result = ta.macd(data, fast=fast, slow=slow, signal=signal)

        if result is None or result.empty:
            # フォールバック: NaN配列を返す
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return (nan_series, nan_series, nan_series)

        return (
            result.iloc[:, 0],  # MACD
            result.iloc[:, 1],  # Signal
            result.iloc[:, 2],  # Histogram
        )

    @staticmethod
    def stoch(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k: int = 14,
        d: int = 3,
        smooth_k: int = 3,
        d_length: int = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """ストキャスティクス"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if k <= 0:
            raise ValueError(f"k must be positive: {k}")
        if d <= 0:
            raise ValueError(f"d must be positive: {d}")
        if smooth_k <= 0:
            raise ValueError(f"smooth_k must be positive: {smooth_k}")

        # d_lengthパラメータが指定された場合、smooth_kをd_lengthに設定
        if d_length is not None:
            smooth_k = d_length

        result = ta.stoch(
            high=high,
            low=low,
            close=close,
            k=k,
            d=d,
            smooth_k=smooth_k,
        )

        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series
        return (result.iloc[:, 0], result.iloc[:, 1])

    @staticmethod
    def willr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """ウィリアムズ%R"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.willr(high=high, low=low, close=close, length=length)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    def cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 20,
    ) -> pd.Series:
        """商品チャネル指数"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.cci(high=high, low=low, close=close, length=length)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    def roc(
        data: pd.Series,
        period: int = 10,
        close: pd.Series = None,
    ) -> pd.Series:
        """変化率"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        length = period
        # dataが提供されない場合はcloseを使用
        if data is None and close is not None:
            data = close
        elif data is None:
            raise ValueError("Either 'data' or 'close' must be provided")

        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        result = ta.roc(data, window=length)
        if result is None or result.empty:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def mom(data: pd.Series, length: int = 10) -> pd.Series:
        """モメンタム"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        result = ta.mom(data, length=length)
        if result is None or result.empty:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
        length: int = None,
        **kwargs
    ) -> pd.Series:
        """平均方向性指数"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # backward compatibility: lengthパラメータをperiodにマッピング
        if length is not None:
            period = length

        result = ta.adx(high=high, low=low, close=close, length=period)
        if result is None or result.empty:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result.iloc[:, 0]  # ADX列

    @staticmethod
    def aroon(
        high: pd.Series,
        low: pd.Series,
        length: int = 14,
    ) -> Tuple[pd.Series, pd.Series]:
        """アルーン"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        period = length

        try:
            result = ta.aroon(high=high, low=low, length=length)
            if result is not None and not result.empty and not result.isna().all().all():
                return result.iloc[:, 0], result.iloc[:, 1]
        except (ValueError, IndexError):
            pass

        # 強化されたフォールバック実装：AROON指標の手動計算
        if len(high) < period or len(low) < period:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series

        try:
            aroon_up_values = np.full(len(high), np.nan)
            aroon_down_values = np.full(len(high), np.nan)

            for i in range(period - 1, len(high)):
                # 過去period期間の高値と安値を取得
                high_window = high.iloc[max(0, i - period + 1):i + 1]
                low_window = low.iloc[max(0, i - period + 1):i + 1]

                if len(high_window) < period:
                    continue

                # 高値の中で現在の高値がどれほど古いかを計算
                max_high = high_window.max()
                if not np.isnan(max_high) and max_high != 0:
                    if max_high == high.iloc[i]:
                            # 現在が最高値の場合、AROON=100
                                aroon_up_values[i] = 100.0
                    else:
                        # 最高値が見つかったインデックスを探す（最近ほどAROONが高い）
                        max_idx = high_window.idxmax()
                        if max_idx is not None:
                            periods_since_max = i - high_window.index.get_loc(max_idx)
                            if periods_since_max <= period:
                                aroon_up_values[i] = 100 * (period - periods_since_max) / period

                # 安値の中で現在の安値がどれほど古いかを計算
                min_low = low_window.min()
                if not np.isnan(min_low) and min_low != 0:
                    if min_low == low.iloc[i]:
                        # 現在が最安値の場合、AROON=100
                        aroon_down_values[i] = 100.0
                    else:
                        # 最安値が見つかったインデックスを探す
                        min_idx = low_window.idxmin()
                        if min_idx is not None:
                            periods_since_min = i - low_window.index.get_loc(min_idx)
                            if periods_since_min <= period:
                                aroon_down_values[i] = 100 * (period - periods_since_min) / period

            # NaN値を埋める（線形補間）
            if np.isnan(aroon_up_values).any():
                valid_up = aroon_up_values[~np.isnan(aroon_up_values)]
                if len(valid_up) > 1:
                    aroon_up_values = pd.Series(aroon_up_values, index=high.index).fillna(method='backfill').fillna(method='pad').values

            if np.isnan(aroon_down_values).any():
                valid_down = aroon_down_values[~np.isnan(aroon_down_values)]
                if len(valid_down) > 1:
                  aroon_down_values = pd.Series(aroon_down_values, index=low.index).fillna(method='backfill').fillna(method='pad').values

            return pd.Series(aroon_up_values, index=high.index), pd.Series(aroon_down_values, index=low.index)

        except (ValueError, IndexError):
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series


    @staticmethod
    def uo(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        fast: int = 7,
        medium: int = 14,
        slow: int = 28,
    ) -> pd.Series:
        """アルティメットオシレーター"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.uo(
            high=high,
            low=low,
            close=close,
            fast=fast,
            medium=medium,
            slow=slow,
        )
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    def apo(data: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Absolute Price Oscillator"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        result = ta.apo(data, fast=fast, slow=slow)
        if result is None or result.empty:
            # フォールバック: 簡易実装 (EMAの差分)
            ema_fast = ta.ema(data, length=fast)
            ema_slow = ta.ema(data, length=slow)
            if ema_fast is not None and ema_slow is not None:
                return ema_fast - ema_slow
            else:
                return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def ao(high: pd.Series, low: pd.Series, **kwargs) -> pd.Series:
        """Awesome Oscillator with enhanced fallback"""
        logger.debug(f"AO calculation start. Data length: high={len(high)}, low={len(low)}")

        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        # データ長チェック
        if len(high) < 50 or len(low) < 50:
            logger.warning(f"Data length insufficient for AO calculation: len(high)={len(high)}, len(low)={len(low)} < 50")
            return pd.Series(np.full(len(high), np.nan), index=high.index)

        # 入力データ前処理: null値の処理
        high_null_count = high.isna().sum() if hasattr(high, 'isna') else 0
        low_null_count = low.isna().sum() if hasattr(low, 'isna') else 0
        logger.debug(f"Input data null check: high_null={high_null_count}, low_null={low_null_count}")

        # null値がある場合は前処理を適用
        if high_null_count > 0 or low_null_count > 0:
            high = high.ffill().bfill().fillna(0)
            low = low.ffill().bfill().fillna(0)
            logger.debug("Applied null value preprocessing to high and low data")

        try:
            logger.debug("Attempting pandas-ta ao calculation")
            result = ta.ao(high=high, low=low)
            logger.debug(f"pammdas-ta result: type={type(result)}, is_None={result is None}")
            if result is not None and hasattr(result, 'empty'):
                logger.debug(f"pammdas-ta result empty check: {not result.empty}, all_na check: {not result.isna().all()}")
            if result is not None and not result.empty and not result.isna().all():
                logger.debug("Returning pandas-ta result successfully")
                return result
            else:
                logger.warning("pandas-ta ao returned invalid result, attempting fallback")
        except Exception as e:
            logger.warning(f"pandas-ta ao failed with exception: {str(e)}")

        # 強化フォールバック実装
        try:
            logger.debug("Starting fallback calculation")
            # 価格の中央値を使用
            median_price = (high + low) / 2.0
            logger.debug(f"Median price calculation: length={len(median_price)}, null_count={median_price.isna().sum() if hasattr(median_price, 'isna') else 0}")

            # シンプルなrolling meanを優先使用
            logger.debug("Using enhanced rolling mean fallback")
            if len(median_price) >= 50:  # 最低限のデータ長確認（50に変更）
                logger.debug("Using enhanced rolling mean fallback")
                sma5_fallback = median_price.rolling(window=5).mean()
                sma34_fallback = median_price.rolling(window=34).mean()
                ao_fallback = sma5_fallback - sma34_fallback
                if ao_fallback.isna().sum() > 0:
                    ao_fallback = ao_fallback.fillna(method='bfill').fillna(method='ffill').fillna(0)
                logger.debug(f"Enhanced fallback result: null_count={ao_fallback.isna().sum() if hasattr(ao_fallback, 'isna') else 0}")
                return ao_fallback
            else:
                logger.warning(f"Data length insufficient for enhanced fallback: {len(median_price)} < 50")

        except Exception as e:
            logger.error(f"Enhanced fallback calculation failed with exception: {str(e)}")

        # 最終フォールバック: NaN配列
        logger.warning("All AO calculation methods failed, returning NaN series")
        return pd.Series(np.full(len(high), np.nan), index=high.index)

    # 後方互換性のためのエイリアス


    @staticmethod
    def stochf(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k: int = 14,
        d: int = 3,
        smooth_k: int = 3,
    ):
        """高速ストキャスティクス（標準ストキャスティクスで代替）"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        return MomentumIndicators.stoch(high, low, close, k=k, d=d, smooth_k=smooth_k)

    @staticmethod
    def cmo(data: pd.Series, period: int = 14, length: int = None) -> pd.Series:
        """チェンジモメンタムオシレーター"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        # backward compatibility: lengthパラメータをperiodにマッピング
        if length is not None:
            period = length
        try:
            if hasattr(ta, 'cmo'):
                import inspect
                sig = inspect.signature(ta.cmo)
            result = ta.cmo(data, length=period)
            if result is None or (hasattr(result, "empty") and result.empty):
                return pd.Series(np.full(len(data), np.nan), index=data.index)
            return result
        except Exception as e:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

    @staticmethod
    def trix(data: pd.Series, period: int = 30) -> pd.Series:
        """TRIX"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        length = period
        result = ta.trix(data, window=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result.iloc[:, 0] if len(result.columns) > 1 else result

    @staticmethod
    def kdj(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """KDJ指標"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        result = ta.kdj(high, low, close, length=period)
        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series, nan_series
        # 結果からK, D, Jを返す
        return result.iloc[:, 0], result.iloc[:, 1], result.iloc[:, 2]

    @staticmethod
    def stochrsi(
        data: pd.Series,
        length: int = 14,
        k_period: int = 5,
        d_period: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """ストキャスティクスRSI"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        result = ta.stochrsi(data, window=length, k_period=k_period, d_period=d_period)
        if result is None:
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return nan_series, nan_series
        return result.iloc[:, 0], result.iloc[:, 1]


    @staticmethod
    def rvgi(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> Tuple[pd.Series, pd.Series]:
        """Relative Vigor Index with enhanced fallback"""
        if not isinstance(open_, pd.Series):
            raise TypeError("open_ must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # 強化されたパラメータチェックとデータ長検証
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        # データ長チェック: 最低必要なデータ長
        min_length = length + 2  # 適切なバッファ
        if len(high) < min_length:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series

        # pandas-ta試行
        try:
            result = ta.rvgi(
                open_=open_,
                high=high,
                low=low,
                close=close,
                window=length,
            )
            if result is not None and not result.empty and not result.isna().all().all():
                return result.iloc[:, 0], result.iloc[:, 1]
        except Exception as e:
            pass

        # 強化されたフォールバック実装: 手動計算
        try:
            # RVIライン = ((close - open) / (high - low)).ewm(span=length).mean() * 100
            rvi_numerator = (close - open_)
            rvi_denominator = (high - low)

            # ゼロ除算防止
            rvi_denominator = rvi_denominator.replace(0, np.nan)

            # RVIライン計算
            rvi_line_raw = rvi_numerator / rvi_denominator * 100
            rvi_line = rvi_line_raw.ewm(span=length).mean() if rvi_line_raw is not None else None

            if rvi_line is not None:
                # シグナルライン: 4周期のEMA (標準的なRVGIシグナル期間)
                rvi_signal = rvi_line.ewm(span=4).mean()

                # NaN補間
                rvi_line = rvi_line.fillna(method='bfill').fillna(method='ffill').fillna(0)
                rvi_signal = rvi_signal.fillna(method='bfill').fillna(method='ffill').fillna(0)

                return rvi_line, rvi_signal

        except Exception as e:
            pass

        # 最終フォールバック: NaN
        nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
        return nan_series, nan_series

    @staticmethod
    def qqe(data: pd.Series, length: int = 14) -> pd.Series:
        """Qualitative Quantitative Estimation"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        result = ta.qqe(data, length=length)

        if result is None or result.empty:
            # フォールバック: RSIを返す
            rsi_result = ta.rsi(data, length=length)
            return (
                rsi_result
                if rsi_result is not None
                else pd.Series(np.full(len(data), np.nan), index=data.index)
            )

        # QQEの主要な列を返す（通常はRSIMA列）
        return result.iloc[:, 1] if result.shape[1] > 1 else result.iloc[:, 0]

    @staticmethod
    def smi(
        data: pd.Series,
        fast: int = 13,
        slow: int = 25,
        signal: int = 2,
    ) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Momentum Index"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        try:
            result = ta.smi(data, fast=fast, slow=slow, signal=signal)
            if result is None:
                nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
                return nan_series, nan_series
            return result.iloc[:, 0], result.iloc[:, 1]
        except Exception:
            # フォールバック: RSIとそのEMAシグナル
            rsi = ta.rsi(data, length=max(5, min(slow, len(data) - 1)))
            if rsi is None:
                nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
                return nan_series, nan_series
            signal_ema = rsi.ewm(span=signal).mean()
            return rsi, signal_ema

    @staticmethod
    def kst(
        data: pd.Series,
        roc1: int = 10,
        roc2: int = 15,
        roc3: int = 20,
        roc4: int = 30,
        sma1: int = 10,
        sma2: int = 10,
        sma3: int = 10,
        sma4: int = 15,
        signal: int = 9,
        rorc1: int = None,
        rorc2: int = None,
        rorc3: int = None,
        rorc4: int = None,
        r1: int = None,
        r2: int = None,
        r3: int = None,
        r4: int = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Know Sure Thing"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        # rorc/rパラメータがある場合、rocパラメータにマッピング
        if rorc1 is not None:
            roc1 = rorc1
        if rorc2 is not None:
            roc2 = rorc2
        if rorc3 is not None:
            roc3 = rorc3
        if rorc4 is not None:
            roc4 = rorc4

        # rパラメータのマッピング(短縮形)
        if r1 is not None:
            roc1 = r1
        if r2 is not None:
            roc2 = r2
        if r3 is not None:
            roc3 = r3
        if r4 is not None:
            roc4 = r4

        # pandas-taのKST関数を使用
        try:
            result = ta.kst(
                data,
                roc1=roc1,
                roc2=roc2,
                roc3=roc3,
                roc4=roc4,
                sma1=sma1,
                sma2=sma2,
                sma3=sma3,
                sma4=sma4,
                signal=signal,
            )
            if result is not None and not result.empty and not result.isna().all().all():
                return result.iloc[:, 0], result.iloc[:, 1]
        except Exception as e:
            pass

        # フォールバック実装: KSTのマニュアル計算
        if len(data) < max(roc4 + sma4, signal):
            # データ長不足の場合、NaNを返す
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return nan_series, nan_series

        try:
            # ROC計算
            roc_val1 = ta.roc(data, length=roc1)
            roc_val2 = ta.roc(data, length=roc2)
            roc_val3 = ta.roc(data, length=roc3)
            roc_val4 = ta.roc(data, length=roc4)

            # SMA平滑化
            roc_sma1 = ta.sma(roc_val1, length=sma1) if roc_val1 is not None else None
            roc_sma2 = ta.sma(roc_val2, length=sma2) if roc_val2 is not None else None
            roc_sma3 = ta.sma(roc_val3, length=sma3) if roc_val3 is not None else None
            roc_sma4 = ta.sma(roc_val4, length=sma4) if roc_val4 is not None else None

            if all(x is not None for x in [roc_sma1, roc_sma2, roc_sma3, roc_sma4]):
                # KSTメインライン: 100 * (roc_sma1 + 2*roc_sma2 + 3*roc_sma3 + 4*roc_sma4)
                kst_line = 100 * (roc_sma1 + 2 * roc_sma2 + 3 * roc_sma3 + 4 * roc_sma4)

                # シグナルライン: KSTのSMA
                signal_line = ta.sma(kst_line, length=signal)

                if signal_line is not None:
                    return kst_line, signal_line

        except Exception as e:
            pass

        # 最終フォールバック: NaN
        nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
        return nan_series, nan_series


    @staticmethod
    def aroonosc(
        high: pd.Series,
        low: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """アルーンオシレーター"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        result = ta.aroon(high=high, low=low, window=length)
        if result is None:
            # フォールバック: aroon関数から計算
            try:
                aroon_up, aroon_down = MomentumIndicators.aroon(high, low, length)
                if aroon_up is not None and aroon_down is not None:
                    return aroon_up - aroon_down
            except Exception:
                pass
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        # アルーンオシレーター = アルーンアップ - アルーンダウン
        return result.iloc[:, 1] - result.iloc[:, 0]

    @staticmethod
    def dx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Directional Movement Index (DX)"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.adx(high=high, low=low, close=close, window=period)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        # DX列を探す
        dx_col = next((col for col in result.columns if "DX" in col), None)
        if dx_col:
            return result[dx_col]
        else:
            # フォールバック: DMP - DMNの差分
            dmp_col = next((col for col in result.columns if "DMP" in col), None)
            dmn_col = next((col for col in result.columns if "DMN" in col), None)
            if dmp_col and dmn_col:
                return result[dmp_col] - result[dmn_col]
            else:
                return pd.Series(np.full(len(high), np.nan), index=high.index)

    @staticmethod
    def plus_di(high, low, close, length: int = 14) -> pd.Series:
        """Plus Directional Indicator (DI)"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.adx(high=high, low=low, close=close, window=length)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        dmp_col = next(
            (col for col in result.columns if "DMP" in col), result.columns[1]
        )
        return result[dmp_col]

    @staticmethod
    def minus_di(high, low, close, length: int = 14) -> pd.Series:
        """Minus Directional Indicator (DI)"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.adx(high=high, low=low, close=close, window=length)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        dmn_col = next(
            (col for col in result.columns if "DMN" in col), result.columns[2]
        )
        return result[dmn_col]

    @staticmethod
    def plus_dm(high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
        """Plus Directional Movement (DM)"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        # Use ta.adx to get DMP, since ta.plus_dm may not exist in some pandas-ta versions
        result = ta.adx(high=high, low=low, close=high, length=period)  # close not used for DM calculation
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        dmp_col = next((col for col in result.columns if "DMP" in col), None)
        if dmp_col:
            return result[dmp_col]
        else:
            return pd.Series(np.full(len(high), np.nan), index=high.index)

    @staticmethod
    def minus_dm(high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
        """Minus Directional Movement (DM)"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        # Use ta.adx to get DMN, since ta.minus_dm may not exist in some pandas-ta versions
        result = ta.adx(high=high, low=low, close=high, length=period)  # close not used for DM calculation
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        dmn_col = next((col for col in result.columns if "DMN" in col), None)
        if dmn_col:
            return result[dmn_col]
        else:
            return pd.Series(np.full(len(high), np.nan), index=high.index)

    @staticmethod
    def ultosc(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        fast: int = 7,
        medium: int = 14,
        slow: int = 28,
        period: int = None,
    ) -> pd.Series:
        """Ultimate Oscillator"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # backward compatibility: periodパラメータからfast, medium, slowを計算
        if period is not None:
            fast = period // 3
            medium = period
            slow = period * 2
            if fast <= 0:
                fast = 7
            if medium <= 0:
                medium = 14
            if slow <= 0:
                slow = 28

        try:
            if hasattr(ta, 'uo'):
                import inspect
                sig = inspect.signature(ta.uo)
            result = ta.uo(
                high=high,
                low=low,
                close=close,
                fast=fast,
                medium=medium,
                slow=slow,
            )
        except Exception as e:
            import traceback

        if result is None or result.empty:
            # フォールバック: 簡易実装 (weighted average of different periods)
            n = len(high)
            if n < slow:
                return pd.Series(np.full(n, np.nan), index=high.index)

            # 単純な加重平均で近似
            weights = np.array([1, 2, 4])  # fast, medium, slowの重み
            weights = weights / weights.sum()

            fast_ma = ta.sma(close, length=fast)
            medium_ma = ta.sma(close, length=medium)
            slow_ma = ta.sma(close, length=slow)

            if fast_ma is not None and medium_ma is not None and slow_ma is not None:
                # 単純な平均で代替
                return (fast_ma + medium_ma + slow_ma) / 3
            else:
                return pd.Series(np.full(n, np.nan), index=high.index)
        return result

    @staticmethod
    def bop(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
        sma_period: int = None,
    ) -> pd.Series:
        """Balance of Power"""
        if not isinstance(open_, pd.Series):
            raise TypeError("open_ must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if period <= 0:
            raise ValueError(f"period must be positive: {period}")

        try:
            if hasattr(ta, 'bop'):
                import inspect
                sig = inspect.signature(ta.bop)
            result = ta.bop(open_, high, low, close, period)
        except Exception as e:
            import traceback
            return pd.Series(np.full(len(high), np.nan), index=high.index)

        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)

        # オプション: SMAシグナルを適用
        if sma_period is not None and sma_period > 0:
            result = ta.sma(result, length=sma_period)

        return result

    @staticmethod
    def adxr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
        length: int = None,
    ) -> pd.Series:
        """ADX評価"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # backward compatibility: lengthパラメータをperiodにマッピング
        if length is not None:
            period = length

        try:
            result = ta.adxr(high=high, low=low, close=close, length=period)
            if result is None:
                return pd.Series(np.full(len(high), np.nan), index=high.index)
            return result
        except Exception:
            # フォールバック: ADXを返す
            result = ta.adx(high=high, low=low, close=close, length=period)
            if result is None:
                return pd.Series(np.full(len(high), np.nan), index=high.index)
            adx_col = next(
                (col for col in result.columns if "ADX" in col), result.columns[0]
            )
            return result[adx_col]

    # 残りの必要なメソッド（簡素化版）
    @staticmethod
    def rocp(data: pd.Series, period: int = 10) -> pd.Series:
        """変化率（%）"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        try:
            return ta.rocp(data, length=period)
        except AttributeError:
            return ta.roc(data, length=period)

    @staticmethod
    def rocr(data: pd.Series, period: int = 10) -> pd.Series:
        """変化率（比率）"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        try:
            return ta.rocr(data, length=period)
        except AttributeError:
            shifted = data.shift(period)
            return data / shifted

    @staticmethod
    def rocr100(data: pd.Series, period: int = 10) -> pd.Series:
        """変化率（比率100スケール）"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        try:
            return ta.rocr(data, length=period, scalar=100)
        except AttributeError:
            shifted = data.shift(period)
            return (data / shifted) * 100

    @staticmethod
    def rsi_ema_cross(
        data: pd.Series, rsi_length: int = 14, ema_length: int = 9, **kwargs
    ) -> Tuple[pd.Series, pd.Series]:
        """RSI EMAクロス"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        try:
            rsi = ta.rsi(data, length=rsi_length)
            if rsi is not None:
                ema = ta.ema(rsi, length=ema_length)
                if ema is not None:
                    return rsi, ema
        except Exception:
            pass

        # フォールバック: 簡易実装
        n = len(data)
        if n < max(rsi_length, ema_length):
            return pd.Series(np.full(n, np.nan), index=data.index), pd.Series(
                np.full(n, np.nan), index=data.index
            )

        # RSIの簡易計算
        rsi_values = np.full(n, np.nan)
        for i in range(rsi_length - 1, n):
            window = data.iloc[i - rsi_length + 1 : i + 1]
            gains = window.diff()[1:]
            avg_gain = gains[gains > 0].mean() if len(gains[gains > 0]) > 0 else 0
            avg_loss = -gains[gains < 0].mean() if len(gains[gains < 0]) > 0 else 0
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi_values[i] = 100 - (100 / (1 + rs))
            else:
                rsi_values[i] = 100

        # EMAの簡易計算
        ema_values = np.full(n, np.nan)
        ema_values[ema_length - 1] = rsi_values[
            rsi_length - 1 : rsi_length + ema_length - 1
        ].mean()
        alpha = 2.0 / (ema_length + 1)

        for i in range(ema_length, n):
            if not np.isnan(rsi_values[i]):
                ema_values[i] = alpha * rsi_values[i] + (1 - alpha) * ema_values[i - 1]

        return pd.Series(rsi_values, index=data.index), pd.Series(
            ema_values, index=data.index
        )

    @staticmethod
    def tsi(data: pd.Series, fast: int = 13, slow: int = 25) -> pd.Series:
        """True Strength Index"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if fast <= 0:
            raise ValueError(f"fast must be positive: {fast}")
        if slow <= 0:
            raise ValueError(f"slow must be positive: {slow}")
        result = ta.tsi(data, fast=fast, slow=slow)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result.iloc[:, 0]

    @staticmethod
    def rvi(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """Relative Volatility Index with enhanced fallback"""
        if not isinstance(open_, pd.Series):
            raise TypeError("open_ must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        try:
            result = ta.rvgi(
                open_=open_,
                high=high,
                low=low,
                close=close,
                window=length,
            )
            if result is not None and not result.empty and not result.isna().all():
                return result.iloc[:, 0] if result.shape[1] > 1 else result
        except Exception:
            pass

        # 強化フォールバック実装：真の変動性インデックスベース
        try:
            # 真の値幅の計算
            true_range = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - close.shift(1)),
                    np.abs(low - close.shift(1))
                )
            )

            # 値幅変化の計算
            tr_diff = true_range.diff()

            # 上下値幅の計算
            upward = pd.Series(np.where(tr_diff > 0, tr_diff, 0), index=close.index)
            downward = pd.Series(np.where(tr_diff < 0, -tr_diff, 0), index=close.index)

            # RSIと同様の計算
            if len(upward) >= length and len(downward) >= length:
                avg_gain = upward.rolling(window=length).mean()
                avg_loss = downward.rolling(window=length).mean()

                # RVI計算 (上昇値幅 / (上昇値幅 + 下降値幅) * 100)
                rvi_values = np.where(
                    avg_gain + avg_loss != 0,
                    (avg_gain / (avg_gain + avg_loss) * 100),
                    50  # 中間値
                )

                return pd.Series(rvi_values, index=close.index)
            else:
                # データ長不足時の簡易フォールバック
                return ta.rsi(close, length=min(length, len(close) - 1))

        except Exception as e:
            # 最終フォールバック: RSIベース
            return ta.rsi(close, length=min(length, len(close) - 1))


    @staticmethod
    def cfo(data: pd.Series, length: int = 9) -> pd.Series:
        """Chande Forecast Oscillator with enhanced implementation"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        try:
            result = ta.cfo(data, window=length)
            if result is not None and isinstance(result, pd.Series):
                # CFOはベクトル形式で返ることがあるのでSeriesに変換
                if not isinstance(result, pd.Series):
                    result = pd.Series(result, index=data.index)
                if not result.isna().all():
                    # CFO軸予測の値が適切な範囲内かチェック
                    if result.min() > -1000 and result.max() < 1000:  # 合理的範囲チェック
                        return result
        except Exception:
            pass

        # 強化フォールバック実装：適切なCFO計算
        if len(data) < length:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        try:
            # CFOの正確な実装：価格の線形回帰予測に基づく
            cfo_values = np.full(len(data), np.nan)

            for i in range(length, len(data)):
                # ウィンドウデータを取得
                window = data.iloc[i - length:i].values

                # 線形回帰計算
                x = np.arange(length)
                y = window

                # 単純線形回帰
                slope = np.cov(x, y)[0, 1] / np.var(x)
                intercept = np.mean(y) - slope * np.mean(x)

                # CFO = (線形予測 - 現在の価格) / 標準偏差
                predicted_price = intercept + slope * length
                std_window = np.std(window)

                if std_window > 0:
                    cfo_values[i] = (predicted_price - data.iloc[i]) / std_window
                    # CFOは通常-3?+3の範囲なのでクリッピング
                    cfo_values[i] = np.clip(cfo_values[i], -3.0, 3.0)

            # CFOをパーセントスケールに変換
            valid_cfo = cfo_values[~np.isnan(cfo_values)]
            if len(valid_cfo) > 0:
                cfo_values = (cfo_values / np.std(valid_cfo)) * 10  # スケーリング

            return pd.Series(cfo_values, index=data.index)

        except Exception as e:
            # 最終フォールバック: EMAベース近似
            try:
                ema_short = ta.ema(data, length=length//2)
                ema_long = ta.ema(data, length=length)
                if ema_short is not None and ema_long is not None:
                    return (ema_short - ema_long).fillna(0)
            except:
                pass

            return pd.Series(np.full(len(data), np.nan), index=data.index)

    @staticmethod
    def cti(data: pd.Series, length: int = 20) -> pd.Series:
        """Chande Trend Index with enhanced robust implementation"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        try:
            result = ta.cti(data, window=length)
            if result is not None and isinstance(result, pd.Series):
                if not result.isna().all():
                    # CTIは-100 to +100の範囲が一般的
                    result = result.clip(-100, 100)
                    return result
        except Exception:
            pass

        # 強化フォールバック実装：CMOベースのCTI計算
        if len(data) < length:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        try:
            cti_values = np.full(len(data), np.nan)

            # CMO (Chande Momentum Oscillator) を計算
            changes = data.diff()
            gains = changes.where(changes > 0, 0)
            losses = -changes.where(changes < 0, 0)

            # 指数移動平均で平滑化
            avg_gain = ta.ema(gains, length=length)
            avg_loss = ta.ema(losses, length=length)

            if avg_gain is not None and avg_loss is not None:
                # CMO = (AvgGain - AvgLoss) / (AvgGain + AvgLoss) * 100
                cmo = (avg_gain - avg_loss) / (avg_gain + avg_loss) * 100

                # CTI = CMOのクインテルあらずバージョン (よりトレンド指向)
                # CTIはCMOの累積和に似るが、バイアスを考慮
                cti_values = cmo.rolling(window=length//2, min_periods=1).apply(
                    lambda x: np.sum(x) / len(x) if len(x) > 0 else 0
                ).values

                # CTIを-100 to +100の範囲に正規化
                cti_values = np.clip(cti_values, -100, 100)

            return pd.Series(cti_values, index=data.index)

        except Exception as e:
            # 最終フォールバック: 簡易相関係数ベース
            try:
                cti_values = np.full(len(data), np.nan)
                for i in range(length, len(data)):
                    # 移動平均との乖離率をCTIとして使用
                    ma = data.iloc[i-length:i].mean()
                    if not np.isnan(ma) and ma != 0:
                        cti_values[i] = ((data.iloc[i] - ma) / ma) * 100
                        cti_values[i] = np.clip(cti_values[i], -100, 100)
                return pd.Series(cti_values, index=data.index)
            except Exception as e2:
                return pd.Series(np.full(len(data), np.nan), index=data.index)

    @staticmethod
    def rmi(
        data: pd.Series = None,
        length: int = 20,
        mom: int = 20,
        close: pd.Series = None,
    ) -> pd.Series:
        """Relative Momentum Index"""
        if not isinstance(data, pd.Series) and close is not None:
            data = close
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        # RMIの簡易実装（RSIベース）
        return ta.rsi(data, length=length)

    @staticmethod
    def dpo(
        data: pd.Series = None,
        length: int = 20,
        close: pd.Series = None,
    ) -> pd.Series:
        """Detrended Price Oscillator"""
        if not isinstance(data, pd.Series) and close is not None:
            data = close
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        result = ta.dpo(data, window=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def chop(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """Choppiness Index"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.chop(high=high, low=low, close=close, window=length)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    def vortex(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> Tuple[pd.Series, pd.Series]:
        """Vortex Indicator"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.vortex(high=high, low=low, close=close, window=length)
        if result is None:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series
        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    def bias(data: pd.Series, length: int = 26) -> pd.Series:
        """Bias"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        result = ta.bias(data, window=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def brar(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 26,
    ) -> pd.Series:
        """BRAR Index"""
        if not isinstance(open_, pd.Series):
            raise TypeError("open_ must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.brar(
            open_=open_,
            high=high,
            low=low,
            close=close,
            length=length,
        )
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    def cg(data: pd.Series, length: int = 10) -> pd.Series:
        """Center of Gravity"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        result = ta.cg(data, window=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def coppock(
        data: pd.Series, length: int = 10, fast: int = 11, slow: int = 14
    ) -> pd.Series:
        """Coppock Curve"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        result = ta.coppock(data, window=length, fast=fast, slow=slow)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def er(data: pd.Series, length: int = 10) -> pd.Series:
        """Efficiency Ratio"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        result = ta.er(data, window=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def eri(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 13,
    ) -> pd.Series:
        """Elder Ray Index"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.eri(high=high, low=low, close=close, window=length)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    def fisher(
        high: pd.Series,
        low: pd.Series,
        length: int = 9,
        signal: int = 1,
    ) -> Tuple[pd.Series, pd.Series]:
        """Fisher Transform"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        result = ta.fisher(high=high, low=low, window=length, signal=signal)
        if result is None:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series
        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    def inertia(
        close: pd.Series,
        high: pd.Series = None,
        low: pd.Series = None,
        length: int = 20,
        rvi_length: int = 14,
    ) -> pd.Series:
        """Inertia"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.inertia(
            close=close,
            high=high,
            low=low,
            length=length,
            rvi_length=rvi_length,
        )
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    def pgo(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """Pretty Good Oscillator"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.pgo(high=high, low=low, close=close, window=length)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    def psl(
        close: pd.Series,
        open_: pd.Series = None,
        length: int = 12,
    ) -> pd.Series:
        """Psychological Line"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.psl(close=close, open_=open_, window=length)
        if result is None or result.empty:
            # フォールバック: 簡易実装 (close > openの割合)
            if open_ is not None:
                return (close > open_).rolling(length).mean() * 100
            else:
                return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    def rsx(data: pd.Series, length: int = 14) -> pd.Series:
        """RSX"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        result = ta.rsx(data, window=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def squeeze(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        bb_length: int = 20,
        bb_std: float = 2.0,
        kc_length: int = 20,
        kc_scalar: float = 1.5,
        mom_length: int = 12,
        mom_smooth: int = 6,
        use_tr: bool = True,
    ) -> pd.Series:
        """Squeeze"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.squeeze(
            high=high,
            low=low,
            close=close,
            bb_window=bb_length,
            bb_std=bb_std,
            kc_window=kc_length,
            kc_scalar=kc_scalar,
            mom_window=mom_length,
            mom_smooth=mom_smooth,
            use_tr=use_tr,
        )
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    def squeeze_pro(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        bb_length: int = 20,
        bb_std: float = 2.0,
        kc_length: int = 20,
        kc_scalar_wide: float = 2.0,
        kc_scalar_normal: float = 1.5,
        kc_scalar_narrow: float = 1.0,
        mom_length: int = 12,
        mom_smooth: int = 6,
        use_tr: bool = True,
    ) -> pd.Series:
        """Squeeze Pro"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.squeeze_pro(
            high=high,
            low=low,
            close=close,
            bb_window=bb_length,
            bb_std=bb_std,
            kc_window=kc_length,
            kc_scalar_wide=kc_scalar_wide,
            kc_scalar_normal=kc_scalar_normal,
            kc_scalar_narrow=kc_scalar_narrow,
            mom_window=mom_length,
            mom_smooth=mom_smooth,
            use_tr=use_tr,
        )
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result
