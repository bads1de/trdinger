"""
TA-Lib アダプタークラス

TA-Libライブラリと既存システムの橋渡しを行うアダプタークラスです。
pandas SeriesとTA-Lib間のデータ変換、エラーハンドリング、
後方互換性の確保を担当します。
"""

import logging
import talib
import pandas as pd
import numpy as np
from typing import Dict


logger = logging.getLogger(__name__)


class TALibCalculationError(Exception):
    """TA-Lib計算エラー"""


class TALibAdapter:
    """TA-Libと既存システムの橋渡しクラス"""

    @staticmethod
    def _validate_input(data: pd.Series, period: int) -> None:
        """
        入力データとパラメータの検証

        Args:
            data: 入力データ
            period: 期間

        Raises:
            TALibCalculationError: 入力が無効な場合
        """
        if data is None or len(data) == 0:
            raise TALibCalculationError("入力データが空です")

        if period <= 0:
            raise TALibCalculationError(f"期間は正の整数である必要があります: {period}")

        if len(data) < period:
            raise TALibCalculationError(
                f"データ長({len(data)})が期間({period})より短いです"
            )

    @staticmethod
    def _safe_talib_calculation(func, *args, **kwargs) -> np.ndarray:
        """
        TA-Lib計算の安全な実行

        Args:
            func: TA-Lib関数
            *args: 位置引数
            **kwargs: キーワード引数

        Returns:
            計算結果のnumpy配列

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise TALibCalculationError(f"TA-Lib計算エラー: {e}")

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average (単純移動平均) を計算

        Args:
            data: 価格データ（pandas Series）
            period: 移動平均の期間

        Returns:
            SMA値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TALibAdapter._validate_input(data, period)

        try:
            # TA-Libで計算
            result = TALibAdapter._safe_talib_calculation(
                talib.SMA, data.values, timeperiod=period
            )

            # pandas Seriesとして返す（元のインデックスを保持）
            return pd.Series(result, index=data.index, name=f"SMA_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"SMA計算でエラー: {e}")
            raise TALibCalculationError(f"SMA計算失敗: {e}")

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average (指数移動平均) を計算

        Args:
            data: 価格データ（pandas Series）
            period: 移動平均の期間

        Returns:
            EMA値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TALibAdapter._validate_input(data, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.EMA, data.values, timeperiod=period
            )

            return pd.Series(result, index=data.index, name=f"EMA_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"EMA計算でエラー: {e}")
            raise TALibCalculationError(f"EMA計算失敗: {e}")

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (相対力指数) を計算

        Args:
            data: 価格データ（pandas Series）
            period: RSIの期間（デフォルト: 14）

        Returns:
            RSI値のpandas Series（0-100の範囲）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TALibAdapter._validate_input(data, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.RSI, data.values, timeperiod=period
            )

            return pd.Series(result, index=data.index, name=f"RSI_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"RSI計算でエラー: {e}")
            raise TALibCalculationError(f"RSI計算失敗: {e}")

    @staticmethod
    def macd(
        data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Dict[str, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence) を計算

        Args:
            data: 価格データ（pandas Series）
            fast: 短期EMAの期間（デフォルト: 12）
            slow: 長期EMAの期間（デフォルト: 26）
            signal: シグナル線の期間（デフォルト: 9）

        Returns:
            MACD値を含む辞書（macd_line, signal_line, histogram）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TALibAdapter._validate_input(data, slow)  # 最も長い期間で検証

        try:
            macd_line, signal_line, histogram = TALibAdapter._safe_talib_calculation(
                talib.MACD,
                data.values,
                fastperiod=fast,
                slowperiod=slow,
                signalperiod=signal,
            )

            return {
                "macd_line": pd.Series(macd_line, index=data.index, name="MACD"),
                "signal_line": pd.Series(
                    signal_line, index=data.index, name="MACD_Signal"
                ),
                "histogram": pd.Series(
                    histogram, index=data.index, name="MACD_Histogram"
                ),
            }

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"MACD計算でエラー: {e}")
            raise TALibCalculationError(f"MACD計算失敗: {e}")

    @staticmethod
    def bollinger_bands(
        data: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> Dict[str, pd.Series]:
        """
        Bollinger Bands (ボリンジャーバンド) を計算

        Args:
            data: 価格データ（pandas Series）
            period: 移動平均の期間（デフォルト: 20）
            std_dev: 標準偏差の倍数（デフォルト: 2.0）

        Returns:
            ボリンジャーバンド値を含む辞書（upper, middle, lower）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TALibAdapter._validate_input(data, period)

        try:
            upper, middle, lower = TALibAdapter._safe_talib_calculation(
                talib.BBANDS,
                data.values,
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev,
            )

            return {
                "upper": pd.Series(upper, index=data.index, name=f"BB_Upper_{period}"),
                "middle": pd.Series(
                    middle, index=data.index, name=f"BB_Middle_{period}"
                ),
                "lower": pd.Series(lower, index=data.index, name=f"BB_Lower_{period}"),
            }

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"ボリンジャーバンド計算でエラー: {e}")
            raise TALibCalculationError(f"ボリンジャーバンド計算失敗: {e}")

    @staticmethod
    def atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Average True Range (平均真の値幅) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            period: ATRの期間（デフォルト: 14）

        Returns:
            ATR値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        # 全てのSeriesが同じ長さであることを確認
        if not (len(high) == len(low) == len(close)):
            raise TALibCalculationError("高値、安値、終値のデータ長が一致しません")

        TALibAdapter._validate_input(close, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.ATR, high.values, low.values, close.values, timeperiod=period
            )

            return pd.Series(result, index=close.index, name=f"ATR_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"ATR計算でエラー: {e}")
            raise TALibCalculationError(f"ATR計算失敗: {e}")

    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Dict[str, pd.Series]:
        """
        Stochastic Oscillator (ストキャスティクス) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            k_period: %Kの期間（デフォルト: 14）
            d_period: %Dの期間（デフォルト: 3）

        Returns:
            ストキャスティクス値を含む辞書（k_percent, d_percent）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        if not (len(high) == len(low) == len(close)):
            raise TALibCalculationError("高値、安値、終値のデータ長が一致しません")

        TALibAdapter._validate_input(close, k_period)

        try:
            k_percent, d_percent = TALibAdapter._safe_talib_calculation(
                talib.STOCH,
                high.values,
                low.values,
                close.values,
                fastk_period=k_period,
                slowk_period=d_period,
                slowd_period=d_period,
            )

            return {
                "k_percent": pd.Series(
                    k_percent, index=close.index, name=f"STOCH_K_{k_period}"
                ),
                "d_percent": pd.Series(
                    d_percent, index=close.index, name=f"STOCH_D_{d_period}"
                ),
            }

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"ストキャスティクス計算でエラー: {e}")
            raise TALibCalculationError(f"ストキャスティクス計算失敗: {e}")

    @staticmethod
    def cci(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
    ) -> pd.Series:
        """
        Commodity Channel Index (CCI) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            period: CCIの期間（デフォルト: 20）

        Returns:
            CCI値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        if not (len(high) == len(low) == len(close)):
            raise TALibCalculationError("高値、安値、終値のデータ長が一致しません")

        TALibAdapter._validate_input(close, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.CCI, high.values, low.values, close.values, timeperiod=period
            )

            return pd.Series(result, index=close.index, name=f"CCI_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"CCI計算でエラー: {e}")
            raise TALibCalculationError(f"CCI計算失敗: {e}")

    @staticmethod
    def williams_r(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Williams %R を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            period: Williams %Rの期間（デフォルト: 14）

        Returns:
            Williams %R値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        if not (len(high) == len(low) == len(close)):
            raise TALibCalculationError("高値、安値、終値のデータ長が一致しません")

        TALibAdapter._validate_input(close, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.WILLR, high.values, low.values, close.values, timeperiod=period
            )

            return pd.Series(result, index=close.index, name=f"WILLR_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"Williams %R計算でエラー: {e}")
            raise TALibCalculationError(f"Williams %R計算失敗: {e}")

    @staticmethod
    def adx(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Average Directional Movement Index (ADX) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            period: ADXの期間（デフォルト: 14）

        Returns:
            ADX値のpandas Series（0-100の範囲）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        if not (len(high) == len(low) == len(close)):
            raise TALibCalculationError("高値、安値、終値のデータ長が一致しません")

        TALibAdapter._validate_input(close, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.ADX, high.values, low.values, close.values, timeperiod=period
            )

            return pd.Series(result, index=close.index, name=f"ADX_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"ADX計算でエラー: {e}")
            raise TALibCalculationError(f"ADX計算失敗: {e}")

    @staticmethod
    def aroon(
        high: pd.Series, low: pd.Series, period: int = 14
    ) -> Dict[str, pd.Series]:
        """
        Aroon (アルーン) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            period: アルーンの期間（デフォルト: 14）

        Returns:
            アルーン値を含む辞書（aroon_down, aroon_up）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        if not (len(high) == len(low)):
            raise TALibCalculationError("高値、安値のデータ長が一致しません")

        TALibAdapter._validate_input(high, period)

        try:
            aroon_down, aroon_up = TALibAdapter._safe_talib_calculation(
                talib.AROON, high.values, low.values, timeperiod=period
            )

            return {
                "aroon_down": pd.Series(
                    aroon_down, index=high.index, name=f"AROON_DOWN_{period}"
                ),
                "aroon_up": pd.Series(
                    aroon_up, index=high.index, name=f"AROON_UP_{period}"
                ),
            }

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"AROON計算でエラー: {e}")
            raise TALibCalculationError(f"AROON計算失敗: {e}")

    @staticmethod
    def aroon_osc(high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
        """
        Aroon Oscillator (AROONOSC) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            period: 期間（デフォルト: 14）

        Returns:
            AROONOSC値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        if not (len(high) == len(low)):
            raise TALibCalculationError("高値、安値のデータ長が一致しません")

        TALibAdapter._validate_input(high, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.AROONOSC, high.values, low.values, timeperiod=period
            )

            return pd.Series(result, index=high.index, name=f"AROONOSC_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"AROONOSC計算でエラー: {e}")
            raise TALibCalculationError(f"AROONOSC計算失敗: {e}")

    @staticmethod
    def mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Money Flow Index (MFI) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            volume: 出来高データ（pandas Series）
            period: MFIの期間（デフォルト: 14）

        Returns:
            MFI値のpandas Series（0-100の範囲）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        if not (len(high) == len(low) == len(close) == len(volume)):
            raise TALibCalculationError(
                "高値、安値、終値、出来高のデータ長が一致しません"
            )

        TALibAdapter._validate_input(close, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.MFI,
                high.values,
                low.values,
                close.values,
                volume.values,
                timeperiod=period,
            )

            return pd.Series(result, index=close.index, name=f"MFI_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"MFI計算でエラー: {e}")
            raise TALibCalculationError(f"MFI計算失敗: {e}")

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On Balance Volume (OBV) を計算

        Args:
            close: 終値データ（pandas Series）
            volume: 出来高データ（pandas Series）

        Returns:
            OBV値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        if not (len(close) == len(volume)):
            raise TALibCalculationError("終値、出来高のデータ長が一致しません")

        if len(close) == 0:
            raise TALibCalculationError("入力データが空です")

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.OBV, close.values, volume.values
            )

            return pd.Series(result, index=close.index, name="OBV")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"OBV計算でエラー: {e}")
            raise TALibCalculationError(f"OBV計算失敗: {e}")

    @staticmethod
    def ad(
        high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """
        Chaikin A/D Line (Accumulation/Distribution Line) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            volume: 出来高データ（pandas Series）

        Returns:
            A/D Line値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        if not (len(high) == len(low) == len(close) == len(volume)):
            raise TALibCalculationError(
                "高値、安値、終値、出来高のデータ長が一致しません"
            )

        if len(close) == 0:
            raise TALibCalculationError("入力データが空です")

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.AD, high.values, low.values, close.values, volume.values
            )

            return pd.Series(result, index=close.index, name="AD")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"A/D Line計算でエラー: {e}")
            raise TALibCalculationError(f"A/D Line計算失敗: {e}")

    @staticmethod
    def adosc(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        fast_period: int = 3,
        slow_period: int = 10,
    ) -> pd.Series:
        """
        Chaikin A/D Oscillator (ADOSC) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            volume: 出来高データ（pandas Series）
            fast_period: 高速期間（デフォルト: 3）
            slow_period: 低速期間（デフォルト: 10）

        Returns:
            ADOSC値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        if not (len(high) == len(low) == len(close) == len(volume)):
            raise TALibCalculationError(
                "高値、安値、終値、出来高のデータ長が一致しません"
            )

        TALibAdapter._validate_input(close, slow_period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.ADOSC,
                high.values,
                low.values,
                close.values,
                volume.values,
                fastperiod=fast_period,
                slowperiod=slow_period,
            )

            return pd.Series(
                result, index=close.index, name=f"ADOSC_{fast_period}_{slow_period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"ADOSC計算でエラー: {e}")
            raise TALibCalculationError(f"ADOSC計算失敗: {e}")

    @staticmethod
    def kama(data: pd.Series, period: int = 30) -> pd.Series:
        """
        Kaufman Adaptive Moving Average (KAMA) を計算

        Args:
            data: 価格データ（pandas Series）
            period: KAMAの期間（デフォルト: 30）

        Returns:
            KAMA値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TALibAdapter._validate_input(data, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.KAMA, data.values, timeperiod=period
            )

            return pd.Series(result, index=data.index, name=f"KAMA_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"KAMA計算でエラー: {e}")
            raise TALibCalculationError(f"KAMA計算失敗: {e}")

    @staticmethod
    def t3(data: pd.Series, period: int = 5, vfactor: float = 0.7) -> pd.Series:
        """
        Triple Exponential Moving Average (T3) を計算

        Args:
            data: 価格データ（pandas Series）
            period: T3の期間（デフォルト: 5）
            vfactor: ボリュームファクター（デフォルト: 0.7）

        Returns:
            T3値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TALibAdapter._validate_input(data, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.T3, data.values, timeperiod=period, vfactor=vfactor
            )

            return pd.Series(result, index=data.index, name=f"T3_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"T3計算でエラー: {e}")
            raise TALibCalculationError(f"T3計算失敗: {e}")

    @staticmethod
    def tema(data: pd.Series, period: int = 30) -> pd.Series:
        """
        Triple Exponential Moving Average (TEMA) を計算

        Args:
            data: 価格データ（pandas Series）
            period: TEMAの期間（デフォルト: 30）

        Returns:
            TEMA値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TALibAdapter._validate_input(data, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.TEMA, data.values, timeperiod=period
            )

            return pd.Series(result, index=data.index, name=f"TEMA_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"TEMA計算でエラー: {e}")
            raise TALibCalculationError(f"TEMA計算失敗: {e}")

    @staticmethod
    def wma(data: pd.Series, period: int) -> pd.Series:
        """
        Weighted Moving Average (加重移動平均) を計算

        Args:
            data: 価格データ（pandas Series）
            period: 移動平均の期間

        Returns:
            WMA値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TALibAdapter._validate_input(data, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.WMA, data.values, timeperiod=period
            )

            return pd.Series(result, index=data.index, name=f"WMA_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"WMA計算でエラー: {e}")
            raise TALibCalculationError(f"WMA計算失敗: {e}")

    @staticmethod
    def trima(data: pd.Series, period: int) -> pd.Series:
        """
        Triangular Moving Average (三角移動平均) を計算

        Args:
            data: 価格データ（pandas Series）
            period: 移動平均の期間

        Returns:
            TRIMA値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TALibAdapter._validate_input(data, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.TRIMA, data.values, timeperiod=period
            )

            return pd.Series(result, index=data.index, name=f"TRIMA_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"TRIMA計算でエラー: {e}")
            raise TALibCalculationError(f"TRIMA計算失敗: {e}")

    @staticmethod
    def midpoint(data: pd.Series, period: int) -> pd.Series:
        """
        MidPoint over period (期間中の中点) を計算

        Args:
            data: 価格データ（pandas Series）
            period: 期間

        Returns:
            MIDPOINT値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TALibAdapter._validate_input(data, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.MIDPOINT, data.values, timeperiod=period
            )

            return pd.Series(result, index=data.index, name=f"MIDPOINT_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"MIDPOINT計算でエラー: {e}")
            raise TALibCalculationError(f"MIDPOINT計算失敗: {e}")

    @staticmethod
    def midprice(high: pd.Series, low: pd.Series, period: int) -> pd.Series:
        """
        Midpoint Price over period (期間中の高値と安値の中点価格) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            period: 期間

        Returns:
            MIDPRICE値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        if not (len(high) == len(low)):
            raise TALibCalculationError("高値と安値のデータ長が一致しません")

        TALibAdapter._validate_input(high, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.MIDPRICE, high.values, low.values, timeperiod=period
            )

            return pd.Series(result, index=high.index, name=f"MIDPRICE_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"MIDPRICE計算でエラー: {e}")
            raise TALibCalculationError(f"MIDPRICE計算失敗: {e}")

    @staticmethod
    def mama(data: pd.Series, fastlimit: float = 0.5, slowlimit: float = 0.05) -> dict:
        """
        MESA Adaptive Moving Average (MAMA) を計算

        Args:
            data: 価格データ（pandas Series）
            fastlimit: 高速制限（デフォルト: 0.5）
            slowlimit: 低速制限（デフォルト: 0.05）

        Returns:
            MAMA値を含む辞書（mama, fama）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TALibAdapter._validate_input(data, 1)  # 最小期間チェック

        try:
            mama_result, fama_result = TALibAdapter._safe_talib_calculation(
                talib.MAMA, data.values, fastlimit=fastlimit, slowlimit=slowlimit
            )

            return {
                "mama": pd.Series(mama_result, index=data.index, name="MAMA"),
                "fama": pd.Series(fama_result, index=data.index, name="FAMA"),
            }

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"MAMA計算でエラー: {e}")
            raise TALibCalculationError(f"MAMA計算失敗: {e}")

    @staticmethod
    def hma(data: pd.Series, period: int) -> pd.Series:
        """
        Hull Moving Average (ハル移動平均) を計算

        HMAはTA-Libに直接実装されていないため、WMAを使用して計算します。
        HMA = WMA(2*WMA(data, period/2) - WMA(data, period), sqrt(period))

        Args:
            data: 価格データ（pandas Series）
            period: 期間

        Returns:
            HMA値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TALibAdapter._validate_input(data, period)

        try:
            import math

            # HMA計算のためのWMA計算
            half_period = max(1, period // 2)
            sqrt_period = max(1, int(math.sqrt(period)))

            wma_half = TALibAdapter._safe_talib_calculation(
                talib.WMA, data.values, timeperiod=half_period
            )
            wma_full = TALibAdapter._safe_talib_calculation(
                talib.WMA, data.values, timeperiod=period
            )

            # 2*WMA(period/2) - WMA(period)
            diff_data = 2 * wma_half - wma_full

            # 最終的なWMA計算
            result = TALibAdapter._safe_talib_calculation(
                talib.WMA, diff_data, timeperiod=sqrt_period
            )

            return pd.Series(result, index=data.index, name=f"HMA_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"HMA計算でエラー: {e}")
            raise TALibCalculationError(f"HMA計算失敗: {e}")

    @staticmethod
    def vwma(data: pd.Series, volume: pd.Series, period: int) -> pd.Series:
        """
        Volume Weighted Moving Average (出来高加重移動平均) を計算

        VWMAはTA-Libに直接実装されていないため、手動で計算します。

        Args:
            data: 価格データ（pandas Series）
            volume: 出来高データ（pandas Series）
            period: 期間

        Returns:
            VWMA値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        if not (len(data) == len(volume)):
            raise TALibCalculationError("価格と出来高のデータ長が一致しません")

        TALibAdapter._validate_input(data, period)

        try:
            # 価格×出来高の積
            pv = data * volume

            # 移動平均計算
            pv_sum = pv.rolling(window=period).sum()
            volume_sum = volume.rolling(window=period).sum()

            # VWMA = sum(price * volume) / sum(volume)
            result = pv_sum / volume_sum

            return pd.Series(result, index=data.index, name=f"VWMA_{period}")

        except Exception as e:
            logger.error(f"VWMA計算でエラー: {e}")
            raise TALibCalculationError(f"VWMA計算失敗: {e}")

    @staticmethod
    def zlema(data: pd.Series, period: int) -> pd.Series:
        """
        Zero Lag Exponential Moving Average (ゼロラグ指数移動平均) を計算

        ZLEMAはTA-Libに直接実装されていないため、EMAを使用して計算します。

        Args:
            data: 価格データ（pandas Series）
            period: 期間

        Returns:
            ZLEMA値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TALibAdapter._validate_input(data, period)

        try:
            # ラグ計算
            lag = (period - 1) // 2

            # ラグ調整されたデータ
            if lag > 0:
                adjusted_data = 2 * data - data.shift(lag)
            else:
                adjusted_data = data

            # EMA計算
            result = TALibAdapter._safe_talib_calculation(
                talib.EMA,
                adjusted_data.bfill().values,
                timeperiod=period,
            )

            return pd.Series(result, index=data.index, name=f"ZLEMA_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"ZLEMA計算でエラー: {e}")
            raise TALibCalculationError(f"ZLEMA計算失敗: {e}")

    @staticmethod
    def plus_di(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Plus Directional Indicator (PLUS_DI) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            period: 期間（デフォルト: 14）

        Returns:
            PLUS_DI値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        if not (len(high) == len(low) == len(close)):
            raise TALibCalculationError("高値、安値、終値のデータ長が一致しません")

        TALibAdapter._validate_input(close, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.PLUS_DI, high.values, low.values, close.values, timeperiod=period
            )

            return pd.Series(result, index=close.index, name=f"PLUS_DI_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"PLUS_DI計算でエラー: {e}")
            raise TALibCalculationError(f"PLUS_DI計算失敗: {e}")

    @staticmethod
    def minus_di(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Minus Directional Indicator (MINUS_DI) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            period: 期間（デフォルト: 14）

        Returns:
            MINUS_DI値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        if not (len(high) == len(low) == len(close)):
            raise TALibCalculationError("高値、安値、終値のデータ長が一致しません")

        TALibAdapter._validate_input(close, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.MINUS_DI, high.values, low.values, close.values, timeperiod=period
            )

            return pd.Series(result, index=close.index, name=f"MINUS_DI_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"MINUS_DI計算でエラー: {e}")
            raise TALibCalculationError(f"MINUS_DI計算失敗: {e}")

    @staticmethod
    def rocp(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Rate of change Percentage (ROCP) を計算

        Args:
            data: 価格データ（pandas Series）
            period: 期間（デフォルト: 10）

        Returns:
            ROCP値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TALibAdapter._validate_input(data, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.ROCP, data.values, timeperiod=period
            )

            return pd.Series(result, index=data.index, name=f"ROCP_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"ROCP計算でエラー: {e}")
            raise TALibCalculationError(f"ROCP計算失敗: {e}")

    @staticmethod
    def rocr(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Rate of change ratio (ROCR) を計算

        Args:
            data: 価格データ（pandas Series）
            period: 期間（デフォルト: 10）

        Returns:
            ROCR値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TALibAdapter._validate_input(data, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.ROCR, data.values, timeperiod=period
            )

            return pd.Series(result, index=data.index, name=f"ROCR_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"ROCR計算でエラー: {e}")
            raise TALibCalculationError(f"ROCR計算失敗: {e}")

    @staticmethod
    def stochf(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        fastk_period: int = 5,
        fastd_period: int = 3,
        fastd_matype: int = 0,
    ) -> dict:
        """
        Stochastic Fast (STOCHF) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            fastk_period: Fast %K期間（デフォルト: 5）
            fastd_period: Fast %D期間（デフォルト: 3）
            fastd_matype: Fast %D移動平均タイプ（デフォルト: 0=SMA）

        Returns:
            STOCHF値を含む辞書（fastk, fastd）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        if not (len(high) == len(low) == len(close)):
            raise TALibCalculationError("高値、安値、終値のデータ長が一致しません")

        TALibAdapter._validate_input(close, max(fastk_period, fastd_period))

        try:
            fastk, fastd = TALibAdapter._safe_talib_calculation(
                talib.STOCHF,
                high.values,
                low.values,
                close.values,
                fastk_period=fastk_period,
                fastd_period=fastd_period,
                fastd_matype=fastd_matype,
            )

            return {
                "fastk": pd.Series(fastk, index=close.index, name="STOCHF_K"),
                "fastd": pd.Series(fastd, index=close.index, name="STOCHF_D"),
            }

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"STOCHF計算でエラー: {e}")
            raise TALibCalculationError(f"STOCHF計算失敗: {e}")

    @staticmethod
    def dema(data: pd.Series, period: int = 30) -> pd.Series:
        """
        Double Exponential Moving Average (DEMA) を計算

        Args:
            data: 価格データ（pandas Series）
            period: DEMAの期間（デフォルト: 30）

        Returns:
            DEMA値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TALibAdapter._validate_input(data, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.DEMA, data.values, timeperiod=period
            )

            return pd.Series(result, index=data.index, name=f"DEMA_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"DEMA計算でエラー: {e}")
            raise TALibCalculationError(f"DEMA計算失敗: {e}")

    @staticmethod
    def natr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Normalized Average True Range (NATR) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            period: NATRの期間（デフォルト: 14）

        Returns:
            NATR値のpandas Series（パーセンテージ）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        if not (len(high) == len(low) == len(close)):
            raise TALibCalculationError("高値、安値、終値のデータ長が一致しません")

        TALibAdapter._validate_input(close, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.NATR, high.values, low.values, close.values, timeperiod=period
            )

            return pd.Series(result, index=close.index, name=f"NATR_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"NATR計算でエラー: {e}")
            raise TALibCalculationError(f"NATR計算失敗: {e}")

    @staticmethod
    def trange(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        True Range (TRANGE) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）

        Returns:
            True Range値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        if not (len(high) == len(low) == len(close)):
            raise TALibCalculationError("高値、安値、終値のデータ長が一致しません")

        if len(close) == 0:
            raise TALibCalculationError("入力データが空です")

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.TRANGE, high.values, low.values, close.values
            )

            return pd.Series(result, index=close.index, name="TRANGE")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"TRANGE計算でエラー: {e}")
            raise TALibCalculationError(f"TRANGE計算失敗: {e}")

    @staticmethod
    def momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Momentum (モメンタム) を計算

        Args:
            data: 価格データ（pandas Series）
            period: モメンタムの期間（デフォルト: 10）

        Returns:
            モメンタム値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TALibAdapter._validate_input(data, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.MOM, data.values, timeperiod=period
            )

            return pd.Series(result, index=data.index, name=f"MOM_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"モメンタム計算でエラー: {e}")
            raise TALibCalculationError(f"モメンタム計算失敗: {e}")

    @staticmethod
    def roc(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Rate of Change (ROC) を計算

        Args:
            data: 価格データ（pandas Series）
            period: ROCの期間（デフォルト: 10）

        Returns:
            ROC値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TALibAdapter._validate_input(data, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.ROC, data.values, timeperiod=period
            )

            return pd.Series(result, index=data.index, name=f"ROC_{period}")

        except TALibCalculationError:
            raise
        except Exception as e:
            logger.error(f"ROC計算でエラー: {e}")
            raise TALibCalculationError(f"ROC計算失敗: {e}")

    @staticmethod
    def stochastic_rsi(
        data: pd.Series, period: int, fastk_period: int, fastd_period: int
    ) -> Dict[str, pd.Series]:
        TALibAdapter._validate_input(data, period)
        try:
            fastk, fastd = TALibAdapter._safe_talib_calculation(
                talib.STOCHRSI,
                data.values,
                timeperiod=period,
                fastk_period=fastk_period,
                fastd_period=fastd_period,
            )
            return {
                "fastk": pd.Series(fastk, index=data.index, name="STOCHRSI_K"),
                "fastd": pd.Series(fastd, index=data.index, name="STOCHRSI_D"),
            }
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"Stochastic RSI計算失敗: {e}")

    @staticmethod
    def ultimate_oscillator(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period1: int,
        period2: int,
        period3: int,
    ) -> pd.Series:
        TALibAdapter._validate_input(close, max(period1, period2, period3))
        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.ULTOSC,
                high.values,
                low.values,
                close.values,
                timeperiod1=period1,
                timeperiod2=period2,
                timeperiod3=period3,
            )
            return pd.Series(result, index=close.index, name="ULTOSC")
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"Ultimate Oscillator計算失敗: {e}")

    @staticmethod
    def cmo(data: pd.Series, period: int) -> pd.Series:
        TALibAdapter._validate_input(data, period)
        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.CMO, data.values, timeperiod=period
            )
            return pd.Series(result, index=data.index, name=f"CMO_{period}")
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"CMO計算失敗: {e}")

    @staticmethod
    def trix(data: pd.Series, period: int) -> pd.Series:
        TALibAdapter._validate_input(data, period)
        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.TRIX, data.values, timeperiod=period
            )
            return pd.Series(result, index=data.index, name=f"TRIX_{period}")
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"TRIX計算失敗: {e}")

    @staticmethod
    def bop(
        open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        if not (len(open) == len(high) == len(low) == len(close)):
            raise TALibCalculationError("OHLCデータの長さが一致しません")
        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.BOP, open.values, high.values, low.values, close.values
            )
            return pd.Series(result, index=close.index, name="BOP")
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"BOP計算失敗: {e}")

    @staticmethod
    def apo(
        data: pd.Series, fast_period: int, slow_period: int, matype: int
    ) -> pd.Series:
        TALibAdapter._validate_input(data, slow_period)
        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.APO,
                data.values,
                fastperiod=fast_period,
                slowperiod=slow_period,
                matype=matype,
            )
            return pd.Series(result, index=data.index, name="APO")
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"APO計算失敗: {e}")

    @staticmethod
    def ppo(
        data: pd.Series, fast_period: int, slow_period: int, matype: int
    ) -> pd.Series:
        TALibAdapter._validate_input(data, slow_period)
        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.PPO,
                data.values,
                fastperiod=fast_period,
                slowperiod=slow_period,
                matype=matype,
            )
            return pd.Series(result, index=data.index, name="PPO")
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"PPO計算失敗: {e}")

    @staticmethod
    def dx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        TALibAdapter._validate_input(close, period)
        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.DX, high.values, low.values, close.values, timeperiod=period
            )
            return pd.Series(result, index=close.index, name=f"DX_{period}")
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"DX計算失敗: {e}")

    @staticmethod
    def adxr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        TALibAdapter._validate_input(close, period)
        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.ADXR, high.values, low.values, close.values, timeperiod=period
            )
            return pd.Series(result, index=close.index, name=f"ADXR_{period}")
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"ADXR計算失敗: {e}")


# 後方互換性のためのヘルパー関数
def safe_talib_calculation(func, *args, **kwargs):
    """
    TA-Lib計算の安全な実行（後方互換性用）

    Args:
        func: TA-Lib関数
        *args: 位置引数
        **kwargs: キーワード引数

    Returns:
        計算結果

    Raises:
        TALibCalculationError: 計算エラーの場合
    """
    return TALibAdapter._safe_talib_calculation(func, *args, **kwargs)
