"""
TA-Lib アダプタークラス

TA-Libライブラリと既存システムの橋渡しを行うアダプタークラスです。
pandas SeriesとTA-Lib間のデータ変換、エラーハンドリング、
後方互換性の確保を担当します。
"""

import talib
import pandas as pd
import numpy as np
from typing import Union, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TALibCalculationError(Exception):
    """TA-Lib計算エラー"""

    pass


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
