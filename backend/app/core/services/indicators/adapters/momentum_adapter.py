"""
モメンタム系テクニカル指標アダプター

オシレーター系やモメンタム系の指標計算を担当します。
RSI, MACD, Stochastic, CCI, Williams %R, ADX, Aroon, MOM, ROC, MFI などを提供します。
"""

import talib
import pandas as pd
from typing import Dict
import logging

from .base_adapter import BaseAdapter, TALibCalculationError

logger = logging.getLogger(__name__)


class MomentumAdapter(BaseAdapter):
    """モメンタム系指標のTA-Libアダプター"""

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
        MomentumAdapter._validate_input(data, period)
        MomentumAdapter._log_calculation_start("RSI", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.RSI, data.values, timeperiod=period
            )

            # JSON形式対応の名前生成
            parameters = {"period": period}
            return MomentumAdapter._create_series_result_with_config(
                result, data.index, "RSI", parameters
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("RSI", e)
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
        MomentumAdapter._validate_input(data, slow)  # 最も長い期間で検証
        MomentumAdapter._log_calculation_start(
            "MACD", fast=fast, slow=slow, signal=signal
        )

        try:
            macd_line, signal_line, histogram = MomentumAdapter._safe_talib_calculation(
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
            MomentumAdapter._log_calculation_error("MACD", e)
            raise TALibCalculationError(f"MACD計算失敗: {e}")

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
        MomentumAdapter._validate_multi_input(high, low, close)
        MomentumAdapter._validate_input(close, k_period)
        MomentumAdapter._log_calculation_start(
            "STOCH", k_period=k_period, d_period=d_period
        )

        try:
            k_percent, d_percent = MomentumAdapter._safe_talib_calculation(
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
            MomentumAdapter._log_calculation_error("STOCH", e)
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
        MomentumAdapter._validate_multi_input(high, low, close)
        MomentumAdapter._validate_input(close, period)
        MomentumAdapter._log_calculation_start("CCI", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.CCI, high.values, low.values, close.values, timeperiod=period
            )
            return MomentumAdapter._create_series_result(
                result, close.index, f"CCI_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("CCI", e)
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
        MomentumAdapter._validate_multi_input(high, low, close)
        MomentumAdapter._validate_input(close, period)
        MomentumAdapter._log_calculation_start("WILLR", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.WILLR, high.values, low.values, close.values, timeperiod=period
            )
            return MomentumAdapter._create_series_result(
                result, close.index, f"WILLR_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("WILLR", e)
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
        MomentumAdapter._validate_multi_input(high, low, close)
        MomentumAdapter._validate_input(close, period)
        MomentumAdapter._log_calculation_start("ADX", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.ADX, high.values, low.values, close.values, timeperiod=period
            )
            return MomentumAdapter._create_series_result(
                result, close.index, f"ADX_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("ADX", e)
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
        MomentumAdapter._validate_multi_input(high, low)
        MomentumAdapter._validate_input(high, period)
        MomentumAdapter._log_calculation_start("AROON", period=period)

        try:
            aroon_down, aroon_up = MomentumAdapter._safe_talib_calculation(
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
            MomentumAdapter._log_calculation_error("AROON", e)
            raise TALibCalculationError(f"アルーン計算失敗: {e}")

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
        MomentumAdapter._validate_input(data, period)
        MomentumAdapter._log_calculation_start("MOM", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.MOM, data.values, timeperiod=period
            )

            # JSON形式対応の名前生成
            parameters = {"period": period}
            return MomentumAdapter._create_series_result_with_config(
                result, data.index, "MOM", parameters
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("MOM", e)
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
        MomentumAdapter._validate_input(data, period)
        MomentumAdapter._log_calculation_start("ROC", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.ROC, data.values, timeperiod=period
            )
            return MomentumAdapter._create_series_result(
                result, data.index, f"ROC_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("ROC", e)
            raise TALibCalculationError(f"ROC計算失敗: {e}")

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
        MomentumAdapter._validate_multi_input(high, low, close, volume)
        MomentumAdapter._validate_input(close, period)
        MomentumAdapter._log_calculation_start("MFI", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.MFI,
                high.values,
                low.values,
                close.values,
                volume.values,
                timeperiod=period,
            )
            return MomentumAdapter._create_series_result(
                result, close.index, f"MFI_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("MFI", e)
            raise TALibCalculationError(f"MFI計算失敗: {e}")

    @staticmethod
    def stochastic_rsi(
        close: pd.Series, period: int, fastk_period: int = 3, fastd_period: int = 3
    ) -> pd.DataFrame:
        """
        Stochastic RSI (ストキャスティクスRSI) を計算

        Stochastic RSI = Stochastic(RSI(close, period), fastk_period, fastd_period)

        Args:
            close: 終値データ（pandas Series）
            period: RSI期間
            fastk_period: Fast %K期間（デフォルト: 3）
            fastd_period: Fast %D期間（デフォルト: 3）

        Returns:
            Stochastic RSIのpandas DataFrame (fastk, fastd)

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        MomentumAdapter._validate_input(close, period)
        MomentumAdapter._log_calculation_start(
            "STOCHRSI",
            period=period,
            fastk_period=fastk_period,
            fastd_period=fastd_period,
        )

        try:
            # 最小データ数の確認（RSI + Stochastic計算のため）
            min_required = period + max(fastk_period, fastd_period) + 10
            if len(close) < min_required:
                raise TALibCalculationError(
                    f"Stochastic RSI計算には最低{min_required}個のデータが必要です（現在: {len(close)}個）"
                )

            # Step 1: RSI計算
            rsi_values = MomentumAdapter._safe_talib_calculation(
                talib.RSI, close.values, timeperiod=period
            )

            # Step 2: RSI値にStochasticを適用
            # RSI値を高値・安値・終値として使用
            fastk, fastd = MomentumAdapter._safe_talib_calculation(
                talib.STOCH,
                rsi_values,  # high
                rsi_values,  # low
                rsi_values,  # close
                fastk_period=fastk_period,
                slowk_period=1,  # Fast Stochasticなので1
                slowk_matype=0,  # SMA
                slowd_period=fastd_period,
                slowd_matype=0,  # SMA
            )

            # 結果のDataFrame作成
            result = pd.DataFrame({"fastk": fastk, "fastd": fastd}, index=close.index)

            return result

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("STOCHRSI", e)
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
        """
        Ultimate Oscillator (アルティメットオシレーター) を計算

        Ultimate Oscillator = 100 * [(4*BP1/TR1) + (2*BP2/TR2) + (BP3/TR3)] / (4+2+1)
        BP = Buying Pressure = Close - min(Low, Previous Close)
        TR = True Range

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            period1: 短期期間（通常7）
            period2: 中期期間（通常14）
            period3: 長期期間（通常28）

        Returns:
            Ultimate Oscillator値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        MomentumAdapter._validate_input(close, max(period1, period2, period3))
        MomentumAdapter._log_calculation_start(
            "ULTOSC", period1=period1, period2=period2, period3=period3
        )

        try:
            # データの長さチェック
            data_lengths = [len(high), len(low), len(close)]
            if len(set(data_lengths)) > 1:
                raise TALibCalculationError(
                    f"全てのデータの長さが一致しません（高値: {len(high)}, 安値: {len(low)}, 終値: {len(close)}）"
                )

            # 最小データ数の確認
            min_required = max(period1, period2, period3) + 10
            if len(close) < min_required:
                raise TALibCalculationError(
                    f"Ultimate Oscillator計算には最低{min_required}個のデータが必要です（現在: {len(close)}個）"
                )

            # TA-Libを使用したUltimate Oscillator計算
            result = MomentumAdapter._safe_talib_calculation(
                talib.ULTOSC,
                high.values,
                low.values,
                close.values,
                timeperiod1=period1,
                timeperiod2=period2,
                timeperiod3=period3,
            )

            return MomentumAdapter._create_series_result(
                result, close.index, f"ULTOSC_{period1}_{period2}_{period3}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("ULTOSC", e)
            raise TALibCalculationError(f"Ultimate Oscillator計算失敗: {e}")

    @staticmethod
    def cmo(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Chande Momentum Oscillator (CMO) を計算

        CMO = 100 * (Sum(Up) - Sum(Down)) / (Sum(Up) + Sum(Down))
        Up = 前日比上昇分, Down = 前日比下降分

        Args:
            data: 価格データ（pandas Series）
            period: CMOの期間（デフォルト: 14）

        Returns:
            CMO値のpandas Series（-100から100の範囲）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        MomentumAdapter._validate_input(data, period)
        MomentumAdapter._log_calculation_start("CMO", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.CMO, data.values, timeperiod=period
            )
            return MomentumAdapter._create_series_result(
                result, data.index, f"CMO_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("CMO", e)
            raise TALibCalculationError(f"CMO計算失敗: {e}")

    @staticmethod
    def trix(data: pd.Series, period: int = 30) -> pd.Series:
        """
        TRIX (1-day Rate-Of-Change of a Triple Smooth EMA) を計算

        TRIX = 1日前のTEMAに対する現在のTEMAの変化率（%）
        TEMA = Triple Exponential Moving Average

        Args:
            data: 価格データ（pandas Series）
            period: TRIXの期間（デフォルト: 30）

        Returns:
            TRIX値のpandas Series（パーセンテージ値）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        MomentumAdapter._validate_input(data, period)
        MomentumAdapter._log_calculation_start("TRIX", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.TRIX, data.values, timeperiod=period
            )
            return MomentumAdapter._create_series_result(
                result, data.index, f"TRIX_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("TRIX", e)
            raise TALibCalculationError(f"TRIX計算失敗: {e}")

    @staticmethod
    def bop(
        open_prices: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """
        BOP（Balance Of Power）を計算

        Args:
            open_prices: 始値のSeries
            high: 高値のSeries
            low: 安値のSeries
            close: 終値のSeries

        Returns:
            BOP値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        MomentumAdapter._log_calculation_start("BOP")

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.BOP, open_prices.values, high.values, low.values, close.values
            )
            return MomentumAdapter._create_series_result(result, close.index, "BOP")
        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("BOP", e)
            raise TALibCalculationError(f"BOP計算失敗: {e}")

    @staticmethod
    def apo(
        close: pd.Series, fast_period: int, slow_period: int, matype: int = 0
    ) -> pd.Series:
        """
        APO（Absolute Price Oscillator）を計算

        Args:
            close: 終値のSeries
            fast_period: 短期期間
            slow_period: 長期期間
            matype: 移動平均タイプ（0=SMA, 1=EMA, etc.）

        Returns:
            APO値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        MomentumAdapter._validate_input(close, fast_period)
        MomentumAdapter._log_calculation_start(
            "APO", fast_period=fast_period, slow_period=slow_period, matype=matype
        )

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.APO,
                close.values,
                fastperiod=fast_period,
                slowperiod=slow_period,
                matype=matype,
            )

            # JSON形式対応の名前生成
            parameters = {
                "fast_period": fast_period,
                "slow_period": slow_period,
                "matype": matype,
            }
            return MomentumAdapter._create_series_result_with_config(
                result, close.index, "APO", parameters
            )
        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("APO", e)
            raise TALibCalculationError(f"APO計算失敗: {e}")

    @staticmethod
    def ppo(
        close: pd.Series, fast_period: int, slow_period: int, matype: int = 0
    ) -> pd.Series:
        """
        PPO（Percentage Price Oscillator）を計算

        Args:
            close: 終値のSeries
            fast_period: 短期期間
            slow_period: 長期期間
            matype: 移動平均タイプ（0=SMA, 1=EMA, etc.）

        Returns:
            PPO値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        MomentumAdapter._validate_input(close, fast_period)
        MomentumAdapter._log_calculation_start(
            "PPO", fast_period=fast_period, slow_period=slow_period, matype=matype
        )

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.PPO,
                close.values,
                fastperiod=fast_period,
                slowperiod=slow_period,
                matype=matype,
            )

            # JSON形式対応の名前生成
            parameters = {
                "fast_period": fast_period,
                "slow_period": slow_period,
                "matype": matype,
            }
            return MomentumAdapter._create_series_result_with_config(
                result, close.index, "PPO", parameters
            )
        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("PPO", e)
            raise TALibCalculationError(f"PPO計算失敗: {e}")

    @staticmethod
    def plus_di(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
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
        MomentumAdapter._validate_multi_input(high, low, close)
        MomentumAdapter._log_calculation_start("PLUS_DI", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.PLUS_DI,
                high.values,
                low.values,
                close.values,
                timeperiod=period,
            )
            return MomentumAdapter._create_series_result(
                result, close.index, f"PLUS_DI_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("PLUS_DI", e)
            raise TALibCalculationError(f"PLUS_DI計算失敗: {e}")

    @staticmethod
    def minus_di(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
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
        MomentumAdapter._validate_multi_input(high, low, close)
        MomentumAdapter._log_calculation_start("MINUS_DI", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.MINUS_DI,
                high.values,
                low.values,
                close.values,
                timeperiod=period,
            )
            return MomentumAdapter._create_series_result(
                result, close.index, f"MINUS_DI_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("MINUS_DI", e)
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
        MomentumAdapter._validate_input(data, period)
        MomentumAdapter._log_calculation_start("ROCP", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.ROCP, data.values, timeperiod=period
            )
            return MomentumAdapter._create_series_result(
                result, data.index, f"ROCP_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("ROCP", e)
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
        MomentumAdapter._validate_input(data, period)
        MomentumAdapter._log_calculation_start("ROCR", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.ROCR, data.values, timeperiod=period
            )
            return MomentumAdapter._create_series_result(
                result, data.index, f"ROCR_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("ROCR", e)
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
            fastk_period: FastK期間（デフォルト: 5）
            fastd_period: FastD期間（デフォルト: 3）
            fastd_matype: FastD移動平均タイプ（デフォルト: 0=SMA）

        Returns:
            STOCHF値を含む辞書（fastk, fastd）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        MomentumAdapter._validate_multi_input(high, low, close)
        MomentumAdapter._log_calculation_start(
            "STOCHF",
            fastk_period=fastk_period,
            fastd_period=fastd_period,
            fastd_matype=fastd_matype,
        )

        try:
            fastk, fastd = MomentumAdapter._safe_talib_calculation(
                talib.STOCHF,
                high.values,
                low.values,
                close.values,
                fastk_period=fastk_period,
                fastd_period=fastd_period,
                fastd_matype=fastd_matype,
            )

            return {
                "fastk": MomentumAdapter._create_series_result(
                    fastk, close.index, f"STOCHF_K_{fastk_period}_{fastd_period}"
                ),
                "fastd": MomentumAdapter._create_series_result(
                    fastd, close.index, f"STOCHF_D_{fastk_period}_{fastd_period}"
                ),
            }

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("STOCHF", e)
            raise TALibCalculationError(f"STOCHF計算失敗: {e}")

    @staticmethod
    def aroonosc(high: pd.Series, low: pd.Series, period: int) -> pd.Series:
        """
        AROONOSC（Aroon Oscillator）を計算

        Args:
            high: 高値のSeries
            low: 安値のSeries
            period: 期間

        Returns:
            AROONOSC値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        MomentumAdapter._validate_input(high, period)
        MomentumAdapter._log_calculation_start("AROONOSC", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.AROONOSC, high.values, low.values, timeperiod=period
            )
            return MomentumAdapter._create_series_result(
                result, high.index, f"AROONOSC_{period}"
            )
        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("AROONOSC", e)
            raise TALibCalculationError(f"AROONOSC計算失敗: {e}")

    @staticmethod
    def dx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """
        DX（Directional Movement Index）を計算

        Args:
            high: 高値のSeries
            low: 安値のSeries
            close: 終値のSeries
            period: 期間

        Returns:
            DX値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        MomentumAdapter._validate_input(close, period)
        MomentumAdapter._log_calculation_start("DX", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.DX, high.values, low.values, close.values, timeperiod=period
            )
            return MomentumAdapter._create_series_result(
                result, close.index, f"DX_{period}"
            )
        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("DX", e)
            raise TALibCalculationError(f"DX計算失敗: {e}")

    @staticmethod
    def adxr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        """
        ADXR（Average Directional Movement Index Rating）を計算

        Args:
            high: 高値のSeries
            low: 安値のSeries
            close: 終値のSeries
            period: 期間

        Returns:
            ADXR値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        MomentumAdapter._validate_input(close, period)
        MomentumAdapter._log_calculation_start("ADXR", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.ADXR, high.values, low.values, close.values, timeperiod=period
            )
            return MomentumAdapter._create_series_result(
                result, close.index, f"ADXR_{period}"
            )
        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("ADXR", e)
            raise TALibCalculationError(f"ADXR計算失敗: {e}")
