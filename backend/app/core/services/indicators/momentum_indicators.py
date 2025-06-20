"""
モメンタム系テクニカル指標

RSI、ストキャスティクス、CCI、Williams %R、モメンタム、ROC の実装を提供します。
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import logging

from .abstract_indicator import BaseIndicator
from .adapters import MomentumAdapter

logger = logging.getLogger(__name__)


class RSIIndicator(BaseIndicator):
    """相対力指数（Relative Strength Index）指標"""

    def __init__(self):
        super().__init__(indicator_type="RSI", supported_periods=[14, 21, 30])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        相対力指数（RSI）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            RSI値のSeries
        """
        # TA-Libを使用した高速計算
        return MomentumAdapter.rsi(df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "相対力指数 - 買われすぎ・売られすぎを示すオシレーター（0-100）"


class StochasticIndicator(BaseIndicator):
    """ストキャスティクス（Stochastic Oscillator）指標"""

    def __init__(self):
        super().__init__(indicator_type="STOCH", supported_periods=[14])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.DataFrame:
        """
        ストキャスティクス（Stochastic Oscillator）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常14）

        Returns:
            ストキャスティクス値を含むDataFrame（%K, %D）
        """
        # TA-Libを使用した高速計算
        stoch_result = MomentumAdapter.stochastic(
            df["high"], df["low"], df["close"], k_period=period, d_period=3
        )

        # DataFrameに変換して返す
        result = pd.DataFrame(
            {
                "k_percent": stoch_result["k_percent"],
                "d_percent": stoch_result["d_percent"],
            }
        )

        return result

    async def calculate_and_format(
        self,
        symbol: str,
        timeframe: str,
        period: int,
        limit: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        ストキャスティクス指標を計算してフォーマットされた結果を返す（オーバーライド）
        """
        try:
            # パラメータ検証
            self.validate_parameters(period, **kwargs)

            # OHLCVデータを取得
            df = await self.get_ohlcv_data(symbol, timeframe, limit)

            # データ検証
            self.validate_data(df, period)

            # 指標を計算
            result = self.calculate(df, period, **kwargs)

            # ストキャスティクス専用のフォーマット
            value_columns = {"value": "k_percent", "signal_value": "d_percent"}

            formatted_result = self.format_multi_value_result(
                result, symbol, timeframe, period, value_columns
            )

            return formatted_result

        except Exception:
            raise

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "ストキャスティクス - 買われすぎ・売られすぎを示すオシレーター（0-100）"


class CCIIndicator(BaseIndicator):
    """CCI（Commodity Channel Index）指標"""

    def __init__(self):
        super().__init__(indicator_type="CCI", supported_periods=[20])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        CCI（Commodity Channel Index）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常20）

        Returns:
            CCI値のSeries
        """
        # TA-Libを使用した高速計算
        return MomentumAdapter.cci(df["high"], df["low"], df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "CCI - 商品チャネル指数、トレンドの強さを測定"


class WilliamsRIndicator(BaseIndicator):
    """Williams %R 指標"""

    def __init__(self):
        super().__init__(indicator_type="WILLR", supported_periods=[14])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        Williams %R を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常14）

        Returns:
            Williams %R値のSeries
        """
        # TA-Libを使用した高速計算
        return MomentumAdapter.williams_r(df["high"], df["low"], df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "Williams %R - 逆張りシグナルに有効なオシレーター（-100-0）"


class MomentumIndicator(BaseIndicator):
    """モメンタム（Momentum）指標"""

    def __init__(self):
        super().__init__(indicator_type="MOM", supported_periods=[10, 14])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        モメンタム（Momentum）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常10または14）

        Returns:
            モメンタム値のSeries
        """
        # TA-Libを使用した高速計算
        return MomentumAdapter.momentum(df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "モメンタム - 価格変化の勢いを測定する指標"


class ROCIndicator(BaseIndicator):
    """ROC（Rate of Change）指標"""

    def __init__(self):
        super().__init__(indicator_type="ROC", supported_periods=[10, 14])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        ROC（Rate of Change）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常10または14）

        Returns:
            ROC値のSeries
        """
        # TA-Libを使用した高速計算
        return MomentumAdapter.roc(df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "ROC - 変化率、価格の変化をパーセンテージで表示"


class ADXIndicator(BaseIndicator):
    """ADX（Average Directional Movement Index）指標"""

    def __init__(self):
        super().__init__(indicator_type="ADX", supported_periods=[14, 21])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        ADX（Average Directional Movement Index）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常14または21）

        Returns:
            ADX値のSeries（0-100の範囲）
        """
        # TA-Libを使用した高速計算
        return MomentumAdapter.adx(df["high"], df["low"], df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "ADX - 平均方向性指数、トレンドの強さを測定（0-100）"


class AroonIndicator(BaseIndicator):
    """Aroon（アルーン）指標"""

    def __init__(self):
        super().__init__(indicator_type="AROON", supported_periods=[14, 25])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.DataFrame:
        """
        Aroon（アルーン）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常14または25）

        Returns:
            Aroon値を含むDataFrame（aroon_down, aroon_up）
        """
        # TA-Libを使用した高速計算
        aroon_result = MomentumAdapter.aroon(df["high"], df["low"], period)

        # DataFrameに変換して返す
        result = pd.DataFrame(
            {
                "aroon_down": aroon_result["aroon_down"],
                "aroon_up": aroon_result["aroon_up"],
            }
        )

        return result

    async def calculate_and_format(
        self,
        symbol: str,
        timeframe: str,
        period: int,
        limit: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Aroon指標を計算してフォーマットされた結果を返す（オーバーライド）
        """
        try:
            # パラメータ検証
            self.validate_parameters(period, **kwargs)

            # OHLCVデータを取得
            df = await self.get_ohlcv_data(symbol, timeframe, limit)

            # データ検証
            self.validate_data(df, period)

            # 指標を計算
            result = self.calculate(df, period, **kwargs)

            # Aroon専用のフォーマット
            value_columns = {"value": "aroon_up", "signal_value": "aroon_down"}

            formatted_result = self.format_multi_value_result(
                result, symbol, timeframe, period, value_columns
            )

            return formatted_result

        except Exception:
            raise

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "Aroon - アルーン、トレンドの変化を検出（0-100）"


class MFIIndicator(BaseIndicator):
    """MFI（Money Flow Index）指標"""

    def __init__(self):
        super().__init__(indicator_type="MFI", supported_periods=[14, 21])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        MFI（Money Flow Index）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常14または21）

        Returns:
            MFI値のSeries（0-100の範囲）
        """
        # 出来高データの存在確認
        if "volume" not in df.columns:
            raise ValueError("MFI計算には出来高データが必要です")

        # TA-Libを使用した高速計算
        return MomentumAdapter.mfi(
            df["high"], df["low"], df["close"], df["volume"], period
        )

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "MFI - マネーフローインデックス、出来高を考慮したRSI（0-100）"


class StochasticRSIIndicator(BaseIndicator):
    """Stochastic RSI（ストキャスティクスRSI）指標"""

    def __init__(self):
        super().__init__(indicator_type="STOCHRSI", supported_periods=[14, 21])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.DataFrame:
        """
        Stochastic RSI（ストキャスティクスRSI）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: RSI期間
            **kwargs: 追加パラメータ
                - fastk_period: Fast %K期間（デフォルト: 3）
                - fastd_period: Fast %D期間（デフォルト: 3）

        Returns:
            Stochastic RSIのDataFrame (fastk, fastd)

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        # パラメータの取得
        fastk_period = kwargs.get("fastk_period", 3)
        fastd_period = kwargs.get("fastd_period", 3)

        # MomentumAdapterを使用したStochastic RSI計算
        return MomentumAdapter.stochastic_rsi(
            df["close"], period, fastk_period, fastd_period
        )

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "Stochastic RSI - ストキャスティクスRSI、RSIにストキャスティクスを適用した高感度オシレーター"


class UltimateOscillatorIndicator(BaseIndicator):
    """Ultimate Oscillator（アルティメットオシレーター）指標"""

    def __init__(self):
        super().__init__(indicator_type="ULTOSC", supported_periods=[7, 14, 28])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        Ultimate Oscillator（アルティメットオシレーター）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 短期期間（中期=period*2, 長期=period*4で計算）
            **kwargs: 追加パラメータ
                - period2: 中期期間（デフォルト: period*2）
                - period3: 長期期間（デフォルト: period*4）

        Returns:
            Ultimate Oscillator値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        # パラメータの取得
        period2 = kwargs.get("period2", period * 2)
        period3 = kwargs.get("period3", period * 4)

        # MomentumAdapterを使用したUltimate Oscillator計算
        return MomentumAdapter.ultimate_oscillator(
            df["high"], df["low"], df["close"], period, period2, period3
        )

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "Ultimate Oscillator - アルティメットオシレーター、複数期間のTrue Rangeベースのモメンタム指標"


class CMOIndicator(BaseIndicator):
    """CMO（Chande Momentum Oscillator）指標"""

    def __init__(self):
        super().__init__(indicator_type="CMO", supported_periods=[7, 14, 21, 28])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        CMO（Chande Momentum Oscillator）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            CMO値のSeries（-100から100の範囲）

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        # MomentumAdapterを使用したCMO計算
        return MomentumAdapter.cmo(df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return (
            "CMO - Chande Momentum Oscillator、改良されたモメンタム指標（-100から100）"
        )


class TRIXIndicator(BaseIndicator):
    """TRIX（Triple Exponential Moving Average）指標"""

    def __init__(self):
        super().__init__(indicator_type="TRIX", supported_periods=[14, 21, 30])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        TRIX（Triple Exponential Moving Average）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            TRIX値のSeries（パーセンテージ値）

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        # MomentumAdapterを使用したTRIX計算
        return MomentumAdapter.trix(df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "TRIX - 三重平滑化されたモメンタム指標、ノイズを除去したトレンド分析"


class BOPIndicator(BaseIndicator):
    """BOP（Balance Of Power）指標"""

    def __init__(self):
        super().__init__(indicator_type="BOP", supported_periods=[1])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        BOP（Balance Of Power）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（BOPは期間を使用しないが、統一性のため）

        Returns:
            BOP値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
            ValueError: 必要なデータが存在しない場合
        """
        # 必要なデータの存在確認
        required_columns = ["open", "high", "low", "close"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"BOP計算には{col}データが必要です")

        # MomentumAdapterを使用したBOP計算
        return MomentumAdapter.bop(df["open"], df["high"], df["low"], df["close"])

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "BOP - Balance Of Power、買い圧力と売り圧力のバランスを測定"


class APOIndicator(BaseIndicator):
    """APO（Absolute Price Oscillator）指標"""

    def __init__(self):
        super().__init__(indicator_type="APO", supported_periods=[12, 26])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        APO（Absolute Price Oscillator）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（短期期間として使用）
            **kwargs: 追加パラメータ
                - slow_period: 長期期間（デフォルト: period * 2）
                - matype: 移動平均タイプ（デフォルト: 0 = SMA）

        Returns:
            APO値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        # パラメータの取得
        slow_period = kwargs.get("slow_period", period * 2)
        matype = kwargs.get("matype", 0)

        # MomentumAdapterを使用したAPO計算
        return MomentumAdapter.apo(df["close"], period, slow_period, matype)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "APO - Absolute Price Oscillator、短期と長期移動平均の絶対差"


class PPOIndicator(BaseIndicator):
    """PPO（Percentage Price Oscillator）指標"""

    def __init__(self):
        super().__init__(indicator_type="PPO", supported_periods=[12, 26])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        PPO（Percentage Price Oscillator）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（短期期間として使用）
            **kwargs: 追加パラメータ
                - slow_period: 長期期間（デフォルト: period * 2）
                - matype: 移動平均タイプ（デフォルト: 0 = SMA）

        Returns:
            PPO値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        # パラメータの取得
        slow_period = kwargs.get("slow_period", period * 2)
        matype = kwargs.get("matype", 0)

        # MomentumAdapterを使用したPPO計算
        return MomentumAdapter.ppo(df["close"], period, slow_period, matype)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "PPO - Percentage Price Oscillator、短期と長期移動平均のパーセンテージ差"


class AROONOSCIndicator(BaseIndicator):
    """AROONOSC（Aroon Oscillator）指標"""

    def __init__(self):
        super().__init__(indicator_type="AROONOSC", supported_periods=[14, 25])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        AROONOSC（Aroon Oscillator）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            AROONOSC値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
            ValueError: 必要なデータが存在しない場合
        """
        # 必要なデータの存在確認
        required_columns = ["high", "low"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"AROONOSC計算には{col}データが必要です")

        # MomentumAdapterを使用したAROONOSC計算
        return MomentumAdapter.aroonosc(df["high"], df["low"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "AROONOSC - Aroon Oscillator、Aroon UpとAroon Downの差"


class DXIndicator(BaseIndicator):
    """DX（Directional Movement Index）指標"""

    def __init__(self):
        super().__init__(indicator_type="DX", supported_periods=[14, 21])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        DX（Directional Movement Index）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            DX値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
            ValueError: 必要なデータが存在しない場合
        """
        # 必要なデータの存在確認
        required_columns = ["high", "low", "close"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"DX計算には{col}データが必要です")

        # MomentumAdapterを使用したDX計算
        return MomentumAdapter.dx(df["high"], df["low"], df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "DX - Directional Movement Index、方向性の強さを測定"


class ADXRIndicator(BaseIndicator):
    """ADXR（Average Directional Movement Index Rating）指標"""

    def __init__(self):
        super().__init__(indicator_type="ADXR", supported_periods=[14, 21])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        ADXR（Average Directional Movement Index Rating）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            ADXR値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
            ValueError: 必要なデータが存在しない場合
        """
        # 必要なデータの存在確認
        required_columns = ["high", "low", "close"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"ADXR計算には{col}データが必要です")

        # MomentumAdapterを使用したADXR計算
        return MomentumAdapter.adxr(df["high"], df["low"], df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "ADXR - Average Directional Movement Index Rating、ADXの平滑化版"


class PLUSDIIndicator(BaseIndicator):
    """PLUS_DI（Plus Directional Indicator）指標"""

    def __init__(self):
        super().__init__(indicator_type="PLUS_DI", supported_periods=[14, 20, 30])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        PLUS_DI（Plus Directional Indicator）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            PLUS_DI値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
            ValueError: 高値・安値・終値データが存在しない場合
        """
        # 高値・安値・終値データの存在確認
        if (
            "high" not in df.columns
            or "low" not in df.columns
            or "close" not in df.columns
        ):
            raise ValueError("PLUS_DI計算には高値・安値・終値データが必要です")

        # MomentumAdapterを使用したPLUS_DI計算
        return MomentumAdapter.plus_di(df["high"], df["low"], df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "PLUS_DI - Plus Directional Indicator、ADXと組み合わせて使用する上昇方向性指標"


class MINUSDIIndicator(BaseIndicator):
    """MINUS_DI（Minus Directional Indicator）指標"""

    def __init__(self):
        super().__init__(indicator_type="MINUS_DI", supported_periods=[14, 20, 30])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        MINUS_DI（Minus Directional Indicator）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            MINUS_DI値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
            ValueError: 高値・安値・終値データが存在しない場合
        """
        # 高値・安値・終値データの存在確認
        if (
            "high" not in df.columns
            or "low" not in df.columns
            or "close" not in df.columns
        ):
            raise ValueError("MINUS_DI計算には高値・安値・終値データが必要です")

        # MomentumAdapterを使用したMINUS_DI計算
        return MomentumAdapter.minus_di(df["high"], df["low"], df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "MINUS_DI - Minus Directional Indicator、ADXと組み合わせて使用する下降方向性指標"


class ROCPIndicator(BaseIndicator):
    """ROCP（Rate of change Percentage）指標"""

    def __init__(self):
        super().__init__(indicator_type="ROCP", supported_periods=[10, 14, 20])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        ROCP（Rate of change Percentage）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            ROCP値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        # MomentumAdapterを使用したROCP計算
        return MomentumAdapter.rocp(df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "ROCP - Rate of change Percentage、ROCのパーセンテージ版、価格変化率をパーセンテージで表示"


class ROCRIndicator(BaseIndicator):
    """ROCR（Rate of change ratio）指標"""

    def __init__(self):
        super().__init__(indicator_type="ROCR", supported_periods=[10, 14, 20])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        ROCR（Rate of change ratio）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            ROCR値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        # MomentumAdapterを使用したROCR計算
        return MomentumAdapter.rocr(df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "ROCR - Rate of change ratio、ROCの比率版、価格変化率を比率で表示"


class STOCHFIndicator(BaseIndicator):
    """STOCHF（Stochastic Fast）指標"""

    def __init__(self):
        super().__init__(indicator_type="STOCHF", supported_periods=[5, 14])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> dict:
        """
        STOCHF（Stochastic Fast）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: FastK期間
            **kwargs: 追加パラメータ
                - fastd_period: FastD期間（デフォルト: 3）
                - fastd_matype: FastD移動平均タイプ（デフォルト: 0=SMA）

        Returns:
            STOCHF値を含む辞書（fastk, fastd）

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
            ValueError: 高値・安値・終値データが存在しない場合
        """
        # 高値・安値・終値データの存在確認
        if (
            "high" not in df.columns
            or "low" not in df.columns
            or "close" not in df.columns
        ):
            raise ValueError("STOCHF計算には高値・安値・終値データが必要です")

        # パラメータの取得
        fastd_period = kwargs.get("fastd_period", 3)
        fastd_matype = kwargs.get("fastd_matype", 0)

        # MomentumAdapterを使用したSTOCHF計算
        return MomentumAdapter.stochf(
            df["high"], df["low"], df["close"], period, fastd_period, fastd_matype
        )

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "STOCHF - Stochastic Fast、高速ストキャスティクス、短期的な価格モメンタムを測定"


# 指標インスタンスのファクトリー関数
def get_momentum_indicator(indicator_type: str) -> BaseIndicator:
    """
    モメンタム系指標のインスタンスを取得

    Args:
        indicator_type: 指標タイプ（'RSI', 'STOCH', 'CCI', 'WILLR', 'MOM', 'ROC', 'ADX', 'AROON', 'MFI', 'STOCHRSI', 'ULTOSC', 'CMO', 'TRIX'）

    Returns:
        指標インスタンス

    Raises:
        ValueError: サポートされていない指標タイプの場合
    """
    indicators = {
        "RSI": RSIIndicator,
        "STOCH": StochasticIndicator,
        "CCI": CCIIndicator,
        "WILLR": WilliamsRIndicator,
        "MOM": MomentumIndicator,
        "ROC": ROCIndicator,
        "ADX": ADXIndicator,
        "AROON": AroonIndicator,
        "MFI": MFIIndicator,
        "STOCHRSI": StochasticRSIIndicator,
        "ULTOSC": UltimateOscillatorIndicator,
        "CMO": CMOIndicator,
        "TRIX": TRIXIndicator,
        "BOP": BOPIndicator,
        "APO": APOIndicator,
        "PPO": PPOIndicator,
        "AROONOSC": AROONOSCIndicator,
        "DX": DXIndicator,
        "ADXR": ADXRIndicator,
        "PLUS_DI": PLUSDIIndicator,
        "MINUS_DI": MINUSDIIndicator,
        "ROCP": ROCPIndicator,
        "ROCR": ROCRIndicator,
        "STOCHF": STOCHFIndicator,
    }

    if indicator_type not in indicators:
        raise ValueError(
            f"サポートされていないモメンタム系指標です: {indicator_type}. "
            f"サポート対象: {list(indicators.keys())}"
        )

    return indicators[indicator_type]()


# サポートされている指標の情報
MOMENTUM_INDICATORS_INFO = {
    "RSI": {
        "periods": [14, 21, 30],
        "description": "相対力指数 - 買われすぎ・売られすぎを示すオシレーター（0-100）",
        "category": "momentum",
    },
    "STOCH": {
        "periods": [14],
        "description": "ストキャスティクス - 買われすぎ・売られすぎを示すオシレーター（0-100）",
        "category": "momentum",
    },
    "CCI": {
        "periods": [20],
        "description": "CCI - 商品チャネル指数、トレンドの強さを測定",
        "category": "momentum",
    },
    "WILLR": {
        "periods": [14],
        "description": "Williams %R - 逆張りシグナルに有効なオシレーター（-100-0）",
        "category": "momentum",
    },
    "MOM": {
        "periods": [10, 14],
        "description": "モメンタム - 価格変化の勢いを測定する指標",
        "category": "momentum",
    },
    "ROC": {
        "periods": [10, 14],
        "description": "ROC - 変化率、価格の変化をパーセンテージで表示",
        "category": "momentum",
    },
    "ADX": {
        "periods": [14, 21],
        "description": "ADX - 平均方向性指数、トレンドの強さを測定（0-100）",
        "category": "momentum",
    },
    "AROON": {
        "periods": [14, 25],
        "description": "Aroon - アルーン、トレンドの変化を検出（0-100）",
        "category": "momentum",
    },
    "MFI": {
        "periods": [14, 21],
        "description": "MFI - マネーフローインデックス、出来高を考慮したRSI（0-100）",
        "category": "momentum",
    },
    "STOCHRSI": {
        "periods": [14, 21],
        "description": "Stochastic RSI - ストキャスティクスRSI、RSIにストキャスティクスを適用した高感度オシレーター",
        "category": "momentum",
    },
    "ULTOSC": {
        "periods": [7, 14, 28],
        "description": "Ultimate Oscillator - アルティメットオシレーター、複数期間のTrue Rangeベースのモメンタム指標",
        "category": "momentum",
    },
    "CMO": {
        "periods": [7, 14, 21, 28],
        "description": "CMO - Chande Momentum Oscillator、改良されたモメンタム指標（-100から100）",
        "category": "momentum",
    },
    "TRIX": {
        "periods": [14, 21, 30],
        "description": "TRIX - 三重平滑化されたモメンタム指標、ノイズを除去したトレンド分析",
        "category": "momentum",
    },
    "BOP": {
        "periods": [1],
        "description": "BOP - Balance Of Power、買い圧力と売り圧力のバランスを測定",
        "category": "momentum",
    },
    "APO": {
        "periods": [12, 26],
        "description": "APO - Absolute Price Oscillator、短期と長期移動平均の絶対差",
        "category": "momentum",
    },
    "PPO": {
        "periods": [12, 26],
        "description": "PPO - Percentage Price Oscillator、短期と長期移動平均のパーセンテージ差",
        "category": "momentum",
    },
    "AROONOSC": {
        "periods": [14, 25],
        "description": "AROONOSC - Aroon Oscillator、Aroon UpとAroon Downの差",
        "category": "momentum",
    },
    "DX": {
        "periods": [14, 21],
        "description": "DX - Directional Movement Index、方向性の強さを測定",
        "category": "momentum",
    },
    "ADXR": {
        "periods": [14, 21],
        "description": "ADXR - Average Directional Movement Index Rating、ADXの平滑化版",
        "category": "momentum",
    },
    "PLUS_DI": {
        "periods": [14, 20, 30],
        "description": "PLUS_DI - Plus Directional Indicator、ADXと組み合わせて使用する上昇方向性指標",
        "category": "momentum",
    },
    "MINUS_DI": {
        "periods": [14, 20, 30],
        "description": "MINUS_DI - Minus Directional Indicator、ADXと組み合わせて使用する下降方向性指標",
        "category": "momentum",
    },
    "ROCP": {
        "periods": [10, 14, 20],
        "description": "ROCP - Rate of change Percentage、ROCのパーセンテージ版、価格変化率をパーセンテージで表示",
        "category": "momentum",
    },
    "ROCR": {
        "periods": [10, 14, 20],
        "description": "ROCR - Rate of change ratio、ROCの比率版、価格変化率を比率で表示",
        "category": "momentum",
    },
    "STOCHF": {
        "periods": [5, 14],
        "description": "STOCHF - Stochastic Fast、高速ストキャスティクス、短期的な価格モメンタムを測定",
        "category": "momentum",
    },
}
