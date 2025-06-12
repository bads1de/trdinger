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

        except Exception as e:
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

        except Exception as e:
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
}
