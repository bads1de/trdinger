"""
モメンタム系テクニカル指標

RSI、ストキャスティクス、CCI、Williams %R、モメンタム、ROC の実装を提供します。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union

from .base_indicator import BaseIndicator


class RSIIndicator(BaseIndicator):
    """相対力指数（Relative Strength Index）指標"""

    def __init__(self):
        super().__init__(
            indicator_type="RSI",
            supported_periods=[14, 21, 30]
        )

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        相対力指数（RSI）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            RSI値のSeries
        """
        close = df["close"]
        delta = close.diff()

        # 上昇と下降を分離
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # 平均上昇と平均下降を計算
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # RSを計算
        rs = avg_gain / avg_loss

        # RSIを計算
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "相対力指数 - 買われすぎ・売られすぎを示すオシレーター（0-100）"


class StochasticIndicator(BaseIndicator):
    """ストキャスティクス（Stochastic Oscillator）指標"""

    def __init__(self):
        super().__init__(
            indicator_type="STOCH",
            supported_periods=[14]
        )

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.DataFrame:
        """
        ストキャスティクス（Stochastic Oscillator）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常14）

        Returns:
            ストキャスティクス値を含むDataFrame（%K, %D）
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # 指定期間の最高値・最安値
        highest_high = high.rolling(window=period, min_periods=period).max()
        lowest_low = low.rolling(window=period, min_periods=period).min()

        # %K = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        k_percent = ((close - lowest_low) / (highest_high - lowest_low)) * 100

        # %D = %Kの3期間移動平均
        d_percent = k_percent.rolling(window=3, min_periods=3).mean()

        # 結果をDataFrameで返す
        result = pd.DataFrame({
            'k_percent': k_percent,
            'd_percent': d_percent
        })

        return result

    async def calculate_and_format(
        self,
        symbol: str,
        timeframe: str,
        period: int,
        limit: Optional[int] = None,
        **kwargs
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
            value_columns = {
                "value": "k_percent",
                "signal_value": "d_percent"
            }

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
        super().__init__(
            indicator_type="CCI",
            supported_periods=[20]
        )

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        CCI（Commodity Channel Index）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常20）

        Returns:
            CCI値のSeries
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Typical Price = (high + low + close) / 3
        typical_price = (high + low + close) / 3

        # Typical Priceの移動平均
        sma_tp = typical_price.rolling(window=period, min_periods=period).mean()

        # Mean Deviation = Typical Priceと移動平均の差の絶対値の移動平均
        mean_deviation = (typical_price - sma_tp).abs().rolling(window=period, min_periods=period).mean()

        # CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)

        return cci

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "CCI - 商品チャネル指数、トレンドの強さを測定"


class WilliamsRIndicator(BaseIndicator):
    """Williams %R 指標"""

    def __init__(self):
        super().__init__(
            indicator_type="WILLR",
            supported_periods=[14]
        )

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        Williams %R を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常14）

        Returns:
            Williams %R値のSeries
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # 指定期間の最高値・最安値
        highest_high = high.rolling(window=period, min_periods=period).max()
        lowest_low = low.rolling(window=period, min_periods=period).min()

        # Williams %R = ((highest_high - close) / (highest_high - lowest_low)) * -100
        williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100

        return williams_r

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "Williams %R - 逆張りシグナルに有効なオシレーター（-100-0）"


class MomentumIndicator(BaseIndicator):
    """モメンタム（Momentum）指標"""

    def __init__(self):
        super().__init__(
            indicator_type="MOM",
            supported_periods=[10, 14]
        )

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        モメンタム（Momentum）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常10または14）

        Returns:
            モメンタム値のSeries
        """
        close = df["close"]

        # Momentum = 現在の終値 - N期間前の終値
        momentum = close - close.shift(period)

        return momentum

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "モメンタム - 価格変化の勢いを測定する指標"


class ROCIndicator(BaseIndicator):
    """ROC（Rate of Change）指標"""

    def __init__(self):
        super().__init__(
            indicator_type="ROC",
            supported_periods=[10, 14]
        )

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        ROC（Rate of Change）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常10または14）

        Returns:
            ROC値のSeries
        """
        close = df["close"]

        # ROC = ((現在の終値 - N期間前の終値) / N期間前の終値) * 100
        prev_close = close.shift(period)
        roc = ((close - prev_close) / prev_close) * 100

        return roc

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "ROC - 変化率、価格の変化をパーセンテージで表示"


# 指標インスタンスのファクトリー関数
def get_momentum_indicator(indicator_type: str) -> BaseIndicator:
    """
    モメンタム系指標のインスタンスを取得

    Args:
        indicator_type: 指標タイプ（'RSI', 'STOCH', 'CCI', 'WILLR', 'MOM', 'ROC'）

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
        "category": "momentum"
    },
    "STOCH": {
        "periods": [14],
        "description": "ストキャスティクス - 買われすぎ・売られすぎを示すオシレーター（0-100）",
        "category": "momentum"
    },
    "CCI": {
        "periods": [20],
        "description": "CCI - 商品チャネル指数、トレンドの強さを測定",
        "category": "momentum"
    },
    "WILLR": {
        "periods": [14],
        "description": "Williams %R - 逆張りシグナルに有効なオシレーター（-100-0）",
        "category": "momentum"
    },
    "MOM": {
        "periods": [10, 14],
        "description": "モメンタム - 価格変化の勢いを測定する指標",
        "category": "momentum"
    },
    "ROC": {
        "periods": [10, 14],
        "description": "ROC - 変化率、価格の変化をパーセンテージで表示",
        "category": "momentum"
    },
}
