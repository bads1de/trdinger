"""
モメンタム系テクニカル指標

RSI、ストキャスティクス、CCI、Williams %R、モメンタム、ROC の実装を提供します。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging

from .base_indicator import BaseIndicator
from .talib_adapter import TALibAdapter, TALibCalculationError

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
        return TALibAdapter.rsi(df["close"], period)

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
        stoch_result = TALibAdapter.stochastic(
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
        return TALibAdapter.cci(df["high"], df["low"], df["close"], period)

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
        return TALibAdapter.williams_r(df["high"], df["low"], df["close"], period)

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
        return TALibAdapter.momentum(df["close"], period)

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
        return TALibAdapter.roc(df["close"], period)

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
}
