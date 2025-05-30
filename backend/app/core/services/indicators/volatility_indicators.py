"""
ボラティリティ系テクニカル指標

ボリンジャーバンド（BB）、ATR の実装を提供します。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union

from .base_indicator import BaseIndicator


class BollingerBandsIndicator(BaseIndicator):
    """ボリンジャーバンド（Bollinger Bands）指標"""

    def __init__(self):
        super().__init__(
            indicator_type="BB",
            supported_periods=[20]
        )

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.DataFrame:
        """
        ボリンジャーバンド（Bollinger Bands）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常20）

        Returns:
            ボリンジャーバンド値を含むDataFrame（middle, upper, lower）
        """
        close = df["close"]

        # 中央線（SMA）
        middle = close.rolling(window=period, min_periods=period).mean()

        # 標準偏差
        std = close.rolling(window=period, min_periods=period).std()

        # 上限・下限（標準偏差の2倍）
        upper = middle + (std * 2)
        lower = middle - (std * 2)

        # 結果をDataFrameで返す
        result = pd.DataFrame({
            'middle': middle,
            'upper': upper,
            'lower': lower
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
        ボリンジャーバンド指標を計算してフォーマットされた結果を返す（オーバーライド）

        Args:
            symbol: 取引ペア
            timeframe: 時間枠
            period: 期間
            limit: OHLCVデータの取得件数制限
            **kwargs: 追加パラメータ

        Returns:
            フォーマットされた計算結果のリスト
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

            # ボリンジャーバンド専用のフォーマット
            value_columns = {
                "value": "middle",
                "upper_band": "upper",
                "lower_band": "lower"
            }

            formatted_result = self.format_multi_value_result(
                result, symbol, timeframe, period, value_columns
            )

            return formatted_result

        except Exception as e:
            raise

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "ボリンジャーバンド - ボラティリティとサポート・レジスタンスを示す"


class ATRIndicator(BaseIndicator):
    """ATR（Average True Range）指標"""

    def __init__(self):
        super().__init__(
            indicator_type="ATR",
            supported_periods=[14, 21]
        )

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        ATR（Average True Range）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常14）

        Returns:
            ATR値のSeries
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # 前日終値
        prev_close = close.shift(1)

        # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR = True Rangeの移動平均
        atr = true_range.rolling(window=period, min_periods=period).mean()

        return atr

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "ATR - 平均真の値幅、ボラティリティを測定する指標"


# 指標インスタンスのファクトリー関数
def get_volatility_indicator(indicator_type: str) -> BaseIndicator:
    """
    ボラティリティ系指標のインスタンスを取得

    Args:
        indicator_type: 指標タイプ（'BB', 'ATR'）

    Returns:
        指標インスタンス

    Raises:
        ValueError: サポートされていない指標タイプの場合
    """
    indicators = {
        "BB": BollingerBandsIndicator,
        "ATR": ATRIndicator,
    }

    if indicator_type not in indicators:
        raise ValueError(
            f"サポートされていないボラティリティ系指標です: {indicator_type}. "
            f"サポート対象: {list(indicators.keys())}"
        )

    return indicators[indicator_type]()


# サポートされている指標の情報
VOLATILITY_INDICATORS_INFO = {
    "BB": {
        "periods": [20],
        "description": "ボリンジャーバンド - ボラティリティとサポート・レジスタンスを示す",
        "category": "volatility"
    },
    "ATR": {
        "periods": [14, 21],
        "description": "ATR - 平均真の値幅、ボラティリティを測定する指標",
        "category": "volatility"
    },
}
