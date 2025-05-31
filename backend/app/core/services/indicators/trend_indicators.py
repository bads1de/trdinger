"""
トレンド系テクニカル指標

SMA（単純移動平均）、EMA（指数移動平均）、MACD の実装を提供します。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union

from .base_indicator import BaseIndicator


class SMAIndicator(BaseIndicator):
    """単純移動平均（Simple Moving Average）指標"""

    def __init__(self):
        super().__init__(
            indicator_type="SMA",
            supported_periods=[5, 10, 20, 50, 100, 200]
        )

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        単純移動平均（SMA）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            SMA値のSeries
        """
        return df["close"].rolling(window=period, min_periods=period).mean()

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "単純移動平均 - 指定期間の終値の平均値"


class EMAIndicator(BaseIndicator):
    """指数移動平均（Exponential Moving Average）指標"""

    def __init__(self):
        super().__init__(
            indicator_type="EMA",
            supported_periods=[5, 10, 20, 50, 100, 200]
        )

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        指数移動平均（EMA）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            EMA値のSeries
        """
        return df["close"].ewm(span=period, adjust=False).mean()

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "指数移動平均 - 直近の価格により重みを置いた移動平均"


class MACDIndicator(BaseIndicator):
    """MACD（Moving Average Convergence Divergence）指標"""

    def __init__(self):
        super().__init__(
            indicator_type="MACD",
            supported_periods=[12]  # 標準的な設定
        )

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.DataFrame:
        """
        MACD（Moving Average Convergence Divergence）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常12、26、9の組み合わせ）

        Returns:
            MACD値を含むDataFrame（macd_line, signal_line, histogram）
        """
        close = df["close"]

        # EMA12とEMA26を計算
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()

        # MACD線 = EMA12 - EMA26
        macd_line = ema12 - ema26

        # シグナル線 = MACD線のEMA9
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        # ヒストグラム = MACD線 - シグナル線
        histogram = macd_line - signal_line

        # 結果をDataFrameで返す
        result = pd.DataFrame({
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
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
        MACD指標を計算してフォーマットされた結果を返す（オーバーライド）

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

            # データ検証（MACD は26期間必要）
            self.validate_data(df, 26)

            # 指標を計算
            result = self.calculate(df, period, **kwargs)

            # MACD専用のフォーマット
            value_columns = {
                "value": "macd_line",
                "signal_value": "signal_line", 
                "histogram_value": "histogram"
            }

            formatted_result = self.format_multi_value_result(
                result, symbol, timeframe, period, value_columns
            )

            return formatted_result

        except Exception as e:
            raise

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "MACD - トレンドの方向性と強さを示すオシレーター"


# 指標インスタンスのファクトリー関数
def get_trend_indicator(indicator_type: str) -> BaseIndicator:
    """
    トレンド系指標のインスタンスを取得

    Args:
        indicator_type: 指標タイプ（'SMA', 'EMA', 'MACD'）

    Returns:
        指標インスタンス

    Raises:
        ValueError: サポートされていない指標タイプの場合
    """
    indicators = {
        "SMA": SMAIndicator,
        "EMA": EMAIndicator,
        "MACD": MACDIndicator,
    }

    if indicator_type not in indicators:
        raise ValueError(
            f"サポートされていないトレンド系指標です: {indicator_type}. "
            f"サポート対象: {list(indicators.keys())}"
        )

    return indicators[indicator_type]()


# サポートされている指標の情報
TREND_INDICATORS_INFO = {
    "SMA": {
        "periods": [5, 10, 20, 50, 100, 200],
        "description": "単純移動平均 - 指定期間の終値の平均値",
        "category": "trend"
    },
    "EMA": {
        "periods": [5, 10, 20, 50, 100, 200],
        "description": "指数移動平均 - 直近の価格により重みを置いた移動平均",
        "category": "trend"
    },
    "MACD": {
        "periods": [12],
        "description": "MACD - トレンドの方向性と強さを示すオシレーター",
        "category": "trend"
    },
}
