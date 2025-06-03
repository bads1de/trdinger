"""
トレンド系テクニカル指標

SMA（単純移動平均）、EMA（指数移動平均）、MACD の実装を提供します。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging

from .abstract_indicator import BaseIndicator
from .adapters import TrendAdapter, MomentumAdapter, TALibCalculationError

logger = logging.getLogger(__name__)


class SMAIndicator(BaseIndicator):
    """単純移動平均（Simple Moving Average）指標"""

    def __init__(self):
        super().__init__(
            indicator_type="SMA", supported_periods=[5, 10, 20, 50, 100, 200]
        )

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        単純移動平均（SMA）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            SMA値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        # TA-Libを使用した高速計算
        return TrendAdapter.sma(df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "単純移動平均 - 指定期間の終値の平均値"


class EMAIndicator(BaseIndicator):
    """指数移動平均（Exponential Moving Average）指標"""

    def __init__(self):
        super().__init__(
            indicator_type="EMA", supported_periods=[5, 10, 20, 50, 100, 200]
        )

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        指数移動平均（EMA）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            EMA値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        # TA-Libを使用した高速計算
        return TrendAdapter.ema(df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "指数移動平均 - 直近の価格により重みを置いた移動平均"


class MACDIndicator(BaseIndicator):
    """MACD（Moving Average Convergence Divergence）指標"""

    def __init__(self):
        super().__init__(indicator_type="MACD", supported_periods=[12])  # 標準的な設定

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.DataFrame:
        """
        MACD（Moving Average Convergence Divergence）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常12、26、9の組み合わせ）

        Returns:
            MACD値を含むDataFrame（macd_line, signal_line, histogram）
        """
        # TA-Libを使用した高速計算
        macd_result = MomentumAdapter.macd(df["close"], fast=12, slow=26, signal=9)

        # DataFrameに変換して返す
        result = pd.DataFrame(
            {
                "macd_line": macd_result["macd_line"],
                "signal_line": macd_result["signal_line"],
                "histogram": macd_result["histogram"],
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
                "histogram_value": "histogram",
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


class KAMAIndicator(BaseIndicator):
    """KAMA（Kaufman Adaptive Moving Average）指標"""

    def __init__(self):
        super().__init__(indicator_type="KAMA", supported_periods=[20, 30])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        KAMA（Kaufman Adaptive Moving Average）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常20または30）

        Returns:
            KAMA値のSeries
        """
        # TA-Libを使用した高速計算
        return TrendAdapter.kama(df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "KAMA - カウフマン適応型移動平均、市場の効率性に応じて調整"


class T3Indicator(BaseIndicator):
    """T3（Triple Exponential Moving Average）指標"""

    def __init__(self):
        super().__init__(indicator_type="T3", supported_periods=[5, 14, 21])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        T3（Triple Exponential Moving Average）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常5、14、21）
            **kwargs: 追加パラメータ
                - vfactor: ボリュームファクター（デフォルト: 0.7）

        Returns:
            T3値のSeries
        """
        # パラメータ取得
        vfactor = kwargs.get("vfactor", 0.7)

        # TA-Libを使用した高速計算
        return TrendAdapter.t3(df["close"], period, vfactor)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "T3 - 三重指数移動平均（T3）、滑らかで応答性の高いトレンド指標"


class TEMAIndicator(BaseIndicator):
    """TEMA（Triple Exponential Moving Average）指標"""

    def __init__(self):
        super().__init__(indicator_type="TEMA", supported_periods=[14, 21, 30])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        TEMA（Triple Exponential Moving Average）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常14、21、30）

        Returns:
            TEMA値のSeries
        """
        # TA-Libを使用した高速計算
        return TrendAdapter.tema(df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "TEMA - 三重指数移動平均、ラグを減らした高応答性移動平均"


class DEMAIndicator(BaseIndicator):
    """DEMA（Double Exponential Moving Average）指標"""

    def __init__(self):
        super().__init__(indicator_type="DEMA", supported_periods=[14, 21, 30])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        DEMA（Double Exponential Moving Average）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常14、21、30）

        Returns:
            DEMA値のSeries
        """
        # TA-Libを使用した高速計算
        return TrendAdapter.dema(df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "DEMA - 二重指数移動平均、ラグを減らした応答性の高い移動平均"


# 指標インスタンスのファクトリー関数
def get_trend_indicator(indicator_type: str) -> BaseIndicator:
    """
    トレンド系指標のインスタンスを取得

    Args:
        indicator_type: 指標タイプ（'SMA', 'EMA', 'MACD', 'KAMA', 'T3', 'TEMA', 'DEMA'）

    Returns:
        指標インスタンス

    Raises:
        ValueError: サポートされていない指標タイプの場合
    """
    indicators = {
        "SMA": SMAIndicator,
        "EMA": EMAIndicator,
        "MACD": MACDIndicator,
        "KAMA": KAMAIndicator,
        "T3": T3Indicator,
        "TEMA": TEMAIndicator,
        "DEMA": DEMAIndicator,
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
        "category": "trend",
    },
    "EMA": {
        "periods": [5, 10, 20, 50, 100, 200],
        "description": "指数移動平均 - 直近の価格により重みを置いた移動平均",
        "category": "trend",
    },
    "MACD": {
        "periods": [12],
        "description": "MACD - トレンドの方向性と強さを示すオシレーター",
        "category": "trend",
    },
    "KAMA": {
        "periods": [20, 30],
        "description": "KAMA - カウフマン適応型移動平均、市場の効率性に応じて調整",
        "category": "trend",
    },
    "T3": {
        "periods": [5, 14, 21],
        "description": "T3 - 三重指数移動平均（T3）、滑らかで応答性の高いトレンド指標",
        "category": "trend",
    },
    "TEMA": {
        "periods": [14, 21, 30],
        "description": "TEMA - 三重指数移動平均、ラグを減らした高応答性移動平均",
        "category": "trend",
    },
    "DEMA": {
        "periods": [14, 21, 30],
        "description": "DEMA - 二重指数移動平均、ラグを減らした応答性の高い移動平均",
        "category": "trend",
    },
}
