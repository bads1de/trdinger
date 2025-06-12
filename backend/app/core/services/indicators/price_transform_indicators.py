"""
価格変換系テクニカル指標

AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE の実装を提供します。
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import logging

from .abstract_indicator import BaseIndicator
from .adapters.price_transform_adapter import PriceTransformAdapter

logger = logging.getLogger(__name__)


class AVGPRICEIndicator(BaseIndicator):
    """AVGPRICE（Average Price）指標"""

    def __init__(self):
        super().__init__(indicator_type="AVGPRICE", supported_periods=[1])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        AVGPRICE（Average Price）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（AVGPRICEは期間を使用しないが、統一性のため）

        Returns:
            AVGPRICE値のSeries

        Raises:
            ValueError: 必要なデータが存在しない場合
        """
        # 必要なデータの存在確認
        required_columns = ["open", "high", "low", "close"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"AVGPRICE計算には{col}データが必要です")

        return PriceTransformAdapter.avgprice(
            df["open"], df["high"], df["low"], df["close"]
        )

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "AVGPRICE - Average Price、OHLC価格の平均値"


class MEDPRICEIndicator(BaseIndicator):
    """MEDPRICE（Median Price）指標"""

    def __init__(self):
        super().__init__(indicator_type="MEDPRICE", supported_periods=[1])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        MEDPRICE（Median Price）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（MEDPRICEは期間を使用しないが、統一性のため）

        Returns:
            MEDPRICE値のSeries

        Raises:
            ValueError: 必要なデータが存在しない場合
        """
        # 必要なデータの存在確認
        required_columns = ["high", "low"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"MEDPRICE計算には{col}データが必要です")

        return PriceTransformAdapter.medprice(df["high"], df["low"])

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "MEDPRICE - Median Price、高値と安値の中央値"


class TYPPRICEIndicator(BaseIndicator):
    """TYPPRICE（Typical Price）指標"""

    def __init__(self):
        super().__init__(indicator_type="TYPPRICE", supported_periods=[1])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        TYPPRICE（Typical Price）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（TYPPRICEは期間を使用しないが、統一性のため）

        Returns:
            TYPPRICE値のSeries

        Raises:
            ValueError: 必要なデータが存在しない場合
        """
        # 必要なデータの存在確認
        required_columns = ["high", "low", "close"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"TYPPRICE計算には{col}データが必要です")

        return PriceTransformAdapter.typprice(df["high"], df["low"], df["close"])

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "TYPPRICE - Typical Price、高値・安値・終値の平均"


class WCLPRICEIndicator(BaseIndicator):
    """WCLPRICE（Weighted Close Price）指標"""

    def __init__(self):
        super().__init__(indicator_type="WCLPRICE", supported_periods=[1])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        WCLPRICE（Weighted Close Price）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（WCLPRICEは期間を使用しないが、統一性のため）

        Returns:
            WCLPRICE値のSeries

        Raises:
            ValueError: 必要なデータが存在しない場合
        """
        # 必要なデータの存在確認
        required_columns = ["high", "low", "close"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"WCLPRICE計算には{col}データが必要です")

        return PriceTransformAdapter.wclprice(df["high"], df["low"], df["close"])

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "WCLPRICE - Weighted Close Price、終値に重みを付けた価格"


# 指標インスタンスのファクトリー関数
def get_price_transform_indicator(indicator_type: str) -> BaseIndicator:
    """
    価格変換系指標のインスタンスを取得

    Args:
        indicator_type: 指標タイプ（'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE'）

    Returns:
        指標インスタンス

    Raises:
        ValueError: 未対応の指標タイプの場合
    """
    indicators = {
        "AVGPRICE": AVGPRICEIndicator,
        "MEDPRICE": MEDPRICEIndicator,
        "TYPPRICE": TYPPRICEIndicator,
        "WCLPRICE": WCLPRICEIndicator,
    }

    if indicator_type not in indicators:
        raise ValueError(f"未対応の価格変換指標タイプ: {indicator_type}")

    return indicators[indicator_type]()


# サポートされている指標の情報
PRICE_TRANSFORM_INDICATORS_INFO = {
    "AVGPRICE": {
        "periods": [1],
        "description": "AVGPRICE - Average Price、OHLC価格の平均値",
        "category": "price_transform",
    },
    "MEDPRICE": {
        "periods": [1],
        "description": "MEDPRICE - Median Price、高値と安値の中央値",
        "category": "price_transform",
    },
    "TYPPRICE": {
        "periods": [1],
        "description": "TYPPRICE - Typical Price、高値・安値・終値の平均",
        "category": "price_transform",
    },
    "WCLPRICE": {
        "periods": [1],
        "description": "WCLPRICE - Weighted Close Price、終値に重みを付けた価格",
        "category": "price_transform",
    },
}
