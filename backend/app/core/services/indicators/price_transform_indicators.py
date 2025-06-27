"""
価格変換系テクニカル指標

AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE の実装を提供します。
"""

import pandas as pd
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


class HTDCPERIODIndicator(BaseIndicator):
    """HT_DCPERIOD（Hilbert Transform - Dominant Cycle Period）指標"""

    def __init__(self):
        super().__init__(indicator_type="HT_DCPERIOD", supported_periods=[1])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        HT_DCPERIOD（Hilbert Transform - Dominant Cycle Period）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（HT_DCPERIODは期間を使用しないが、統一性のため）

        Returns:
            HT_DCPERIOD値のSeries
        """
        return PriceTransformAdapter.ht_dcperiod(df["close"])

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "HT_DCPERIOD - ヒルベルト変換による支配的サイクル期間"


class HTDCPHASEIndicator(BaseIndicator):
    """HT_DCPHASE（Hilbert Transform - Dominant Cycle Phase）指標"""

    def __init__(self):
        super().__init__(indicator_type="HT_DCPHASE", supported_periods=[1])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        HT_DCPHASE（Hilbert Transform - Dominant Cycle Phase）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（HT_DCPHASEは期間を使用しないが、統一性のため）

        Returns:
            HT_DCPHASE値のSeries
        """
        return PriceTransformAdapter.ht_dcphase(df["close"])

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "HT_DCPHASE - ヒルベルト変換による支配的サイクル位相"


class HTPHASORIndicator(BaseIndicator):
    """HT_PHASOR（Hilbert Transform - Phasor Components）指標"""

    def __init__(self):
        super().__init__(indicator_type="HT_PHASOR", supported_periods=[1])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        HT_PHASOR（Hilbert Transform - Phasor Components）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（HT_PHASORは期間を使用しないが、統一性のため）

        Returns:
            HT_PHASOR値のSeries（複数値の場合は最初の値）
        """
        return PriceTransformAdapter.ht_phasor(df["close"])

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "HT_PHASOR - ヒルベルト変換によるフェーザー成分"


class HTSINEIndicator(BaseIndicator):
    """HT_SINE（Hilbert Transform - SineWave）指標"""

    def __init__(self):
        super().__init__(indicator_type="HT_SINE", supported_periods=[1])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        HT_SINE（Hilbert Transform - SineWave）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（HT_SINEは期間を使用しないが、統一性のため）

        Returns:
            HT_SINE値のSeries（複数値の場合は最初の値）
        """
        return PriceTransformAdapter.ht_sine(df["close"])

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "HT_SINE - ヒルベルト変換による正弦波"


class HTTRENDMODEIndicator(BaseIndicator):
    """HT_TRENDMODE（Hilbert Transform - Trend vs Cycle Mode）指標"""

    def __init__(self):
        super().__init__(indicator_type="HT_TRENDMODE", supported_periods=[1])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        HT_TRENDMODE（Hilbert Transform - Trend vs Cycle Mode）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（HT_TRENDMODEは期間を使用しないが、統一性のため）

        Returns:
            HT_TRENDMODE値のSeries
        """
        return PriceTransformAdapter.ht_trendmode(df["close"])

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "HT_TRENDMODE - ヒルベルト変換によるトレンド対サイクルモード"


class FAMAIndicator(BaseIndicator):
    """FAMA（Following Adaptive Moving Average）指標"""

    def __init__(self):
        super().__init__(indicator_type="FAMA", supported_periods=[14, 20, 30])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        FAMA（Following Adaptive Moving Average）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            FAMA値のSeries
        """
        return PriceTransformAdapter.fama(df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "FAMA - Following Adaptive Moving Average、MAMA指標の追従成分"


class SAREXTIndicator(BaseIndicator):
    """SAREXT（Parabolic SAR - Extended）指標"""

    def __init__(self):
        super().__init__(indicator_type="SAREXT", supported_periods=[1])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        SAREXT（Parabolic SAR - Extended）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（SAREXTは期間を使用しないが、統一性のため）

        Returns:
            SAREXT値のSeries
        """
        return PriceTransformAdapter.sarext(df["high"], df["low"])

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "SAREXT - Parabolic SAR Extended、拡張版パラボリックSAR"


class SARIndicator(BaseIndicator):
    """SAR（Parabolic SAR）指標"""

    def __init__(self):
        super().__init__(indicator_type="SAR", supported_periods=[1])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        SAR（Parabolic SAR）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（SARは期間を使用しないが、統一性のため）

        Returns:
            SAR値のSeries
        """
        return PriceTransformAdapter.sar(df["high"], df["low"])

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "SAR - Parabolic SAR、パラボリック・ストップ・アンド・リバース"


# 指標インスタンスのファクトリー関数
def get_price_transform_indicator(indicator_type: str) -> BaseIndicator:
    """
    価格変換系指標のインスタンスを取得

    Args:
        indicator_type: 指標タイプ

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
        "HT_DCPERIOD": HTDCPERIODIndicator,
        "HT_DCPHASE": HTDCPHASEIndicator,
        "HT_PHASOR": HTPHASORIndicator,
        "HT_SINE": HTSINEIndicator,
        "HT_TRENDMODE": HTTRENDMODEIndicator,
        "FAMA": FAMAIndicator,
        "SAREXT": SAREXTIndicator,
        "SAR": SARIndicator,
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
    "HT_DCPERIOD": {
        "periods": [1],
        "description": "HT_DCPERIOD - ヒルベルト変換による支配的サイクル期間",
        "category": "price_transform",
    },
    "HT_DCPHASE": {
        "periods": [1],
        "description": "HT_DCPHASE - ヒルベルト変換による支配的サイクル位相",
        "category": "price_transform",
    },
    "HT_PHASOR": {
        "periods": [1],
        "description": "HT_PHASOR - ヒルベルト変換によるフェーザー成分",
        "category": "price_transform",
    },
    "HT_SINE": {
        "periods": [1],
        "description": "HT_SINE - ヒルベルト変換による正弦波",
        "category": "price_transform",
    },
    "HT_TRENDMODE": {
        "periods": [1],
        "description": "HT_TRENDMODE - ヒルベルト変換によるトレンド対サイクルモード",
        "category": "price_transform",
    },
    "FAMA": {
        "periods": [14, 20, 30],
        "description": "FAMA - Following Adaptive Moving Average、MAMA指標の追従成分",
        "category": "price_transform",
    },
    "SAREXT": {
        "periods": [1],
        "description": "SAREXT - Parabolic SAR Extended、拡張版パラボリックSAR",
        "category": "price_transform",
    },
    "SAR": {
        "periods": [1],
        "description": "SAR - Parabolic SAR、パラボリック・ストップ・アンド・リバース",
        "category": "price_transform",
    },
}
