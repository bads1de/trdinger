"""
出来高系テクニカル指標

OBV、Chaikin A/D Line、Chaikin A/D Oscillator の実装を提供します。
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import logging

from .abstract_indicator import BaseIndicator
from .adapters import VolumeAdapter

logger = logging.getLogger(__name__)


class OBVIndicator(BaseIndicator):
    """OBV（On Balance Volume）指標"""

    def __init__(self):
        super().__init__(
            indicator_type="OBV", supported_periods=[1]
        )  # OBVは期間を使用しない

    def calculate(self, df: pd.DataFrame, period: int = 1, **kwargs) -> pd.Series:
        """
        OBV（On Balance Volume）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（OBVでは使用しないが、インターフェース統一のため）

        Returns:
            OBV値のSeries
        """
        # 出来高データの存在確認
        if "volume" not in df.columns:
            raise ValueError("OBV計算には出来高データが必要です")

        # TA-Libを使用した高速計算
        return VolumeAdapter.obv(df["close"], df["volume"])

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "OBV - オンバランスボリューム、出来高の累積による価格予測"


class ADIndicator(BaseIndicator):
    """AD（Chaikin A/D Line）指標"""

    def __init__(self):
        super().__init__(
            indicator_type="AD", supported_periods=[1]
        )  # A/D Lineは期間を使用しない

    def calculate(self, df: pd.DataFrame, period: int = 1, **kwargs) -> pd.Series:
        """
        AD（Chaikin A/D Line）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（A/D Lineでは使用しないが、インターフェース統一のため）

        Returns:
            A/D Line値のSeries
        """
        # 出来高データの存在確認
        if "volume" not in df.columns:
            raise ValueError("A/D Line計算には出来高データが必要です")

        # TA-Libを使用した高速計算
        return VolumeAdapter.ad(df["high"], df["low"], df["close"], df["volume"])

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "A/D Line - チャイキン蓄積/分散ライン、買い圧力と売り圧力のバランス"


class ADOSCIndicator(BaseIndicator):
    """ADOSC（Chaikin A/D Oscillator）指標"""

    def __init__(self):
        super().__init__(
            indicator_type="ADOSC", supported_periods=[3, 10]
        )  # 高速期間と低速期間

    def calculate(self, df: pd.DataFrame, period: int = 3, **kwargs) -> pd.Series:
        """
        ADOSC（Chaikin A/D Oscillator）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 高速期間（デフォルト: 3）
            **kwargs: 追加パラメータ
                - slow_period: 低速期間（デフォルト: 10）

        Returns:
            ADOSC値のSeries
        """
        # 出来高データの存在確認
        if "volume" not in df.columns:
            raise ValueError("ADOSC計算には出来高データが必要です")

        # パラメータ取得
        fast_period = period
        slow_period = kwargs.get("slow_period", 10)

        # TA-Libを使用した高速計算
        return VolumeAdapter.adosc(
            df["high"], df["low"], df["close"], df["volume"], fast_period, slow_period
        )

    async def calculate_and_format(
        self,
        symbol: str,
        timeframe: str,
        period: int,
        limit: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        ADOSC指標を計算してフォーマットされた結果を返す（オーバーライド）
        """
        try:
            # パラメータ検証
            self.validate_parameters(period, **kwargs)

            # OHLCVデータを取得
            df = await self.get_ohlcv_data(symbol, timeframe, limit)

            # データ検証
            slow_period = kwargs.get("slow_period", 10)
            self.validate_data(df, max(period, slow_period))

            # 指標を計算
            result = self.calculate(df, period, **kwargs)

            # 単一値としてフォーマット
            formatted_result = self.format_single_value_result(
                result, symbol, timeframe, period
            )

            return formatted_result

        except Exception as e:
            raise

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "ADOSC - チャイキンA/Dオシレーター、A/D Lineの移動平均の差"


# 指標インスタンスのファクトリー関数
def get_volume_indicator(indicator_type: str) -> BaseIndicator:
    """
    出来高系指標のインスタンスを取得

    Args:
        indicator_type: 指標タイプ（'OBV', 'AD', 'ADOSC'）

    Returns:
        指標インスタンス

    Raises:
        ValueError: サポートされていない指標タイプの場合
    """
    indicators = {
        "OBV": OBVIndicator,
        "AD": ADIndicator,
        "ADOSC": ADOSCIndicator,
    }

    if indicator_type not in indicators:
        raise ValueError(
            f"サポートされていない出来高系指標です: {indicator_type}. "
            f"サポート対象: {list(indicators.keys())}"
        )

    return indicators[indicator_type]()


# サポートされている指標の情報
VOLUME_INDICATORS_INFO = {
    "OBV": {
        "periods": [1],
        "description": "OBV - オンバランスボリューム、出来高の累積による価格予測",
        "category": "volume",
    },
    "AD": {
        "periods": [1],
        "description": "A/D Line - チャイキン蓄積/分散ライン、買い圧力と売り圧力のバランス",
        "category": "volume",
    },
    "ADOSC": {
        "periods": [3, 10],
        "description": "ADOSC - チャイキンA/Dオシレーター、A/D Lineの移動平均の差",
        "category": "volume",
    },
}
