"""
ボラティリティ系テクニカル指標

ボリンジャーバンド（BB）、ATR の実装を提供します。
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import logging

from .abstract_indicator import BaseIndicator
from .adapters import VolatilityAdapter

logger = logging.getLogger(__name__)


class BollingerBandsIndicator(BaseIndicator):
    """ボリンジャーバンド（Bollinger Bands）指標"""

    def __init__(self):
        super().__init__(indicator_type="BB", supported_periods=[20])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.DataFrame:
        """
        ボリンジャーバンド（Bollinger Bands）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常20）

        Returns:
            ボリンジャーバンド値を含むDataFrame（middle, upper, lower）
        """
        # TA-Libを使用した高速計算
        bb_result = VolatilityAdapter.bollinger_bands(
            df["close"], period=period, std_dev=2.0
        )

        # DataFrameに変換して返す
        result = pd.DataFrame(
            {
                "middle": bb_result["middle"],
                "upper": bb_result["upper"],
                "lower": bb_result["lower"],
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
                "lower_band": "lower",
            }

            formatted_result = self.format_multi_value_result(
                result, symbol, timeframe, period, value_columns
            )

            return formatted_result

        except Exception:
            raise

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "ボリンジャーバンド - ボラティリティとサポート・レジスタンスを示す"


class ATRIndicator(BaseIndicator):
    """ATR（Average True Range）指標"""

    def __init__(self):
        super().__init__(indicator_type="ATR", supported_periods=[14, 21])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        ATR（Average True Range）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常14）

        Returns:
            ATR値のSeries
        """
        # TA-Libを使用した高速計算
        return VolatilityAdapter.atr(df["high"], df["low"], df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "ATR - 平均真の値幅、ボラティリティを測定する指標"


class NATRIndicator(BaseIndicator):
    """NATR（Normalized Average True Range）指標"""

    def __init__(self):
        super().__init__(indicator_type="NATR", supported_periods=[14, 21])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        NATR（Normalized Average True Range）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常14または21）

        Returns:
            NATR値のSeries（パーセンテージ）
        """
        # TA-Libを使用した高速計算
        return VolatilityAdapter.natr(df["high"], df["low"], df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "NATR - 正規化平均真の値幅、価格に対する相対的なボラティリティ（%）"


class TRANGEIndicator(BaseIndicator):
    """TRANGE（True Range）指標"""

    def __init__(self):
        super().__init__(
            indicator_type="TRANGE", supported_periods=[1]
        )  # TRANGEは期間を使用しない

    def calculate(self, df: pd.DataFrame, period: int = 1, **kwargs) -> pd.Series:
        """
        TRANGE（True Range）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（TRANGEでは使用しないが、インターフェース統一のため）

        Returns:
            True Range値のSeries
        """
        # TA-Libを使用した高速計算
        return VolatilityAdapter.trange(df["high"], df["low"], df["close"])

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "TRANGE - 真の値幅、各期間の実際の価格変動幅"


class KeltnerChannelsIndicator(BaseIndicator):
    """Keltner Channels（ケルトナーチャネル）指標"""

    def __init__(self):
        super().__init__(indicator_type="KELTNER", supported_periods=[10, 14, 20])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.DataFrame:
        """
        Keltner Channels（ケルトナーチャネル）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間
            **kwargs: 追加パラメータ
                - multiplier: ATRの倍数（デフォルト: 2.0）

        Returns:
            Keltner ChannelsのDataFrame (upper, middle, lower)

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        # multiplierパラメータの取得
        multiplier = kwargs.get("multiplier", 2.0)

        # VolatilityAdapterを使用したKeltner Channels計算
        return VolatilityAdapter.keltner_channels(
            df["high"], df["low"], df["close"], period, multiplier
        )

    def get_description(self) -> str:
        """指標の説明を取得"""
        return (
            "Keltner Channels - ケルトナーチャネル、ATRベースのボラティリティチャネル"
        )


class STDDEVIndicator(BaseIndicator):
    """STDDEV（Standard Deviation）指標"""

    def __init__(self):
        super().__init__(indicator_type="STDDEV", supported_periods=[5, 10, 20, 30])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        STDDEV（Standard Deviation）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間
            **kwargs: 追加パラメータ
                - nbdev: 標準偏差の倍数（デフォルト: 1.0）

        Returns:
            STDDEV値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        # パラメータの取得
        nbdev = kwargs.get("nbdev", 1.0)

        # VolatilityAdapterを使用したSTDDEV計算
        return VolatilityAdapter.stddev(df["close"], period, nbdev)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "STDDEV - 標準偏差、価格のばらつきを測定するボラティリティ指標"


class DonchianChannelsIndicator(BaseIndicator):
    """Donchian Channels（ドンチャンチャネル）指標"""

    def __init__(self):
        super().__init__(indicator_type="DONCHIAN", supported_periods=[10, 20, 50])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> dict:
        """
        ドンチャンチャネル（Donchian Channels）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            Donchian Channelsを含む辞書（upper, lower, middle）

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        # VolatilityAdapterを使用したDonchian Channels計算
        return VolatilityAdapter.donchian_channels(df["high"], df["low"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "Donchian Channels - ドンチャンチャネル、指定期間の最高値・最低値によるチャネル指標"


# 指標インスタンスのファクトリー関数
def get_volatility_indicator(indicator_type: str) -> BaseIndicator:
    """
    ボラティリティ系指標のインスタンスを取得

    Args:
        indicator_type: 指標タイプ（'BB', 'ATR', 'NATR', 'TRANGE', 'KELTNER', 'STDDEV', 'DONCHIAN'）

    Returns:
        指標インスタンス

    Raises:
        ValueError: サポートされていない指標タイプの場合
    """
    indicators = {
        "BB": BollingerBandsIndicator,
        "ATR": ATRIndicator,
        "NATR": NATRIndicator,
        "TRANGE": TRANGEIndicator,
        "KELTNER": KeltnerChannelsIndicator,
        "STDDEV": STDDEVIndicator,
        "DONCHIAN": DonchianChannelsIndicator,
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
        "category": "volatility",
    },
    "ATR": {
        "periods": [14, 21],
        "description": "ATR - 平均真の値幅、ボラティリティを測定する指標",
        "category": "volatility",
    },
    "NATR": {
        "periods": [14, 21],
        "description": "NATR - 正規化平均真の値幅、価格に対する相対的なボラティリティ（%）",
        "category": "volatility",
    },
    "TRANGE": {
        "periods": [1],
        "description": "TRANGE - 真の値幅、各期間の実際の価格変動幅",
        "category": "volatility",
    },
    "KELTNER": {
        "periods": [10, 14, 20],
        "description": "Keltner Channels - ケルトナーチャネル、ATRベースのボラティリティチャネル",
        "category": "volatility",
    },
    "STDDEV": {
        "periods": [5, 10, 20, 30],
        "description": "STDDEV - 標準偏差、価格のばらつきを測定するボラティリティ指標",
        "category": "volatility",
    },
    "DONCHIAN": {
        "periods": [10, 20, 50],
        "description": "Donchian Channels - ドンチャンチャネル、指定期間の最高値・最低値によるチャネル指標",
        "category": "volatility",
    },
}
