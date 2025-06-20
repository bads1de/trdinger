"""
トレンド系テクニカル指標

SMA（単純移動平均）、EMA（指数移動平均）、MACD の実装を提供します。
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import logging

from .abstract_indicator import BaseIndicator
from .adapters import TrendAdapter, MomentumAdapter

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

        except Exception:
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


class WMAIndicator(BaseIndicator):
    """WMA（Weighted Moving Average）指標"""

    def __init__(self):
        super().__init__(
            indicator_type="WMA", supported_periods=[5, 10, 20, 50, 100, 200]
        )

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        加重移動平均（WMA）を計算（TA-Lib使用）

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            WMA値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        # TA-Libを使用した高速計算
        return TrendAdapter.wma(df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "加重移動平均 - 直近の価格により大きな重みを置いた移動平均"


class HMAIndicator(BaseIndicator):
    """HMA（Hull Moving Average）指標"""

    def __init__(self):
        super().__init__(indicator_type="HMA", supported_periods=[9, 14, 21, 30, 50])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        ハル移動平均（HMA）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            HMA値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        # TrendAdapterを使用したHMA計算
        return TrendAdapter.hma(df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "HMA - ハル移動平均、ラグを最小化した高応答性移動平均"


class VWMAIndicator(BaseIndicator):
    """VWMA（Volume Weighted Moving Average）指標"""

    def __init__(self):
        super().__init__(indicator_type="VWMA", supported_periods=[10, 20, 30, 50])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        出来高加重移動平均（VWMA）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            VWMA値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
            ValueError: 出来高データが存在しない場合
        """
        # 出来高データの存在確認
        if "volume" not in df.columns:
            raise ValueError("VWMA計算には出来高データが必要です")

        # TrendAdapterを使用したVWMA計算
        return TrendAdapter.vwma(df["close"], df["volume"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "VWMA - 出来高加重移動平均、出来高を重みとした移動平均"


class ZLEMAIndicator(BaseIndicator):
    """ZLEMA（Zero Lag Exponential Moving Average）指標"""

    def __init__(self):
        super().__init__(indicator_type="ZLEMA", supported_periods=[9, 14, 21, 30, 50])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        ゼロラグ指数移動平均（ZLEMA）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            ZLEMA値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        # TrendAdapterを使用したZLEMA計算
        return TrendAdapter.zlema(df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "ZLEMA - ゼロラグ指数移動平均、従来のEMAのラグを削減した高応答性移動平均"


class MAMAIndicator(BaseIndicator):
    """MAMA（MESA Adaptive Moving Average）指標"""

    def __init__(self):
        super().__init__(indicator_type="MAMA", supported_periods=[20, 30, 50])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> dict:
        """
        MAMA（MESA Adaptive Moving Average）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（fastlimitの計算に使用）
            **kwargs: 追加パラメータ
                - fastlimit: 高速制限（デフォルト: 0.5）
                - slowlimit: 低速制限（デフォルト: 0.05）

        Returns:
            MAMA値を含む辞書（mama, fama）

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        # パラメータの取得
        fastlimit = kwargs.get("fastlimit", 0.5)
        slowlimit = kwargs.get("slowlimit", 0.05)

        # TrendAdapterを使用したMAMA計算
        return TrendAdapter.mama(df["close"], fastlimit, slowlimit)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "MAMA - MESA適応型移動平均、市場の効率性に応じて自動調整される移動平均"


class MIDPOINTIndicator(BaseIndicator):
    """MIDPOINT（MidPoint over period）指標"""

    def __init__(self):
        super().__init__(indicator_type="MIDPOINT", supported_periods=[14, 20, 30])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        MIDPOINT（MidPoint over period）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            MIDPOINT値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        # TrendAdapterを使用したMIDPOINT計算
        return TrendAdapter.midpoint(df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "MIDPOINT - MidPoint over period、指定期間の最高値と最安値の中点"


class MIDPRICEIndicator(BaseIndicator):
    """MIDPRICE（Midpoint Price over period）指標"""

    def __init__(self):
        super().__init__(indicator_type="MIDPRICE", supported_periods=[14, 20, 30])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        MIDPRICE（Midpoint Price over period）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            MIDPRICE値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
            ValueError: 高値・安値データが存在しない場合
        """
        # 高値・安値データの存在確認
        if "high" not in df.columns or "low" not in df.columns:
            raise ValueError("MIDPRICE計算には高値・安値データが必要です")

        # TrendAdapterを使用したMIDPRICE計算
        return TrendAdapter.midprice(df["high"], df["low"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "MIDPRICE - Midpoint Price over period、指定期間の高値と安値の中点価格"


class TRIMAIndicator(BaseIndicator):
    """TRIMA（Triangular Moving Average）指標"""

    def __init__(self):
        super().__init__(indicator_type="TRIMA", supported_periods=[14, 20, 30, 50])

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        TRIMA（Triangular Moving Average）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            TRIMA値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        # TrendAdapterを使用したTRIMA計算
        return TrendAdapter.trima(df["close"], period)

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "TRIMA - Triangular Moving Average、三角移動平均、より滑らかでノイズの少ない移動平均"


# 指標インスタンスのファクトリー関数
def get_trend_indicator(indicator_type: str) -> BaseIndicator:
    """
    トレンド系指標のインスタンスを取得

    Args:
        indicator_type: 指標タイプ（'SMA', 'EMA', 'MACD', 'KAMA', 'T3', 'TEMA', 'DEMA', 'WMA', 'HMA', 'VWMA', 'ZLEMA', 'MAMA', 'MIDPOINT', 'MIDPRICE', 'TRIMA'）

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
        "WMA": WMAIndicator,
        "HMA": HMAIndicator,
        "VWMA": VWMAIndicator,
        "ZLEMA": ZLEMAIndicator,
        "MAMA": MAMAIndicator,
        "MIDPOINT": MIDPOINTIndicator,
        "MIDPRICE": MIDPRICEIndicator,
        "TRIMA": TRIMAIndicator,
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
    "WMA": {
        "periods": [5, 10, 20, 50, 100, 200],
        "description": "加重移動平均 - 直近の価格により大きな重みを置いた移動平均",
        "category": "trend",
    },
    "HMA": {
        "periods": [9, 14, 21, 30, 50],
        "description": "HMA - ハル移動平均、ラグを最小化した高応答性移動平均",
        "category": "trend",
    },
    "VWMA": {
        "periods": [10, 20, 30, 50],
        "description": "VWMA - 出来高加重移動平均、出来高を重みとした移動平均",
        "category": "trend",
    },
    "ZLEMA": {
        "periods": [9, 14, 21, 30, 50],
        "description": "ZLEMA - ゼロラグ指数移動平均、従来のEMAのラグを削減した高応答性移動平均",
        "category": "trend",
    },
    "MAMA": {
        "periods": [20, 30, 50],
        "description": "MAMA - MESA適応型移動平均、市場の効率性に応じて自動調整される移動平均",
        "category": "trend",
    },
    "MIDPOINT": {
        "periods": [14, 20, 30],
        "description": "MIDPOINT - MidPoint over period、指定期間の最高値と最安値の中点",
        "category": "trend",
    },
    "MIDPRICE": {
        "periods": [14, 20, 30],
        "description": "MIDPRICE - Midpoint Price over period、指定期間の高値と安値の中点価格",
        "category": "trend",
    },
    "TRIMA": {
        "periods": [14, 20, 30, 50],
        "description": "TRIMA - Triangular Moving Average、三角移動平均、より滑らかでノイズの少ない移動平均",
        "category": "trend",
    },
}
