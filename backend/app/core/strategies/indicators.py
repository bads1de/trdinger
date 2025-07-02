"""
テクニカル指標関数（オートストラテジー最適化版）

backtesting.pyで使用するテクニカル指標を定義します。
numpy配列ベースの新しいアーキテクチャを使用し、最大限のパフォーマンスを実現します。

主な特徴:
- backtesting.pyとの完全な互換性
- numpy配列ネイティブ処理による最高速度
- Ta-lib直接呼び出しによる最大パフォーマンス
- pandas Seriesの変換を最小限に抑制
"""

import logging
import pandas as pd
import numpy as np

from typing import Union, List


from ..services.indicators.trend import TrendIndicators
from ..services.indicators.momentum import MomentumIndicators
from ..services.indicators.volatility import VolatilityIndicators
from ..services.indicators.utils import ensure_numpy_array
from ..utils.data_utils import ensure_series

logger = logging.getLogger(__name__)


def SMA(data: Union[pd.Series, List, np.ndarray], period: int) -> pd.Series:
    """
    Simple Moving Average (単純移動平均) - numpy配列最適化版

    Args:
        data: 価格データ（通常はClose価格）
        period: 移動平均の期間

    Returns:
        SMAの値を含むpandas.Series

    Note:
        numpy配列ベースの高速計算を使用し、結果をpandas.Seriesに変換して
        backtesting.pyとの互換性を維持します。
    """
    try:
        # numpy配列に変換して高速計算
        data_array = ensure_numpy_array(data)
        result_array = TrendIndicators.sma(data_array, period=period)

        # backtesting.py互換性のためpandas.Seriesに変換
        if isinstance(data, pd.Series):
            return pd.Series(result_array, index=data.index, name=f"SMA_{period}")
        else:
            return pd.Series(result_array, name=f"SMA_{period}")

    except Exception as e:
        logger.error(f"SMA計算エラー: {e}")
        raise


def EMA(data: Union[pd.Series, List, np.ndarray], period: int) -> pd.Series:
    """
    Exponential Moving Average (指数移動平均) - numpy配列最適化版

    Args:
        data: 価格データ（通常はClose価格）
        period: 移動平均の期間

    Returns:
        EMAの値を含むpandas.Series

    Note:
        numpy配列ベースの高速計算を使用し、結果をpandas.Seriesに変換して
        backtesting.pyとの互換性を維持します。
    """
    try:
        # numpy配列に変換して高速計算
        data_array = ensure_numpy_array(data)
        result_array = TrendIndicators.ema(data_array, period=period)

        # backtesting.py互換性のためpandas.Seriesに変換
        if isinstance(data, pd.Series):
            return pd.Series(result_array, index=data.index, name=f"EMA_{period}")
        else:
            return pd.Series(result_array, name=f"EMA_{period}")

    except Exception as e:
        logger.error(f"EMA計算エラー: {e}")
        raise


def RSI(data: Union[pd.Series, List, np.ndarray], period: int = 14) -> pd.Series:
    """
    Relative Strength Index (相対力指数) - numpy配列最適化版

    Args:
        data: 価格データ（通常はClose価格）
        period: RSIの期間（デフォルト: 14）

    Returns:
        RSIの値を含むpandas.Series（0-100の範囲）

    Note:
        numpy配列ベースの高速計算を使用し、結果をpandas.Seriesに変換して
        backtesting.pyとの互換性を維持します。
    """
    try:
        # numpy配列に変換して高速計算
        data_array = ensure_numpy_array(data)
        result_array = MomentumIndicators.rsi(data_array, period=period)

        # backtesting.py互換性のためpandas.Seriesに変換
        if isinstance(data, pd.Series):
            return pd.Series(result_array, index=data.index, name=f"RSI_{period}")
        else:
            return pd.Series(result_array, name=f"RSI_{period}")

    except Exception as e:
        logger.error(f"RSI計算エラー: {e}")
        raise


def MACD(
    data: Union[pd.Series, List, np.ndarray],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple:
    """
    Moving Average Convergence Divergence (MACD) - numpy配列最適化版

    Args:
        data: 価格データ（通常はClose価格）
        fast_period: 短期EMAの期間（デフォルト: 12）
        slow_period: 長期EMAの期間（デフォルト: 26）
        signal_period: シグナル線の期間（デフォルト: 9）

    Returns:
        tuple: (MACD線, シグナル線, ヒストグラム) - 全てpandas.Series

    Note:
        numpy配列ベースの高速計算を使用し、結果をpandas.Seriesに変換して
        backtesting.pyとの互換性を維持します。
    """
    try:
        # numpy配列に変換して高速計算
        data_array = ensure_numpy_array(data)
        macd_array, signal_array, histogram_array = MomentumIndicators.macd(
            data_array, fast=fast_period, slow=slow_period, signal=signal_period
        )

        # backtesting.py互換性のためpandas.Seriesに変換
        if isinstance(data, pd.Series):
            index = data.index
        else:
            index = None

        macd_series = pd.Series(macd_array, index=index, name="MACD")
        signal_series = pd.Series(signal_array, index=index, name="MACD_Signal")
        histogram_series = pd.Series(
            histogram_array, index=index, name="MACD_Histogram"
        )

        return macd_series, signal_series, histogram_series

    except Exception as e:
        logger.error(f"MACD計算エラー: {e}")
        raise


def BollingerBands(
    data: Union[pd.Series, List, np.ndarray], period: int = 20, std_dev: float = 2.0
) -> tuple:
    """
    Bollinger Bands (ボリンジャーバンド) - numpy配列最適化版

    Args:
        data: 価格データ（通常はClose価格）
        period: 移動平均の期間（デフォルト: 20）
        std_dev: 標準偏差の倍数（デフォルト: 2.0）

    Returns:
        tuple: (上限バンド, 中央線(SMA), 下限バンド) - 全てpandas.Series

    Note:
        numpy配列ベースの高速計算を使用し、結果をpandas.Seriesに変換して
        backtesting.pyとの互換性を維持します。
    """
    try:
        # numpy配列に変換して高速計算
        data_array = ensure_numpy_array(data)
        upper_array, middle_array, lower_array = VolatilityIndicators.bollinger_bands(
            data_array, period=period, std_dev=std_dev
        )

        # backtesting.py互換性のためpandas.Seriesに変換
        if isinstance(data, pd.Series):
            index = data.index
        else:
            index = None

        upper_series = pd.Series(upper_array, index=index, name=f"BB_Upper_{period}")
        middle_series = pd.Series(middle_array, index=index, name=f"BB_Middle_{period}")
        lower_series = pd.Series(lower_array, index=index, name=f"BB_Lower_{period}")

        return upper_series, middle_series, lower_series

    except Exception as e:
        logger.error(f"BollingerBands計算エラー: {e}")
        raise


def Stochastic(
    high: Union[pd.Series, List, np.ndarray],
    low: Union[pd.Series, List, np.ndarray],
    close: Union[pd.Series, List, np.ndarray],
    k_period: int = 14,
    d_period: int = 3,
) -> tuple:
    """
    Stochastic Oscillator (ストキャスティクス) - TA-Lib使用

    Args:
        high: 高値データ
        low: 安値データ
        close: 終値データ
        k_period: %Kの期間（デフォルト: 14）
        d_period: %Dの期間（デフォルト: 3）

    Returns:
        tuple: (%K, %D)

    Raises:
        ImportError: TA-Libが利用できない場合
        TALibCalculationError: TA-Lib計算エラーの場合
    """
    orchestrator = TechnicalIndicatorService()
    high_series = ensure_series(high)
    low_series = ensure_series(low)
    close_series = ensure_series(close)

    # DataFrameに変換（IndicatorOrchestratorはDataFrameを期待）
    df = pd.DataFrame({"high": high_series, "low": low_series, "close": close_series})

    # IndicatorOrchestratorでStochasticを計算（DataFrameで返される）
    stoch_result = orchestrator.calculate_indicator(
        df, indicator_type="STOCH", period=k_period, d_period=d_period
    )

    # 既存のAPIに合わせてtupleで返す
    return stoch_result["k_percent"], stoch_result["d_percent"]


def ATR(
    high: Union[pd.Series, List, np.ndarray],
    low: Union[pd.Series, List, np.ndarray],
    close: Union[pd.Series, List, np.ndarray],
    period: int = 14,
) -> pd.Series:
    """
    Average True Range (平均真の値幅) - numpy配列最適化版

    Args:
        high: 高値データ
        low: 安値データ
        close: 終値データ
        period: ATRの期間（デフォルト: 14）

    Returns:
        ATRの値を含むpandas.Series

    Note:
        numpy配列ベースの高速計算を使用し、結果をpandas.Seriesに変換して
        backtesting.pyとの互換性を維持します。
    """
    try:
        # numpy配列に変換して高速計算
        high_array = ensure_numpy_array(high)
        low_array = ensure_numpy_array(low)
        close_array = ensure_numpy_array(close)

        result_array = VolatilityIndicators.atr(
            high_array, low_array, close_array, period=period
        )

        # backtesting.py互換性のためpandas.Seriesに変換
        if isinstance(close, pd.Series):
            return pd.Series(result_array, index=close.index, name=f"ATR_{period}")
        else:
            return pd.Series(result_array, name=f"ATR_{period}")

    except Exception as e:
        logger.error(f"ATR計算エラー: {e}")
        raise
