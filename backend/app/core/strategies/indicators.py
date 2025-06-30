"""
テクニカル指標関数

backtesting.pyで使用するテクニカル指標を定義します。
既存のテクニカル指標計算機能と統合可能です。
"""

import pandas as pd
import numpy as np
from typing import Union, List, cast
import logging

from ..services.indicators.indicator_orchestrator import TechnicalIndicatorService
from ..utils.data_utils import ensure_series

logger = logging.getLogger(__name__)


def SMA(data: Union[pd.Series, List, np.ndarray], period: int) -> pd.Series:
    """
    Simple Moving Average (単純移動平均) - IndicatorOrchestrator使用

    Args:
        data: 価格データ（通常はClose価格）
        period: 移動平均の期間

    Returns:
        SMAの値を含むpandas.Series

    Raises:
        ImportError: TA-Libが利用できない場合
        TALibCalculationError: TA-Lib計算エラーの場合
    """
    orchestrator = TechnicalIndicatorService()
    data_series = ensure_series(data)

    # DataFrameに変換（IndicatorOrchestratorはDataFrameを期待）
    df = pd.DataFrame({"close": data_series})

    result = orchestrator.calculate_indicator(df, indicator_type="SMA", period=period)
    return cast(pd.Series, result)


def EMA(data: Union[pd.Series, List, np.ndarray], period: int) -> pd.Series:
    """
    Exponential Moving Average (指数移動平均) - IndicatorOrchestrator使用

    Args:
        data: 価格データ（通常はClose価格）
        period: 移動平均の期間

    Returns:
        EMAの値を含むpandas.Series

    Raises:
        ImportError: TA-Libが利用できない場合
        TALibCalculationError: TA-Lib計算エラーの場合
    """
    orchestrator = TechnicalIndicatorService()
    data_series = ensure_series(data)

    # DataFrameに変換（IndicatorOrchestratorはDataFrameを期待）
    df = pd.DataFrame({"close": data_series})

    result = orchestrator.calculate_indicator(df, indicator_type="EMA", period=period)
    return cast(pd.Series, result)


def RSI(data: Union[pd.Series, List, np.ndarray], period: int = 14) -> pd.Series:
    """
    Relative Strength Index (相対力指数) - IndicatorOrchestrator使用

    Args:
        data: 価格データ（通常はClose価格）
        period: RSIの期間（デフォルト: 14）

    Returns:
        RSIの値を含むpandas.Series（0-100の範囲）

    Raises:
        ImportError: TA-Libが利用できない場合
        TALibCalculationError: TA-Lib計算エラーの場合
    """
    orchestrator = TechnicalIndicatorService()
    data_series = ensure_series(data)

    # DataFrameに変換（IndicatorOrchestratorはDataFrameを期待）
    df = pd.DataFrame({"close": data_series})

    result = orchestrator.calculate_indicator(df, indicator_type="RSI", period=period)
    return cast(pd.Series, result)


def MACD(
    data: Union[pd.Series, List, np.ndarray],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple:
    """
    Moving Average Convergence Divergence (MACD) - TA-Lib使用

    Args:
        data: 価格データ（通常はClose価格）
        fast_period: 短期EMAの期間（デフォルト: 12）
        slow_period: 長期EMAの期間（デフォルト: 26）
        signal_period: シグナル線の期間（デフォルト: 9）

    Returns:
        tuple: (MACD線, シグナル線, ヒストグラム)

    Raises:
        ImportError: TA-Libが利用できない場合
        TALibCalculationError: TA-Lib計算エラーの場合
    """
    orchestrator = TechnicalIndicatorService()
    data_series = ensure_series(data)

    # DataFrameに変換（IndicatorOrchestratorはDataFrameを期待）
    df = pd.DataFrame({"close": data_series})

    # IndicatorOrchestratorでMACDを計算（辞書で返される）
    macd_result = orchestrator.calculate_indicator(
        df,
        indicator_type="MACD",
        fast_period=fast_period,
        slow_period=slow_period,
        signal_period=signal_period,
    )

    # 既存のAPIに合わせてtupleで返す
    return (
        macd_result["macd_line"],
        macd_result["signal_line"],
        macd_result["histogram"],
    )


def BollingerBands(
    data: Union[pd.Series, List, np.ndarray], period: int = 20, std_dev: float = 2.0
) -> tuple:
    """
    Bollinger Bands (ボリンジャーバンド) - TA-Lib使用

    Args:
        data: 価格データ（通常はClose価格）
        period: 移動平均の期間（デフォルト: 20）
        std_dev: 標準偏差の倍数（デフォルト: 2.0）

    Returns:
        tuple: (上限バンド, 中央線(SMA), 下限バンド)

    Raises:
        ImportError: TA-Libが利用できない場合
        TALibCalculationError: TA-Lib計算エラーの場合
    """
    orchestrator = TechnicalIndicatorService()
    data_series = ensure_series(data)

    # DataFrameに変換（IndicatorOrchestratorはDataFrameを期待）
    df = pd.DataFrame({"close": data_series})

    # IndicatorOrchestratorでBollinger Bandsを計算（DataFrameで返される）
    bb_result = orchestrator.calculate_indicator(
        df, indicator_type="BB", period=period, std_dev=std_dev
    )

    # 既存のAPIに合わせてtupleで返す
    return bb_result["upper"], bb_result["middle"], bb_result["lower"]


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
    Average True Range (平均真の値幅) - TA-Lib使用

    Args:
        high: 高値データ
        low: 安値データ
        close: 終値データ
        period: ATRの期間（デフォルト: 14）

    Returns:
        ATRの値を含むpandas.Series

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

    result = orchestrator.calculate_indicator(df, indicator_type="ATR", period=period)
    return cast(pd.Series, result)
