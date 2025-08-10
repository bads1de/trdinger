"""
互換レイヤ: pandas_ta_utils

過去コードやテストが参照するインターフェースを維持するための薄いラッパー。
内部実装は technical_indicators.* クラスに委譲します。
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import pandas as pd

from .technical_indicators.trend import TrendIndicators
from .technical_indicators.momentum import MomentumIndicators
from .technical_indicators.volatility import VolatilityIndicators
from .utils import PandasTAError

ArrayLike = Union[np.ndarray, pd.Series]


def sma(data: ArrayLike, period: int) -> np.ndarray:
    return TrendIndicators.sma(data, length=period)


def ema(data: ArrayLike, period: int) -> np.ndarray:
    return TrendIndicators.ema(data, length=period)


def rsi(data: ArrayLike, period: int = 14) -> np.ndarray:
    return MomentumIndicators.rsi(data, length=period)


def macd(
    data: ArrayLike,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return MomentumIndicators.macd(
        data, fast=fastperiod, slow=slowperiod, signal=signalperiod
    )


def atr(
    high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 14
) -> np.ndarray:
    return VolatilityIndicators.atr(high, low, close, length=period)


def bbands(
    data: ArrayLike, period: int = 20, std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return VolatilityIndicators.bbands(data, length=period, std=std_dev)


__all__ = [
    "sma",
    "ema",
    "rsi",
    "macd",
    "atr",
    "bbands",
    "PandasTAError",
]

