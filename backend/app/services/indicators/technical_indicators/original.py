"""
独自テクニカル指標モジュール

現在の実装:
- FRAMA (Fractal Adaptive Moving Average)
- SUPER_SMOOTHER (Ehlers 2-Pole Super Smoother Filter)
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class OriginalIndicators:
    """新規の独自指標を提供するクラス"""

    _ALPHA_MIN: Final[float] = 0.01
    _ALPHA_MAX: Final[float] = 1.0

    @staticmethod
    def frama(close: pd.Series, length: int = 16, slow: int = 200) -> pd.Series:
        """Fractal Adaptive Moving Average (FRAMA)

        John Ehlers が提案した適応型移動平均で、ウィンドウのフラクタル次元に応じてスムージング係数を調整する。
        ETFHQ による改良案に従い、slow パラメータで最大スムージング長を調整できるようにする。

        参考文献:
            - "Ehler's Fractal Adaptive Moving Average (FRAMA)", ProRealCode, 2016-11-24
              https://www.prorealcode.com/prorealtime-indicators/ehlers-fractal-adaptive-moving-average/

        Args:
            close: クローズ価格の系列
            length: フラクタル次元を評価するローリングウィンドウ長（偶数、>=4）
            slow: スムージング係数の下限を決める最大期間（>=1）

        Returns:
            FRAMA 値を表す Pandas Series
        """

        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if length < 4:
            raise ValueError("length must be >= 4")
        if length % 2 != 0:
            raise ValueError("length must be an even number")
        if slow < 1:
            raise ValueError("slow must be >= 1")

        if close.empty:
            return pd.Series(np.full(0, np.nan), index=close.index, name="FRAMA")
        if len(close) < length:
            logger.warning(
                "FRAMA: insufficient data length (%s) for window size %s", len(close), length
            )
            return pd.Series(np.full(len(close), np.nan), index=close.index, name="FRAMA")

        prices = close.astype(float).to_numpy(copy=True)
        result = np.empty_like(prices)
        result[:] = np.nan

        half = length // 2
        log2 = np.log(2.0)
        slow_float = float(slow)
        w = 2.303 * np.log(2.0 / (slow_float + 1.0))

        # ウォームアップ期間は元の価格をそのまま返す
        warmup_end = length - 1
        result[:warmup_end] = prices[:warmup_end]

        for idx in range(warmup_end, len(prices)):
            window = prices[idx - length + 1 : idx + 1]
            first_half = window[:half]
            second_half = window[half:]

            n1 = (np.max(first_half) - np.min(first_half)) / half
            n2 = (np.max(second_half) - np.min(second_half)) / half
            n3 = (np.max(window) - np.min(window)) / length

            if n1 > 0 and n2 > 0 and n3 > 0:
                dimen = (np.log(n1 + n2) - np.log(n3)) / log2
            else:
                dimen = 1.0

            alpha = float(np.exp(w * (dimen - 1.0)))
            alpha = float(np.clip(alpha, OriginalIndicators._ALPHA_MIN, OriginalIndicators._ALPHA_MAX))

            prev_value = result[idx - 1] if np.isfinite(result[idx - 1]) else window[-1]
            current_price = window[-1]
            result[idx] = alpha * current_price + (1.0 - alpha) * prev_value

        return pd.Series(result, index=close.index, name="FRAMA")

    @staticmethod
    def super_smoother(close: pd.Series, length: int = 10) -> pd.Series:
        """Ehlers 2-Pole Super Smoother Filter

        John Ehlers が提案したバターワース型の2極フィルターで、0.5*(x[n] + x[n-1])を入力としたIIR構造を用いて
        高周波ノイズを抑制しつつ遅延を最小化する。

        参考実装:
            - "Ehler´s Super Smoothers.", ProRealCode Forum, 2018-04-06
              https://prorealcode.com/topic/ehlers-super-smoothers/

        Args:
            close: クローズ価格の系列
            length: フィルター期間（>=2）

        Returns:
            Super Smoother による平滑化結果
        """

        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if length < 2:
            raise ValueError("length must be >= 2")

        if close.empty:
            return pd.Series(np.full(0, np.nan), index=close.index, name="SUPER_SMOOTHER")

        prices = close.astype(float).to_numpy(copy=True)
        result = np.empty_like(prices)
        result[:] = np.nan

        warmup = min(len(prices), 2)
        result[:warmup] = prices[:warmup]

        sqrt_two = np.sqrt(2.0)
        f = (sqrt_two * np.pi) / float(length)
        a = float(np.exp(-f))
        c2 = 2.0 * a * float(np.cos(f))
        c3 = -(a**2)
        c1 = 1.0 - c2 - c3

        for idx in range(2, len(prices)):
            current = prices[idx]
            previous = prices[idx - 1]
            result[idx] = (
                0.5 * c1 * (current + previous)
                + c2 * result[idx - 1]
                + c3 * result[idx - 2]
            )

        return pd.Series(result, index=close.index, name="SUPER_SMOOTHER")
