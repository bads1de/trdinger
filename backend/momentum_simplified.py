"""
モメンタム系テクニカル指標（簡素化版）

pandas-taを直接活用し、冗長なラッパーを削除した効率的な実装。
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd
import pandas_ta as ta


class MomentumIndicators:
    """
    モメンタム系指標クラス（簡素化版）

    pandas-taを直接活用し、不要なラッパーを削除。
    """

    @staticmethod
    def rsi(data: Union[np.ndarray, pd.Series], length: int = 14) -> np.ndarray:
        """相対力指数"""
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return ta.rsi(series, length=length).values

    @staticmethod
    def macd(
        data: Union[np.ndarray, pd.Series],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        result = ta.macd(series, fast=fast, slow=slow, signal=signal)

        return (
            result.iloc[:, 0].values,  # MACD
            result.iloc[:, 1].values,  # Signal
            result.iloc[:, 2].values,  # Histogram
        )

    @staticmethod
    def stoch(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        k: int = 14,
        d: int = 3,
        smooth_k: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ストキャスティクス"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.stoch(
            high=high_series,
            low=low_series,
            close=close_series,
            k=k,
            d=d,
            smooth_k=smooth_k,
        )

        return (result.iloc[:, 0].values, result.iloc[:, 1].values)

    @staticmethod
    def willr(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """ウィリアムズ%R"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        return ta.willr(
            high=high_series, low=low_series, close=close_series, length=length
        ).values

    @staticmethod
    def cci(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 20,
    ) -> np.ndarray:
        """商品チャネル指数"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        return ta.cci(
            high=high_series, low=low_series, close=close_series, length=length
        ).values

    @staticmethod
    def roc(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """変化率"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return ta.roc(series, length=length).values

    @staticmethod
    def mom(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """モメンタム"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return ta.mom(series, length=length).values

    @staticmethod
    def adx(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """平均方向性指数"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.adx(
            high=high_series, low=low_series, close=close_series, length=length
        )
        return result.iloc[:, 0].values  # ADX列

    @staticmethod
    def aroon(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """アルーン"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low

        result = ta.aroon(high=high_series, low=low_series, length=length)
        return result.iloc[:, 0].values, result.iloc[:, 1].values

    @staticmethod
    def mfi(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        volume: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """マネーフローインデックス"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close
        volume_series = pd.Series(volume) if isinstance(volume, np.ndarray) else volume

        return ta.mfi(
            high=high_series,
            low=low_series,
            close=close_series,
            volume=volume_series,
            length=length,
        ).values

    @staticmethod
    def apo(
        data: Union[np.ndarray, pd.Series], fast: int = 12, slow: int = 26
    ) -> np.ndarray:
        """Absolute Price Oscillator"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return ta.apo(series, fast=fast, slow=slow).values

    @staticmethod
    def ao(
        high: Union[np.ndarray, pd.Series], low: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """Awesome Oscillator"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        return ta.ao(high=high_series, low=low_series).values

    # 後方互換性のためのエイリアス
    @staticmethod
    def macdext(*args, **kwargs):
        """MACD拡張版（標準MACDで代替）"""
        return MomentumIndicators.macd(*args, **kwargs)

    @staticmethod
    def macdfix(*args, **kwargs):
        """MACD固定版（標準MACDで代替）"""
        return MomentumIndicators.macd(*args, **kwargs)

    @staticmethod
    def stochf(*args, **kwargs):
        """高速ストキャスティクス（標準ストキャスティクスで代替）"""
        return MomentumIndicators.stoch(*args, **kwargs)

    @staticmethod
    def cmo(data: Union[np.ndarray, pd.Series], length: int = 14) -> np.ndarray:
        """チェンジモメンタムオシレーター"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return ta.cmo(series, length=length).values

    @staticmethod
    def trix(data: Union[np.ndarray, pd.Series], length: int = 30) -> np.ndarray:
        """TRIX"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        result = ta.trix(series, length=length)
        return result.iloc[:, 0].values if len(result.columns) > 1 else result.values
