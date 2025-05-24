"""
テクニカル指標計算モジュール
"""
import pandas as pd
import numpy as np


class TechnicalIndicators:
    """テクニカル指標を計算するクラス"""

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """単純移動平均線 (Simple Moving Average)"""
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """指数移動平均線 (Exponential Moving Average)"""
        return data.ewm(span=period).mean()

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """RSI (Relative Strength Index)"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(data, fast_period)
        ema_slow = TechnicalIndicators.ema(data, slow_period)
        macd_line = ema_fast - ema_slow
        macd_signal = TechnicalIndicators.ema(macd_line, signal_period)
        macd_histogram = macd_line - macd_signal
        return macd_line.values, macd_signal.values, macd_histogram.values

    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple:
        """ボリンジャーバンド"""
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper.values, sma.values, lower.values

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> tuple:
        """ストキャスティクス"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent.values, d_percent.values

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """ATR (Average True Range)"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    @staticmethod
    def calculate_indicator(data: pd.DataFrame, indicator_name: str, params: dict) -> pd.Series:
        """
        指標名とパラメータに基づいて指標を計算する汎用メソッド

        Args:
            data: OHLCV データ
            indicator_name: 指標名 ('SMA', 'EMA', 'RSI', 'MACD', etc.)
            params: パラメータ辞書

        Returns:
            計算された指標のSeries
        """
        indicator_name = indicator_name.upper()

        if indicator_name == 'SMA':
            return TechnicalIndicators.sma(data['close'], params.get('period', 20))

        elif indicator_name == 'EMA':
            return TechnicalIndicators.ema(data['close'], params.get('period', 20))

        elif indicator_name == 'RSI':
            return TechnicalIndicators.rsi(data['close'], params.get('period', 14))

        elif indicator_name == 'MACD':
            macd_line, macd_signal, macd_histogram = TechnicalIndicators.macd(
                data['close'],
                params.get('fast_period', 12),
                params.get('slow_period', 26),
                params.get('signal_period', 9)
            )
            # MACDの場合は辞書で返す
            return {
                'macd': macd_line,
                'signal': macd_signal,
                'histogram': macd_histogram
            }

        elif indicator_name == 'BB' or indicator_name == 'BOLLINGER_BANDS':
            upper, middle, lower = TechnicalIndicators.bollinger_bands(
                data['close'],
                params.get('period', 20),
                params.get('std_dev', 2.0)
            )
            return {
                'upper': upper,
                'middle': middle,
                'lower': lower
            }

        elif indicator_name == 'STOCH' or indicator_name == 'STOCHASTIC':
            slowk, slowd = TechnicalIndicators.stochastic(
                data['high'],
                data['low'],
                data['close'],
                params.get('k_period', 14),
                params.get('d_period', 3)
            )
            return {
                'k': slowk,
                'd': slowd
            }

        elif indicator_name == 'ATR':
            return TechnicalIndicators.atr(
                data['high'],
                data['low'],
                data['close'],
                params.get('period', 14)
            )

        else:
            raise ValueError(f"Unsupported indicator: {indicator_name}")
