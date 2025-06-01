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
    def _normalize_column_names(data: pd.DataFrame) -> pd.DataFrame:
        """
        データフレームの列名を正規化（大文字・小文字の統一）

        Args:
            data: 入力データフレーム

        Returns:
            列名が正規化されたデータフレーム
        """
        # 列名のマッピング（小文字 → 大文字）
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }

        # 現在の列名を確認して適切にマッピング
        normalized_data = data.copy()
        for old_name, new_name in column_mapping.items():
            if old_name in normalized_data.columns:
                normalized_data = normalized_data.rename(columns={old_name: new_name})

        return normalized_data

    @staticmethod
    def _get_price_column(data: pd.DataFrame, column_type: str) -> pd.Series:
        """
        価格列を取得（列名の大文字・小文字を自動判定）

        Args:
            data: データフレーム
            column_type: 'close', 'open', 'high', 'low', 'volume'

        Returns:
            価格データのSeries
        """
        # 正規化されたデータを使用
        normalized_data = TechnicalIndicators._normalize_column_names(data)

        # 大文字の列名で取得
        column_name = column_type.capitalize()
        if column_name in normalized_data.columns:
            return normalized_data[column_name]
        else:
            raise KeyError(f"Column '{column_name}' not found in data. Available columns: {list(normalized_data.columns)}")

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

        # データの列名を正規化
        normalized_data = TechnicalIndicators._normalize_column_names(data)

        if indicator_name == 'SMA':
            return TechnicalIndicators.sma(
                TechnicalIndicators._get_price_column(data, 'close'),
                params.get('period', 20)
            )

        elif indicator_name == 'EMA':
            return TechnicalIndicators.ema(
                TechnicalIndicators._get_price_column(data, 'close'),
                params.get('period', 20)
            )

        elif indicator_name == 'RSI':
            return TechnicalIndicators.rsi(
                TechnicalIndicators._get_price_column(data, 'close'),
                params.get('period', 14)
            )

        elif indicator_name == 'MACD':
            macd_line, macd_signal, macd_histogram = TechnicalIndicators.macd(
                TechnicalIndicators._get_price_column(data, 'close'),
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
                TechnicalIndicators._get_price_column(data, 'close'),
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
                TechnicalIndicators._get_price_column(data, 'high'),
                TechnicalIndicators._get_price_column(data, 'low'),
                TechnicalIndicators._get_price_column(data, 'close'),
                params.get('k_period', 14),
                params.get('d_period', 3)
            )
            return {
                'k': slowk,
                'd': slowd
            }

        elif indicator_name == 'ATR':
            return TechnicalIndicators.atr(
                TechnicalIndicators._get_price_column(data, 'high'),
                TechnicalIndicators._get_price_column(data, 'low'),
                TechnicalIndicators._get_price_column(data, 'close'),
                params.get('period', 14)
            )

        else:
            raise ValueError(f"Unsupported indicator: {indicator_name}")
