"""
テクニカル指標関数

backtesting.pyで使用するテクニカル指標を定義します。
既存のテクニカル指標計算機能と統合可能です。
"""

import pandas as pd
import numpy as np
from typing import Union, List


def SMA(data: Union[pd.Series, List, np.ndarray], period: int) -> pd.Series:
    """
    Simple Moving Average (単純移動平均)
    
    Args:
        data: 価格データ（通常はClose価格）
        period: 移動平均の期間
        
    Returns:
        SMAの値を含むpandas.Series
    """
    if isinstance(data, (list, np.ndarray)):
        data = pd.Series(data)
    
    return data.rolling(window=period, min_periods=period).mean()


def EMA(data: Union[pd.Series, List, np.ndarray], period: int) -> pd.Series:
    """
    Exponential Moving Average (指数移動平均)
    
    Args:
        data: 価格データ（通常はClose価格）
        period: 移動平均の期間
        
    Returns:
        EMAの値を含むpandas.Series
    """
    if isinstance(data, (list, np.ndarray)):
        data = pd.Series(data)
    
    return data.ewm(span=period, adjust=False).mean()


def RSI(data: Union[pd.Series, List, np.ndarray], period: int = 14) -> pd.Series:
    """
    Relative Strength Index (相対力指数)
    
    Args:
        data: 価格データ（通常はClose価格）
        period: RSIの期間（デフォルト: 14）
        
    Returns:
        RSIの値を含むpandas.Series（0-100の範囲）
    """
    if isinstance(data, (list, np.ndarray)):
        data = pd.Series(data)
    
    # 価格変化を計算
    delta = data.diff()
    
    # 上昇と下降を分離
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 移動平均を計算
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    # RSIを計算
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def MACD(data: Union[pd.Series, List, np.ndarray], 
         fast_period: int = 12, 
         slow_period: int = 26, 
         signal_period: int = 9) -> tuple:
    """
    Moving Average Convergence Divergence (MACD)
    
    Args:
        data: 価格データ（通常はClose価格）
        fast_period: 短期EMAの期間（デフォルト: 12）
        slow_period: 長期EMAの期間（デフォルト: 26）
        signal_period: シグナル線の期間（デフォルト: 9）
        
    Returns:
        tuple: (MACD線, シグナル線, ヒストグラム)
    """
    if isinstance(data, (list, np.ndarray)):
        data = pd.Series(data)
    
    # EMAを計算
    ema_fast = EMA(data, fast_period)
    ema_slow = EMA(data, slow_period)
    
    # MACD線を計算
    macd_line = ema_fast - ema_slow
    
    # シグナル線を計算
    signal_line = EMA(macd_line, signal_period)
    
    # ヒストグラムを計算
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def BollingerBands(data: Union[pd.Series, List, np.ndarray], 
                   period: int = 20, 
                   std_dev: float = 2.0) -> tuple:
    """
    Bollinger Bands (ボリンジャーバンド)
    
    Args:
        data: 価格データ（通常はClose価格）
        period: 移動平均の期間（デフォルト: 20）
        std_dev: 標準偏差の倍数（デフォルト: 2.0）
        
    Returns:
        tuple: (上限バンド, 中央線(SMA), 下限バンド)
    """
    if isinstance(data, (list, np.ndarray)):
        data = pd.Series(data)
    
    # 中央線（SMA）を計算
    middle_band = SMA(data, period)
    
    # 標準偏差を計算
    rolling_std = data.rolling(window=period, min_periods=period).std()
    
    # 上限・下限バンドを計算
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    
    return upper_band, middle_band, lower_band


def Stochastic(high: Union[pd.Series, List, np.ndarray],
               low: Union[pd.Series, List, np.ndarray],
               close: Union[pd.Series, List, np.ndarray],
               k_period: int = 14,
               d_period: int = 3) -> tuple:
    """
    Stochastic Oscillator (ストキャスティクス)
    
    Args:
        high: 高値データ
        low: 安値データ
        close: 終値データ
        k_period: %Kの期間（デフォルト: 14）
        d_period: %Dの期間（デフォルト: 3）
        
    Returns:
        tuple: (%K, %D)
    """
    if isinstance(high, (list, np.ndarray)):
        high = pd.Series(high)
    if isinstance(low, (list, np.ndarray)):
        low = pd.Series(low)
    if isinstance(close, (list, np.ndarray)):
        close = pd.Series(close)
    
    # 最高値・最安値を計算
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    
    # %Kを計算
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    
    # %Dを計算（%Kの移動平均）
    d_percent = k_percent.rolling(window=d_period, min_periods=d_period).mean()
    
    return k_percent, d_percent


def ATR(high: Union[pd.Series, List, np.ndarray],
        low: Union[pd.Series, List, np.ndarray],
        close: Union[pd.Series, List, np.ndarray],
        period: int = 14) -> pd.Series:
    """
    Average True Range (平均真の値幅)
    
    Args:
        high: 高値データ
        low: 安値データ
        close: 終値データ
        period: ATRの期間（デフォルト: 14）
        
    Returns:
        ATRの値を含むpandas.Series
    """
    if isinstance(high, (list, np.ndarray)):
        high = pd.Series(high)
    if isinstance(low, (list, np.ndarray)):
        low = pd.Series(low)
    if isinstance(close, (list, np.ndarray)):
        close = pd.Series(close)
    
    # 前日終値
    prev_close = close.shift(1)
    
    # True Rangeを計算
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATRを計算（True Rangeの移動平均）
    atr = true_range.rolling(window=period, min_periods=period).mean()
    
    return atr
