import pandas as pd
import numpy as np
import pandas_ta as ta

def calculate_choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Choppiness Index (CI) を計算します。
    CIは、市場がトレンド状態（低い値）か、レンジ状態（高い値）かを示す指標です。

    Args:
        high (pd.Series): 高値のSeries
        low (pd.Series): 安値のSeries
        close (pd.Series): 終値のSeries
        window (int): 計算期間

    Returns:
        pd.Series: Choppiness Index のSeries
    """
    # ATRの計算 (pandas_taを使用)
    atr = ta.atr(high=high, low=low, close=close, length=1)

    # Sum of ATR over n periods
    sum_atr = atr.rolling(window=window).sum()

    # Highest High and Lowest Low over n periods
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()

    # Range
    price_range = highest_high - lowest_low

    # Avoid division by zero
    ci = pd.Series(np.nan, index=close.index)
    valid_indices = (price_range != 0) & (sum_atr != 0)
    
    # NaNを避けて計算
    numerator = np.log10(sum_atr.loc[valid_indices] / price_range.loc[valid_indices])
    denominator = np.log10(window)
    
    ci.loc[valid_indices] = 100 * numerator / denominator

    # 0から100の範囲にスケーリング
    # CIの標準的な計算は0-100の範囲になりますが、
    # np.log10(window)で割ることでこの範囲に収束するはずです。
    return ci

def calculate_fractal_dimension_index(close: pd.Series, window: int = 10) -> pd.Series:
    """
    Fractal Dimension Index (FDI) を計算します。
    FDIは、時系列データの複雑性または自己相似性の度合いを示します。
    トレンド市場では値が低く（1に近い）、レンジ市場では値が高く（1.5に近い）なります。

    参考実装:
    https://www.quantconnect.com/forum/discussion/11497/fractal-dimension-indicator-fdi/p1

    Args:
        close (pd.Series): 終値のSeries
        window (int): 計算期間

    Returns:
        pd.Series: Fractal Dimension Index のSeries
    """
    fdi = pd.Series(np.nan, index=close.index)

    for i in range(window - 1, len(close)):
        
        # 現在のウィンドウ内の価格データ
        segment = close.iloc[i - window + 1 : i + 1]
        
        # 範囲の最大値と最小値
        max_price = segment.max()
        min_price = segment.min()
        
        # 距離 L (直線の長さ)
        # 各バー間の変化を合計
        l = (segment.diff().abs().dropna()).sum()
        
        if l == 0: # 価格が全く動かない場合
            fdi.iloc[i] = 1.0
            continue
            
        # N (観測点数)
        N = window
        
        # d (最大距離)
        d = max_price - min_price
        
        if d == 0: # レンジ幅が0の場合
            fdi.iloc[i] = 1.0
            continue

        try:
            # Hurst exponent (H) based calculation (simplified)
            # More common approach based on log(L) / log(N)
            # L: path length, N: window size (or max_price - min_price for normalization)
            
            # Simplified approach for FDI (more common in trading context)
            # (log(N-1) + log(sum(abs(close[i]-close[i-1]))) - log(max(close)-min(close))) / log(2)
            # This is one of many ways to approximate fractal dimension
            
            # A common formula for fractal dimension (Higuchi, or similar approximations)
            # Is roughly 1 + (log(L) - log(d)) / log(N) for normalized path length
            
            # A different interpretation for trading, more like efficiency ratio:
            # log(window) / log(window / (L / d))
            # Or simplified: log(sum(abs(diff))) / log(window)
            
            # Let's use a simpler, more direct interpretation related to complexity/smoothness
            # FD = (log(path length) - log(straight line distance)) / log(time_span) + 1
            
            # Here's a common formula used in trading indicators, more related to efficiency ratio
            # log(number of steps) / log(sum of distances / straight line distance)
            
            # Re-evaluating standard fractal dimension for financial time series
            # A robust method is Higuchi, but that's complex.
            # Simpler methods often rely on how noisy the series is relative to its range.
            
            # Let's reconsider the "QuantConnect" forum reference.
            # FD = (Log10(window) + Log10(AverageTrueRange(window)) - Log10(High-Low)) / Log10(window)
            # This looks more like CI than pure FD.
            
            # Let's implement a more direct interpretation of "how jagged is the line"
            # N_max = max(abs(prices[i] - prices[i-2])) for window
            # N_min = min(abs(prices[i] - prices[i-2])) for window
            
            # A common way to get FD related to Hurst:
            # Range / Std Dev
            
            # Given the request for "Fractal Dimension Index", a simple implementation derived from technical analysis literature:
            # The definition is quite varied. Let's use an approach common in TA:
            # FD = (log(C_max) - log(C_min)) / log(N)
            # Where C_max = max(length_of_path_in_window)
            #       C_min = min(length_of_path_in_window)
            # This is not directly what's typically meant.
            
            # Reverting to the simpler, more common definition used in TA indicators (like in TradingView's "Fractal Adaptive Moving Average" logic):
            # FD = log(N) / log(N * (sum(abs(price_diff)) / (max_price - min_price)))
            # This also has issues.
            
            # Let's stick to the simplest interpretation for trading:
            # The ratio of the sum of absolute price changes to the total range within the window.
            # This ratio, when normalized, gives an idea of "jaggedness".
            
            # From De Prado's book, it is often discussed in context of Hurst exponent (H)
            # H relates to 1 - FD.
            # A simple approximation for H is:
            # log(Range/StdDev) / log(window)
            
            # Let's try the definition as:
            # FD = 1 + (log(L) - log(d)) / log(N) where L is path length, d is max span, N is time
            
            # Path length (L)
            path_length = np.sum(np.abs(np.diff(segment)))
            
            # Straight line distance (d) - max_price - min_price
            straight_line_distance = max_price - min_price
            
            if straight_line_distance == 0 or path_length == 0:
                fdi.iloc[i] = 1.0 # Or NaN, depending on desired behavior for flat lines
                continue
            
            # Calculate Fractal Dimension
            # A common definition for FD in financial time series (like in "Trading Chaos")
            # is related to log(Length) / log(span).
            
            # Let's use a common interpretation of FD in trading indicators,
            # which essentially measures how much "space" the price takes up.
            
            # FD = (log(C_high - C_low) - log(N)) / log(2)  -- not this
            
            # Re-checking standard TA-Lib like FDI/FD calculations.
            # There isn't a single "standard" FDI in most TA libraries that is easy to implement.
            # Many use approximations or derivations from Hurst exponent.
            
            # Let's try a common, simpler formula for price series fractal dimension:
            # FD = (log(length_of_path) - log(max_price - min_price)) / log(window) + 1
            # Source: Often seen in discussions of "Generalized Hurst Exponent" or similar.
            
            # length_of_path = sum(abs(segment.diff()))
            # max_price - min_price = straight_line_distance
            
            if path_length <= straight_line_distance: # Should not happen for noisy data, but for flat lines
                 fdi.iloc[i] = 1.0
                 continue
                 
            # Log(L/d) / Log(N)
            fdi.iloc[i] = 1 + (np.log(path_length) - np.log(straight_line_distance)) / np.log(window)
            
        except ValueError: # log(0) など
            fdi.iloc[i] = np.nan
    
    return fdi