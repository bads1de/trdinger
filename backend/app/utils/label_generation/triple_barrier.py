import numpy as np
import pandas as pd
from typing import Optional, List, Union

class TripleBarrier:
    """
    Triple Barrier Method for labeling financial data.
    Based on Marcos Lopez de Prado's "Advances in Financial Machine Learning".
    """
    def __init__(self, pt: float = 1.0, sl: float = 1.0, min_ret: float = 0.001, num_threads: int = 1):
        """
        Args:
            pt (float): Profit Taking multiplier.
            sl (float): Stop Loss multiplier.
            min_ret (float): Minimum return required to be considered a label.
            num_threads (int): Number of threads for parallel processing (not used in simplified version).
        """
        self.pt = pt
        self.sl = sl
        self.min_ret = min_ret
        self.num_threads = num_threads

    def get_events(
        self, 
        close: pd.Series, 
        t_events: pd.DatetimeIndex, 
        pt_sl: List[float], 
        target: pd.Series, 
        min_ret: float, 
        vertical_barrier_times: Optional[pd.Series] = None, 
        side: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Finds the time of the first barrier touch.

        Args:
            close (pd.Series): Close prices.
            t_events (pd.DatetimeIndex): Timestamps of the events (e.g. from CUSUM filter or all timestamps).
            pt_sl (List[float]): [pt multiplier, sl multiplier].
            target (pd.Series): Volatility for dynamic barrier width.
            min_ret (float): Minimum target return.
            vertical_barrier_times (pd.Series): Timestamps of vertical barriers (time horizon).
            side (pd.Series): Optional side of the bet (1 for buy, -1 for sell). If None, checks both sides.

        Returns:
            pd.DataFrame: Events with 't1' (touch time), 'trgt' (target return), and 'side' (if provided).
        """
        # 1. Get target volatility for each event
        target = target.loc[t_events]
        target = target[target > min_ret] # Filter by min_ret

        if target.empty:
            return pd.DataFrame(columns=['t1', 'trgt'])

        # 2. Get vertical barriers
        if vertical_barrier_times is None:
            vertical_barrier_times = pd.Series(pd.NaT, index=t_events)
        else:
            vertical_barrier_times = vertical_barrier_times.loc[t_events]

        # 3. Find touch times
        events = pd.DataFrame(index=target.index)
        events['t1'] = pd.NaT # Time of barrier touch
        events['trgt'] = target
        
        if side is None:
            # Check both sides (volatility based)
            side_ = pd.Series(1.0, index=target.index)
            pt_mult = pt_sl[0]
            sl_mult = pt_sl[1]
        else:
            side_ = side.loc[target.index]
            pt_mult = pt_sl[0]
            sl_mult = pt_sl[1]

        # Loop through events (can be parallelized, but using loop for simplicity/safety)
        for loc, t0 in enumerate(events.index):
            t1_vertical = vertical_barrier_times.iloc[loc]
            trgt = target.iloc[loc]
            
            # Slice price data from t0 to t1_vertical (or end if NaT)
            df0 = close[t0:t1_vertical] # Includes t0
            
            # Calculate returns relative to t0
            # returns = (df0 / close[t0]) - 1 
            # Optimized: avoid division by scalar in loop if possible, but pandas series op is fast enough
            returns = (df0 / close.at[t0]) - 1
            
            out_bounds = pd.DataFrame()
            
            # Check upper barrier (Profit Taking)
            if pt_mult > 0:
                # Upper barrier threshold
                upper_thresh = trgt * pt_mult
                # Find first time return exceeds threshold
                # We want the first index where returns > upper_thresh
                # Note: returns includes t0 where ret=0. So we check from t0+1
                touch_upper = returns[returns > upper_thresh].index.min()
                if pd.notna(touch_upper):
                    out_bounds.loc[touch_upper, 'touch_type'] = 'pt'

            # Check lower barrier (Stop Loss)
            if sl_mult > 0:
                # Lower barrier threshold
                lower_thresh = -trgt * sl_mult
                touch_lower = returns[returns < lower_thresh].index.min()
                if pd.notna(touch_lower):
                    out_bounds.loc[touch_lower, 'touch_type'] = 'sl'

            # Determine which barrier was touched first
            if not out_bounds.empty:
                events.at[t0, 't1'] = out_bounds.index.min()
            else:
                # No barrier touched, use vertical barrier
                events.at[t0, 't1'] = t1_vertical

        return events

    def get_bins(self, events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """
        Generates labels based on the barrier touch events.

        Args:
            events (pd.DataFrame): Output from get_events.
            close (pd.Series): Close prices.

        Returns:
            pd.DataFrame: Labels with 'ret' (return) and 'bin' (label).
        """
        # 1. Drop events with no t1 (should not happen if vertical barrier exists)
        events_ = events.dropna(subset=['t1'])
        
        # 2. Get prices at t0 and t1
        # t0 is the index of events_
        # t1 is events_['t1']
        
        # Join close prices for t0
        px_init = close.loc[events_.index]
        
        # Join close prices for t1 using mapping
        # Use indices to map to prices. t1 might contain timestamps not in close index?
        # Assuming t1 comes from close index.
        
        # Handling potential missing keys if t1 is not in close (e.g. vertical barrier slightly off)
        # Ideally t1 is exactly in close.index.
        # If vertical barrier was constructed from timestamps, it might not match close index exactly.
        # We use 'asof' or reindex logic if needed, but here we assume alignment.
        
        # To be safe, use searchsorted or asof
        t1_prices = []
        for t1_val in events_['t1']:
            # Exact match
            if t1_val in close.index:
                t1_prices.append(close.at[t1_val])
            else:
                # Find nearest prior price
                idx = close.index.get_indexer([t1_val], method='pad')[0]
                if idx != -1:
                    t1_prices.append(close.iloc[idx])
                else:
                    t1_prices.append(np.nan)
        
        px_end = pd.Series(t1_prices, index=events_.index)
        
        # 3. Calculate return
        out = pd.DataFrame(index=events_.index)
        out['ret'] = px_end / px_init - 1
        
        # 4. Assign labels (bins)
        out['bin'] = 0.0 # Default to 0 (Vertical Barrier / Time expiration with small return)
        
        # Assign 1 if PT touched
        # Check if return exceeded target * pt
        # We need the target and pt info.
        # Instead of re-calculating, we can infer from return sign and magnitude relative to target?
        # Or simpler: 
        # - If touched PT (high return) -> 1
        # - If touched SL (low return) -> -1
        # - If vertical barrier -> 0 (or sign of return if configured)
        
        # Standard implementation:
        # If ret > 0 -> 1 ? No, we want to know WHICH barrier was touched.
        # But get_events logic already filtered by barrier.
        
        # Let's use the logic: 
        # if ret > 0 and ret > target * pt -> 1 (PT)
        # if ret < 0 and ret < -target * sl -> -1 (SL)
        # else -> 0
        
        # However, we don't have target/pt available in get_bins easily unless passed or stored.
        # Simpler logic used in practice:
        # 1 if ret > 0
        # -1 if ret < 0
        # 0 if ret == 0 (rare)
        # But this ignores the "Vertical Barrier = 0" logic which is useful for conservative labeling.
        
        # Better Logic:
        # If ret > target * pt * 0.99 (allow tolerance) -> 1
        # If ret < -target * sl * 0.99 -> -1
        # Else -> 0
        
        # Join target from events
        out['trgt'] = events_['trgt']
        
        # Labeling
        # Using min_ret as a filter for 0 class
        # If abs(ret) < min_ret, treat as 0
        
        # Upper barrier check
        if self.pt > 0:
            out.loc[out['ret'] > out['trgt'] * self.pt * 0.999, 'bin'] = 1.0
            
        # Lower barrier check
        if self.sl > 0:
            out.loc[out['ret'] < -out['trgt'] * self.sl * 0.999, 'bin'] = -1.0
            
        return out
