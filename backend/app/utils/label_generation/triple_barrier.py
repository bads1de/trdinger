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
        events['side'] = None # Barrier side touched ('pt', 'sl', 'vertical')
        
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
            t1_vertical = vertical_barrier_times.loc[t0]
            trgt = target.iloc[loc]
            
            # Slice price data from t0 to t1_vertical (or end if NaT)
            df0 = close[t0:t1_vertical] # Includes t0
            
            # Calculate returns relative to t0
            # returns = (df0 / close[t0]) - 1 
            # Optimized: avoid division by scalar in loop if possible, but pandas series op is fast enough
            if pd.isna(close.at[t0]):
                 continue
                 
            returns = (df0 / close.at[t0]) - 1
            
            out_bounds = pd.DataFrame(columns=['touch_type'])
            
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
                first_touch_time = out_bounds.index.min()
                events.at[t0, 't1'] = first_touch_time
                
                touch_type = out_bounds.loc[first_touch_time, 'touch_type']
                if isinstance(touch_type, pd.Series):
                     # Pick first one (unlikely)
                     touch_type = touch_type.iloc[0]
                events.at[t0, 'side'] = touch_type
            else:
                # No barrier touched, use vertical barrier
                events.at[t0, 't1'] = t1_vertical
                events.at[t0, 'side'] = 'vertical'

        return events

    def get_bins(self, events: pd.DataFrame, close: pd.Series, binary_label: bool = False) -> pd.DataFrame:
        """
        Generates labels based on the barrier touch events.

        Args:
            events (pd.DataFrame): Output from get_events.
            close (pd.Series): Close prices.
            binary_label (bool): If True, returns 0/1 labels for Meta-Labeling (1=Trend/PT, 0=Fake/SL/Vertical).

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
        out['bin'] = 0.0 # Default to 0
        
        # Join target and side from events
        out['trgt'] = events_['trgt']
        
        has_side_info = 'side' in events_.columns
        
        if binary_label:
            # Meta-Labeling Mode: 1 if PT touched, 0 otherwise
            if has_side_info:
                # Use explicit side info if available
                out.loc[events_['side'] == 'pt', 'bin'] = 1.0
                out.loc[events_['side'] == 'sl', 'bin'] = 0.0
                out.loc[events_['side'] == 'vertical', 'bin'] = 0.0
            else:
                # Fallback to return-based logic
                # Upper barrier check (PT)
                if self.pt > 0:
                    # Strict check: Must be > PT threshold
                    out.loc[out['ret'] > out['trgt'] * self.pt * 0.999, 'bin'] = 1.0
                # SL and Vertical are implicitly 0
        else:
            # Standard Mode: 1 (PT), -1 (SL), 0 (Vertical/Range)
            
            if has_side_info:
                out.loc[events_['side'] == 'pt', 'bin'] = 1.0
                out.loc[events_['side'] == 'sl', 'bin'] = -1.0
                out.loc[events_['side'] == 'vertical', 'bin'] = 0.0
            else:
                # Fallback to return-based logic
                # Upper barrier check
                if self.pt > 0:
                    out.loc[out['ret'] > out['trgt'] * self.pt * 0.999, 'bin'] = 1.0
                    
                # Lower barrier check
                if self.sl > 0:
                    out.loc[out['ret'] < -out['trgt'] * self.sl * 0.999, 'bin'] = -1.0
            
        return out
