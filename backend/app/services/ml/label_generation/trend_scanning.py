import numpy as np
import pandas as pd
from typing import Optional
from numba import jit
import logging

logger = logging.getLogger(__name__)


@jit(nopython=True)
def _find_best_window_numba(
    close_np: np.ndarray, t0_idx: int, min_window: int, max_window: int, step: int
) -> tuple[float, float, float]:
    """
    Numba optimized function to find the best window with maximum t-value.

    Returns:
        tuple: (best_L, best_t_val, best_slope)
        If no valid window found, returns (0.0, 0.0, 0.0)
    """
    max_len = len(close_np)

    # Initialize best values
    best_t_val = 0.0
    best_L = 0.0
    best_slope = 0.0
    max_abs_t = -1.0
    found = False

    # Iterate over windows
    for L in range(min_window, max_window + 1, step):
        if t0_idx + L >= max_len:
            break

        # Price segment
        # y = close_np[t0_idx : t0_idx + L + 1]
        # x = np.arange(len(y))

        # Manual regression calculation to avoid allocations inside loop
        n = L + 1

        # Calculate sums manually or using slice (slice is okay in numba)
        # Using slices in loop is fine in recent numba, but let's be safe and efficient

        # x is 0, 1, ..., L
        # sum_x = L * (L + 1) / 2
        sum_x = L * (L + 1) * 0.5

        # sum_xx = L * (L + 1) * (2*L + 1) / 6
        sum_xx = L * (L + 1) * (2 * L + 1) / 6.0

        # Calculate sum_y and sum_xy
        sum_y = 0.0
        sum_xy = 0.0

        # Numba loop for sums
        for i in range(n):
            val = close_np[t0_idx + i]
            sum_y += val
            sum_xy += i * val

        denominator = n * sum_xx - sum_x * sum_x
        if abs(denominator) < 1e-9:
            continue

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

        # Residuals and Standard Error
        sum_res_sq = 0.0
        for i in range(n):
            val = close_np[t0_idx + i]
            pred = slope * i + intercept
            res = val - pred
            sum_res_sq += res * res

        if n <= 2:
            continue

        sigma_eps_sq = sum_res_sq / (n - 2)
        if sigma_eps_sq < 0:  # Should not happen
            sigma_eps = 0.0
        else:
            sigma_eps = np.sqrt(sigma_eps_sq)

        # sum((x - mean_x)^2) = ss_x
        ss_x = sum_xx - (sum_x * sum_x) / n

        if ss_x <= 1e-9 or sigma_eps < 1e-9:
            # Perfect fit
            max_t = 100.0
            t_val = max_t if slope > 0 else -max_t
        else:
            se_slope = sigma_eps / np.sqrt(ss_x)
            t_val = slope / se_slope
            # Clip t-value
            if t_val > 100.0:
                t_val = 100.0
            elif t_val < -100.0:
                t_val = -100.0

        # Track best (max absolute t-value)
        abs_t = abs(t_val)
        if abs_t > max_abs_t:
            max_abs_t = abs_t
            best_t_val = t_val
            best_L = float(L)
            best_slope = slope
            found = True

    if not found:
        return 0.0, 0.0, 0.0

    return best_L, best_t_val, best_slope


class TrendScanning:
    """
    Trend Scanning Method for labeling.
    Based on Marcos Lopez de Prado's "Advances in Financial Machine Learning".

    Fits a linear regression to price series over multiple forward windows
    and selects the window with the highest t-statistic for the slope.
    """

    def __init__(
        self,
        min_window: int = 5,
        max_window: int = 20,
        step: int = 1,
        min_t_value: float = 2.0,
    ):
        """
        Args:
            min_window (int): Minimum look-forward window size.
            max_window (int): Maximum look-forward window size.
            step (int): Step size for window iteration.
            min_t_value (float): Minimum t-value threshold to consider a trend significant.
                                 If max t-value < min_t_value, label is 0.
        """
        self.min_window = min_window
        self.max_window = max_window
        self.step = step
        self.min_t_value = min_t_value

    def get_labels(
        self,
        close: pd.Series,
        t_events: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """
        Generate labels using Trend Scanning.

        Args:
            close (pd.Series): Close prices.
            t_events (pd.DatetimeIndex): timestamps to label. If None, label all.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - t1: Timestamp of the end of the selected trend window.
                - t_value: The t-statistic of the selected trend.
                - bin: Label (1 for Up, -1 for Down, 0 for No Trend).
                - ret: Return over the selected window.
        """
        if t_events is None:
            t_events = close.index

        # Ensure t_events are in close.index
        t_events = t_events[t_events.isin(close.index)]

        out = pd.DataFrame(index=t_events, columns=["t1", "t_value", "bin", "ret"])

        # Convert close to numpy for speed
        # Ensure float64
        close_np = close.values.astype(np.float64)

        # Map index to integer positions
        # Optimization: accessing index is slow inside loop, map is better but
        # if t_events matches close.index subset, we can just find indices.
        # But close might be time-series with gaps?
        # Assuming timestamps match exactly.

        # To map t_events timestamps to integer locations in close_np efficiently:
        # If both are sorted, we can use searchsorted.
        # But let's assume they might not be perfectly aligned or unique without work.
        # Using a dictionary map is O(N) setup and O(1) lookup.
        idx_map = {idx: i for i, idx in enumerate(close.index)}

        # Pre-allocate arrays for results to construct DataFrame at once (faster than loc setters)
        n_events = len(t_events)
        t1_results = np.empty(n_events, dtype=object)
        t_val_results = np.zeros(n_events, dtype=np.float64)
        bin_results = np.zeros(n_events, dtype=int)
        ret_results = np.zeros(n_events, dtype=np.float64)

        # Iterate over t_events
        # This loop is still in Python but the heavy lifting is in Numba

        for i, t0 in enumerate(t_events):
            if t0 not in idx_map:
                continue

            t0_idx = idx_map[t0]

            # Call Numba optimized function
            best_L, best_t_val, best_slope = _find_best_window_numba(
                close_np, t0_idx, self.min_window, self.max_window, self.step
            )

            if best_L == 0.0 and best_t_val == 0.0 and best_slope == 0.0:
                # No valid window found
                continue

            best_L_int = int(best_L)

            # Determine label
            label = 0
            if best_t_val > self.min_t_value:
                label = 1
            elif best_t_val < -self.min_t_value:
                label = -1

            # Calculate indices
            t1_idx = t0_idx + best_L_int

            # Set results
            t1_results[i] = close.index[t1_idx]
            t_val_results[i] = best_t_val
            bin_results[i] = label

            # Return
            ret = (close_np[t1_idx] / close_np[t0_idx]) - 1
            ret_results[i] = ret

        out["t1"] = t1_results
        out["t_value"] = t_val_results
        out["bin"] = bin_results
        out["ret"] = ret_results

        # Drop rows where execution failed (t1 is None or similar)
        # In our pre-allocation, None is default for object array? No, unintialized.
        # Better to filter.

        # Filter out invalid rows (where we didn't write anything)
        # Checking t1_results for None/Null
        mask = pd.notna(t1_results)  # boolean array
        out = out[mask]

        return out
