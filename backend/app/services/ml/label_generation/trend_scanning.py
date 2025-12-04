import numpy as np
import pandas as pd
from typing import Optional, List, Tuple


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

        # Pre-calculate X values (time indices) for regressions?
        # Since we do rolling regressions of different lengths, we can't easily vectorize everything
        # without significant memory usage. We'll loop for now.
        # Optimization: We can use numpy sliding windows or just loop efficiently.

        # Convert close to numpy for speed
        close_np = close.values
        # Map index to integer positions
        idx_map = {idx: i for i, idx in enumerate(close.index)}

        for t0 in t_events:
            t0_idx = idx_map[t0]

            best_t_val = 0.0
            best_window = 0
            best_slope = 0.0

            # Iterate over windows
            # We need at least min_window points.
            # Window L means we look at prices from t0 to t0+L (inclusive or exclusive?)
            # Usually regression is on t0, t0+1, ..., t0+L.

            max_idx = min(t0_idx + self.max_window, len(close_np) - 1)
            min_idx = t0_idx + self.min_window

            if min_idx > max_idx:
                continue

            # We will iterate through possible end indices
            # Using a loop here might be slow in Python.
            # For a production system, this inner loop should be optimized or numba-jitted.
            # For now, we'll implement it in pure numpy/python.

            # Extract the maximum possible chunk once
            max_chunk = close_np[t0_idx : max_idx + 1]

            # We can iterate L from min_window to max_window
            # L is the number of bars *after* t0.
            # So regression size is L+1 points.

            found_trend = False

            # Store candidates
            candidates = []

            for L in range(self.min_window, self.max_window + 1, self.step):
                if t0_idx + L >= len(close_np):
                    break

                # Price segment
                y = close_np[t0_idx : t0_idx + L + 1]
                x = np.arange(len(y))

                # Linear Regression
                # slope = cov(x, y) / var(x)
                # intercept = mean(y) - slope * mean(x)
                # t-stat calculation

                n = len(y)
                sum_x = np.sum(x)
                sum_y = np.sum(y)
                sum_xx = np.sum(x * x)
                sum_xy = np.sum(x * y)

                denominator = n * sum_xx - sum_x * sum_x
                if denominator == 0:
                    continue

                slope = (n * sum_xy - sum_x * sum_y) / denominator
                intercept = (sum_y - slope * sum_x) / n

                # Residuals
                y_pred = slope * x + intercept
                residuals = y - y_pred

                # Standard Error of Slope
                # s_err = sqrt( sum(residuals^2) / (n-2) ) / sqrt( sum((x - mean_x)^2) )

                sum_res_sq = np.sum(residuals**2)
                if n <= 2:
                    continue

                sigma_eps = np.sqrt(sum_res_sq / (n - 2))

                # sum((x - mean_x)^2) = sum_xx - sum_x^2/n
                ss_x = sum_xx - (sum_x * sum_x) / n

                if ss_x <= 0 or sigma_eps == 0:
                    # Perfect fit or vertical line?
                    # If sigma_eps is 0, t-value is inf.
                    t_val = np.inf if slope > 0 else -np.inf
                else:
                    se_slope = sigma_eps / np.sqrt(ss_x)
                    t_val = slope / se_slope

                candidates.append((L, t_val, slope))

            if not candidates:
                continue

            # Select best window based on max absolute t-value
            # Note: Some implementations prefer the *first* window that crosses a threshold.
            # But standard Trend Scanning picks the max t-value.

            best_candidate = max(candidates, key=lambda x: abs(x[1]))
            best_L, best_t_val, best_slope = best_candidate

            # Determine label
            label = 0
            if best_t_val > self.min_t_value:
                label = 1
            elif best_t_val < -self.min_t_value:
                label = -1

            # Set result
            t1_idx = t0_idx + best_L
            out.at[t0, "t1"] = close.index[t1_idx]
            out.at[t0, "t_value"] = best_t_val
            out.at[t0, "bin"] = label

            # Calculate return over the selected window
            # ret = (price_at_t1 / price_at_t0) - 1
            ret = (close_np[t1_idx] / close_np[t0_idx]) - 1
            out.at[t0, "ret"] = ret

        return out
