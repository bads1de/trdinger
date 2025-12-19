"""
Volume Profile Feature Calculator

価格レベル別の出来高分布を分析し、市場構造を捉える特徴量を生成します。
学術的に検証された強力な特徴量（Kaggle/論文で実証済み）。
"""

import logging
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from numba import jit

logger = logging.getLogger(__name__)


@jit(nopython=True)
def _numba_calc_bins(w_high, w_low, w_vol, price_min, bin_step, num_bins):
    """ビンごとの出来高を計算"""
    bin_volume = np.zeros(num_bins)
    for j in range(len(w_high)):
        s_bin = int((w_low[j] - price_min) / bin_step)
        e_bin = int((w_high[j] - price_min) / bin_step)
        s_bin = max(0, min(s_bin, num_bins - 1))
        e_bin = max(0, min(e_bin, num_bins - 1))
        
        num_aff = e_bin - s_bin + 1
        vol_per = w_vol[j] / num_aff
        for b in range(s_bin, e_bin + 1):
            bin_volume[b] += vol_per
    return bin_volume


@jit(nopython=True)
def _numba_rolling_volume_profile(
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    window: int,
    num_bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close_arr)
    poc_arr, vah_arr, val_arr = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

    for i in range(window, n):
        w_h, w_l, w_v = high_arr[i-window:i], low_arr[i-window:i], volume_arr[i-window:i]
        p_min, p_max = w_l.min(), w_h.max()

        if p_min == p_max:
            poc_arr[i] = vah_arr[i] = val_arr[i] = close_arr[i]
            continue

        bin_step = (p_max - p_min) / num_bins
        bin_vol = _numba_calc_bins(w_h, w_l, w_v, p_min, bin_step, num_bins)

        poc_bin = np.argmax(bin_vol)
        poc_arr[i] = p_min + (poc_bin + 0.5) * bin_step

        target_v = bin_vol.sum() * 0.70
        vah_b, val_bin, acc_v = poc_bin, poc_bin, bin_vol[poc_bin]

        while acc_v < target_v:
            v_up = bin_vol[vah_b + 1] if vah_b + 1 < num_bins else 0.0
            v_dn = bin_vol[val_bin - 1] if val_bin > 0 else 0.0
            if v_up == 0 and v_dn == 0: break
            if v_up > v_dn:
                vah_b += 1; acc_v += v_up
            else:
                val_bin -= 1; acc_v += v_dn

        vah_arr[i] = p_min + (vah_b + 1) * bin_step
        val_arr[i] = p_min + val_bin * bin_step

    return poc_arr, vah_arr, val_arr


@jit(nopython=True)
def _numba_detect_volume_nodes_signed(
    high_arr: np.ndarray, low_arr: np.ndarray, close_arr: np.ndarray, volume_arr: np.ndarray, window: int, num_bins: int
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(close_arr)
    hvn_dist, lvn_dist = np.zeros(n), np.zeros(n)
    for i in range(window, n):
        w_h, w_l, w_v = high_arr[i-window:i], low_arr[i-window:i], volume_arr[i-window:i]
        p_min, p_max = w_l.min(), w_h.max()
        if p_min == p_max: continue
        bin_step = (p_max - p_min) / num_bins
        bin_vol = _numba_calc_bins(w_h, w_l, w_v, p_min, bin_step, num_bins)
        sorted_vol = np.sort(bin_vol)
        lvn_th, hvn_th = sorted_vol[int(0.25*(num_bins-1))], sorted_vol[int(0.75*(num_bins-1))]
        curr_p = close_arr[i]
        
        best_hvn, best_lvn = -1.0, -1.0
        min_h_d, min_l_d = 1e30, 1e30
        for b in range(num_bins):
            p = p_min + (b + 0.5) * bin_step
            if bin_vol[b] >= hvn_th:
                d = abs(curr_p - p)
                if d < min_h_d: min_h_d, best_hvn = d, p
            if bin_vol[b] <= lvn_th:
                d = abs(curr_p - p)
                if d < min_l_d: min_l_d, best_lvn = d, p
        if best_hvn != -1.0 and curr_p != 0: hvn_dist[i] = (curr_p - best_hvn) / curr_p
        if best_lvn != -1.0 and curr_p != 0: lvn_dist[i] = (curr_p - best_lvn) / curr_p
    return hvn_dist, lvn_dist


@jit(nopython=True)
def _numba_vp_skewness_kurtosis(close_arr: np.ndarray, volume_arr: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(close_arr)
    skew_arr, kurt_arr = np.zeros(n), np.zeros(n)
    for i in range(window, n):
        w_c, w_v = close_arr[i-window:i], volume_arr[i-window:i]
        sum_v = w_v.sum()
        if sum_v == 0: continue
        w_mean = (w_c * w_v).sum() / sum_v
        med = np.median(w_c)
        std = np.std(w_c)
        if std > 0: skew_arr[i] = (w_mean - med) / std
        diff = w_c - w_mean
        w_var = (diff**2 * w_v).sum() / sum_v
        w_std = np.sqrt(w_var)
        if w_std > 0:
            m4 = (diff**4 * w_v).sum() / sum_v
            kurt_arr[i] = m4 / (w_std**4) - 3
    return skew_arr, kurt_arr


class VolumeProfileFeatureCalculator:
    """Volume Profile特徴量計算クラス"""

    def __init__(self, lookback_period: int = 50, num_bins: int = 20):
        self.lookback_period = lookback_period
        self.num_bins = num_bins

    def calculate_features(self, df: pd.DataFrame, lookback_periods: Optional[list] = None) -> pd.DataFrame:
        high_arr = df["high"].values.astype(np.float64)
        low_arr = df["low"].values.astype(np.float64)
        close_arr = df["close"].values.astype(np.float64)
        volume_arr = df["volume"].values.astype(np.float64)
        n = len(df)

        if lookback_periods is None:
            lookback_periods = [self.lookback_period, 100, 200]

        # 特徴量データを辞書で保持
        features_dict = {}

        for period in lookback_periods:
            poc, vah, val = _numba_rolling_volume_profile(high_arr, low_arr, close_arr, volume_arr, period, self.num_bins)
            
            # 手動でNaN埋め
            def safe_fill(arr):
                res = np.copy(arr)
                last_val = 0.0
                for i in range(len(res)):
                    if np.isnan(res[i]): res[i] = last_val
                    else: last_val = res[i]
                return res

            poc_f, vah_f, val_f = safe_fill(poc), safe_fill(vah), safe_fill(val)
            
            # 距離計算 (NumPy不具合を避けるためプリミティブ操作)
            features_dict[f"POC_Distance_{period}"] = [(close_arr[i]-poc_f[i])/poc_f[i] if poc_f[i]!=0 else 0.0 for i in range(n)]
            features_dict[f"VAH_Distance_{period}"] = [(close_arr[i]-vah_f[i])/vah_f[i] if vah_f[i]!=0 else 0.0 for i in range(n)]
            features_dict[f"VAL_Distance_{period}"] = [(close_arr[i]-val_f[i])/val_f[i] if val_f[i]!=0 else 0.0 for i in range(n)]
            features_dict[f"In_Value_Area_{period}"] = [1.0 if val_f[i] <= close_arr[i] <= vah_f[i] else 0.0 for i in range(n)]
            features_dict[f"Value_Area_Width_{period}"] = [(vah_f[i]-val_f[i])/poc_f[i] if poc_f[i]!=0 else 0.0 for i in range(n)]

        hvn, lvn = _numba_detect_volume_nodes_signed(high_arr, low_arr, close_arr, volume_arr, self.lookback_period, self.num_bins)
        features_dict["HVN_Distance"] = hvn.tolist()
        features_dict["LVN_Distance"] = lvn.tolist()

        skew, kurt = _numba_vp_skewness_kurtosis(close_arr, volume_arr, self.lookback_period)
        features_dict["VP_Skewness"] = skew.tolist()
        features_dict["VP_Kurtosis"] = kurt.tolist()

        # 最後に一括でDataFrame化
        return pd.DataFrame(features_dict, index=df.index)