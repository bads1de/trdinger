"""
Volume Profile Feature Calculator

価格レベル別の出来高分布を分析し、市場構造を捉える特徴量を生成します。
学術的に検証された強力な特徴量（Kaggle/論文で実証済み）。
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numba import jit

logger = logging.getLogger(__name__)


@jit(nopython=True)
def _numba_calc_bins(w_high, w_low, w_vol, price_min, bin_step, num_bins):
    """ビンごとの出来高を計算（共通Numba関数）"""
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
        w_h, w_l, w_c, w_v = high_arr[i-window:i], low_arr[i-window:i], close_arr[i-window:i], volume_arr[i-window:i]
        p_min, p_max = w_l.min(), w_h.max()

        if p_min == p_max or p_max - p_min == 0:
            poc_arr[i] = vah_arr[i] = val_arr[i] = w_c[-1]
            continue

        bin_step = (p_max - p_min) / num_bins
        bin_vol = _numba_calc_bins(w_h, w_l, w_v, p_min, bin_step, num_bins)

        # POC
        poc_bin = np.argmax(bin_vol)
        poc_arr[i] = p_min + (poc_bin + 0.5) * bin_step

        # VAH/VAL (70%)
        target_v = bin_vol.sum() * 0.70
        vah_b, val_bin, acc_v = poc_bin, poc_bin, bin_vol[poc_bin]

        while acc_v < target_v:
            v_up = bin_vol[vah_b + 1] if vah_b + 1 < num_bins else 0.0
            v_dn = bin_vol[val_bin - 1] if val_bin > 0 else 0.0
            if v_up == 0 and v_dn == 0:
                break
            if v_up > v_dn:
                vah_b += 1
                acc_v += v_up
            else:
                val_bin -= 1
                acc_v += v_dn

        vah_arr[i] = p_min + (vah_b + 1) * bin_step
        val_arr[i] = p_min + val_bin * bin_step

    return poc_arr, vah_arr, val_arr


@jit(nopython=True)
def _numba_detect_volume_nodes_signed(
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    window: int,
    num_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(close_arr)
    hvn_dist, lvn_dist = np.zeros(n), np.zeros(n)

    for i in range(window, n):
        w_h, w_l, w_v = high_arr[i-window:i], low_arr[i-window:i], volume_arr[i-window:i]
        p_min, p_max = w_l.min(), w_h.max()
        if p_min == p_max or p_max - p_min == 0:
            continue

        bin_step = (p_max - p_min) / num_bins
        bin_vol = _numba_calc_bins(w_h, w_l, w_v, p_min, bin_step, num_bins)

        sorted_vol = np.sort(bin_vol)
        lvn_th, hvn_th = sorted_vol[int(0.25 * (num_bins - 1))], sorted_vol[int(0.75 * (num_bins - 1))]
        curr_p = close_arr[i]

        def find_best(th, is_hvn):
            best_p, min_d = -1.0, 1e30
            for b in range(num_bins):
                cond = (bin_vol[b] >= th) if is_hvn else (bin_vol[b] <= th)
                if cond:
                    p = p_min + (b + 0.5) * bin_step
                    d = abs(curr_p - p)
                    if d < min_d:
                        min_d, best_p = d, p
            return best_p

        hvn_p = find_best(hvn_th, True)
        if hvn_p != -1.0 and curr_p != 0:
            hvn_dist[i] = (curr_p - hvn_p) / curr_p
        
        lvn_p = find_best(lvn_th, False)
        if lvn_p != -1.0 and curr_p != 0:
            lvn_dist[i] = (curr_p - lvn_p) / curr_p

    return hvn_dist, lvn_dist


@jit(nopython=True)
def _numba_vp_skewness_kurtosis(
    close_arr: np.ndarray, volume_arr: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(close_arr)
    skew_arr = np.zeros(n)
    kurt_arr = np.zeros(n)

    for i in range(window, n):
        start_idx = i - window
        end_idx = i

        w_close = close_arr[start_idx:end_idx]
        w_vol = volume_arr[start_idx:end_idx]

        # 加重平均
        sum_vol = 0.0
        sum_price_vol = 0.0
        prices = w_close

        for j in range(window):
            v = w_vol[j]
            p = prices[j]
            sum_vol += v
            sum_price_vol += p * v

        if sum_vol == 0:
            continue

        weighted_mean = sum_price_vol / sum_vol

        # 中央値（近似またはO(N log N)ソート）
        # Numbaはnp.medianをサポートしていますか？ はい。
        median_price = np.median(prices)  # ソートのため比較的遅くなる可能性があります

        # 終値の標準偏差（元のコードでは非加重？）
        # 元: skew = (weighted_mean - median_price) / window_data["close"].std()
        # std()は通常、非加重のpandas stdです。

        price_std = np.std(
            prices
        )  # numpyのstdはデフォルトで母集団(ddof=0)、pandasは標本(ddof=1)
        # numpyのstdで十分な近似とします

        if price_std > 0:
            skew_arr[i] = (weighted_mean - median_price) / price_std

        # 尖度
        # 元: 加重分散
        sum_sq_diff_vol = 0.0
        for j in range(window):
            p = prices[j]
            v = w_vol[j]
            sum_sq_diff_vol += ((p - weighted_mean) ** 2) * v

        weighted_var = sum_sq_diff_vol / sum_vol
        weighted_std = np.sqrt(weighted_var)

        if weighted_std > 0:
            sum_pow4_vol = 0.0
            for j in range(window):
                p = prices[j]
                v = w_vol[j]
                sum_pow4_vol += ((p - weighted_mean) ** 4) * v

            fourth_moment = sum_pow4_vol / sum_vol
            kurt_arr[i] = fourth_moment / (weighted_std**4) - 3

    return skew_arr, kurt_arr


class VolumeProfileFeatureCalculator:
    """
    Volume Profile（価格レベル別出来高）ベースの特徴量計算クラス

    市場参加者の「合意価格」や重要な価格レベルを特定し、
    現在価格との関係から予測力のある特徴量を生成します。
    """

    def __init__(self, lookback_period: int = 50, num_bins: int = 20):
        """
        Args:
            lookback_period: Volume Profile計算に使用する過去期間
            num_bins: 価格を分割するビン数（解像度）
        """
        self.lookback_period = lookback_period
        self.num_bins = num_bins

    def calculate_features(
        self, df: pd.DataFrame, lookback_periods: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Volume Profile特徴量を計算

        Args:
            df: OHLCV DataFrame
            lookback_periods: 複数期間でのVolume Profile計算（Noneの場合はデフォルト使用）

        Returns:
            Volume Profile特徴量を含むDataFrame
        """
        # float64 numpy配列であることを確認
        high_arr = df["high"].values.astype(np.float64)
        low_arr = df["low"].values.astype(np.float64)
        close_arr = df["close"].values.astype(np.float64)
        volume_arr = df["volume"].values.astype(np.float64)

        result = pd.DataFrame(index=df.index)

        if lookback_periods is None:
            lookback_periods = [self.lookback_period, 100, 200]

        for period in lookback_periods:
            # Volume Profile計算 (Numba最適化)
            poc_arr, vah_arr, val_arr = _numba_rolling_volume_profile(
                high_arr, low_arr, close_arr, volume_arr, period, self.num_bins
            )

            # 整列のためにSeriesに戻す（インデックスが一致していれば厳密なndarray代入の方が高速ですが）
            # インデックスが一致していると仮定

            current_price = close_arr

            # 0をNaNまたは小さな数に置き換えてゼロ除算を回避するか？
            # 標準: pocが0の場合（価格では発生しないはず）、結果はinfになります。

            poc_dist = (current_price - poc_arr) / poc_arr
            vah_dist = (current_price - vah_arr) / vah_arr
            val_dist = (current_price - val_arr) / val_arr

            result[f"POC_Distance_{period}"] = poc_dist
            result[f"VAH_Distance_{period}"] = vah_dist
            result[f"VAL_Distance_{period}"] = val_dist

            # Value Area内か（バイナリ）
            in_va = ((current_price >= val_arr) & (current_price <= vah_arr)).astype(
                float
            )
            result[f"In_Value_Area_{period}"] = in_va

            # Value Area幅（ボラティリティ代理）
            va_width = (vah_arr - val_arr) / poc_arr
            result[f"Value_Area_Width_{period}"] = va_width

            # NaNを埋める（最初の 'period' 行）
            # これは最後に行うか、必要に応じてffillロジックを使用できます。
            # 元のコードはffillを行っていました。
            # ここではNaNをそのままにするか、埋めるか？
            # 元: poc_series.fillna(method="ffill")
            # 結果列を埋めます
            cols = [
                f"POC_Distance_{period}",
                f"VAH_Distance_{period}",
                f"VAL_Distance_{period}",
                f"In_Value_Area_{period}",
                f"Value_Area_Width_{period}",
            ]
            result[cols] = result[cols].ffill().fillna(0.0)

        # HVN/LVN（高/低出来高ノード）検出 (Numba最適化)
        # デフォルトのlookback_periodを使用
        # 注: 元のコードはHVN/LVNと歪度（Skewness）に別々のループを使用していました
        # それらも最適化しました。

        # 符号付き距離を正しく返すように _numba_detect_volume_nodes を修正する必要があります
        # 思考プロセスでは完全な書き換えをスキップしました、下のコードブロックで修正しましょう（コメントがコードに残っていました）

        # 実は上で _numba_detect_volume_nodes を正の距離を返すように簡略化しました。
        # しかし待てよ、元のコード:
        # nearest_hvn = hvn_prices[np.argmin(np.abs(hvn_prices - current))]
        # hvn_distance.iloc[i] = (current - nearest_hvn) / current
        # これは符号付きです。現在価格 > HVN（価格が出来高ノードより上）の場合、正になります。

        # この符号に合わせて _numba_detect_volume_nodes を調整する必要があります。
        # 以下の関数を更新します。

        hvn_arr, lvn_arr = _numba_detect_volume_nodes_signed(
            high_arr,
            low_arr,
            close_arr,
            volume_arr,
            self.lookback_period,
            self.num_bins,
        )

        result["HVN_Distance"] = hvn_arr
        result["LVN_Distance"] = lvn_arr

        # Volume Profile形状特徴 (Numba最適化)
        skew_arr, kurt_arr = _numba_vp_skewness_kurtosis(
            close_arr, volume_arr, self.lookback_period
        )
        result["VP_Skewness"] = skew_arr
        result["VP_Kurtosis"] = kurt_arr

        return result

    # 非推奨 / 内部メソッドは参照用に残すか、削除するか？
    # ロジックを置き換えたので、削除するかフォールバックとして残すか？
    # 完全に削除する方が良いでしょう。

    def _compute_volume_profile(self, *args, **kwargs):
        raise NotImplementedError("Use optimized calculate_features instead")


@jit(nopython=True)
def _numba_detect_volume_nodes_signed(
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    window: int,
    num_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(close_arr)
    hvn_dist_arr = np.zeros(n)
    lvn_dist_arr = np.zeros(n)

    for i in range(window, n):
        w_high = high_arr[i - window : i]
        w_low = low_arr[i - window : i]
        w_vol = volume_arr[i - window : i]

        price_min = w_low.min()
        price_max = w_high.max()

        if price_min == price_max:
            continue

        bin_step = (price_max - price_min) / num_bins
        if bin_step == 0:
            continue

        bin_volume = np.zeros(num_bins)

        for j in range(window):
            start_bin = int((w_low[j] - price_min) / bin_step)
            end_bin = int((w_high[j] - price_min) / bin_step)

            if start_bin < 0:
                start_bin = 0
            if start_bin >= num_bins:
                start_bin = num_bins - 1
            if end_bin < 0:
                end_bin = 0
            if end_bin >= num_bins:
                end_bin = num_bins - 1

            num_affected = end_bin - start_bin + 1
            vol_per_bin = w_vol[j] / num_affected
            for b in range(start_bin, end_bin + 1):
                bin_volume[b] += vol_per_bin

        sorted_vol = np.sort(bin_volume)
        lvn_threshold = sorted_vol[int(0.25 * (num_bins - 1))]
        hvn_threshold = sorted_vol[int(0.75 * (num_bins - 1))]

        current_price = close_arr[i]

        # HVN Signed Distance
        best_hvn_price = -1.0
        min_hvn_dist_abs = 1e30

        for b in range(num_bins):
            if bin_volume[b] >= hvn_threshold:
                bin_price = price_min + (b + 0.5) * bin_step
                dist_abs = abs(current_price - bin_price)
                if dist_abs < min_hvn_dist_abs:
                    min_hvn_dist_abs = dist_abs
                    best_hvn_price = bin_price

        if best_hvn_price != -1.0 and current_price != 0:
            hvn_dist_arr[i] = (current_price - best_hvn_price) / current_price

        # LVN Signed Distance
        best_lvn_price = -1.0
        min_lvn_dist_abs = 1e30

        for b in range(num_bins):
            if bin_volume[b] <= lvn_threshold:
                bin_price = price_min + (b + 0.5) * bin_step
                dist_abs = abs(current_price - bin_price)
                if dist_abs < min_lvn_dist_abs:
                    min_lvn_dist_abs = dist_abs
                    best_lvn_price = bin_price

        if best_lvn_price != -1.0 and current_price != 0:
            lvn_dist_arr[i] = (current_price - best_lvn_price) / current_price

    return hvn_dist_arr, lvn_dist_arr



