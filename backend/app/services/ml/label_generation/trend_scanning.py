import logging
from typing import Optional

import numpy as np
import pandas as pd
from numba import jit

logger = logging.getLogger(__name__)


@jit(nopython=True)
def _trend_scanning_loop_numba(
    close_vals: np.ndarray,
    t0_indices: np.ndarray,
    min_window: int,
    max_window: int,
    step: int,
    min_t_value: float,
    return_t_value_as_label: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    トレンドスキャンのメインループ（Numba最適化 O(N*W)版）

    Returns:
        out_t_vals (np.ndarray): t値
        out_bins (np.ndarray): ラベル（1, 0, -1 または t値）
        out_t1_idxs (np.ndarray): 終了位置のインデックス (-1なら無効)
    """
    n_events = len(t0_indices)
    max_len = len(close_vals)

    # 累計和の事前計算 (O(L))
    cumsum_y = np.zeros(max_len + 1)
    cumsum_xy = np.zeros(max_len + 1)
    cumsum_yy = np.zeros(max_len + 1)
    for i in range(max_len):
        val = close_vals[i]
        cumsum_y[i + 1] = cumsum_y[i] + val
        cumsum_xy[i + 1] = cumsum_xy[i] + i * val
        cumsum_yy[i + 1] = cumsum_yy[i] + val * val

    out_t_vals = np.zeros(n_events, dtype=np.float64)
    out_bins = np.zeros(n_events, dtype=np.float64)
    out_t1_idxs = np.full(n_events, -1, dtype=np.int64)

    for i in range(n_events):
        t0_idx = t0_indices[i]
        if t0_idx == -1:
            continue

        best_t_val = 0.0
        best_L = 0.0
        max_abs_t = -1.0
        found = False

        # Window Loop (O(W))
        for L in range(min_window, max_window + 1, step):
            t1_idx = t0_idx + L
            if t1_idx >= max_len:
                break

            n_val = float(L + 1)

            # 区間統計量を累計和から取得 (O(1))
            sum_y = cumsum_y[t1_idx + 1] - cumsum_y[t0_idx]
            sum_yy = cumsum_yy[t1_idx + 1] - cumsum_yy[t0_idx]
            sum_ty = cumsum_xy[t1_idx + 1] - cumsum_xy[t0_idx]
            sum_xy = sum_ty - t0_idx * sum_y

            # X = 0, 1, ..., L の統計量 (定数)
            sum_x = L * (L + 1) * 0.5
            sum_xx = L * (L + 1) * (2 * L + 1) / 6.0

            denominator = n_val * sum_xx - sum_x * sum_x
            if abs(denominator) < 1e-12:
                continue

            # 区間の偏差平方和 (ss_y) で価格が一定かチェック
            ss_y = sum_yy - (sum_y * sum_y) / n_val
            if ss_y < 1e-12:
                slope = 0.0
                intercept = sum_y / n_val
                sum_res_sq = 0.0
            else:
                # 回帰係数
                slope = (n_val * sum_xy - sum_x * sum_y) / denominator
                intercept = (sum_y - slope * sum_x) / n_val
                # 残差平方和 (RSS)
                sum_res_sq = max(0.0, sum_yy - intercept * sum_y - slope * sum_xy)

            if n_val <= 2:
                continue

            sigma_eps_sq = sum_res_sq / (n_val - 2)
            sigma_eps = np.sqrt(max(0.0, sigma_eps_sq))

            ss_x = sum_xx - (sum_x * sum_x) / n_val

            t_val = 0.0
            # 傾きがほぼ0、または残差がほぼ0（完全一致）の場合はガード
            if abs(slope) < 1e-11 or sum_res_sq < 1e-11:
                if abs(slope) < 1e-14:
                    t_val = 0.0
                else:
                    t_val = 100.0 if slope > 0 else -100.0
            elif ss_x > 1e-12 and sigma_eps > 1e-12:
                se_slope = sigma_eps / np.sqrt(ss_x)
                t_val = slope / se_slope
                if t_val > 100.0:
                    t_val = 100.0
                if t_val < -100.0:
                    t_val = -100.0
            elif abs(slope) > 1e-12:
                t_val = 100.0 if slope > 0 else -100.0
            else:
                t_val = 0.0

            abs_t = abs(t_val)
            if abs_t > max_abs_t:
                max_abs_t = abs_t
                best_t_val = t_val
                best_L = float(L)
                found = True

        if found:
            out_t_vals[i] = best_t_val
            out_t1_idxs[i] = t0_idx + int(best_L)

            if return_t_value_as_label:
                out_bins[i] = best_t_val
            else:
                if best_t_val > min_t_value:
                    out_bins[i] = 1.0
                elif best_t_val < -min_t_value:
                    out_bins[i] = -1.0
                else:
                    out_bins[i] = 0.0

    return out_t_vals, out_bins, out_t1_idxs


class TrendScanning:
    """
    ラベリングのためのトレンドスキャン手法。
    Marcos Lopez de Pradoの "Advances in Financial Machine Learning" に基づく。

    複数の将来のウィンドウにわたる価格系列に線形回帰を適合させ、
    傾きのt統計量が最も高いウィンドウを選択します。
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
            min_window (int): 最小の前方ウィンドウサイズ。
            max_window (int): 最大の前方ウィンドウサイズ。
            step (int): ウィンドウ反復のステップサイズ。
            min_t_value (float): トレンドが有意であると見なすための最小t値しきい値。
                                 最大t値 < min_t_value の場合、ラベルは0になります。
        """
        self.min_window = min_window
        self.max_window = max_window
        self.step = step
        self.min_t_value = min_t_value

    def get_labels(
        self,
        close: pd.Series,
        t_events: Optional[pd.DatetimeIndex] = None,
        use_log_price: bool = True,  # 対数価格を使用するオプション
        return_t_value: bool = False,  # t値をそのまま返すオプション(回帰用)
    ) -> pd.DataFrame:
        """
        トレンドスキャンを使用してラベルを生成（Numba完全ベクトル化）

        Args:
            return_t_value (bool): Trueの場合、bin列に離散ラベルではなくt値(連続値)を格納します。

        Returns:
            pd.DataFrame: columns=["t1", "t_value", "bin", "ret"]
                          binは離散ラベル(1, 0, -1) または t値そのもの
        """
        t_events = t_events if t_events is not None else close.index
        # t_eventsがclose.indexに含まれるものだけにフィルタ
        t_events = t_events[t_events.isin(close.index)]
        if t_events.empty:
            return pd.DataFrame(columns=["t1", "t_value", "bin", "ret"])

        # 対数価格の使用 (トレンド強度の一貫性向上のため推奨)
        if use_log_price:
            close_values = np.log(close.values.astype(np.float64))
        else:
            close_values = close.values.astype(np.float64)

        idxs = close.index.get_indexer(t_events)

        # Numbaで一括計算
        t_vals, bins, t1_idxs = _trend_scanning_loop_numba(
            close_values,
            idxs,
            self.min_window,
            self.max_window,
            self.step,
            self.min_t_value,
            return_t_value,
        )

        # 結果構築（有効な結果のみ抽出）
        valid_mask = t1_idxs != -1

        if not np.any(valid_mask):
            return pd.DataFrame(columns=["t1", "t_value", "bin", "ret"])

        valid_t0 = t_events[valid_mask]
        valid_t1_idxs = t1_idxs[valid_mask]
        valid_t_vals = t_vals[valid_mask]
        valid_bins = bins[valid_mask]
        valid_t0_idxs = idxs[valid_mask]

        # t1 timestamp取得
        valid_t1 = close.index[valid_t1_idxs]

        # return計算 (元の価格ベース)
        p1 = close.values[valid_t1_idxs]
        p0 = close.values[valid_t0_idxs]
        returns = (p1 / p0) - 1.0

        out = pd.DataFrame(index=valid_t0)
        out["t1"] = valid_t1
        out["t_value"] = valid_t_vals
        out["bin"] = valid_bins
        out["ret"] = returns

        out.index.name = None
        return out
