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
    トレンドスキャンのメインループ（Numba最適化）

    Returns:
        out_t_vals (np.ndarray): t値
        out_bins (np.ndarray): ラベル（1, 0, -1 または t値）
        out_t1_idxs (np.ndarray): 終了位置のインデックス (-1なら無効)
    """
    n_events = len(t0_indices)
    max_len = len(close_vals)

    out_t_vals = np.zeros(n_events, dtype=np.float64)
    out_bins = np.zeros(n_events, dtype=np.float64)
    # -1で初期化
    out_t1_idxs = np.full(n_events, -1, dtype=np.int64)

    for i in range(n_events):
        t0_idx = t0_indices[i]
        if t0_idx == -1:
            continue

        # 最適化ロジック (Inline _find_best_window_numba)
        best_t_val = 0.0
        best_L = 0.0
        max_abs_t = -1.0
        found = False

        # Window Loop
        for L in range(min_window, max_window + 1, step):
            if t0_idx + L >= max_len:
                break

            n = L + 1

            # Regression: Y = alpha + beta * X
            # X = 0, 1, ..., L

            # Sum X, Sum XX
            sum_x = L * (L + 1) * 0.5
            sum_xx = L * (L + 1) * (2 * L + 1) / 6.0

            sum_y = 0.0
            sum_xy = 0.0

            # Data Loop (Calculate Sum Y, Sum XY)
            for k in range(n):
                val = close_vals[t0_idx + k]
                sum_y += val
                sum_xy += k * val

            denominator = n * sum_xx - sum_x * sum_x

            # 分母がほぼ0の場合、回帰不能
            if abs(denominator) < 1e-9:
                continue

            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n

            # Calculate Residuals & t-value
            sum_res_sq = 0.0
            for k in range(n):
                val = close_vals[t0_idx + k]
                pred = slope * k + intercept
                res = val - pred
                sum_res_sq += res * res

            if n <= 2:
                continue

            sigma_eps_sq = sum_res_sq / (n - 2)
            sigma_eps = np.sqrt(sigma_eps_sq) if sigma_eps_sq > 0 else 0.0

            ss_x = sum_xx - (sum_x * sum_x) / n

            t_val = 0.0
            # 標準誤差の計算
            if ss_x > 1e-9 and sigma_eps > 1e-9:
                se_slope = sigma_eps / np.sqrt(ss_x)
                t_val = slope / se_slope
                # t値をクリップ
                if t_val > 100.0:
                    t_val = 100.0
                if t_val < -100.0:
                    t_val = -100.0
            elif abs(slope) > 1e-9:
                # 完全一致直線の場合など
                t_val = 100.0 if slope > 0 else -100.0

            abs_t = abs(t_val)
            if abs_t > max_abs_t:
                max_abs_t = abs_t
                best_t_val = t_val
                best_L = float(L)
                found = True

        # End Window Loop

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
        # NOTE: isinは遅い場合があるが、ここでは安全性優先
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
        # min_window, max_window, stepはクラスメンバだが、Numbaに渡すためローカル変数化または直接渡す
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
        # t1_idxs != -1 の要素のみ
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
        # 対数価格を渡した場合でも、ここでのretは実価格の変動率として出すのが一般的
        # close.values は元の価格
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
