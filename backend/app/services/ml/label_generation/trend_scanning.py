import logging
from typing import Optional

import numpy as np
import pandas as pd
from numba import jit

logger = logging.getLogger(__name__)


@jit(nopython=True)
def _find_best_window_numba(
    close_np: np.ndarray, t0_idx: int, min_window: int, max_window: int, step: int
) -> tuple[float, float, float]:
    """
    最大のt値を持つ最適なウィンドウを見つけるためのNumba最適化関数。

    Returns:
        tuple: (best_L, best_t_val, best_slope)
        有効なウィンドウが見つからない場合は (0.0, 0.0, 0.0) を返します。
    """
    max_len = len(close_np)

    # 最適値の初期化
    best_t_val = 0.0
    best_L = 0.0
    best_slope = 0.0
    max_abs_t = -1.0
    found = False

    # ウィンドウごとの反復処理
    for L in range(min_window, max_window + 1, step):
        if t0_idx + L >= max_len:
            break

        # 価格セグメント
        # y = close_np[t0_idx : t0_idx + L + 1]
        # x = np.arange(len(y))

        # ループ内の割り当てを避けるための手動回帰計算
        n = L + 1

        # 合計を手動で計算するか、スライスを使用します（Numbaではスライスも問題ありません）
        # 最近のNumbaではループ内のスライスは問題ありませんが、安全かつ効率的に行います

        # x は 0, 1, ..., L
        # sum_x = L * (L + 1) / 2
        sum_x = L * (L + 1) * 0.5

        # sum_xx = L * (L + 1) * (2*L + 1) / 6
        sum_xx = L * (L + 1) * (2 * L + 1) / 6.0

        # sum_y と sum_xy を計算
        sum_y = 0.0
        sum_xy = 0.0

        # 合計計算用のNumbaループ
        for i in range(n):
            val = close_np[t0_idx + i]
            sum_y += val
            sum_xy += i * val

        denominator = n * sum_xx - sum_x * sum_x
        if abs(denominator) < 1e-9:
            continue

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

        # 残差と標準誤差
        sum_res_sq = 0.0
        for i in range(n):
            val = close_np[t0_idx + i]
            pred = slope * i + intercept
            res = val - pred
            sum_res_sq += res * res

        if n <= 2:
            continue

        sigma_eps_sq = sum_res_sq / (n - 2)
        if sigma_eps_sq < 0:  # 発生しないはず
            sigma_eps = 0.0
        else:
            sigma_eps = np.sqrt(sigma_eps_sq)

        # sum((x - mean_x)^2) = ss_x
        ss_x = sum_xx - (sum_x * sum_x) / n

        if ss_x <= 1e-9 or sigma_eps < 1e-9:
            # 完全一致
            if abs(slope) < 1e-9:
                t_val = 0.0
            else:
                max_t = 100.0
                t_val = max_t if slope > 0 else -max_t
        else:
            se_slope = sigma_eps / np.sqrt(ss_x)
            t_val = slope / se_slope
            # t値をクリップ
            if t_val > 100.0:
                t_val = 100.0
            elif t_val < -100.0:
                t_val = -100.0

        # ベスト（最大絶対t値）を追跡
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
        トレンドスキャンを使用してラベルを生成

        Args:
            return_t_value (bool): Trueの場合、bin列に離散ラベルではなくt値(連続値)を格納します。

        Returns:
            pd.DataFrame: columns=["t1", "t_value", "bin", "ret"]
                          binは離散ラベル(1, 0, -1) または t値そのもの
        """
        t_events = t_events if t_events is not None else close.index
        t_events = t_events[t_events.isin(close.index)]
        if t_events.empty:
            return pd.DataFrame(columns=["t1", "t_value", "bin", "ret"])

        # 対数価格の使用 (トレンド強度の一貫性向上のため推奨)
        if use_log_price:
            close_values = np.log(close.values.astype(np.float64))
        else:
            close_values = close.values.astype(np.float64)

        idxs = close.index.get_indexer(t_events)

        results = []
        for i, t0_idx in enumerate(idxs):
            if t0_idx == -1:
                continue

            best_L, best_t, best_slope = _find_best_window_numba(
                close_values, t0_idx, self.min_window, self.max_window, self.step
            )
            if best_L == 0:
                continue

            # ラベル生成
            if return_t_value:
                # 回帰問題やウェイト付け用にt値をそのまま使う
                label = best_t
            else:
                # 従来の3値分類
                label = (
                    1
                    if best_t > self.min_t_value
                    else (-1 if best_t < -self.min_t_value else 0)
                )

            t1_idx = t0_idx + int(best_L)

            # リターン計算 (これは常に対数ではなく元の価格の実リターンが良いことが多いが、
            # 分析の一貫性のためにここも対数差分にする手もある。
            # ここでは実務的な分かりやすさ優先で、元の価格の単純リターンを計算する)
            t1_price = close.values[t1_idx]
            t0_price = close.values[t0_idx]
            actual_ret = (t1_price / t0_price) - 1.0

            results.append(
                {
                    "t0": t_events[i],
                    "t1": close.index[t1_idx],
                    "t_value": best_t,
                    "bin": label,
                    "ret": actual_ret,
                }
            )

        if not results:
            return pd.DataFrame(columns=["t1", "t_value", "bin", "ret"])

        # resultsからインデックス(t0)とデータを分離
        t0_list = [r["t0"] for r in results]
        data_list = [{k: v for k, v in r.items() if k != "t0"} for r in results]

        out = pd.DataFrame(data_list, index=t0_list)
        out.index.name = None
        return out
