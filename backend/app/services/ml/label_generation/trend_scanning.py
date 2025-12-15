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
    ) -> pd.DataFrame:
        """
        トレンドスキャンを使用してラベルを生成します。

        Args:
            close (pd.Series): 終値。
            t_events (pd.DatetimeIndex): ラベル付けするタイムスタンプ。Noneの場合、すべてをラベル付けします。

        Returns:
            pd.DataFrame: 以下の列を持つDataFrame:
                - t1: 選択されたトレンドウィンドウの終了タイムスタンプ。
                - t_value: 選択されたトレンドのt統計量。
                - bin: ラベル（上昇の場合は1、下降の場合は-1、トレンドなしの場合は0）。
                - ret: 選択されたウィンドウでの収益率。
        """
        if t_events is None:
            t_events = close.index

        # t_events が close.index に存在することを確認
        t_events = t_events[t_events.isin(close.index)]

        out = pd.DataFrame(index=t_events, columns=["t1", "t_value", "bin", "ret"])

        # 速度向上のために close を numpy に変換
        # float64 を保証
        close_np = close.values.astype(np.float64)

        # インデックスを整数位置にマップ
        # 最適化：ループ内でのインデックスアクセスは遅いため、マップの方が適しています。
        # t_events が close.index のサブセットと一致する場合は、インデックスを見つけるだけで済みます。
        # しかし、close はギャップのある時系列かもしれません。
        # タイムスタンプが正確に一致すると仮定します。

        # t_events のタイムスタンプを close_np の整数位置に効率的にマップするには：
        # 両方がソートされている場合、searchsorted を使用できます。
        # しかし、処理なしでは完全に整列していないか、一意でない可能性があると仮定しましょう。
        # 辞書マップの使用は、セットアップが O(N)、ルックアップが O(1) です。
        idx_map = {idx: i for i, idx in enumerate(close.index)}

        # DataFrame を一度に構築するために結果用の配列を事前に割り当て（loc セッターよりも高速）
        n_events = len(t_events)
        t1_results = np.empty(n_events, dtype=object)
        t_val_results = np.zeros(n_events, dtype=np.float64)
        bin_results = np.zeros(n_events, dtype=int)
        ret_results = np.zeros(n_events, dtype=np.float64)

        # t_events に対する反復処理
        # このループはPythonのままですが、重い処理はNumbaで行われます

        for i, t0 in enumerate(t_events):
            if t0 not in idx_map:
                continue

            t0_idx = idx_map[t0]

            # Numba最適化関数の呼び出し
            best_L, best_t_val, best_slope = _find_best_window_numba(
                close_np, t0_idx, self.min_window, self.max_window, self.step
            )

            if best_L == 0.0 and best_t_val == 0.0 and best_slope == 0.0:
                # 有効なウィンドウが見つかりません
                continue

            best_L_int = int(best_L)

            # ラベルの決定
            label = 0
            if best_t_val > self.min_t_value:
                label = 1
            elif best_t_val < -self.min_t_value:
                label = -1

            # インデックスの計算
            t1_idx = t0_idx + best_L_int

            # 結果の設定
            t1_results[i] = close.index[t1_idx]
            t_val_results[i] = best_t_val
            bin_results[i] = label

            # リターン
            ret = (close_np[t1_idx] / close_np[t0_idx]) - 1
            ret_results[i] = ret

        out["t1"] = t1_results
        out["t_value"] = t_val_results
        out["bin"] = bin_results
        out["ret"] = ret_results

        # 実行に失敗した行を削除（t1がNoneなど）
        # 事前割り当てでは、オブジェクト配列のデフォルトはNoneですか？いいえ、未初期化です。
        # フィルタリングする方が良いです。

        # 無効な行（何も書き込まなかった行）を除外
        # t1_results の None/Null チェック
        mask = pd.notna(t1_results)  # boolean array
        out = out[mask]

        return out



