from typing import List, Optional

import numpy as np
import pandas as pd
from numba import jit


@jit(nopython=True)
def _process_events_numba(
    close_vals: np.ndarray,
    close_times: np.ndarray,  # int64 inputs
    t0_indices: np.ndarray,
    v_bar_indices: np.ndarray,
    v_bar_times: np.ndarray,  # int64 inputs
    targets: np.ndarray,
    sides: np.ndarray,
    pt: float,
    sl: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    トリプルバリアイベントを一括処理するNumba関数

    Returns:
        out_t1 (np.ndarray): 接触時刻（int64, NaT=min）
        out_side (np.ndarray): 接触タイプ (1=pt, 2=sl, 3=vertical, 0=none)
    """
    n_events = len(t0_indices)

    # NaT (int64 min) で初期化
    out_t1 = np.full(n_events, np.iinfo(np.int64).min, dtype=np.int64)
    out_side = np.zeros(n_events, dtype=np.int8)

    close_len = len(close_vals)

    for i in range(n_events):
        start_idx = t0_indices[i]
        if start_idx == -1:
            continue

        # 垂直バリアのインデックス。 -1 または範囲外ならデータの最後まで
        end_idx = v_bar_indices[i]
        limit_idx = end_idx if end_idx != -1 else close_len

        p0 = close_vals[start_idx]
        if np.isnan(p0) or p0 == 0:
            continue

        trgt = targets[i]
        s = sides[i]

        pt_thresh = trgt * pt
        sl_thresh = trgt * sl

        hit_idx = -1
        hit_type = 0  # 0: none, 1: pt, -1: sl

        # バリア判定ループ
        # start_idx + 1 から limit_idx まで
        for j in range(start_idx + 1, limit_idx):
            ret = (close_vals[j] / p0) - 1.0

            if s >= 0:  # Long or neutral(assume long)
                if pt > 0 and ret >= pt_thresh:
                    hit_idx = j
                    hit_type = 1
                    break
                if sl > 0 and ret <= -sl_thresh:
                    hit_idx = j
                    hit_type = -1
                    break
            else:  # Short checking
                if pt > 0 and ret <= -pt_thresh:
                    hit_idx = j
                    hit_type = 1
                    break
                if sl > 0 and ret >= sl_thresh:
                    hit_idx = j
                    hit_type = -1
                    break

        if hit_idx != -1:
            out_t1[i] = close_times[hit_idx]
            out_side[i] = 1 if hit_type == 1 else 2  # 1: pt, 2: sl
        else:
            # ヒットなし -> 垂直バリアの確認
            t_vertical = v_bar_times[i]
            # NaT (=min int64) でなければ垂直バリア到達とみなす
            if t_vertical != np.iinfo(np.int64).min:
                out_t1[i] = t_vertical
                out_side[i] = 3  # 3: vertical

    return out_t1, out_side


class TripleBarrier:
    """
    金融データのラベリングのためのトリプルバリア法。
    Marcos Lopez de Pradoの「金融機械学習の進歩」に基づいています。
    """

    def __init__(
        self,
        pt: float = 1.0,
        sl: float = 1.0,
        min_ret: float = 0.001,
        num_threads: int = 1,
    ):
        """
        Args:
            pt (float): 利食い（Profit Taking）の乗数。
            sl (float): 損切り（Stop Loss）の乗数。
            min_ret (float): ラベルと見なされるために必要な最小リターン。
            num_threads (int): 並列処理のスレッド数（簡易版では使用されません）。
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
        side: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """バリア接触時刻を見つける（Numba完全ベクトル化バージョン）"""
        # ターゲットのフィルタリング
        target = target.loc[t_events]
        target = target[target > min_ret]
        if target.empty:
            return pd.DataFrame(columns=["t1", "trgt", "side"])

        # 準備
        if vertical_barrier_times is None:
            vertical_barrier_times = pd.Series(pd.NaT, index=target.index)

        # targetのインデックスに合わせてアラインメント
        v_bar = vertical_barrier_times.reindex(target.index).fillna(pd.NaT)

        if side is None:
            side_series = pd.Series(1.0, index=target.index)
        else:
            side_series = side.reindex(target.index).fillna(1.0)

        # Numba入力用のデータ準備 (すべてNumPy配列/Viewに変換)

        # 1. 価格データと時刻
        close_vals = close.values.astype(np.float64)
        # 時刻はint64(ns)として扱う
        close_times = close.index.values.astype(np.int64)

        # 2. イベント開始位置のインデックス
        # get_indexer は見つからない場合 -1 を返す
        t0_idxs = close.index.get_indexer(target.index)

        # 3. 垂直バリアの位置インデックス
        # v_bar自体はTimestampだが、これがcloseのどこに該当するかを探す必要がある
        # 一旦、v_barの値をclose.indexから探す
        # v_barにはNaTが含まれる可能性がある => NaTの場合は -1 になるよう処理
        valid_v_mask = v_bar.notna()
        v_bar_idxs = np.full(len(target), -1, dtype=np.int64)

        if valid_v_mask.any():
            # 有効な垂直バリア時刻のみインデックス検索
            valid_times = v_bar[valid_v_mask]
            found_idxs = close.index.get_indexer(valid_times)
            v_bar_idxs[valid_v_mask] = found_idxs

        # 4. 垂直バリアの時刻 (int64)
        v_bar_times_int = v_bar.values.astype(np.int64)

        # 5. その他パラメータ配列
        target_vals = target.values.astype(np.float64)
        side_vals = side_series.values.astype(np.float64)
        pt_mult, sl_mult = pt_sl

        # Numba関数の実行
        out_t1_int, out_side_int = _process_events_numba(
            close_vals,
            close_times,
            t0_idxs,
            v_bar_idxs,
            v_bar_times_int,
            target_vals,
            side_vals,
            pt_mult
            * self.pt,  # クラスのptと引数のpt_slを掛け合わせる（元実装の挙動に準拠）
            sl_mult * self.sl,
        )

        # 結果の構築
        # int64 -> datetime64[ns]
        out_t1 = pd.to_datetime(out_t1_int)
        # NaT (int64 min) は pd.to_datetime で NaT に変換されるはずだが、
        # 明示的に min value を NaT に置換念のため
        # pandasの仕様では int64の最小値を渡すとNaTになることが多いが、
        # ここでは mask で処理するのが確実
        is_nat = out_t1_int == np.iinfo(np.int64).min
        out_t1.values[is_nat] = np.datetime64("NaT")

        # sideの数値マッピングを文字列に戻す
        # 0: None, 1: pt, 2: sl, 3: vertical
        side_map = np.array([np.nan, "pt", "sl", "vertical"], dtype=object)
        out_side_str = side_map[out_side_int]

        events = pd.DataFrame(index=target.index)
        events["t1"] = out_t1
        events["trgt"] = target
        events["side"] = out_side_str

        return events

    def get_bins(
        self, events: pd.DataFrame, close: pd.Series, binary_label: bool = False
    ) -> pd.DataFrame:
        """バリア接触イベントに基づいてラベルを生成"""
        ev = events.dropna(subset=["t1"])
        if ev.empty:
            return pd.DataFrame(columns=["ret", "bin", "trgt"])

        px_init = close.loc[ev.index]

        # t1の価格取得 (ベクトル化)
        # reindexは遅い場合があるので、mapの方が速いかもしれないが、
        # ここではシンプルさを維持
        px_end = close.reindex(ev["t1"], method="pad")
        px_end.index = ev.index

        out = pd.DataFrame(index=ev.index)
        out["ret"] = px_end / px_init - 1
        out["bin"] = 0.0
        out["trgt"] = ev["trgt"]

        if "side" in ev.columns:
            if binary_label:
                # ptなら1、それ以外(sl/vertical)は0でいいのか？
                # 元実装: pt->1, 他は暗黙0
                out.loc[ev["side"] == "pt", "bin"] = 1.0
            else:
                out.loc[ev["side"] == "pt", "bin"] = 1.0
                out.loc[ev["side"] == "sl", "bin"] = -1.0
                # verticalは0のまま
        else:
            # フォールバック (リターンベース)
            if self.pt > 0:
                out.loc[out["ret"] > out["trgt"] * self.pt * 0.999, "bin"] = 1.0
            if not binary_label and self.sl > 0:
                out.loc[out["ret"] < -out["trgt"] * self.sl * 0.999, "bin"] = -1.0

        return out
