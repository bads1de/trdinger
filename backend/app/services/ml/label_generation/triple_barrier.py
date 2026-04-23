import logging
from typing import Any, List, Optional, cast

import numpy as np
import pandas as pd
from numba import jit

logger = logging.getLogger(__name__)


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
    金融機械学習における「トリプルバリア法」を用いて、価格データにラベルを付与するクラスです。
    Marcos Lopez de Prado の「Advances in Financial Machine Learning」に基づいています。

    この手法は、以下の3つの「バリア（壁）」のいずれかに価格が最初に接触したタイミングでラベルを決定します：
    1. **利食いバリア (Upper Barrier)**: 価格が上方に一定距離動いた場合に接触。ラベル +1（買い）またはリターンを付与。
    2. **損切りバリア (Lower Barrier)**: 価格が下方に一定距離動いた場合に接触。ラベル -1（売り）またはリターンを付与。
    3. **時間制限バリア (Vertical Barrier)**: 一定時間経過しても上下のバリアに接触しなかった場合に接触。その時点のリターンを付与。

    これにより、単純な固定期間リターンよりも、実際の取引戦略（TP/SLを設定した注文）に近い形での学習が可能になります。
    """

    def __init__(
        self,
        pt: float = 1.0,
        sl: float = 1.0,
        min_ret: float = 0.001,
        num_threads: int = 1,
    ):
        """
        TripleBarrierを初期化します。

        Args:
            pt (float): 利食い（Profit Taking）の幅を決定する乗数。
            sl (float): 損切り（Stop Loss）の幅を決定する乗数。
            min_ret (float): ラベルとして有効と見なすための最小リターン。これ以下の変動は 0 (No Trade) とされます。
            num_threads (int): 並列処理のスレッド数。
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
        """
        指定されたイベント（エントリー候補点）に対して、各バリアの接触時刻とリターンを特定します。

        このメソッドは Numba により高速化されており、大規模な時系列データに対しても高速に動作します。

        Args:
            close (pd.Series): 市場の終値データ（DatetimeIndex）。
            t_events (pd.DatetimeIndex): ラベル付けの対象となる時刻（エントリー候補点）のリスト。
            pt_sl (List[float]): 利食い・損切り幅のリスト（通常は `[pt, sl]`。本クラスでは `self.pt`, `self.sl` が優先されます）。
            target (pd.Series): 各時刻のボラティリティ等に基づく動的なバリア幅。
            min_ret (float): ラベル付与に必要な最小リターン閾値。
            vertical_barrier_times (Optional[pd.Series]): 各イベントに対する時間制限バリア（決済期限）の時刻。
            side (Optional[pd.Series]): 各イベントの方向（1: Longのみ、-1: Shortのみ、None: 両方）。

        Returns:
            pd.DataFrame: 発生したイベント情報のDataFrame。
                - "t1" (datetime): 最初にバリアに接触した時刻。
                - "trgt" (float): 使用されたバリア幅（ターゲット）。
                - "side" (int): 接触したバリアの種類（1: 利食い, 2: 損切り, 3: 時間制限）。
        """
        # ターゲットのフィルタリング
        target = cast(pd.Series, target.loc[t_events])
        target = cast(pd.Series, target[target > min_ret])
        if target.empty:
            return pd.DataFrame(columns=pd.Index(["t1", "trgt", "side"]))

        # 準備
        if vertical_barrier_times is None:
            vertical_barrier_times = pd.Series(pd.NaT, index=target.index)

        # targetのインデックスに合わせてアラインメント
        v_bar_reindexed = vertical_barrier_times.reindex(target.index)
        v_bar = pd.Series(v_bar_reindexed.values, index=target.index).fillna(
            cast(Any, pd.NaT)
        )

        if side is None:
            side_series = pd.Series(1.0, index=target.index)
        else:
            side_series = side.reindex(target.index).fillna(1.0)

        # Numba入力用のデータ準備 (すべてNumPy配列/Viewに変換)

        # 1. 価格データと時刻
        close_vals = close.to_numpy(dtype=np.float64)
        # 時刻はint64(ns)として扱う (明示的にns解像度に変換)
        close_times = (
            close.index.astype("datetime64[ns]").astype(np.int64).to_numpy()
        )

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
            valid_v_bar_series = cast(pd.Series, v_bar[valid_v_mask])
            valid_times = pd.Series(
                valid_v_bar_series.values, index=valid_v_bar_series.index
            )
            # get_indexer requires exact match which can fail. Use searchsorted to find position.
            idx_obj = cast(pd.DatetimeIndex, close.index)
            found_idxs = idx_obj.searchsorted(
                valid_times.to_numpy(dtype="datetime64[ns]"), side="left"
            )
            found_idxs = np.clip(found_idxs, 0, len(close))
            v_bar_idxs[valid_v_mask] = found_idxs

        # 4. 垂直バリアの時刻 (int64)
        v_bar_times_int = (
            v_bar.astype("datetime64[ns]").astype(np.int64).to_numpy()
        )

        # 5. その他パラメータ配列
        target_vals = target.to_numpy(dtype=np.float64)
        side_vals = side_series.to_numpy(dtype=np.float64)
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
        # pandas.to_datetime may return an immutable DatetimeIndex or a read-only array.
        # We explicitly convert to a numpy array and ensure it is a writable copy.
        out_t1 = pd.to_datetime(out_t1_int, unit="ns").to_numpy(copy=True)

        # NaT (int64 min) は pd.to_datetime で NaT に変換されるはずだが、
        # 明示的に min value を NaT に置換念のため
        # pandasの仕様では int64の最小値を渡すとNaTになることが多いが、
        # ここでは mask で処理するのが確実
        is_nat = out_t1_int == np.iinfo(np.int64).min
        out_t1[is_nat] = np.datetime64("NaT")

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
        self,
        events: pd.DataFrame,
        close: pd.Series,
        binary_label: bool = False,
    ) -> pd.DataFrame:
        """バリア接触イベントに基づいてラベルを生成

        インデックスの整合性を確保し、特徴量とラベルのズレを防ぎます。
        """
        ev = events.dropna(subset=["t1"])
        if ev.empty:
            return pd.DataFrame(columns=pd.Index(["ret", "bin", "trgt"]))

        # 開始価格の取得（インデックスの整合性確認）
        # eventsのインデックスがcloseのインデックスと一致するとは限らないため
        # 明示的にreindexで位置合わせを行う
        px_init_series = close.reindex(ev.index)
        # NaNがある行を除去
        valid_price_mask = px_init_series.notna()
        if not valid_price_mask.all():
            logger.warning(
                f"{(~valid_price_mask).sum()}個のイベントで開始価格が見つかりません。"
                f"これらの行を除外します。"
            )
            ev = ev[valid_price_mask]
            px_init_series = px_init_series[valid_price_mask]

        if ev.empty:
            return pd.DataFrame(columns=pd.Index(["ret", "bin", "trgt"]))

        px_init = px_init_series

        # t1の価格取得（ベクトル化）
        # t1の時刻がcloseのインデックスに存在しない場合、パディングで近似
        px_end_series = close.reindex(ev["t1"], method="pad")
        # t1に対応する価格が見つからない場合、その行は除外
        valid_end_price_mask = px_end_series.notna()
        if not valid_end_price_mask.all():
            logger.warning(
                f"{(~valid_end_price_mask).sum()}個のイベントで終了価格が見つかりません。"
                f"これらの行を除外します。"
            )
            ev = ev[valid_end_price_mask]
            px_end_series = px_end_series[valid_end_price_mask]
            px_init = px_init[valid_end_price_mask]

        ev_df = cast(pd.DataFrame, ev)
        if ev_df.empty:
            return pd.DataFrame(columns=pd.Index(["ret", "bin", "trgt"]))

        # インデックスをevに明示的に設定（位置合わせの保証）
        ev_df_for_index = cast(pd.DataFrame, ev)
        px_end = pd.Series(
            cast(pd.Series, px_end_series).values, index=ev_df_for_index.index
        )

        out = pd.DataFrame(index=ev_df_for_index.index)
        out["ret"] = px_end / px_init - 1
        out["bin"] = 0.0
        out["trgt"] = ev["trgt"]

        if "side" in ev_df_for_index.columns:
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
                out.loc[out["ret"] > out["trgt"] * self.pt * 0.999, "bin"] = (
                    1.0
                )
            if not binary_label and self.sl > 0:
                out.loc[out["ret"] < -out["trgt"] * self.sl * 0.999, "bin"] = (
                    -1.0
                )

        return out
