from typing import List, Optional

import pandas as pd


class TripleBarrier:
    """
    金融データのラベリングのためのトリプルバリア法。
    Marcos Lopez de Pradoの「Advances in Financial Machine Learning」に基づいています。
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
        """最初のバリア接触時刻を見つける"""
        target = target.loc[t_events]
        target = target[target > min_ret]
        if target.empty:
            return pd.DataFrame(columns=["t1", "trgt", "side"])

        v_bar = vertical_barrier_times.loc[target.index] if vertical_barrier_times is not None else pd.Series(pd.NaT, index=target.index)
        side_ = side.loc[target.index] if side is not None else pd.Series(1.0, index=target.index)

        events = pd.DataFrame(index=target.index, columns=["t1", "trgt", "side"])
        events["trgt"] = target
        pt_mult, sl_mult = pt_sl

        for t0 in target.index:
            t1_v = v_bar.loc[t0]
            trgt = target.loc[t0]
            
            # データスライス
            df0 = close[t0:t1_v] if pd.notna(t1_v) else close[t0:]
            if len(df0) < 2 or pd.isna(close.at[t0]):
                continue
            
            rets = (df0 / close.at[t0]) - 1
            s = side_.loc[t0]
            
            # バリア設定 (1:ロング, -1:ショート)
            up_m, up_t = (sl_mult, "sl") if s == -1 else (pt_mult, "pt")
            dn_m, dn_t = (pt_mult, "pt") if s == -1 else (sl_mult, "sl")

            # 接触判定
            touch_times = {}
            if up_m > 0:
                t = rets[rets > trgt * up_m].index.min()
                if pd.notna(t):
                    touch_times[t] = up_t
            if dn_m > 0:
                t = rets[rets < -trgt * dn_m].index.min()
                if pd.notna(t):
                    touch_times[t] = dn_t

            if touch_times:
                first_t = min(touch_times.keys())
                events.at[t0, "t1"], events.at[t0, "side"] = first_t, touch_times[first_t]
            else:
                events.at[t0, "t1"], events.at[t0, "side"] = t1_v, "vertical"

        return events

    def get_bins(
        self, events: pd.DataFrame, close: pd.Series, binary_label: bool = False
    ) -> pd.DataFrame:
        """バリア接触イベントに基づいてラベルを生成"""
        ev = events.dropna(subset=["t1"])
        px_init = close.loc[ev.index]
        
        # t1の価格取得 (ベクトル化)
        px_end = close.reindex(ev["t1"], method="pad")
        px_end.index = ev.index

        out = pd.DataFrame(index=ev.index)
        out["ret"] = px_end / px_init - 1
        out["bin"] = 0.0
        out["trgt"] = ev["trgt"]

        if "side" in ev.columns:
            if binary_label:
                out.loc[ev["side"] == "pt", "bin"] = 1.0
            else:
                out.loc[ev["side"] == "pt", "bin"] = 1.0
                out.loc[ev["side"] == "sl", "bin"] = -1.0
        else:
            # フォールバック (リターンベース)
            if self.pt > 0:
                out.loc[out["ret"] > out["trgt"] * self.pt * 0.999, "bin"] = 1.0
            if not binary_label and self.sl > 0:
                out.loc[out["ret"] < -out["trgt"] * self.sl * 0.999, "bin"] = -1.0

        return out



