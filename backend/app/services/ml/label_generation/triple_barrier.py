from typing import List, Optional

import numpy as np
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
        """
        最初のバリア接触時刻を見つけます。

        Args:
            close (pd.Series): 終値。
            t_events (pd.DatetimeIndex): イベントのタイムスタンプ（例：CUSUMフィルターまたは全タイムスタンプ）。
            pt_sl (List[float]): [pt乗数, sl乗数]。
            target (pd.Series): 動的バリア幅のためのボラティリティ。
            min_ret (float): 最小ターゲットリターン。
            vertical_barrier_times (pd.Series): 垂直バリアのタイムスタンプ（タイムホライズン）。
            side (pd.Series): オプションの賭けの方向（買いは1、売りは-1）。Noneの場合、両側を確認します。

        Returns:
            pd.DataFrame: 't1'（接触時刻）、'trgt'（ターゲットリターン）、'side'（提供された場合）を含むイベント。
        """
        # 1. 各イベントのターゲットボラティリティを取得
        target = target.loc[t_events]
        target = target[target > min_ret]  # min_retでフィルタリング

        if target.empty:
            return pd.DataFrame(columns=["t1", "trgt"])

        # 2. 垂直バリアを取得
        if vertical_barrier_times is None:
            vertical_barrier_times = pd.Series(pd.NaT, index=t_events)
        else:
            vertical_barrier_times = vertical_barrier_times.loc[t_events]

        # 3. 接触時刻を見つける
        events = pd.DataFrame(index=target.index)
        events["t1"] = pd.NaT  # バリア接触時刻
        events["trgt"] = target
        events["side"] = None  # 接触したバリアの種類（'pt', 'sl', 'vertical'）

        # 乗数を抽出
        pt_mult = pt_sl[0]
        sl_mult = pt_sl[1]

        if side is None:
            # 両側をチェック（ボラティリティに基づく）- デフォルトは1（ロング）のロジック
            side_ = pd.Series(1.0, index=target.index)
        else:
            side_ = side.loc[target.index]

        # イベントをループ処理（並列化可能だが、簡潔さ/安全性のためにループを使用）
        for loc, t0 in enumerate(events.index):
            t1_vertical = vertical_barrier_times.loc[t0]
            trgt = target.iloc[loc]

            # タイムスタンプがpd.Timestampであることを確認
            t0 = pd.Timestamp(t0)
            if pd.notna(t1_vertical):
                t1_vertical = pd.Timestamp(t1_vertical)

            # close.indexとタイムゾーンを一致させる
            if close.index.tz is not None:
                # closeがタイムゾーン対応
                if t0.tz is None:
                    t0 = t0.tz_localize(close.index.tz)
                else:
                    t0 = t0.tz_convert(close.index.tz)

                if pd.notna(t1_vertical):
                    if t1_vertical.tz is None:
                        t1_vertical = t1_vertical.tz_localize(close.index.tz)
                    else:
                        t1_vertical = t1_vertical.tz_convert(close.index.tz)
            else:
                # closeがタイムゾーン非対応
                if t0.tz is not None:
                    t0 = t0.tz_localize(None)
                if pd.notna(t1_vertical) and t1_vertical.tz is not None:
                    t1_vertical = t1_vertical.tz_localize(None)

            # t0からt1_verticalまでの価格データをスライス（t1_verticalがNaTの場合は末尾まで）
            if pd.isna(t1_vertical):
                df0 = close[t0:]
            else:
                try:
                    df0 = close[t0:t1_vertical]  # t0を含む
                except Exception as e:
                    print("終値データのスライス中にエラーが発生しました:")
                    print(f"t0: {t0} (tz={t0.tz})")
                    print(
                        f"t1_vertical: {t1_vertical} (tz={t1_vertical.tz if pd.notna(t1_vertical) else 'NaT'})"
                    )
                    print(f"close.index.tz: {close.index.tz}")
                    raise e

            # t0に対するリターンを計算
            # returns = (df0 / close[t0]) - 1
            # 最適化: ループ内のスカラー除算を避ける（ただし、pandasのシリーズ演算は十分高速）
            if pd.isna(close.at[t0]):
                continue

            returns = (df0 / close.at[t0]) - 1

            out_bounds = pd.DataFrame(columns=["touch_type"])

            # sideに基づいてバリア設定を決定
            # side_val: 1 (ロング) または -1 (ショート)
            side_val = side_.iloc[loc]

            if side_val == -1:
                # ショートポジション
                # 上限バリア（価格上昇）-> 損失 (SL)
                # 下限バリア（価格下落）-> 利益 (PT)
                up_mult, up_type = sl_mult, "sl"
                down_mult, down_type = pt_mult, "pt"
            else:
                # ロングポジション（デフォルト）
                # 上限バリア（価格上昇）-> 利益 (PT)
                # 下限バリア（価格下落）-> 損失 (SL)
                up_mult, up_type = pt_mult, "pt"
                down_mult, down_type = sl_mult, "sl"

            # 上限バリアをチェック
            if up_mult > 0:
                # 上限バリアしきい値
                upper_thresh = trgt * up_mult

                # しきい値を超えた最初の時刻を見つける
                touch_upper = returns[returns > upper_thresh].index.min()
                if pd.notna(touch_upper):
                    out_bounds.loc[touch_upper, "touch_type"] = up_type

            # 下限バリアをチェック
            if down_mult > 0:
                # 下限バリアしきい値
                lower_thresh = -trgt * down_mult

                # しきい値を下回った最初の時刻を見つける
                touch_lower = returns[returns < lower_thresh].index.min()
                if pd.notna(touch_lower):
                    out_bounds.loc[touch_lower, "touch_type"] = down_type

            # どのバリアが最初に接触されたかを決定
            if not out_bounds.empty:
                first_touch_time = out_bounds.index.min()
                events.at[t0, "t1"] = first_touch_time

                touch_type = out_bounds.loc[first_touch_time, "touch_type"]
                if isinstance(touch_type, pd.Series):
                    # 最初のものを選択（まれなケース）
                    touch_type = touch_type.iloc[0]
                events.at[t0, "side"] = touch_type
            else:
                # バリアが接触しなかった場合、垂直バリアを使用
                events.at[t0, "t1"] = t1_vertical
                events.at[t0, "side"] = "vertical"

        return events

    def get_bins(
        self, events: pd.DataFrame, close: pd.Series, binary_label: bool = False
    ) -> pd.DataFrame:
        """
        バリア接触イベントに基づいてラベルを生成します。

        Args:
            events (pd.DataFrame): get_eventsの出力。
            close (pd.Series): 終値。
            binary_label (bool): Trueの場合、メタラベリング用の0/1ラベルを返します（1=トレンド/PT、0=ダマシ/SL/垂直）。

        Returns:
            pd.DataFrame: 'ret'（リターン）と'bin'（ラベル）を含むラベル。
        """
        # 1. t1がないイベントを削除（垂直バリアが存在する場合は発生しないはず）
        events_ = events.dropna(subset=["t1"])

        # 2. t0とt1の終値を取得
        # t0はevents_のインデックス
        # t1はevents_['t1']

        # t0の終値に参加
        px_init = close.loc[events_.index]

        # マッピングを使用してt1の終値に参加
        # 価格にマッピングするためにインデックスを使用。t1にはcloseインデックスにないタイムスタンプが含まれる可能性があるか？
        # t1はcloseインデックスから来ると仮定。

        # t1がcloseにない場合の潜在的なキー不足の処理（例：垂直バリアがわずかにずれている場合）
        # 理想的にはt1はclose.indexと完全に一致する。
        # 垂直バリアがタイムスタンプから構築された場合、closeインデックスと正確に一致しない可能性がある。
        # 必要に応じて 'asof' または再インデックスロジックを使用するが、ここではアライメントを仮定。

        # 安全のため、searchsortedまたはasofを使用
        t1_prices = []
        for t1_val in events_["t1"]:
            # 完全一致
            if t1_val in close.index:
                t1_prices.append(close.at[t1_val])
            else:
                # 直前の価格を見つける
                idx = close.index.get_indexer([t1_val], method="pad")[0]
                if idx != -1:
                    t1_prices.append(close.iloc[idx])
                else:
                    t1_prices.append(np.nan)

        px_end = pd.Series(t1_prices, index=events_.index)

        # 3. リターンを計算
        out = pd.DataFrame(index=events_.index)
        out["ret"] = px_end / px_init - 1

        # 4. ラベルを割り当てる（ビン）
        out["bin"] = 0.0  # デフォルトは0

        # イベントからターゲットとサイドを結合
        out["trgt"] = events_["trgt"]

        has_side_info = "side" in events_.columns

        if binary_label:
            # メタラベリングモード: PTが接触した場合は1、それ以外は0
            if has_side_info:
                # 明示的なサイド情報がある場合
                out.loc[events_["side"] == "pt", "bin"] = 1.0
                out.loc[events_["side"] == "sl", "bin"] = 0.0
                out.loc[events_["side"] == "vertical", "bin"] = 0.0
            else:
                # リターンベースのロジックにフォールバック
                # 上限バリアチェック (PT)
                if self.pt > 0:
                    # 厳密なチェック: PTしきい値より大きい必要がある
                    out.loc[out["ret"] > out["trgt"] * self.pt * 0.999, "bin"] = 1.0
                # SLと垂直は暗黙的に0
        else:
            # 標準モード: 1 (PT), -1 (SL), 0 (垂直/レンジ)

            if has_side_info:
                out.loc[events_["side"] == "pt", "bin"] = 1.0
                out.loc[events_["side"] == "sl", "bin"] = -1.0
                out.loc[events_["side"] == "vertical", "bin"] = 0.0
            else:
                # リターンベースのロジックにフォールバック
                # 上限バリアチェック
                if self.pt > 0:
                    out.loc[out["ret"] > out["trgt"] * self.pt * 0.999, "bin"] = 1.0

                # 下限バリアチェック
                if self.sl > 0:
                    out.loc[out["ret"] < -out["trgt"] * self.sl * 0.999, "bin"] = -1.0

        return out



