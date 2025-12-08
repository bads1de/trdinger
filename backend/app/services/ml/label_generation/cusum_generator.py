import pandas as pd
import numpy as np
from typing import Optional


class CusumSignalGenerator:
    """
    Symmetric CUSUM Filter を用いてイベントを生成するクラス。
    Marcos Lopez de Prado "Advances in Financial Machine Learning" Chapter 2.5.2.1 準拠。
    """

    def get_events(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
        volatility: Optional[pd.Series] = None,
        price_column: str = "close",
        vol_multiplier: float = 1.0,
    ) -> pd.DatetimeIndex:
        """
        CUSUMフィルターを実行し、イベントが発生したタイムスタンプを返す。

        Args:
            df: OHLCV DataFrame
            threshold: 固定閾値（対数収益率ベース）。Noneの場合はvolatilityを使用。
            volatility: 動的閾値として使用するボラティリティSeries。
            price_column: 価格カラム名
            vol_multiplier: 動的閾値の乗数（volatility * multiplier）

        Returns:
            pd.DatetimeIndex: イベント発生時刻のインデックス
        """
        prices = df[price_column]

        # 対数収益率を計算
        # r_t = ln(p_t / p_{t-1})
        log_returns = np.log(prices / prices.shift(1))
        log_returns = log_returns.fillna(0)  # 最初は0

        # 閾値の準備
        if threshold is not None:
            # 固定閾値
            h = pd.Series(threshold, index=df.index)
        elif volatility is not None:
            # 動的閾値（ボラティリティ）
            h = volatility * vol_multiplier
        else:
            raise ValueError("Either threshold or volatility must be provided.")

        # CUSUM計算
        t_events = []
        s_pos = 0.0
        s_neg = 0.0

        # 収益率と閾値をnumpy配列に変換して高速化
        arr_returns = log_returns.values
        arr_h = h.values
        arr_index = df.index

        # ループ処理
        # Note: CUSUMは直前の状態に依存するためベクトル化が難しい
        for i in range(1, len(arr_returns)):
            r_t = arr_returns[i]
            h_t = arr_h[i]

            # 閾値がNaNの場合はスキップ（ボラティリティ計算初期など）
            if np.isnan(h_t) or h_t <= 0:
                continue

            # S_t^+ = max(0, S_{t-1}^+ + r_t)
            # S_t^- = min(0, S_{t-1}^- + r_t)
            # ※ 本来は r_t - E[r_t] だが、E[r_t] (期待収益率) は通常0と仮定されることが多い

            s_pos = max(0, s_pos + r_t)
            s_neg = min(0, s_neg + r_t)

            if s_pos > h_t:
                s_pos = 0  # リセット
                t_events.append(arr_index[i])
            elif s_neg < -h_t:
                s_neg = 0  # リセット
                t_events.append(arr_index[i])

        return pd.DatetimeIndex(t_events)

    def get_daily_volatility(self, close: pd.Series, span: int = 100) -> pd.Series:
        """
        日次ボラティリティを計算する。
        Exponential Weighted Moving Standard Deviation of Returns.

        Args:
            close: 終値シリーズ
            span: EWMのspan（期間）

        Returns:
            pd.Series: 日次ボラティリティ
        """
        # 前日比（対数収益率ではないが、近似として変化率を使用）
        # 本によっては対数収益率を使う場合もあるが、ここではシンプルにpct_change
        # ただし、CUSUMで対数収益率を使うならここも合わせるべき

        # 1. 前日（1日前のデータ）とのインデックスを合わせる
        # データが1時間足の場合、"1日"は24本分ではない。
        # ここでは単純に「直前の足とのリターン」の標準偏差を計算し、
        # それをイベント検知の閾値として使う。
        # "Daily" Volatilityという名前だが、実際には "Local" Volatility として機能する。

        returns = np.log(close / close.shift(1))

        # 指数加重移動標準偏差
        vol = returns.ewm(span=span).std()

        return vol
