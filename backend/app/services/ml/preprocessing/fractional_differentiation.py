import logging
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FractionalDifferentiation:
    """
    分数次差分 (Fractional Differentiation) を計算するクラス。

    Marcos Lopez de Prado著 "Advances in Financial Machine Learning" に基づく実装。
    時系列の定常性を確保しつつ、過去の記憶（メモリ）を最大限保持することを目的とする。
    NumPyのconvolveを用いた高速ベクトル化実装。
    """

    def __init__(
        self, d: float = 1.0, window_size: int = 0, weight_threshold: float = 1e-4
    ):
        """
        初期化

        Args:
            d (float): 差分の次数。0 < d < 1 の範囲が一般的だが、任意の正の実数。
            window_size (int): 固定ウィンドウサイズ。0の場合は閾値に基づいて動的に決定（未実装）。
                               パフォーマンスと一貫性のために正の整数を指定することを推奨。
            weight_threshold (float): 重みの絶対値がこの値を下回った時点で計算を打ち切る閾値。
                                      window_sizeが指定されている場合は無視される場合がある。
        """
        self.d = d
        self.window_size = window_size
        self.weight_threshold = weight_threshold

    def _get_weights(self, d: float, size: int) -> np.ndarray:
        """
        分数次差分の重み係数を計算する。

        w_k = -w_{k-1} * (d - k + 1) / k

        Args:
            d (float): 差分次数
            size (int): 重みの数（ウィンドウサイズ）

        Returns:
            np.ndarray: 重み係数の配列 (w_0, w_1, ..., w_{size-1})
        """
        # 重み計算もベクトル化可能だが、サイズが小さいのでループでも十分
        # しかし一応改善しておく
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] * (d - k + 1) / k
            w.append(w_k)
        return np.array(w)

    def transform(
        self, data: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        """データに対して分数次差分を適用する"""
        if isinstance(data, pd.Series):
            return self._transform_series(data)
        if isinstance(data, pd.DataFrame):
            # Applyは遅いので、列ごとにループして結合するほうが安全かつ制御しやすいが、
            # np.convolveならapplyでapply(raw=False)を使わなくても高速化できるので
            # ここではシンプルに各列処理
            # DataFrame全体を一度に処理するのは難しい（長さが違うとNaNが入るので）

            # ベクトル化されたSeries変換を適用
            return data.apply(self._transform_series)

        raise TypeError("Input must be a Series or DataFrame")

    def _transform_series(self, series: pd.Series) -> pd.Series:
        """単一のSeriesに対して分数次差分を適用する（固定ウィンドウ方式・高速版）"""
        window_size = int(self.window_size)

        # データ不足時の処理
        if len(series) < window_size:
            return pd.Series(np.nan, index=series.index, name=series.name)

        # 欠損値の前処理: 先頭にあるNaNはスキップするか、ffillする必要がある
        # ここではシンプルに、入力にNaNが含まれているとconvolve結果もNaNになる仕様とする
        # 必要であれば呼び出し側でffillすることを期待、あるいはここでffill
        series_vals = series.ffill().values.astype(np.float64)

        # 重みの取得 (新しいデータが先頭に来るように反転する等、convolveの仕様に合わせる)
        # get_weights は [w0, w1, ..., w_k] を返す (w0が現在の値の重み、w1が1期前...)
        # np.convolve(mode='valid') を使う場合、
        # filter h と signal x の畳み込みは sum(h[k] * x[n-k])
        # w = [w0, w1, ... w_size-1]
        # 重み係数は w0*x[t] + w1*x[t-1] + ... という定義。
        # convolve(x, h)の実装:
        # (x * h)[n] = \sum_{m} x[m] h[n-m]
        # 一方、フィルタリングとしては、y[n] = \sum_{k} w[k] x[n-k]
        # これは畳み込みの定義そのものなので、wをそのまま h として渡せばよい。

        # 検証：d=1 のとき、weights=[1, -1]
        # x=[10, 20]
        # convolve([10, 20], [1, -1])
        # -> n=1のとき: x[0]*h[1] + x[1]*h[0] = 10*(-1) + 20*(1) = 10. Correct.
        # よって、反転させてはいけない。

        weights = self._get_weights(self.d, window_size)

        # np.convolve(x, w) で正しい順序計算になる
        diff_vals = np.convolve(series_vals, weights, mode="valid")

        # 結果を格納する配列 (初期値NaN)

        result = np.full(len(series), np.nan)

        # validモードの結果は配列の後ろの方（最新）に位置する
        # index: window_size-1 から最後まで
        result[window_size - 1 :] = diff_vals

        return pd.Series(result, index=series.index, name=series.name)
