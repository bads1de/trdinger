"""
暗号資産のための高度な特徴量エンジニアリング
戦略ドキュメント (features_strategy.pdf) からの「強力な特徴量」を実装します。
"""

import logging

import numpy as np
import pandas as pd

from ..utils import handle_pandas_ta_errors

logger = logging.getLogger(__name__)


class AdvancedFeatures:
    """
    分数差分、清算カスケードなどを含む高度な特徴量。
    """

    @staticmethod
    def get_weights_ffd(d: float, thres: float, lim: int) -> np.ndarray:
        """
        分数微分（固定ウィンドウ）の重みを計算します。

        Args:
            d (float): 差分次数 (例: 0.4)
            thres (float): 重みのカットオフ閾値 (例: 1e-5)
            lim (int): 重みの最大長

        Returns:
            np.ndarray: 重みの配列 (逆順: w[-1] が w_0)
        """
        w, k = [1.0], 1
        while True:
            w_k = -w[-1] / k * (d - k + 1)
            if abs(w_k) < thres or len(w) >= lim:
                break
            w.append(w_k)
            k += 1
        return np.array(w[::-1]).reshape(-1, 1)

    @staticmethod
    @handle_pandas_ta_errors
    def frac_diff_ffd(
        series: pd.Series,
        d: float = 0.4,
        thres: float = 1e-5,
        window: int = 2000,
    ) -> pd.Series:
        """
        分数微分（固定ウィンドウFFD）。
        時系列データの定常性を達成しつつ、メモリ（長期記憶）を保持します。

        Args:
            series (pd.Series): 入力価格シリーズ (通常は対数価格)
            d (float): 差分の次数 (0 < d < 1)
            thres (float): 重みのカットオフ閾値
            window (int): 重みの最大ウィンドウサイズ
        """
        if not isinstance(series, pd.Series):
            raise TypeError("series must be pandas Series")

        # 1. 重みの計算
        w = AdvancedFeatures.get_weights_ffd(d, thres, window)
        width = len(w)
        w = w.flatten()

        # 2. 重みの適用
        # 安全性と明確さのためにループを使用しますが、ストライドによるベクトル化も可能です
        series_filled = series.ffill()
        series_vals = series_filled.values

        result = np.full(len(series), np.nan)

        if len(series) >= width:
            for i in range(width - 1, len(series)):
                # データのウィンドウを取得: X_{t-width+1} ... X_{t}
                # window_valは、[w_K, ..., w_0]である重みwと一致させる必要があります
                # ここで、w_0はX_tに対応し、w_1はX_{t-1}に対応します
                # get_weights_ffdは[w_K, ..., w_0]を返します
                # したがって、window_valが[X_{t-K}, ..., X_{t}]であれば、単純なドット積が機能します

                window_val = series_vals[i - width + 1 : i + 1]
                result[i] = np.dot(window_val, w)

        return pd.Series(result, index=series.index)

    @staticmethod
    @handle_pandas_ta_errors
    def liquidation_cascade_score(
        close: pd.Series,
        open_interest: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        清算カスケードスコア

        反転のシグナルとなることが多い強制清算（ロングの投げ売りやショートの買い戻し）を検出します。

        計算式: -1 * sign(Delta P) * Delta OI * Volume
        価格が下落(sign=-1)し、OIが減少(Delta OI < 0)し、出来高が大きい場合、スコアは負の大きな値となり、ロングの清算を示唆します。
        """
        # 変化量の計算
        delta_p = close.diff()
        delta_oi = open_interest.diff()

        # 価格変化の符号 (+1, 0, -1)
        sign_p = np.sign(delta_p)

        # 計算式
        # 注: 清算中はOIが減少するため delta_oi は負になります
        # 価格下落(-1) * OI減少(-) * Volume(+) -> 全体として負の値になるべきか？
        # 原論文に従うと、スコアの絶対値が大きいほどカスケードの可能性が高い
        score = -1 * sign_p * delta_oi * volume

        return score.fillna(0.0)

    @staticmethod
    @handle_pandas_ta_errors
    def squeeze_probability(
        close: pd.Series,
        funding_rate: pd.Series,
        open_interest: pd.Series,
        low: pd.Series,
        lookback: int = 20,
    ) -> pd.Series:
        """
        ショートスクイーズ確率指数

        以下の条件からショートスクイーズの発生確率（セットアップ）を検出します:
        1. 負のファンディングレート（ショートがロングに金利を支払っている状態）
        2. OIの増加（ショートポジションが積み上がっている状態）
        3. 価格が直近安値に対して相対的に高い位置にある（価格が下がらず維持されている＝吸収されている）
        """
        # 1. 負のファンディングレート要因: I(FR < 0) * |FR|
        neg_fr_factor = np.where(funding_rate < 0, funding_rate.abs(), 0)

        # 2. OI変化要因: Delta OI
        delta_oi = open_interest.diff()
        # OIの増加（ショートの積み増し）のみに着目するため、負の値は0にクリップ
        delta_oi_factor = delta_oi.clip(lower=0)

        # 3. 価格位置要因: Price - Low_n
        # 価格が安値圏で停滞せず、ショートの積み増しにも関わらず上昇または維持している場合、
        # ショート勢にとっての「痛み（含み損）」が増していることを示唆します。

        low_n = low.rolling(window=lookback).min()
        price_location = close - low_n

        # 総合スコア（確率プロキシ）
        prob = (
            pd.Series(neg_fr_factor, index=close.index)
            * delta_oi_factor
            * price_location
        )

        return prob.fillna(0.0)

    @staticmethod
    @handle_pandas_ta_errors
    def trend_quality(
        close: pd.Series,
        open_interest: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """
        トレンド品質（VWAP/OIダイバージェンスの代替指標）

        価格変化とOI変化の相関関係を計算します。
        - 正の相関: 健全なトレンド（新規資金の流入によって価格が動いている）
        - 負の相関: 弱いトレンド（既存ポジションの解消によって価格が動いている）
        """
        delta_p = close.diff()
        delta_oi = open_interest.diff()

        correlation = delta_p.rolling(window=window).corr(delta_oi)

        return correlation.fillna(0.0)

    @staticmethod
    @handle_pandas_ta_errors
    def oi_weighted_funding_rate(
        funding_rate: pd.Series,
        open_interest: pd.Series,
    ) -> pd.Series:
        """
        OI加重ファンディングレート

        計算式: FR * OI
        市場全体で支払われている/受け取られているファンディング手数料の総額（ドル価値）を表します。
        市場の偏りとコスト圧力を測る指標です。
        """
        return funding_rate * open_interest

    @staticmethod
    def liquidity_efficiency(open_interest: pd.Series, volume: pd.Series) -> pd.Series:
        """
        流動性効率（Open Interest / Volume）
        """
        return (open_interest / volume).replace([np.inf, -np.inf], 0).fillna(0)

    @staticmethod
    def leverage_ratio(open_interest: pd.Series, market_cap: pd.Series) -> pd.Series:
        """
        推定レバレッジ比率（Total OI / Estimated Market Cap）
        """
        return (open_interest / market_cap).replace([np.inf, -np.inf], 0).fillna(0)




