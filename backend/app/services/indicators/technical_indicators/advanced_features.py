"""
暗号資産のための高度な特徴量エンジニアリング
戦略ドキュメント (features_strategy.pdf) からの「強力な特徴量」を実装します。
"""

import logging

import numpy as np
import pandas as pd

from ..data_validation import (
    handle_pandas_ta_errors,
    validate_multi_series_params,
    validate_series_params,
)

logger = logging.getLogger(__name__)


class AdvancedFeatures:
    """
    分数差分、清算カスケードなどを含む高度な特徴量。
    """

    @staticmethod
    def get_weights_ffd(d: float, thres: float, lim: int) -> np.ndarray:
        """
        分数微分（固定ウィンドウ）の重みを計算します。
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
    def z_score(series: pd.Series, window: int = 20) -> pd.Series:
        """
        Zスコアを計算 (x - mean) / std
        """
        roll = series.rolling(window=window)
        return ((series - roll.mean()) / (roll.std() + 1e-9)).fillna(0.0)

    @staticmethod
    @handle_pandas_ta_errors
    def void_oscillator(
        close: pd.Series,
        volume: pd.Series,
        window: int = 20,
        volume_threshold_quantile: float = 0.2
    ) -> pd.Series:
        """
        流動性真空検知器 (Void Oscillator)
        
        出来高が薄い中での価格急変動（真空地帯）を検知。
        大きな値は、薄商いの中での急騰・急落（ダマシの可能性大）を示す。
        """
        validation = validate_multi_series_params(
            {"close": close, "volume": volume}, window
        )
        if validation is not None:
            return validation

        # 1. 価格変動のZスコア
        ret = np.log(close / close.shift(1)).fillna(0)
        z_ret = AdvancedFeatures.z_score(ret, window)
        
        # 2. 出来高ショック（閾値以下の出来高を検知）
        # 過去window期間の分位点を計算
        vol_threshold = volume.rolling(window=window).quantile(volume_threshold_quantile)
        is_low_volume = (volume < vol_threshold).astype(int)
        
        return z_ret * is_low_volume

    @staticmethod
    @handle_pandas_ta_errors
    def crypto_leverage_index(
        open_interest: pd.Series,
        funding_rate: pd.Series,
        ls_ratio_divergence: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Crypto Leverage Index (CLI)
        
        市場全体のレバレッジ過熱感を示す複合インデックス。
        Norm(OI Z-Score) + Norm(|FR| * sign(FR)) + Norm(L/S Divergence)
        """
        validation = validate_multi_series_params(
            {
                "open_interest": open_interest,
                "funding_rate": funding_rate,
                "ls_ratio_divergence": ls_ratio_divergence
            }, window
        )
        if validation is not None:
            return validation

        # 1. OI Z-Score
        z_oi = AdvancedFeatures.z_score(open_interest, window)
        
        # 2. FR Impact (|FR| * sign(FR) = FRそのものだが、概念的に強調)
        # FRの絶対値が大きいほどレバレッジコストが高い
        fr_impact = funding_rate # そのまま使用
        z_fr = AdvancedFeatures.z_score(fr_impact, window)
        
        # 3. L/S Divergence (Whale Divergenceなど)
        # 1.0からの乖離を評価
        ls_div = ls_ratio_divergence - 1.0
        z_ls = AdvancedFeatures.z_score(ls_div, window)
        
        # 単純加算（重み付けなし）
        cli = z_oi + z_fr + z_ls
        
        return cli

    @staticmethod
    @handle_pandas_ta_errors
    def triplet_imbalance(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.Series:
        """
        Triplet Imbalance (価格構造の不均衡)
        
        (Max - Mid) / (Mid - Min)
        ここでは (High - Close) / (Close - Low) として実装し、
        上ヒゲと下ヒゲのバランス（売り圧力 vs 買い圧力）を評価する。
        値が大きいほど売り圧力（上ヒゲ）が強い。
        """
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}
        )
        if validation is not None:
            return validation

        upper_shadow = high - close
        lower_shadow = close - low
        
        # 0除算回避
        imbalance = upper_shadow / (lower_shadow + 1e-9)
        
        # 対数変換で分布を正規化
        return np.log(imbalance + 1e-9).fillna(0.0)

    @staticmethod
    @handle_pandas_ta_errors
    def volume_divergence_fakeout(
        close: pd.Series,
        volume: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Volume Divergence Fakeout (出来高ダイバージェンス)
        
        価格が過去N期間の高値を更新したにもかかわらず、
        出来高が平均以下である場合、ダマシ（Fakeout）の可能性が高い。
        
        Returns:
            Fakeout Score (1.0 = 強いダマシシグナル, 0.0 = 正常)
        """
        validation = validate_multi_series_params(
            {"close": close, "volume": volume}, window
        )
        if validation is not None:
            return validation

        # 1. 新高値フラグ
        rolling_max = close.rolling(window=window).max().shift(1)
        is_new_high = (close > rolling_max)
        
        # 2. 出来高減衰フラグ (Volume < SMA_Volume)
        vol_ma = volume.rolling(window=window).mean()
        is_low_volume = (volume < vol_ma)
        
        # 3. 乖離度 (価格の上昇幅 * 出来高の不足分)
        price_gain = (close - rolling_max) / rolling_max
        vol_drop = (vol_ma - volume) / vol_ma
        
        fakeout_score = np.where(
            is_new_high & is_low_volume,
            price_gain * vol_drop * 100, # スコアを見やすくスケーリング
            0.0
        )
        
        return pd.Series(fakeout_score, index=close.index).fillna(0.0)

    @staticmethod
    def _calculate_rs_hurst(series_chunk: np.ndarray) -> float:
        """R/S分析によるハースト指数の単一ウィンドウ計算（ヘルパー）"""
        if len(series_chunk) < 2:
            return 0.5
            
        # 対数リターン
        # series_chunk は価格データそのもの想定
        # 簡易計算のため、ここでは差分（増分）を用いる
        incs = series_chunk[1:] - series_chunk[:-1]
        
        # 平均からの偏差
        mean_inc = np.mean(incs)
        centered = incs - mean_inc
        
        # 累積偏差 (Cumulative Deviations)
        z = np.cumsum(centered)
        
        # 範囲 (Range)
        r = np.max(z) - np.min(z)
        
        # 標準偏差 (Standard Deviation)
        s = np.std(incs, ddof=1)
        
        if s == 0:
            return 0.5
            
        # R/S
        rs = r / s
        
        # Hurst = log(R/S) / log(n) の簡易推定
        # 本来は複数のnで回帰分析するが、ローリング計算用に簡易化
        n = len(incs)
        if n < 2 or rs <= 0:
            return 0.5
            
        h = np.log(rs) / np.log(n / 2) # n/2は経験的な調整項
        return float(np.clip(h, 0.0, 1.0))

    @staticmethod
    @handle_pandas_ta_errors
    def hurst_exponent(
        close: pd.Series,
        window: int = 100
    ) -> pd.Series:
        """
        ハースト指数 (Hurst Exponent)
        
        時系列の長期記憶性を測定。
        H < 0.5: 平均回帰的（レンジ）
        H = 0.5: ランダムウォーク
        H > 0.5: トレンド持続적
        
        注意: 計算コストが高いため、ローリング適用にはNumba等の最適化が望ましいが、
        ここではPandasのrolling.applyを使用（やや遅い）。
        """
        validation = validate_series_params(close, window)
        if validation is not None:
            return validation

        # rolling apply で計算
        # raw=True で numpy array を渡す
        hurst = close.rolling(window=window).apply(
            AdvancedFeatures._calculate_rs_hurst, raw=True
        )
        
        return hurst.fillna(0.5)

    @staticmethod
    def _calculate_sample_entropy(L: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """サンプル・エントロピーの計算（ヘルパー）"""
        N = len(L)
        if N <= m:
            return 0.0
            
        # 標準偏差でスケーリングされた閾値
        r_thresh = r * np.std(L)
        if r_thresh == 0:
            return 0.0

        def count_matches(m_len):
            count = 0
            # テンプレートベクトルの作成
            X = np.array([L[i : i + m_len] for i in range(N - m_len + 1)])
            # すべてのペア間の距離（無限遠ノルム）を計算
            for i in range(len(X)):
                # 自分自身を除外した距離
                dist = np.max(np.abs(X - X[i]), axis=1)
                count += np.sum(dist <= r_thresh) - 1
            return count

        A = count_matches(m + 1)
        B = count_matches(m)
        
        if A == 0 or B == 0:
            return 0.0
            
        return -np.log(A / B)

    @staticmethod
    @handle_pandas_ta_errors
    def sample_entropy(
        close: pd.Series,
        window: int = 50,
        m: int = 2,
        r: float = 0.2
    ) -> pd.Series:
        """
        サンプル・エントロピー (Sample Entropy)
        
        時系列の複雑さと規則性を測定。低いほど規則的（トレンドの可能性）、高いほどランダム。
        """
        validation = validate_series_params(close, window)
        if validation is not None:
            return validation
            
        # 計算コストが高いため、小さな窓幅を推奨
        entropy = close.rolling(window=window).apply(
            lambda x: AdvancedFeatures._calculate_sample_entropy(x, m, r),
            raw=True
        )
        return entropy.fillna(0.0)

    @staticmethod
    def _calculate_katz_fd(x: np.ndarray) -> float:
        """Katzのフラクタル次元計算（ヘルパー）"""
        n = len(x) - 1
        if n <= 0:
            return 1.0
            
        # 総経路長 (Total path length)
        dists = np.abs(np.diff(x))
        L = np.sum(dists)
        
        # 平均ステップ長
        a = L / n
        
        # 平面距離 (Planar distance between first point and the farthest point)
        d = np.max(np.abs(x - x[0]))
        
        if L == 0 or d == 0 or a == 0:
            return 1.0
            
        # Katz formula: D = log10(n) / (log10(d/L) + log10(n))
        # ここでは log10(L/a) = log10(n) を利用
        return np.log10(n) / (np.log10(d / L) + np.log10(n))

    @staticmethod
    @handle_pandas_ta_errors
    def fractal_dimension(
        close: pd.Series,
        window: int = 30
    ) -> pd.Series:
        """
        Katzのフラクタル次元 (Fractal Dimension)
        
        チャートの「粗さ」を測定。1.0（直線）〜 2.0（平面を埋め尽くすほど複雑）。
        トレンドが発生すると次元が低下する傾向がある。
        """
        validation = validate_series_params(close, window)
        if validation is not None:
            return validation
            
        fd = close.rolling(window=window).apply(
            AdvancedFeatures._calculate_katz_fd, raw=True
        )
        # 理論上の範囲 [1.0, 2.0] にクリップして安定させる
        return fd.clip(lower=1.0, upper=2.0).fillna(1.0)

    @staticmethod
    @handle_pandas_ta_errors
    def vpin_approximation(
        close: pd.Series,
        volume: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        近似VPIN (Volume-Induced Probability of Informed Trading)
        
        出来高の不均衡から「情報に基づいた取引」の確率を推定。
        高いほど需給が偏っており、急変動の前兆となる。
        """
        validation = validate_multi_series_params({"close": close, "volume": volume}, window)
        if validation is not None:
            return validation

        # 価格変化の方向
        delta_p = close.diff().fillna(0)
        
        # 簡易的なバイ/セル分割
        # CDF(標準正規分布)を用いて、価格変化の大きさに応じて出来高を按分
        # 面倒なCDF計算の代わりに、シグモイド関数や単純な符号付き割合で近似
        vol_std = delta_p.rolling(window=window).std() + 1e-9
        z_score = delta_p / vol_std
        
        # 買い出来高の割合 (0.0 to 1.0)
        buy_ratio = 1 / (1 + np.exp(-z_score))
        
        buy_vol = volume * buy_ratio
        sell_vol = volume * (1 - buy_ratio)
        
        # VPIN = sum(|V_buy - V_sell|) / sum(V_total)
        abs_imbalance = np.abs(buy_vol - sell_vol)
        
        vpin = abs_imbalance.rolling(window=window).sum() / (volume.rolling(window=window).sum() + 1e-9)
        
        return vpin.fillna(0.0)

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
        """
        validation = validate_series_params(series, window)
        if validation is not None:
            return validation

        # d=0 の場合は元のシリーズをそのまま返す（計算コスト削減と精度維持）
        if abs(d) < 1e-9:
            return series.copy()

        # 1. 重みの計算
        w = AdvancedFeatures.get_weights_ffd(d, thres, window)
        width = len(w)
        w = w.flatten()

        # 2. 重みの適用
        series_filled = series.ffill()
        series_vals = series_filled.values

        result = np.full(len(series), np.nan)

        if len(series) >= width:
            for i in range(width - 1, len(series)):
                window_val = series_vals[i - width + 1 : i + 1]
                result[i] = np.dot(window_val, w)

        return pd.Series(result, index=series.index, name=series.name)

    @staticmethod
    @handle_pandas_ta_errors
    def liquidation_cascade_score(
        close: pd.Series,
        open_interest: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        清算カスケードスコア
        """
        validation = validate_multi_series_params(
            {"close": close, "open_interest": open_interest, "volume": volume}
        )
        if validation is not None:
            return validation

        # 変化量の計算
        delta_p = close.diff()
        delta_oi = open_interest.diff()

        # 価格変化の符号 (+1, 0, -1)
        sign_p = np.sign(delta_p)

        # 計算式
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
        """
        validation = validate_multi_series_params(
            {
                "close": close,
                "funding_rate": funding_rate,
                "open_interest": open_interest,
                "low": low,
            },
            lookback,
        )
        if validation is not None:
            return validation

        # 1. 負のファンディングレート要因: I(FR < 0) * |FR|
        neg_fr_factor = np.where(funding_rate < 0, funding_rate.abs(), 0)

        # 2. OI変化要因: Delta OI
        delta_oi = open_interest.diff()
        delta_oi_factor = delta_oi.clip(lower=0)

        # 3. 価格位置要因: Price - Low_n
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
        """
        validation = validate_multi_series_params(
            {"close": close, "open_interest": open_interest}, window
        )
        if validation is not None:
            return validation

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
        """
        validation = validate_multi_series_params(
            {"funding_rate": funding_rate, "open_interest": open_interest}
        )
        if validation is not None:
            return validation

        return funding_rate * open_interest

    @staticmethod
    @handle_pandas_ta_errors
    def regime_quadrant(
        close: pd.Series,
        open_interest: pd.Series,
    ) -> pd.Series:
        """
        価格とOIの変化に基づく4象限レジーム分析
        
        0: 強気トレンド (Price↑, OI↑) - 新規買い
        1: ショートカバー (Price↑, OI↓) - 売り決済（ダマシ予兆）
        2: 弱気トレンド (Price↓, OI↑) - 新規売り
        3: ロング清算 (Price↓, OI↓) - 買い決済（反発予兆）
        """
        validation = validate_multi_series_params(
            {"close": close, "open_interest": open_interest}
        )
        if validation is not None:
            return validation

        delta_p = close.diff().fillna(0)
        delta_oi = open_interest.diff().fillna(0)

        # 条件分岐をベクトル化
        # デフォルトは -1 (不明/変化なし)
        regime = pd.Series(-1, index=close.index, dtype=int)
        
        # 0: Bull Trend
        mask_bull = (delta_p > 0) & (delta_oi > 0)
        regime[mask_bull] = 0
        
        # 1: Short Cover
        mask_cover = (delta_p > 0) & (delta_oi < 0)
        regime[mask_cover] = 1
        
        # 2: Bear Trend
        mask_bear = (delta_p < 0) & (delta_oi > 0)
        regime[mask_bear] = 2
        
        # 3: Long Liquidation
        mask_liq = (delta_p < 0) & (delta_oi < 0)
        regime[mask_liq] = 3
        
        return regime

    @staticmethod
    @handle_pandas_ta_errors
    def whale_divergence(
        ls_ratio_positions: pd.Series,
        ls_ratio_accounts: pd.Series,
    ) -> pd.Series:
        """
        クジラ乖離スコア (Whale Divergence Score)
        
        Top Trader L/S Ratio (Positions) / Global Account L/S Ratio
        > 1.0: 大口が個人より強気（スマートマネーの買い）
        < 1.0: 大口が個人より弱気（天井圏の可能性大）
        """
        validation = validate_multi_series_params(
            {"ls_ratio_positions": ls_ratio_positions, "ls_ratio_accounts": ls_ratio_accounts}
        )
        if validation is not None:
            return validation

        return (ls_ratio_positions / ls_ratio_accounts).replace([np.inf, -np.inf], 1.0).fillna(1.0)

    @staticmethod
    @handle_pandas_ta_errors
    def oi_price_confirmation(
        close: pd.Series,
        open_interest: pd.Series,
    ) -> pd.Series:
        """
        OI-Price Confirmation
        
        sign(Delta P) * Delta OI
        正: トレンドはOI増加によって支持されている
        負: トレンドとOIが逆行（ダイバージェンス）
        """
        validation = validate_multi_series_params(
            {"close": close, "open_interest": open_interest}
        )
        if validation is not None:
            return validation

        delta_p_sign = np.sign(close.diff().fillna(0))
        delta_oi = open_interest.diff().fillna(0)
        
        return delta_p_sign * delta_oi

    @staticmethod
    @handle_pandas_ta_errors
    def liquidity_efficiency(open_interest: pd.Series, volume: pd.Series) -> pd.Series:
        """
        流動性効率（Open Interest / Volume）
        """
        validation = validate_multi_series_params(
            {"open_interest": open_interest, "volume": volume}
        )
        if validation is not None:
            return validation

        return (open_interest / volume).replace([np.inf, -np.inf], 0).fillna(0)

    @staticmethod
    @handle_pandas_ta_errors
    def leverage_ratio(open_interest: pd.Series, market_cap: pd.Series) -> pd.Series:
        """
        推定レバレッジ比率（Total OI / Estimated Market Cap）
        """
        validation = validate_multi_series_params(
            {"open_interest": open_interest, "market_cap": market_cap}
        )
        if validation is not None:
            return validation

        return (open_interest / market_cap).replace([np.inf, -np.inf], 0).fillna(0)
