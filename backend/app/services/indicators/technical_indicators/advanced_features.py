"""
暗号資産のための高度な特徴量エンジニアリング
戦略ドキュメント (features_strategy.pdf) からの「強力な特徴量」を実装します。

登録してある特徴量の一覧:
- Z-Score
- Void Oscillator
- Crypto Leverage Index (CLI)
- Triplet Imbalance
- Volume Divergence Fakeout
- Hurst Exponent
- Sample Entropy
- Fractal Dimension (Katz)
- VPIN Approximation
- Fractional Differentiation (FFD)
- Liquidation Cascade Score
- Squeeze Probability
- Trend Quality
- OI Weighted Funding Rate
- Regime Quadrant
- Whale Divergence
- OI-Price Confirmation
- Liquidity Efficiency
- Leverage Ratio
"""

import logging
from typing import cast

import numpy as np
import pandas as pd
from numba import njit, prange

from ..data_validation import (
    handle_pandas_ta_errors,
    normalize_non_finite,
    run_multi_series_indicator,
    run_series_indicator,
)

logger = logging.getLogger(__name__)


@njit(parallel=True, cache=True)  # type: ignore[untyped-decorator]
def _njit_hurst_loop(data: np.ndarray, win: int) -> np.ndarray:
    """
    ハースト指数を計算する Numba 加速ループ。

    Args:
        data: 入力データ
        win: ウィンドウサイズ

    Returns:
        np.ndarray: 計算されたハースト指数の配列
    """
    n = len(data)
    res = np.full(n, 0.5)
    for i in prange(win - 1, n):
        chunk = data[i - win + 1 : i + 1]
        incs = chunk[1:] - chunk[:-1]

        # Standard deviation
        mean_inc = 0.0
        for val in incs:
            mean_inc += val
        mean_inc /= len(incs)

        sq_diff_sum = 0.0
        for val in incs:
            sq_diff_sum += (val - mean_inc) ** 2
        s = np.sqrt(sq_diff_sum / len(incs))

        if s < 1e-12:
            res[i] = 0.5
            continue

        # Range
        centered_sum = 0.0
        max_z = 0.0
        min_z = 0.0
        for val in incs:
            centered_sum += val - mean_inc
            if centered_sum > max_z:
                max_z = centered_sum
            if centered_sum < min_z:
                min_z = centered_sum

        r = max_z - min_z
        rs = r / s

        nn = len(incs)
        if rs > 0 and nn > 1:
            h = np.log(rs) / np.log(nn / 2.0)
            if h < 0:
                h = 0.0
            if h > 1:
                h = 1.0
            res[i] = h
    return res


@njit(cache=True)  # type: ignore[untyped-decorator]
def _njit_count_matches(data: np.ndarray, m_len: int, threshold: float) -> int:
    """
    サンプル・エントロピー計算のためのパターン一致数をカウントする Numba 加速関数。

    Args:
        data: 入力データ
        m_len: パターンの長さ
        threshold: 一致判定の閾値

    Returns:
        int: 一致したパターン数
    """
    n = len(data)
    count = 0
    for i in range(n - m_len):
        for j in range(i + 1, n - m_len):
            match = True
            for k in range(m_len):
                if abs(data[i + k] - data[j + k]) > threshold:
                    match = False
                    break
            if match:
                count += 1
    return count


@njit(parallel=True, cache=True)  # type: ignore[untyped-decorator]
def _njit_sample_entropy_loop(
    data: np.ndarray, win: int, m_val: int, r_val: float
) -> np.ndarray:
    """
    サンプル・エントロピーを計算する Numba 加速ループ。

    Args:
        data: 入力データ
        win: ウィンドウサイズ
        m_val: パターンの長さ
        r_val: 閾値係数

    Returns:
        np.ndarray: 計算されたサンプル・エントロピーの配列
    """
    n = len(data)
    res = np.zeros(n)
    for i in prange(win - 1, n):
        chunk = data[i - win + 1 : i + 1]

        # Std dev
        m_c = 0.0
        for val in chunk:
            m_c += val
        m_c /= len(chunk)

        v_c = 0.0
        for val in chunk:
            v_c += (val - m_c) ** 2
        std = np.sqrt(v_c / len(chunk))

        if std == 0:
            res[i] = 0.0
            continue

        thresh = r_val * std
        # _njit_count_matches is called sequentially within prange for the chunk
        a_count = _njit_count_matches(chunk, m_val + 1, thresh)
        b_count = _njit_count_matches(chunk, m_val, thresh)

        if b_count == 0:
            # m次元でもペアが無い: 未知数 0/0 → 0.0 (従来挙動を維持)
            res[i] = 0.0
        elif a_count > 0:
            res[i] = -np.log(a_count / b_count)
        else:
            # m次元は一致するが(m+1)次元に一致しない = 完全に規則的
            # サンプルエントロピーの理論値は +∞
            res[i] = np.inf
    return res


@njit(parallel=True, cache=True)  # type: ignore[untyped-decorator]
def _njit_katz_loop(data: np.ndarray, win: int) -> np.ndarray:
    """
    Katz フラクタル次元を計算する Numba 加速ループ。

    Args:
        data: 入力データ
        win: ウィンドウサイズ

    Returns:
        np.ndarray: 計算されたフラクタル次元の配列
    """
    n_obs = len(data)
    res = np.full(n_obs, 1.0)
    for i in prange(win - 1, n_obs):
        x = data[i - win + 1 : i + 1]
        n = len(x) - 1
        if n <= 0:
            continue

        l_val = 0.0
        for j in range(n):
            l_val += abs(x[j + 1] - x[j])

        max_dist = 0.0
        x0 = x[0]
        for val in x:
            dist = abs(val - x0)
            if dist > max_dist:
                max_dist = dist

        if l_val > 0 and max_dist > 0:
            fd = np.log10(n) / (np.log10(max_dist / l_val) + np.log10(n))
            if fd < 1.0:
                fd = 1.0
            if fd > 2.0:
                fd = 2.0
            res[i] = fd
    return res


class AdvancedFeatures:
    """
    分数差分、清算カスケードなどを含む高度な特徴量。
    """

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
        volume_threshold_quantile: float = 0.2,
    ) -> pd.Series:
        """
        流動性真空検知器 (Void Oscillator)

        出来高が薄い中での価格急変動（真空地帯）を検知。
        大きな値は、薄商いの中での急騰・急落（ダマシの可能性大）を示す。
        """

        def _calculate_void_oscillator() -> pd.Series:
            """
            Void Oscillator の計算ロジック。
            """
            ret = (
                cast(pd.Series, np.log(close / close.shift(1)))
                .fillna(0.0)
                .astype(float)
            )
            z_ret = AdvancedFeatures.z_score(ret, window)
            vol_threshold = volume.rolling(window=window).quantile(
                volume_threshold_quantile
            )
            is_low_volume = (volume < vol_threshold).astype(int)
            return z_ret * is_low_volume

        return cast(
            pd.Series,
            run_multi_series_indicator(
                {"close": close, "volume": volume},
                window,
                _calculate_void_oscillator,
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def crypto_leverage_index(
        open_interest: pd.Series,
        funding_rate: pd.Series,
        ls_ratio_divergence: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """
        Crypto Leverage Index (CLI)

        市場全体のレバレッジ過熱感を示す複合インデックス。
        Norm(OI Z-Score) + Norm(|FR| * sign(FR)) + Norm(L/S Divergence)
        """
        return cast(
            pd.Series,
            run_multi_series_indicator(
                {
                    "open_interest": open_interest,
                    "funding_rate": funding_rate,
                    "ls_ratio_divergence": ls_ratio_divergence,
                },
                window,
                lambda: (
                    AdvancedFeatures.z_score(open_interest, window)
                    + AdvancedFeatures.z_score(funding_rate, window)
                    + AdvancedFeatures.z_score(ls_ratio_divergence - 1.0, window)
                ),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def triplet_imbalance(
        high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """
        Triplet Imbalance (価格構造の不均衡)

        (Max - Mid) / (Mid - Min)
        ここでは (High - Close) / (Close - Low) として実装し、
        上ヒゲと下ヒゲのバランス（売り圧力 vs 買い圧力）を評価する。
        値が大きいほど売り圧力（上ヒゲ）が強い。
        """

        def _calculate_triplet_imbalance() -> pd.Series:
            """
            Triplet Imbalance の計算ロジック。
            """
            upper_shadow = high - close
            lower_shadow = close - low
            imbalance = upper_shadow / (lower_shadow + 1e-9)
            # imbalanceが負になる異常データに備えてクリップ
            safe_imbalance = np.clip(imbalance, a_min=0.0, a_max=None)
            return pd.Series(np.log(safe_imbalance + 1e-9), index=close.index).fillna(
                0.0
            )

        return cast(
            pd.Series,
            run_multi_series_indicator(
                {"high": high, "low": low, "close": close},
                None,
                _calculate_triplet_imbalance,
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def volume_divergence_fakeout(
        close: pd.Series, volume: pd.Series, window: int = 20
    ) -> pd.Series:
        """
        Volume Divergence Fakeout (出来高ダイバージェンス)

        価格が過去N期間の高値を更新したにもかかわらず、
        出来高が平均以下である場合、ダマシ（Fakeout）の可能性が高い。

        Returns:
            Fakeout Score (1.0 = 強いダマシシグナル, 0.0 = 正常)
        """

        def _calculate_volume_divergence_fakeout() -> pd.Series:
            """
            Volume Divergence Fakeout の計算ロジック。
            """
            rolling_max = close.rolling(window=window).max().shift(1)
            vol_ma = volume.rolling(window=window).mean()

            is_new_high = close > rolling_max
            is_low_volume = volume < vol_ma

            price_gain = (close - rolling_max) / rolling_max
            vol_drop = (vol_ma - volume) / vol_ma

            fakeout_score = np.where(
                is_new_high & is_low_volume,
                price_gain * vol_drop * 100,
                0.0,
            )
            return pd.Series(fakeout_score, index=close.index).fillna(0.0)

        return cast(
            pd.Series,
            run_multi_series_indicator(
                {"close": close, "volume": volume},
                window,
                _calculate_volume_divergence_fakeout,
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def hurst_exponent(close: pd.Series, window: int = 100) -> pd.Series:
        """
        ハースト指数 (Hurst Exponent)

        時系列の長期記憶性を測定。
        H < 0.5: 平均回帰的（レンジ）
        H = 0.5: ランダムウォーク
        H > 0.5: トレンド持続的

        Numbaによる高速化を試みる。
        """
        return cast(
            pd.Series,
            run_series_indicator(
                close,
                window,
                lambda: pd.Series(
                    _njit_hurst_loop(close.values.astype(np.float64), window),
                    index=close.index,
                ).fillna(0.5),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def sample_entropy(
        close: pd.Series, window: int = 50, m: int = 2, r: float = 0.2
    ) -> pd.Series:
        """
        サンプル・エントロピー (Sample Entropy)

        時系列の複雑さと規則性を測定。低いほど規則的（トレンドの可能性）、高いほどランダム。
        Numbaによる劇的な高速化。
        """
        return cast(
            pd.Series,
            run_series_indicator(
                close,
                window,
                lambda: pd.Series(
                    _njit_sample_entropy_loop(
                        close.values.astype(np.float64), window, m, r
                    ),
                    index=close.index,
                ).fillna(0.0),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def fractal_dimension(close: pd.Series, window: int = 30) -> pd.Series:
        """
        Katzのフラクタル次元 (Fractal Dimension)

        チャートの「粗さ」を測定。1.0（直線）〜 2.0（平面を埋め尽くすほど複雑）。
        トレンドが発生すると次元が低下する傾向がある。
        """
        return cast(
            pd.Series,
            run_series_indicator(
                close,
                window,
                lambda: pd.Series(
                    _njit_katz_loop(close.values.astype(np.float64), window),
                    index=close.index,
                ).fillna(1.0),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def vpin_approximation(
        close: pd.Series, volume: pd.Series, window: int = 20
    ) -> pd.Series:
        """
        近似VPIN (Volume-Induced Probability of Informed Trading)

        出来高の不均衡から「情報に基づいた取引」の確率を推定。
        高いほど需給が偏っており、急変動の前兆となる。
        """

        def _calculate_vpin() -> pd.Series:
            """
            VPIN 近似の計算ロジック。
            """
            delta_p = close.diff().fillna(0)
            vol_std = delta_p.rolling(window=window).std() + 1e-9
            z_score = delta_p / vol_std
            buy_ratio = 1 / (1 + np.exp(-z_score))
            buy_vol = volume * buy_ratio
            sell_vol = volume * (1 - buy_ratio)
            abs_imbalance = pd.Series(np.abs(buy_vol - sell_vol), index=close.index)
            result = abs_imbalance.rolling(window=window).sum() / (
                volume.rolling(window=window).sum() + 1e-9
            )
            return pd.Series(
                np.where(np.isnan(result), 0.0, result),
                index=close.index,
                dtype=float,
            )

        return cast(
            pd.Series,
            run_multi_series_indicator(
                {"close": close, "volume": volume},
                window,
                _calculate_vpin,
            ),
        )

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

        def _calculate_liquidation_cascade_score() -> pd.Series:
            """
            清算カスケードスコアの計算ロジック。
            """
            result = -np.sign(close.diff()) * open_interest.diff() * volume
            return pd.Series(result, index=close.index).fillna(0.0)

        return cast(
            pd.Series,
            run_multi_series_indicator(
                {
                    "close": close,
                    "open_interest": open_interest,
                    "volume": volume,
                },
                None,
                _calculate_liquidation_cascade_score,
            ),
        )

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

        def _calculate_squeeze_probability() -> pd.Series:
            """
            スクイーズ確率の計算ロジック。
            """
            neg_fr_factor = pd.Series(
                np.where(funding_rate < 0, funding_rate.abs(), 0.0),
                index=close.index,
                dtype=float,
            )
            delta_oi_factor = open_interest.diff().clip(lower=0.0).astype(float)
            price_location = (close - low.rolling(window=lookback).min()).astype(float)
            result = neg_fr_factor * delta_oi_factor * price_location
            return pd.Series(
                np.where(np.isnan(result), 0.0, result),
                index=close.index,
                dtype=float,
            )

        return cast(
            pd.Series,
            run_multi_series_indicator(
                {
                    "close": close,
                    "funding_rate": funding_rate,
                    "open_interest": open_interest,
                    "low": low,
                },
                lookback,
                _calculate_squeeze_probability,
            ),
        )

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

        def _calculate_trend_quality() -> pd.Series:
            """
            トレンド品質の計算ロジック。
            """
            result = close.diff().rolling(window=window).corr(open_interest.diff())
            if isinstance(result, pd.DataFrame):
                return result.iloc[:, 0].fillna(0.0)
            return result.fillna(0.0)

        return cast(
            pd.Series,
            run_multi_series_indicator(
                {"close": close, "open_interest": open_interest},
                window,
                _calculate_trend_quality,
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def oi_weighted_funding_rate(
        funding_rate: pd.Series,
        open_interest: pd.Series,
    ) -> pd.Series:
        """
        OI加重ファンディングレート
        """
        return cast(
            pd.Series,
            run_multi_series_indicator(
                {"funding_rate": funding_rate, "open_interest": open_interest},
                None,
                lambda: funding_rate * open_interest,
            ),
        )

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

        def _calculate_regime_quadrant() -> pd.Series:
            """
            価格とOIの変化に基づく4象限レジーム分析の計算ロジック。
            """
            delta_p = close.diff().fillna(0.0).astype(float)
            delta_oi = open_interest.diff().fillna(0.0).astype(float)

            return pd.Series(
                np.select(
                    [
                        (delta_p > 0) & (delta_oi > 0),
                        (delta_p > 0) & (delta_oi < 0),
                        (delta_p < 0) & (delta_oi > 0),
                        (delta_p < 0) & (delta_oi < 0),
                    ],
                    [0, 1, 2, 3],
                    default=-1,
                ),
                index=close.index,
                dtype=int,
            )

        return cast(
            pd.Series,
            run_multi_series_indicator(
                {"close": close, "open_interest": open_interest},
                None,
                _calculate_regime_quadrant,
            ),
        )

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
        return cast(
            pd.Series,
            run_multi_series_indicator(
                {
                    "ls_ratio_positions": ls_ratio_positions,
                    "ls_ratio_accounts": ls_ratio_accounts,
                },
                None,
                lambda: normalize_non_finite(
                    ls_ratio_positions / ls_ratio_accounts,
                    fill_value=1.0,
                ),
            ),
        )

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
        return cast(
            pd.Series,
            run_multi_series_indicator(
                {"close": close, "open_interest": open_interest},
                None,
                lambda: (
                    np.sign(close.diff().fillna(0.0).astype(float))
                    * open_interest.diff().fillna(0.0).astype(float)
                ),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def long_short_ratio_zscore(
        long_short_ratio: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """
        Long/Short Ratio Z-Score

        ロング/ショート比率の正規化された乖離度を算出します。
        比率が1.0（均衡）からどの程度乖離しているかを、
        直近window期間の平均・標準偏差で標準化します。

        正の大きな値: ロングが過熱（買い戻しの反転リスク）
        負の大きな値: ショートが過熱（ショートカバーの反転リスク）
        """
        return cast(
            pd.Series,
            run_series_indicator(
                long_short_ratio,
                window,
                lambda: AdvancedFeatures.z_score(long_short_ratio, window),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def oi_roc(open_interest: pd.Series, period: int = 20) -> pd.Series:
        """
        Open Interest 変化率（Rate of Change）

        建玉の増減率をそのまま扱う。価格と独立した資金流入・流出シグナル。
        正: 建玉増加（ポジション積み上がり）
        負: 建玉減少（決済・巻き戻し）
        """
        return cast(
            pd.Series,
            run_series_indicator(
                open_interest,
                period,
                lambda: normalize_non_finite(
                    open_interest.pct_change(period).astype(float), fill_value=0.0
                ),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def funding_rate_level(funding_rate: pd.Series) -> pd.Series:
        """
        Funding Rate 水準（そのまま）

        ファンディングレートの生の値を返す。
        正: ロングがショートに支払い（ロング過熱・弱気の逆張り候補）
        負: ショートがロングに支払い（ショート過熱・強気の逆張り候補）
        """
        return cast(
            pd.Series,
            run_series_indicator(
                funding_rate,
                None,
                lambda: normalize_non_finite(
                    funding_rate.astype(float), fill_value=0.0
                ),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def long_short_ratio_level(long_short_ratio: pd.Series) -> pd.Series:
        """
        Long/Short Ratio 水準（そのまま）

        ロング/ショート比率の生の値を返す。
        1.0 超: ロング優勢、1.0 未満: ショート優勢
        """
        return cast(
            pd.Series,
            run_series_indicator(
                long_short_ratio,
                None,
                lambda: normalize_non_finite(
                    long_short_ratio.astype(float), fill_value=1.0
                ),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def lsr_roc(long_short_ratio: pd.Series, period: int = 20) -> pd.Series:
        """
        Long/Short Ratio 変化率（Rate of Change）

        センチメントの急変を捉える。
        正: ロング方向への急傾き、負: ショート方向への急傾き
        """
        return cast(
            pd.Series,
            run_series_indicator(
                long_short_ratio,
                period,
                lambda: normalize_non_finite(
                    long_short_ratio.pct_change(period).astype(float), fill_value=0.0
                ),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def liquidity_efficiency(open_interest: pd.Series, volume: pd.Series) -> pd.Series:
        """
        流動性効率（Open Interest / Volume）
        """
        return cast(
            pd.Series,
            run_multi_series_indicator(
                {"open_interest": open_interest, "volume": volume},
                None,
                lambda: normalize_non_finite(open_interest / volume, fill_value=0),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def leverage_ratio(open_interest: pd.Series, market_cap: pd.Series) -> pd.Series:
        """
        推定レバレッジ比率（Total OI / Estimated Market Cap）
        """
        return cast(
            pd.Series,
            run_multi_series_indicator(
                {"open_interest": open_interest, "market_cap": market_cap},
                None,
                lambda: normalize_non_finite(open_interest / market_cap, fill_value=0),
            ),
        )
