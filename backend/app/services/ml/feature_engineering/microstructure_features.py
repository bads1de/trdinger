import numpy as np
import pandas as pd


class MicrostructureFeatureCalculator:
    """
    市場の微細構造（Microstructure）に関する特徴量を計算するクラス。
    需給の不均衡、流動性、スプレッドなどを推定し、ダマシ検知に役立てる。
    """

    def calculate_features(
        self, ohlcv: pd.DataFrame, window_short: int = 20, window_long: int = 50
    ) -> pd.DataFrame:
        """
        全ての特徴量を計算して結合したDataFrameを返す。
        """
        df = pd.DataFrame(index=ohlcv.index)

        # 1. Amihud Illiquidity (流動性枯渇度)
        df["Amihud_Illiquidity"] = self.calculate_amihud_illiquidity(
            ohlcv, window=window_short
        )

        # 2. Roll Measure (実効スプレッド推定)
        df["Roll_Measure"] = self.calculate_roll_measure(ohlcv, window=window_short)

        # 3. VPIN Proxy (出来高不均衡)
        df["VPIN_Proxy"] = self.calculate_vpin_proxy(ohlcv, window=window_short)

        # 4. Kyle's Lambda (マーケットインパクト)
        df["Kyles_Lambda"] = self.calculate_kyles_lambda(ohlcv, window=window_short)

        # 5. Volume Variance (出来高の分散 - 情報の不確実性)
        df["Volume_CV"] = ohlcv["volume"].rolling(window=window_short).std() / (
            ohlcv["volume"].rolling(window=window_short).mean() + 1e-9
        )

        # 6. Corwin-Schultz Spread (High-Lowベースのスプレッド推定)
        df["Corwin_Schultz_Spread"] = self.calculate_corwin_schultz_spread(
            ohlcv, window=window_short
        )

        return df

    def calculate_amihud_illiquidity(
        self, df: pd.DataFrame, window: int = 20
    ) -> pd.Series:
        """
        Amihud Illiquidity Ratio: |Return| / (Volume * Price)
        値が大きいほど流動性が低く、価格が飛びやすい状態を示す。
        """
        returns = df["close"].pct_change().abs()
        # Dollar Volume (出来高 * 価格) を使用するのが一般的
        dollar_volume = df["volume"] * df["close"]

        illiq = returns / (dollar_volume + 1e-9)  # ゼロ除算防止

        # 移動平均で平滑化
        return illiq.rolling(window=window).mean()

    def calculate_roll_measure(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Roll Measure: 実効スプレッドの推定値。
        Roll = 2 * sqrt(-Cov(Delta P_t, Delta P_{t-1}))
        負の自己相関が強いほど、スプレッド（ノイズ）が大きいことを示唆する。
        """
        delta_p = df["close"].diff()

        # 自己共分散を計算: Cov(x_t, x_{t-1})
        # rolling().cov() は2つのSeries間の共分散だが、ここでは自己共分散が必要
        # shiftして計算する
        cov = delta_p.rolling(window=window).cov(delta_p.shift(1))

        # 共分散が正の場合は0とする（Rollモデルの仮定外）
        cov = cov.fillna(0)
        neg_cov = np.where(cov < 0, -cov, 0)

        roll = 2 * np.sqrt(neg_cov)
        return pd.Series(roll, index=df.index)

    def calculate_vpin_proxy(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """VPINの簡易プロキシ"""
        # 出来高の割り当て
        buy_vol = np.where(df["close"] > df["open"], df["volume"], 
                  np.where(df["close"] == df["open"], df["volume"] * 0.5, 0.0))
        sell_vol = np.where(df["close"] < df["open"], df["volume"], 
                   np.where(df["close"] == df["open"], df["volume"] * 0.5, 0.0))

        buy_sum = pd.Series(buy_vol, index=df.index).rolling(window=window).sum()
        sell_sum = pd.Series(sell_vol, index=df.index).rolling(window=window).sum()
        
        return (buy_sum - sell_sum).abs() / (buy_sum + sell_sum + 1e-9)

    def calculate_kyles_lambda(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Kyle's Lambda: マーケットインパクトの推定値。
        価格変化とOrder Flow（符号付き出来高）の回帰係数として推定される。
        Lambda = Cov(Return, SignedVolume) / Var(SignedVolume)
        """
        returns = df["close"].pct_change()

        # Signed Volume (近似符号付き出来高)
        # Close > Open なら +Volume, Close < Open なら -Volume
        sign = np.sign(df["close"] - df["open"])
        # signが0の場合は前回のsignを使うか、0のままにする。ここでは0のまま。
        signed_volume = df["volume"] * sign

        # ローリング共分散と分散
        cov = returns.rolling(window=window).cov(signed_volume)
        var = signed_volume.rolling(window=window).var()

        lambda_val = cov / (var + 1e-9)

        return lambda_val

    def calculate_corwin_schultz_spread(
        self, df: pd.DataFrame, window: int = 20
    ) -> pd.Series:
        """
        Corwin-Schultz Spread Estimator
        High-Low価格比を用いてスプレッドを推定する強力な手法。
        """
        high = df["high"]
        low = df["low"]

        # ベータの計算
        # beta = E[sum(ln(H/L)^2)]
        hl_ratio = np.log(high / low) ** 2
        beta = hl_ratio.rolling(window=2).sum()

        # ガンマの計算
        # gamma = [ln(H2/L2)]^2 ここで H2, L2 は2日間の高値/安値
        h2 = high.rolling(window=2).max()
        l2 = low.rolling(window=2).min()
        gamma = np.log(h2 / l2) ** 2

        # アルファの計算
        # alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
        # 簡略化式: alpha = (sqrt(2*beta) - sqrt(beta)) / 0.17157 - sqrt(gamma) / 0.4142

        const1 = np.sqrt(2) - 1
        const2 = 3 - 2 * np.sqrt(2)

        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / const1 - np.sqrt(gamma / const2)

        # スプレッドの計算
        # S = 2 * (exp(alpha) - 1) / (1 + exp(alpha))
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

        # 負の値や異常値を処理
        spread = spread.replace([np.inf, -np.inf], np.nan).fillna(0)
        spread = np.where(spread < 0, 0, spread)

        # 指定ウィンドウで平滑化
        return pd.Series(spread, index=df.index).rolling(window=window).mean()



