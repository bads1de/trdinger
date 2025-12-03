"""
Volume Profile Feature Calculator

価格レベル別の出来高分布を分析し、市場構造を捉える特徴量を生成します。
学術的に検証された強力な特徴量（Kaggle/論文で実証済み）。
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class VolumeProfileFeatureCalculator:
    """
    Volume Profile（価格レベル別出来高）ベースの特徴量計算クラス

    市場参加者の「合意価格」や重要な価格レベルを特定し、
    現在価格との関係から予測力のある特徴量を生成します。
    """

    def __init__(self, lookback_period: int = 50, num_bins: int = 20):
        """
        Args:
            lookback_period: Volume Profile計算に使用する過去期間
            num_bins: 価格を分割するビン数（解像度）
        """
        self.lookback_period = lookback_period
        self.num_bins = num_bins

    def calculate_features(
        self, df: pd.DataFrame, lookback_periods: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Volume Profile特徴量を計算

        Args:
            df: OHLCV DataFrame
            lookback_periods: 複数期間でのVolume Profile計算（Noneの場合はデフォルト使用）

        Returns:
            Volume Profile特徴量を含むDataFrame
        """
        result = pd.DataFrame(index=df.index)

        if lookback_periods is None:
            lookback_periods = [self.lookback_period, 100, 200]

        for period in lookback_periods:
            # Volume Profile計算
            poc, vah, val = self._calculate_volume_profile_rolling(df, window=period)

            # 現在価格からの距離（%）
            current_price = df["close"]

            result[f"POC_Distance_{period}"] = (current_price - poc) / poc
            result[f"VAH_Distance_{period}"] = (current_price - vah) / vah
            result[f"VAL_Distance_{period}"] = (current_price - val) / val

            # Value Area内にいるか（バイナリ）
            result[f"In_Value_Area_{period}"] = (
                (current_price >= val) & (current_price <= vah)
            ).astype(float)

            # Value Area幅（ボラティリティ代理）
            result[f"Value_Area_Width_{period}"] = (vah - val) / poc

        # HVN/LVN（高/低出来高ノード）検出
        hvn_distance, lvn_distance = self._detect_volume_nodes(df)
        result["HVN_Distance"] = hvn_distance
        result["LVN_Distance"] = lvn_distance

        # Volume Profile形状特徴
        result["VP_Skewness"] = self._calculate_vp_skewness(df)
        result["VP_Kurtosis"] = self._calculate_vp_kurtosis(df)

        return result

    def _calculate_volume_profile_rolling(
        self, df: pd.DataFrame, window: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        ローリングウィンドウでVolume Profileを計算

        Returns:
            (POC, VAH, VAL)のタプル
        """
        poc_series = pd.Series(index=df.index, dtype=float)
        vah_series = pd.Series(index=df.index, dtype=float)
        val_series = pd.Series(index=df.index, dtype=float)

        for i in range(window, len(df)):
            window_data = df.iloc[i - window : i]

            # Volume Profile計算
            poc, vah, val = self._compute_volume_profile(
                window_data["high"].values,
                window_data["low"].values,
                window_data["close"].values,
                window_data["volume"].values,
                num_bins=self.num_bins,
            )

            poc_series.iloc[i] = poc
            vah_series.iloc[i] = vah
            val_series.iloc[i] = val

        # 欠損値を前方埋め
        poc_series = poc_series.fillna(method="ffill")
        vah_series = vah_series.fillna(method="ffill")
        val_series = val_series.fillna(method="ffill")

        return poc_series, vah_series, val_series

    def _compute_volume_profile(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        num_bins: int = 20,
    ) -> Tuple[float, float, float]:
        """
        単一ウィンドウのVolume Profileを計算

        Returns:
            (POC, VAH, VAL)
        """
        # 価格範囲を取得
        price_min = low.min()
        price_max = high.max()

        if price_min == price_max:
            # 価格変動がない場合
            return close[-1], close[-1], close[-1]

        # 価格をビンに分割
        bins = np.linspace(price_min, price_max, num_bins + 1)
        bin_volume = np.zeros(num_bins)

        # 各バーの出来高を価格ビンに分配
        for i in range(len(high)):
            # 各バーが複数のビンにまたがる可能性を考慮
            bar_low = low[i]
            bar_high = high[i]
            bar_vol = volume[i]

            # このバーが影響するビンを特定
            affected_bins = np.where((bins[:-1] <= bar_high) & (bins[1:] >= bar_low))[0]

            if len(affected_bins) > 0:
                # 出来高を影響するビンに均等分配
                bin_volume[affected_bins] += bar_vol / len(affected_bins)

        # POC: 最大出来高のビン
        poc_bin = np.argmax(bin_volume)
        poc = (bins[poc_bin] + bins[poc_bin + 1]) / 2

        # Value Area: 総出来高の70%を含む価格範囲
        total_volume = bin_volume.sum()
        target_volume = total_volume * 0.70

        # POCから上下に拡張してValue Areaを決定
        vah_bin = poc_bin
        val_bin = poc_bin
        accumulated_volume = bin_volume[poc_bin]

        while accumulated_volume < target_volume:
            # 上下のどちらが出来高が多いかを判断
            vol_above = bin_volume[vah_bin + 1] if vah_bin + 1 < num_bins else 0
            vol_below = bin_volume[val_bin - 1] if val_bin > 0 else 0

            if vol_above > vol_below and vah_bin + 1 < num_bins:
                vah_bin += 1
                accumulated_volume += bin_volume[vah_bin]
            elif val_bin > 0:
                val_bin -= 1
                accumulated_volume += bin_volume[val_bin]
            else:
                break

        vah = bins[vah_bin + 1]  # 上限
        val = bins[val_bin]  # 下限

        return poc, vah, val

    def _detect_volume_nodes(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        HVN（高出来高ノード）とLVN（低出来高ノード）を検出

        Returns:
            (HVN距離, LVN距離)
        """
        window = self.lookback_period
        current_price = df["close"]

        hvn_distance = pd.Series(index=df.index, dtype=float)
        lvn_distance = pd.Series(index=df.index, dtype=float)

        for i in range(window, len(df)):
            window_data = df.iloc[i - window : i]

            # Volume Profile計算
            high_arr = window_data["high"].values
            low_arr = window_data["low"].values
            close_arr = window_data["close"].values
            volume_arr = window_data["volume"].values

            price_min = low_arr.min()
            price_max = high_arr.max()

            if price_min == price_max:
                continue

            bins = np.linspace(price_min, price_max, self.num_bins + 1)
            bin_volume = np.zeros(self.num_bins)

            for j in range(len(high_arr)):
                bar_low = low_arr[j]
                bar_high = high_arr[j]
                bar_vol = volume_arr[j]

                affected_bins = np.where(
                    (bins[:-1] <= bar_high) & (bins[1:] >= bar_low)
                )[0]

                if len(affected_bins) > 0:
                    bin_volume[affected_bins] += bar_vol / len(affected_bins)

            # HVN: 上位25%の出来高ビン
            hvn_threshold = np.percentile(bin_volume, 75)
            hvn_bins = np.where(bin_volume >= hvn_threshold)[0]

            # LVN: 下位25%の出来高ビン
            lvn_threshold = np.percentile(bin_volume, 25)
            lvn_bins = np.where(bin_volume <= lvn_threshold)[0]

            # 現在価格に最も近いHVN/LVNまでの距離
            current = current_price.iloc[i]

            if len(hvn_bins) > 0:
                hvn_prices = (bins[hvn_bins] + bins[hvn_bins + 1]) / 2
                nearest_hvn = hvn_prices[np.argmin(np.abs(hvn_prices - current))]
                hvn_distance.iloc[i] = (current - nearest_hvn) / current

            if len(lvn_bins) > 0:
                lvn_prices = (bins[lvn_bins] + bins[lvn_bins + 1]) / 2
                nearest_lvn = lvn_prices[np.argmin(np.abs(lvn_prices - current))]
                lvn_distance.iloc[i] = (current - nearest_lvn) / current

        return hvn_distance.fillna(0), lvn_distance.fillna(0)

    def _calculate_vp_skewness(self, df: pd.DataFrame) -> pd.Series:
        """Volume Profile分布の歪度を計算"""
        window = self.lookback_period
        skewness = pd.Series(index=df.index, dtype=float)

        for i in range(window, len(df)):
            window_data = df.iloc[i - window : i]

            # 簡易的な歪度計算（価格の出来高加重平均との偏り）
            prices = window_data["close"].values
            volumes = window_data["volume"].values

            weighted_mean = np.average(prices, weights=volumes)
            median_price = np.median(prices)

            # 歪度の代理指標
            skew = (weighted_mean - median_price) / window_data["close"].std()
            skewness.iloc[i] = skew

        return skewness.fillna(0)

    def _calculate_vp_kurtosis(self, df: pd.DataFrame) -> pd.Series:
        """Volume Profile分布の尖度を計算"""
        window = self.lookback_period
        kurtosis = pd.Series(index=df.index, dtype=float)

        for i in range(window, len(df)):
            window_data = df.iloc[i - window : i]

            # 価格の出来高加重分散
            prices = window_data["close"].values
            volumes = window_data["volume"].values

            weighted_mean = np.average(prices, weights=volumes)
            weighted_var = np.average((prices - weighted_mean) ** 2, weights=volumes)
            weighted_std = np.sqrt(weighted_var)

            if weighted_std > 0:
                # 4次モーメント
                fourth_moment = np.average(
                    (prices - weighted_mean) ** 4, weights=volumes
                )
                kurt = fourth_moment / (weighted_std**4) - 3  # Excess kurtosis
                kurtosis.iloc[i] = kurt

        return kurtosis.fillna(0)
