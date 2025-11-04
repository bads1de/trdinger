"""
高度な特徴量エンジニアリング

精度向上のための高度な技術指標とラグ特徴量を実装します。
現在の40.55%から60%以上への精度向上を目指します。
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """高度な特徴量エンジニアリングクラス"""

    def __init__(self):
        """初期化"""
        self.scaler = StandardScaler()

    # 旧API互換（create_features -> create_advanced_features）
    def create_features(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        return self.create_advanced_features(ohlcv_data)

    def create_advanced_features(
        self,
        ohlcv_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        高度な特徴量を生成

        Args:
            ohlcv_data: OHLCVデータ
            funding_rate_data: ファンディングレートデータ
            open_interest_data: 建玉残高データ

        Returns:
            高度な特徴量を含むDataFrame
        """
        logger.info("🚀 高度な特徴量エンジニアリング開始")

        features = ohlcv_data.copy()

        # 1. ラグ特徴量
        features = self._add_lag_features(features)

        # 2. 高度な技術指標
        features = self._add_advanced_technical_indicators(features)

        # 3. 統計的特徴量
        features = self._add_statistical_features(features)

        # 4. 時系列特徴量
        features = self._add_time_series_features(features)

        # 5. ボラティリティ特徴量
        features = self._add_volatility_features(features)

        # 6. 外部データ特徴量
        if funding_rate_data is not None:
            features = self._add_funding_rate_features(features, funding_rate_data)

        if open_interest_data is not None:
            features = self._add_open_interest_features(features, open_interest_data)

        # 7. 相互作用特徴量
        features = self._add_interaction_features(features)

        # 8. 季節性特徴量
        features = self._add_seasonal_features(features)

        logger.info(f"✅ 高度な特徴量生成完了: {features.shape[1]}個の特徴量")

        return features

    def _add_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """ラグ特徴量を追加（最適化版：期間削減）"""
        logger.info("📊 ラグ特徴量を追加中...")

        new_features = {}

        # 価格のラグ特徴量（期間を削減: 6→3期間）
        lag_periods = [1, 6, 24]  # 1h, 6h, 24h

        for period in lag_periods:
            new_features[f"close_lag_{period}"] = data["close"].shift(period)
            new_features[f"volume_lag_{period}"] = data["volume"].shift(period)

        # 価格変化率のラグ
        new_features["returns"] = data["close"].pct_change()
        for period in lag_periods:
            new_features[f"returns_lag_{period}"] = new_features["returns"].shift(period)

        # 累積リターン（主要期間のみ）
        for period in [6, 24]:
            new_features[f"cumulative_returns_{period}"] = new_features["returns"].rolling(period).sum()

        # 一括で結合
        new_df = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
        return new_df

    def _add_advanced_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """高度な技術指標を追加"""
        logger.info("📈 高度な技術指標を追加中...")

        new_features = {}

        data["high"].values
        data["low"].values
        data["close"].values
        data["volume"].values

        try:
            import pandas_ta as ta

            # モメンタム指標（pandas-ta使用）
            stoch_result = ta.stoch(
                high=data["high"], low=data["low"], close=data["close"]
            )
            if stoch_result is not None and not stoch_result.empty:
                new_features["stochastic_k"] = stoch_result.iloc[:, 0]  # STOCHk
                new_features["stochastic_d"] = stoch_result.iloc[:, 1]  # STOCHd

            williams_r_result = ta.willr(
                high=data["high"], low=data["low"], close=data["close"]
            )
            if williams_r_result is not None:
                new_features["williams_r"] = williams_r_result

            cci_result = ta.cci(high=data["high"], low=data["low"], close=data["close"])
            if cci_result is not None:
                new_features["cci"] = cci_result

            mfi_result = ta.mfi(
                high=data["high"],
                low=data["low"],
                close=data["close"],
                volume=data["volume"],
            )
            if mfi_result is not None:
                new_features["mfi"] = mfi_result

            uo_result = ta.uo(high=data["high"], low=data["low"], close=data["close"])
            if uo_result is not None:
                new_features["ultimate_oscillator"] = uo_result

            # トレンド指標（pandas-ta使用）
            adx_result = ta.adx(high=data["high"], low=data["low"], close=data["close"])
            if adx_result is not None and not adx_result.empty:
                new_features["ADX"] = adx_result["ADX_14"]
                new_features["DI_Plus"] = adx_result["DMP_14"]
                new_features["DI_Minus"] = adx_result["DMN_14"]

            aroon_result = ta.aroon(high=data["high"], low=data["low"])
            if aroon_result is not None and not aroon_result.empty:
                new_features["Aroon_Up"] = aroon_result["AROONU_14"]
                new_features["Aroon_Down"] = aroon_result["AROOND_14"]

            aroon_osc_result = ta.aroon(high=data["high"], low=data["low"], scalar=100)
            if aroon_osc_result is not None and not aroon_osc_result.empty:
                new_features["AROONOSC"] = aroon_osc_result["AROONOSC_14"]

            # ボラティリティ指標（pandas-ta使用）
            new_features["ATR"] = ta.atr(
                high=data["high"], low=data["low"], close=data["close"]
            )
            new_features["NATR"] = ta.natr(
                high=data["high"], low=data["low"], close=data["close"]
            )
            new_features["TRANGE"] = ta.true_range(
                high=data["high"], low=data["low"], close=data["close"]
            )

            # 出来高指標（pandas-ta使用）
            new_features["OBV"] = ta.obv(close=data["close"], volume=data["volume"])
            new_features["AD"] = ta.ad(
                high=data["high"],
                low=data["low"],
                close=data["close"],
                volume=data["volume"],
            )
            new_features["ADOSC"] = ta.adosc(
                high=data["high"],
                low=data["low"],
                close=data["close"],
                volume=data["volume"],
            )

        except Exception as e:
            logger.warning(f"pandas-ta指標計算エラー: {e}")

        # 一括で結合
        new_df = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
        return new_df

    def _add_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """統計的特徴量を追加"""
        logger.info("📊 統計的特徴量を追加中...")

        new_features = {}
        windows = [5, 10, 20, 50]

        for window in windows:
            # 移動統計
            new_features[f"Close_mean_{window}"] = data["close"].rolling(window).mean()
            new_features[f"Close_std_{window}"] = data["close"].rolling(window).std()
            new_features[f"Close_skew_{window}"] = data["close"].rolling(window).skew()
            new_features[f"Close_kurt_{window}"] = data["close"].rolling(window).kurt()

            # 分位数
            new_features[f"Close_q25_{window}"] = data["close"].rolling(window).quantile(0.25)
            new_features[f"Close_q75_{window}"] = data["close"].rolling(window).quantile(0.75)
            new_features[f"Close_median_{window}"] = data["close"].rolling(window).median()

            # 範囲統計
            high_max = data["high"].rolling(window).max()
            low_min = data["low"].rolling(window).min()
            new_features[f"Close_range_{window}"] = high_max - low_min

            q75 = new_features[f"Close_q75_{window}"]
            q25 = new_features[f"Close_q25_{window}"]
            new_features[f"Close_iqr_{window}"] = q75 - q25

            # 出来高統計
            new_features[f"Volume_mean_{window}"] = data["volume"].rolling(window).mean()
            new_features[f"Volume_std_{window}"] = data["volume"].rolling(window).std()

        # 一括で結合
        new_df = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
        return new_df

    def _add_time_series_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """時系列特徴量を追加"""
        logger.info("⏰ 時系列特徴量を追加中...")

        new_features = {}

        # 差分特徴量
        for period in [1, 6, 24]:
            new_features[f"Close_diff_{period}"] = data["close"].diff(period)
            new_features[f"Volume_diff_{period}"] = data["volume"].diff(period)

        # 変化率
        for period in [1, 6, 12, 24]:
            new_features[f"Close_pct_change_{period}"] = data["close"].pct_change(period)
            new_features[f"Volume_pct_change_{period}"] = data["volume"].pct_change(period)

        # 移動平均からの乖離
        for window in [5, 10, 20]:
            ma = data["close"].rolling(window).mean()
            new_features[f"Close_deviation_from_ma_{window}"] = (data["close"] - ma) / ma

        # トレンド強度（pandas-ta使用）
        for window in [10, 20, 50]:
            new_features[f"Trend_strength_{window}"] = ta.linreg(
                data["close"], length=window, slope=True
            )

        # 一括で結合
        new_df = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
        return new_df

    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """ボラティリティ特徴量を追加"""
        logger.info("📊 ボラティリティ特徴量を追加中...")

        new_features = {}

        # 実現ボラティリティ
        new_features["Returns"] = data["close"].pct_change()

        for window in [5, 10, 20, 50]:
            new_features[f"Realized_Vol_{window}"] = new_features["Returns"].rolling(
                window
            ).std() * np.sqrt(
                24
            )  # 日次換算
            new_features[f"Vol_of_Vol_{window}"] = (
                new_features[f"Realized_Vol_{window}"].rolling(window).std()
            )

        # Parkinson推定量（高値・安値ベースのボラティリティ）
        for window in [10, 20]:
            hl_ratio = np.log(data["high"] / data["low"])
            new_features[f"Parkinson_Vol_{window}"] = hl_ratio.rolling(window).var() * (
                1 / (4 * np.log(2))
            )

        # ボラティリティレジーム
        vol_20 = new_features["Returns"].rolling(20).std()
        new_features["Vol_Regime"] = pd.cut(
            vol_20, bins=3, labels=[0, 1, 2]
        )  # 低・中・高ボラティリティ

        # 一括で結合
        new_df = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
        return new_df

    def _add_funding_rate_features(
        self, data: pd.DataFrame, fr_data: pd.DataFrame
    ) -> pd.DataFrame:
        """ファンディングレート特徴量を追加"""
        logger.info("💰 ファンディングレート特徴量を追加中...")

        new_features = {}

        if "funding_rate" in fr_data.columns:
            # ファンディングレートの統計
            for window in [3, 7, 14]:  # 3回、7回、14回分（24h, 56h, 112h）
                new_features[f"FR_mean_{window}"] = (
                    fr_data["funding_rate"].rolling(window).mean()
                )
                new_features[f"FR_std_{window}"] = fr_data["funding_rate"].rolling(window).std()
                new_features[f"FR_sum_{window}"] = fr_data["funding_rate"].rolling(window).sum()

            # ファンディングレートの変化
            new_features["FR_change"] = fr_data["funding_rate"].diff()
            new_features["FR_change_abs"] = new_features["FR_change"].abs()

            # 極端なファンディングレート
            new_features["FR_extreme_positive"] = (
                fr_data["funding_rate"] > fr_data["funding_rate"].quantile(0.95)
            ).astype(int)
            new_features["FR_extreme_negative"] = (
                fr_data["funding_rate"] < fr_data["funding_rate"].quantile(0.05)
            ).astype(int)

        # 一括で結合
        new_df = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
        return new_df

    def _add_open_interest_features(
        self, data: pd.DataFrame, oi_data: pd.DataFrame
    ) -> pd.DataFrame:
        """建玉残高特徴量を追加"""
        logger.info("📊 建玉残高特徴量を追加中...")

        new_features = {}

        if "open_interest" in oi_data.columns:
            # 建玉残高の変化率
            for period in [1, 6, 24]:
                new_features[f"OI_pct_change_{period}"] = oi_data["open_interest"].pct_change(
                    period
                )

            # 建玉残高の移動平均
            for window in [6, 24, 168]:  # 6h, 24h, 168h(1週間)
                new_features[f"OI_ma_{window}"] = (
                    oi_data["open_interest"].rolling(window).mean()
                )
                new_features[f"OI_deviation_{window}"] = (
                    oi_data["open_interest"] - new_features[f"OI_ma_{window}"]
                ) / new_features[f"OI_ma_{window}"]

            # 建玉残高と価格の関係
            new_features["OI_Price_Correlation"] = (
                oi_data["open_interest"].rolling(24).corr(data["close"])
            )

        # 一括で結合
        new_df = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
        return new_df

    def _add_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """相互作用特徴量を追加"""
        logger.info("🔗 相互作用特徴量を追加中...")

        new_features = {}

        # 価格と出来高の相互作用
        new_features["Price_Volume_Product"] = data["close"] * data["volume"]
        new_features["Price_Volume_Ratio"] = data["close"] / (data["volume"] + 1e-8)

        # ボラティリティと出来高
        if "Realized_Vol_20" in data.columns:
            new_features["Vol_Volume_Product"] = data["Realized_Vol_20"] * data["volume"]

        # 技術指標の組み合わせ
        if "RSI" in data.columns and "Stochastic_K" in data.columns:
            new_features["RSI_Stoch_Avg"] = (data["RSI"] + data["Stochastic_K"]) / 2
            new_features["RSI_Stoch_Diff"] = data["RSI"] - data["Stochastic_K"]

        # 一括で結合
        new_df = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
        return new_df

    def _add_seasonal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """季節性特徴量を追加"""
        logger.info("📅 季節性特徴量を追加中...")

        new_features = {}

        # DatetimeIndexの確認
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning(
                "インデックスがDatetimeIndexではありません。時間関連特徴量をスキップします。"
            )
            return data

        # 時間特徴量（getattrを使用して属性アクセスエラーを回避）
        try:
            hour = getattr(data.index, "hour", 0)  # type: ignore
            dayofweek = getattr(data.index, "dayofweek", 0)  # type: ignore
            day = getattr(data.index, "day", 1)  # type: ignore
            month = getattr(data.index, "month", 1)  # type: ignore

            new_features["Hour"] = hour
            new_features["DayOfWeek"] = dayofweek
            new_features["DayOfMonth"] = day
            new_features["Month"] = month
        except (AttributeError, TypeError) as e:
            logger.warning(f"時間関連特徴量の生成でエラー: {e}")
            new_features["Hour"] = 0
            new_features["DayOfWeek"] = 0
            new_features["DayOfMonth"] = 1
            new_features["Month"] = 1

        # 周期的エンコーディング
        new_features["Hour_sin"] = np.sin(2 * np.pi * new_features["Hour"] / 24)
        new_features["Hour_cos"] = np.cos(2 * np.pi * new_features["Hour"] / 24)
        new_features["DayOfWeek_sin"] = np.sin(2 * np.pi * new_features["DayOfWeek"] / 7)
        new_features["DayOfWeek_cos"] = np.cos(2 * np.pi * new_features["DayOfWeek"] / 7)

        # 市場時間特徴量
        new_features["Is_Weekend"] = (new_features["DayOfWeek"] >= 5).astype(int)  # 土日
        new_features["Is_Asian_Hours"] = ((new_features["Hour"] >= 0) & (new_features["Hour"] < 8)).astype(int)
        new_features["Is_European_Hours"] = ((new_features["Hour"] >= 8) & (new_features["Hour"] < 16)).astype(
            int
        )
        new_features["Is_American_Hours"] = ((new_features["Hour"] >= 16) & (new_features["Hour"] < 24)).astype(
            int
        )

        # 一括で結合
        new_df = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
        return new_df


# グローバルインスタンス
advanced_feature_engineer = AdvancedFeatureEngineer()
