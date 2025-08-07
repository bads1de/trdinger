"""
高度な特徴量エンジニアリング

精度向上のための高度な技術指標とラグ特徴量を実装します。
現在の40.55%から60%以上への精度向上を目指します。
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """高度な特徴量エンジニアリングクラス"""

    def __init__(self):
        """初期化"""
        self.scaler = StandardScaler()

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
        """ラグ特徴量を追加"""
        logger.info("📊 ラグ特徴量を追加中...")

        # 価格のラグ特徴量
        lag_periods = [1, 3, 6, 12, 24, 48]  # 1h, 3h, 6h, 12h, 24h, 48h

        for period in lag_periods:
            data[f"Close_lag_{period}"] = data["Close"].shift(period)
            data[f"Volume_lag_{period}"] = data["Volume"].shift(period)
            data[f"High_lag_{period}"] = data["High"].shift(period)
            data[f"Low_lag_{period}"] = data["Low"].shift(period)

        # 価格変化率のラグ
        data["Returns"] = data["Close"].pct_change()
        for period in lag_periods:
            data[f"Returns_lag_{period}"] = data["Returns"].shift(period)

        # 累積リターン
        for period in [6, 12, 24]:
            data[f"Cumulative_Returns_{period}"] = data["Returns"].rolling(period).sum()

        return data

    def _add_advanced_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """高度な技術指標を追加"""
        logger.info("📈 高度な技術指標を追加中...")

        high = data["High"].values
        low = data["Low"].values
        close = data["Close"].values
        volume = data["Volume"].values

        try:
            # モメンタム指標
            data["Stochastic_K"], data["Stochastic_D"] = talib.STOCH(high, low, close)
            data["Williams_R"] = talib.WILLR(high, low, close)
            data["CCI"] = talib.CCI(high, low, close)
            data["MFI"] = talib.MFI(high, low, close, volume)
            data["Ultimate_Oscillator"] = talib.ULTOSC(high, low, close)

            # トレンド指標
            data["ADX"] = talib.ADX(high, low, close)
            data["ADXR"] = talib.ADXR(high, low, close)
            data["DI_Plus"] = talib.PLUS_DI(high, low, close)
            data["DI_Minus"] = talib.MINUS_DI(high, low, close)
            data["Aroon_Up"], data["Aroon_Down"] = talib.AROON(high, low)
            data["AROONOSC"] = talib.AROONOSC(high, low)

            # ボラティリティ指標
            data["ATR"] = talib.ATR(high, low, close)
            data["NATR"] = talib.NATR(high, low, close)
            data["TRANGE"] = talib.TRANGE(high, low, close)

            # 出来高指標
            data["OBV"] = talib.OBV(close, volume)
            data["AD"] = talib.AD(high, low, close, volume)
            data["ADOSC"] = talib.ADOSC(high, low, close, volume)

            # パターン認識（一部）
            data["DOJI"] = talib.CDLDOJI(data["Open"], high, low, close)
            data["HAMMER"] = talib.CDLHAMMER(data["Open"], high, low, close)
            data["SHOOTING_STAR"] = talib.CDLSHOOTINGSTAR(
                data["Open"], high, low, close
            )

        except Exception as e:
            logger.warning(f"TALib指標計算エラー: {e}")

        return data

    def _add_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """統計的特徴量を追加"""
        logger.info("📊 統計的特徴量を追加中...")

        windows = [5, 10, 20, 50]

        for window in windows:
            # 移動統計
            data[f"Close_mean_{window}"] = data["Close"].rolling(window).mean()
            data[f"Close_std_{window}"] = data["Close"].rolling(window).std()
            data[f"Close_skew_{window}"] = data["Close"].rolling(window).skew()
            data[f"Close_kurt_{window}"] = data["Close"].rolling(window).kurt()

            # 分位数
            data[f"Close_q25_{window}"] = data["Close"].rolling(window).quantile(0.25)
            data[f"Close_q75_{window}"] = data["Close"].rolling(window).quantile(0.75)
            data[f"Close_median_{window}"] = data["Close"].rolling(window).median()

            # 範囲統計
            data[f"Close_range_{window}"] = (
                data["High"].rolling(window).max() - data["Low"].rolling(window).min()
            )
            data[f"Close_iqr_{window}"] = (
                data[f"Close_q75_{window}"] - data[f"Close_q25_{window}"]
            )

            # 出来高統計
            data[f"Volume_mean_{window}"] = data["Volume"].rolling(window).mean()
            data[f"Volume_std_{window}"] = data["Volume"].rolling(window).std()

        return data

    def _add_time_series_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """時系列特徴量を追加"""
        logger.info("⏰ 時系列特徴量を追加中...")

        # 差分特徴量
        for period in [1, 6, 24]:
            data[f"Close_diff_{period}"] = data["Close"].diff(period)
            data[f"Volume_diff_{period}"] = data["Volume"].diff(period)

        # 変化率
        for period in [1, 6, 12, 24]:
            data[f"Close_pct_change_{period}"] = data["Close"].pct_change(period)
            data[f"Volume_pct_change_{period}"] = data["Volume"].pct_change(period)

        # 移動平均からの乖離
        for window in [5, 10, 20]:
            ma = data["Close"].rolling(window).mean()
            data[f"Close_deviation_from_ma_{window}"] = (data["Close"] - ma) / ma

        # トレンド強度（効率的な実装）
        for window in [10, 20, 50]:
            # NumPyのpolyfitを使用してより効率的に線形回帰の傾きを計算
            def calculate_trend_strength(series):
                if len(series) == window and not series.isna().any():
                    x = np.arange(len(series))
                    # polyfit(degree=1)で線形回帰の傾きを直接計算
                    slope = np.polyfit(x, series, 1)[0]
                    return slope
                return np.nan

            data[f"Trend_strength_{window}"] = (
                data["Close"].rolling(window).apply(calculate_trend_strength, raw=False)
            )

        return data

    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """ボラティリティ特徴量を追加"""
        logger.info("📊 ボラティリティ特徴量を追加中...")

        # 実現ボラティリティ
        data["Returns"] = data["Close"].pct_change()

        for window in [5, 10, 20, 50]:
            data[f"Realized_Vol_{window}"] = data["Returns"].rolling(
                window
            ).std() * np.sqrt(
                24
            )  # 日次換算
            data[f"Vol_of_Vol_{window}"] = (
                data[f"Realized_Vol_{window}"].rolling(window).std()
            )

        # Parkinson推定量（高値・安値ベースのボラティリティ）
        for window in [10, 20]:
            hl_ratio = np.log(data["High"] / data["Low"])
            data[f"Parkinson_Vol_{window}"] = hl_ratio.rolling(window).var() * (
                1 / (4 * np.log(2))
            )

        # ボラティリティレジーム
        vol_20 = data["Returns"].rolling(20).std()
        data["Vol_Regime"] = pd.cut(
            vol_20, bins=3, labels=[0, 1, 2]
        )  # 低・中・高ボラティリティ

        return data

    def _add_funding_rate_features(
        self, data: pd.DataFrame, fr_data: pd.DataFrame
    ) -> pd.DataFrame:
        """ファンディングレート特徴量を追加"""
        logger.info("💰 ファンディングレート特徴量を追加中...")

        if "funding_rate" in fr_data.columns:
            # ファンディングレートの統計
            for window in [3, 7, 14]:  # 3回、7回、14回分（24h, 56h, 112h）
                data[f"FR_mean_{window}"] = (
                    fr_data["funding_rate"].rolling(window).mean()
                )
                data[f"FR_std_{window}"] = fr_data["funding_rate"].rolling(window).std()
                data[f"FR_sum_{window}"] = fr_data["funding_rate"].rolling(window).sum()

            # ファンディングレートの変化
            data["FR_change"] = fr_data["funding_rate"].diff()
            data["FR_change_abs"] = data["FR_change"].abs()

            # 極端なファンディングレート
            data["FR_extreme_positive"] = (
                fr_data["funding_rate"] > fr_data["funding_rate"].quantile(0.95)
            ).astype(int)
            data["FR_extreme_negative"] = (
                fr_data["funding_rate"] < fr_data["funding_rate"].quantile(0.05)
            ).astype(int)

        return data

    def _add_open_interest_features(
        self, data: pd.DataFrame, oi_data: pd.DataFrame
    ) -> pd.DataFrame:
        """建玉残高特徴量を追加"""
        logger.info("📊 建玉残高特徴量を追加中...")

        if "open_interest" in oi_data.columns:
            # 建玉残高の変化率
            for period in [1, 6, 24]:
                data[f"OI_pct_change_{period}"] = oi_data["open_interest"].pct_change(
                    period
                )

            # 建玉残高の移動平均
            for window in [6, 24, 168]:  # 6h, 24h, 168h(1週間)
                data[f"OI_ma_{window}"] = (
                    oi_data["open_interest"].rolling(window).mean()
                )
                data[f"OI_deviation_{window}"] = (
                    oi_data["open_interest"] - data[f"OI_ma_{window}"]
                ) / data[f"OI_ma_{window}"]

            # 建玉残高と価格の関係
            data["OI_Price_Correlation"] = (
                oi_data["open_interest"].rolling(24).corr(data["Close"])
            )

        return data

    def _add_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """相互作用特徴量を追加"""
        logger.info("🔗 相互作用特徴量を追加中...")

        # 価格と出来高の相互作用
        data["Price_Volume_Product"] = data["Close"] * data["Volume"]
        data["Price_Volume_Ratio"] = data["Close"] / (data["Volume"] + 1e-8)

        # ボラティリティと出来高
        if "Realized_Vol_20" in data.columns:
            data["Vol_Volume_Product"] = data["Realized_Vol_20"] * data["Volume"]

        # 技術指標の組み合わせ
        if "RSI" in data.columns and "Stochastic_K" in data.columns:
            data["RSI_Stoch_Avg"] = (data["RSI"] + data["Stochastic_K"]) / 2
            data["RSI_Stoch_Diff"] = data["RSI"] - data["Stochastic_K"]

        return data

    def _add_seasonal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """季節性特徴量を追加"""
        logger.info("📅 季節性特徴量を追加中...")

        # 時間特徴量
        data["Hour"] = data.index.hour
        data["DayOfWeek"] = data.index.dayofweek
        data["DayOfMonth"] = data.index.day
        data["Month"] = data.index.month

        # 周期的エンコーディング
        data["Hour_sin"] = np.sin(2 * np.pi * data["Hour"] / 24)
        data["Hour_cos"] = np.cos(2 * np.pi * data["Hour"] / 24)
        data["DayOfWeek_sin"] = np.sin(2 * np.pi * data["DayOfWeek"] / 7)
        data["DayOfWeek_cos"] = np.cos(2 * np.pi * data["DayOfWeek"] / 7)

        # 市場時間特徴量
        data["Is_Weekend"] = (data["DayOfWeek"] >= 5).astype(int)  # 土日
        data["Is_Asian_Hours"] = ((data["Hour"] >= 0) & (data["Hour"] < 8)).astype(int)
        data["Is_European_Hours"] = ((data["Hour"] >= 8) & (data["Hour"] < 16)).astype(
            int
        )
        data["Is_American_Hours"] = ((data["Hour"] >= 16) & (data["Hour"] < 24)).astype(
            int
        )

        return data

    def clean_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """特徴量のクリーニング"""
        logger.info("🧹 特徴量クリーニング中...")

        # 無限値を除去
        data = data.replace([np.inf, -np.inf], np.nan)

        # 欠損値を前方補完
        data = data.fillna(method="ffill")

        # 残った欠損値を中央値で補完
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(
            data[numeric_columns].median()
        )

        # 異常値のクリッピング（99.5%ile）
        for col in numeric_columns:
            if col not in [
                "Hour",
                "DayOfWeek",
                "DayOfMonth",
                "Month",
                "Is_Weekend",
                "Is_Asian_Hours",
                "Is_European_Hours",
                "Is_American_Hours",
            ]:
                q_low = data[col].quantile(0.005)
                q_high = data[col].quantile(0.995)
                data[col] = data[col].clip(lower=q_low, upper=q_high)

        logger.info(f"✅ 特徴量クリーニング完了: {data.shape[1]}個の特徴量")

        return data


# グローバルインスタンス
advanced_feature_engineer = AdvancedFeatureEngineer()
