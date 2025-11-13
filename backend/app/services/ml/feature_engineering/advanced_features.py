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
        """ラグ特徴量を追加（重要な期間のみ）"""
        logger.info("📊 ラグ特徴量を追加中...")

        new_features = {}

        # 価格のラグ特徴量（最重要期間のみ: 1h, 24h）
        lag_periods = [1, 24]

        for period in lag_periods:
            new_features[f"close_lag_{period}"] = data["close"].shift(period)

        # 価格変化率のラグ（24hのみ）
        # NOTE: "returns" 自体は特徴量重要度分析で完全未使用と判定されたため削除
        # 代わりに returns_lag_24 と cumulative_returns_24 のみ計算
        returns_temp = data["close"].pct_change()
        new_features["returns_lag_24"] = returns_temp.shift(24)

        # 累積リターン（24hのみ）
        new_features["cumulative_returns_24"] = (
            returns_temp.rolling(24).sum()
        )

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
            # Removed: stochastic_k, stochastic_d (低寄与度特徴量削除: 2025-11-13)

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
        """統計的特徴量を追加（主要ウィンドウのみ）"""
        logger.info("📊 統計的特徴量を追加中...")

        new_features = {}
        windows = [20, 50]  # 標準期間とトレンド期間のみ

        for window in windows:
            # Removed: Close_mean_20, Close_mean_50 (低寄与度特徴量削除: 2025-11-13)
            # 移動統計（標準偏差のみ残す）
            new_features[f"Close_std_{window}"] = data["close"].rolling(window).std()

            # 範囲統計（重要な指標のみ）
            high_max = data["high"].rolling(window).max()
            low_min = data["low"].rolling(window).min()
            new_features[f"Close_range_{window}"] = high_max - low_min

        # 一括で結合
        new_df = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
        return new_df

    def _add_time_series_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """時系列特徴量を追加（重要な期間のみ）"""
        logger.info("⏰ 時系列特徴量を追加中...")

        new_features = {}

        # Removed: Close_pct_change_1, Close_pct_change_24 (低寄与度特徴量削除: 2025-11-13)
        # 変化率特徴量は削除

        # 移動平均からの乖離（20期間のみ）
        ma_20 = data["close"].rolling(20).mean()
        new_features["Close_deviation_from_ma_20"] = (data["close"] - ma_20) / ma_20

        # トレンド強度（20期間のみ）
        new_features["Trend_strength_20"] = ta.linreg(
            data["close"], length=20, slope=True
        )

        # 一括で結合
        new_df = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
        return new_df

    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """ボラティリティ特徴量を追加（高寄与度のみ）"""
        logger.info("📊 ボラティリティ特徴量を追加中...")

        new_features = {}

        # 実現ボラティリティ（20期間のみ）- 高寄与度
        # Removed: Returns (低寄与度特徴量削除: 2025-11-13)
        returns_temp = data["close"].pct_change()
        new_features["Realized_Vol_20"] = returns_temp.rolling(20).std() * np.sqrt(24)

        # Parkinson推定量（20期間のみ）- 高寄与度
        hl_ratio = np.log(data["high"] / data["low"])
        new_features["Parkinson_Vol_20"] = hl_ratio.rolling(20).var() * (
            1 / (4 * np.log(2))
        )

        # 削除された特徴量（低寄与度）:
        # - Vol_Regime (スコア: 5.38e-05)
        # - high_vol_regime (スコア: 1.50e-04)

        # 一括で結合
        new_df = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
        return new_df

    def _add_funding_rate_features(
        self, data: pd.DataFrame, fr_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        ファンディングレート特徴量（新設計: Tier 1特徴量）

        新しいFundingRateFeatureCalculatorを使用してTier 1特徴量を生成
        """
        logger.info("💰 ファンディングレート特徴量を追加中...")

        from .funding_rate_features import FundingRateFeatureCalculator

        try:
            # FundingRateFeatureCalculatorを使用
            fr_calculator = FundingRateFeatureCalculator()
            result_df = fr_calculator.calculate_features(data, fr_data)

            fr_features = [col for col in result_df.columns if col.startswith("fr_")]
            logger.info(f"ファンディングレート特徴量を追加: {len(fr_features)}個")

            return result_df
        except Exception as e:
            logger.warning(f"ファンディングレート特徴量の計算エラー: {e}")
            return data

    def _add_open_interest_features(
        self, data: pd.DataFrame, oi_data: pd.DataFrame
    ) -> pd.DataFrame:
        """建玉残高特徴量を追加（主要指標のみ）"""
        logger.info("📊 建玉残高特徴量を追加中...")

        new_features = {}

        if "open_interest" in oi_data.columns:
            # 建玉残高の変化率（24hのみ）
            new_features["OI_pct_change_24"] = oi_data["open_interest"].pct_change(24)

            # 建玉残高の移動平均（24hのみ）
            new_features["OI_ma_24"] = oi_data["open_interest"].rolling(24).mean()
            new_features["OI_deviation_24"] = (
                oi_data["open_interest"] - new_features["OI_ma_24"]
            ) / new_features["OI_ma_24"]

            # 建玉残高と価格の関係
            new_features["OI_Price_Correlation"] = (
                oi_data["open_interest"].rolling(24).corr(data["close"])
            )

        # 一括で結合
        new_df = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
        return new_df

    def _add_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """相互作用特徴量を追加（重要な組み合わせのみ）"""
        logger.info("🔗 相互作用特徴量を追加中...")

        new_features = {}

        # 価格と出来高の相互作用（最も重要）
        new_features["Price_Volume_Ratio"] = data["close"] / (data["volume"] + 1e-8)

        # ボラティリティと出来高
        if "Realized_Vol_20" in data.columns:
            new_features["Vol_Volume_Product"] = (
                data["Realized_Vol_20"] * data["volume"]
            )

        # 一括で結合
        new_df = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
        return new_df

    def _add_seasonal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """季節性特徴量（削除: 暗号通貨市場では24時間取引で時間効果が弱い）"""
        logger.info("📅 季節性特徴量を追加中...")

        # 分析結果: 時間・セッション関連特徴量は全て極めて低い寄与度のため削除
        # 削除された特徴量: Hour, DayOfWeek, Hour_sin, Hour_cos, DayOfWeek_sin, DayOfWeek_cos,
        #                 Is_Weekend, Is_Asian_Hours, Is_American_Hours
        # 理由: 暗号通貨は24時間取引で時間帯効果が弱い（全てスコア < 0.0003）

        return data


# グローバルインスタンス
advanced_feature_engineer = AdvancedFeatureEngineer()
