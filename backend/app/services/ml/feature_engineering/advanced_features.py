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

        # 9. マルチタイムフレーム(MTF)特徴量 (2025-11-23追加: 4h, 24h)
        features = self._add_mtf_features(features, ohlcv_data, timeframe_hours=4)
        features = self._add_mtf_features(features, ohlcv_data, timeframe_hours=24)

        # 10. 市場ダイナミクス特徴量 (OI/FR/Priceの相互作用) (2025-11-23追加)
        if funding_rate_data is not None and open_interest_data is not None:
            features = self._add_market_dynamics_features(
                features, ohlcv_data, funding_rate_data, open_interest_data
            )
            
        # 11. RANGE検出特化特徴量 (2025-11-24追加)
        features = self._add_range_detection_features(features)

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
        # FutureWarning対応: fill_method=None
        returns_temp = data["close"].pct_change(fill_method=None)
        new_features["returns_lag_24"] = returns_temp.shift(24)

        # 累積リターン（24hのみ）
        new_features["cumulative_returns_24"] = returns_temp.rolling(24).sum()

        # 一括で結合
        new_df = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
        return new_df

    def _add_advanced_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """高度な技術指標を追加"""
        logger.info("📈 高度な技術指標を追加中...")

        new_features = {}

        try:
            # モメンタム指標
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

            # === レンジ/トレンド判定強化 ===
            # 1. Choppiness Index (CHOP)
            chop_result = ta.chop(high=data["high"], low=data["low"], close=data["close"])
            if chop_result is not None:
                new_features["CHOP"] = chop_result

            # 2. Vortex Indicator (VI)
            vortex_result = ta.vortex(high=data["high"], low=data["low"], close=data["close"])
            if vortex_result is not None and not vortex_result.empty:
                new_features["VI_Plus"] = vortex_result["VTXP_14"]
                new_features["VI_Minus"] = vortex_result["VTXM_14"]

            # 3. Bollinger Bandwidth (BBW)
            bbands = ta.bbands(close=data["close"])
            if bbands is not None and not bbands.empty:
                if "BBB_5_2.0" in bbands.columns:
                    new_features["BBW"] = bbands["BBB_5_2.0"]
                if "BBP_5_2.0" in bbands.columns:
                    new_features["BB_Percent"] = bbands["BBP_5_2.0"]

            # トレンド指標
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

            # ボラティリティ指標
            new_features["ATR"] = ta.atr(
                high=data["high"], low=data["low"], close=data["close"]
            )
            new_features["NATR"] = ta.natr(
                high=data["high"], low=data["low"], close=data["close"]
            )
            new_features["TRANGE"] = ta.true_range(
                high=data["high"], low=data["low"], close=data["close"]
            )

            # 出来高指標
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
        windows = [20, 50]

        for window in windows:
            # 移動統計（標準偏差のみ残す）
            new_features[f"Close_std_{window}"] = data["close"].rolling(window).std()

            # 範囲統計
            high_max = data["high"].rolling(window).max()
            low_min = data["low"].rolling(window).min()
            new_features[f"Close_range_{window}"] = high_max - low_min

        new_df = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
        return new_df

    def _add_time_series_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """時系列特徴量を追加（重要な期間のみ）"""
        logger.info("⏰ 時系列特徴量を追加中...")

        new_features = {}

        # 移動平均からの乖離（20期間のみ）
        ma_20 = data["close"].rolling(20).mean()
        new_features["Close_deviation_from_ma_20"] = (data["close"] - ma_20) / ma_20

        # トレンド強度（20期間のみ）
        new_features["Trend_strength_20"] = ta.linreg(
            data["close"], length=20, slope=True
        )

        new_df = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
        return new_df

    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """ボラティリティ特徴量を追加（高寄与度のみ）"""
        logger.info("📊 ボラティリティ特徴量を追加中...")

        new_features = {}

        # 実現ボラティリティ（20期間のみ）
        # FutureWarning対応: fill_method=None
        returns_temp = data["close"].pct_change(fill_method=None)
        new_features["Realized_Vol_20"] = returns_temp.rolling(20).std() * np.sqrt(24)

        # Parkinson推定量（20期間のみ）
        hl_ratio = np.log(data["high"] / data["low"])
        new_features["Parkinson_Vol_20"] = hl_ratio.rolling(20).var() * (
            1 / (4 * np.log(2))
        )

        new_df = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
        return new_df

    def _add_funding_rate_features(
        self, data: pd.DataFrame, fr_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        ファンディングレート特徴量（新設計: Tier 1特徴量）
        """
        logger.info("💰 ファンディングレート特徴量を追加中...")

        from .funding_rate_features import FundingRateFeatureCalculator

        try:
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
            # FutureWarning対応
            new_features["OI_pct_change_24"] = oi_data["open_interest"].pct_change(24, fill_method=None)

            # 建玉残高の移動平均（24hのみ）
            new_features["OI_ma_24"] = oi_data["open_interest"].rolling(24).mean()
            new_features["OI_deviation_24"] = (
                oi_data["open_interest"] - new_features["OI_ma_24"]
            ) / new_features["OI_ma_24"]

            # 建玉残高と価格の関係
            new_features["OI_Price_Correlation"] = (
                oi_data["open_interest"].rolling(24).corr(data["close"])
            )

        new_df = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
        return new_df

    def _add_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """相互作用特徴量を追加（重要な組み合わせのみ）"""
        logger.info("🔗 相互作用特徴量を追加中...")

        new_features = {}

        # 価格と出来高の相互作用
        new_features["Price_Volume_Ratio"] = data["close"] / (data["volume"] + 1e-8)

        # ボラティリティと出来高
        if "Realized_Vol_20" in data.columns:
            new_features["Vol_Volume_Product"] = (
                data["Realized_Vol_20"] * data["volume"]
            )

        new_df = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
        return new_df

    def _add_seasonal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """季節性特徴量（削除）"""
        logger.info("📅 季節性特徴量を追加中...")
        return data

    def _add_mtf_features(
        self, features: pd.DataFrame, ohlcv: pd.DataFrame, timeframe_hours: int = 4
    ) -> pd.DataFrame:
        """
        マルチタイムフレーム特徴量を追加
        上位足（4hなど）のトレンドやボラティリティ情報を追加する
        """
        logger.info(f"🌍 MTF特徴量を追加中 ({timeframe_hours}h)...")

        try:
            # リサンプリング (1h -> 4h)
            # 文字列の "first", "last" が内部で NDFrame.first/last を呼び出し、
            # それが offset 引数を要求してエラーになる場合があるため、関数に置き換える
            agg_dict = {
                "open": lambda x: x.iloc[0] if len(x) > 0 else np.nan,
                "high": "max",
                "low": "min",
                "close": lambda x: x.iloc[-1] if len(x) > 0 else np.nan,
                "volume": "sum",
            }
            
            # 確実にDatetimeIndexであることを確認
            if not isinstance(ohlcv.index, pd.DatetimeIndex):
                logger.warning(f"MTF特徴量計算スキップ: インデックスがDatetimeIndexではありません")
                return features

            resampled = ohlcv.resample(f"{timeframe_hours}h").agg(agg_dict).dropna()

            if resampled.empty:
                logger.warning(f"MTF特徴量計算: リサンプリング後のデータが空です ({timeframe_hours}h)")
                return features

            mtf_features = pd.DataFrame(index=resampled.index)
            
            # 上位足のトレンド (RSI, EMA乖離)
            mtf_features[f"MTF_{timeframe_hours}h_RSI"] = ta.rsi(resampled["close"], length=14)
            
            ema_50 = ta.ema(resampled["close"], length=50)
            if ema_50 is not None:
                mtf_features[f"MTF_{timeframe_hours}h_EMA50_Diff"] = (resampled["close"] - ema_50) / ema_50

            # 上位足のボラティリティ (BBW)
            bbands = ta.bbands(resampled["close"], length=20, std=2)
            if bbands is not None and "BBB_20_2.0" in bbands.columns:
                mtf_features[f"MTF_{timeframe_hours}h_BBW"] = bbands["BBB_20_2.0"]

            # 上位足のトレンド方向 (ADX)
            adx = ta.adx(resampled["high"], resampled["low"], resampled["close"], length=14)
            if adx is not None and "ADX_14" in adx.columns:
                mtf_features[f"MTF_{timeframe_hours}h_ADX"] = adx["ADX_14"]

            # 元の時間軸に合わせてリインデックス (ffillで直前の値を採用 = 未来の情報をリークさせない)
            mtf_features_aligned = mtf_features.reindex(features.index).ffill()

            # 結合
            new_df = pd.concat([features, mtf_features_aligned], axis=1)
            
            added_cols = len(new_df.columns) - len(features.columns)
            logger.info(f"MTF特徴量を追加: {added_cols}個")
            
            return new_df

        except Exception as e:
            logger.warning(f"MTF特徴量計算エラー: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            return features

    def _add_market_dynamics_features(
        self,
        features: pd.DataFrame,
        ohlcv: pd.DataFrame,
        fr_data: pd.DataFrame,
        oi_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        市場ダイナミクス特徴量を追加 (OI/FR/Priceの高度な相互作用)
        """
        logger.info("🌪️ 市場ダイナミクス特徴量を追加中...")

        try:
            # データの準備 (リインデックスして結合)
            combined = features[["close"]].copy()
            
            # Funding Rateの結合
            if "funding_rate" in fr_data.columns:
                fr_aligned = fr_data["funding_rate"].reindex(features.index).ffill()
                combined["fr"] = fr_aligned
            
            # Open Interestの結合
            if "open_interest" in oi_data.columns:
                oi_aligned = oi_data["open_interest"].reindex(features.index).ffill()
                combined["oi"] = oi_aligned

            # 必須カラムが揃っているか確認
            if "fr" not in combined.columns or "oi" not in combined.columns:
                logger.warning("FRまたはOIデータが不足しているため、ダイナミクス特徴量をスキップします")
                return features

            new_features_dict = {}

            # 1. OI Weighted FR
            new_features_dict["OI_Weighted_FR"] = combined["fr"] * combined["oi"]
            
            # 2. Cumulative OI Weighted FR
            new_features_dict["Cumulative_OI_Weighted_FR_24h"] = new_features_dict["OI_Weighted_FR"].rolling(24).sum()

            # 3. OI/Price Divergence
            price_pct = combined["close"].pct_change(fill_method=None)
            oi_pct = combined["oi"].pct_change(fill_method=None)
            epsilon = 1e-6
            new_features_dict["OI_Price_Divergence"] = oi_pct / (price_pct.abs() + epsilon)

            # 4. FR/Price Divergence
            new_features_dict["FR_Price_Correlation_24h"] = combined["fr"].rolling(24).corr(combined["close"])

            # 結合
            new_features_df = pd.DataFrame(new_features_dict, index=features.index)
            new_df = pd.concat([features, new_features_df], axis=1)
            
            added_cols = len(new_df.columns) - len(features.columns)
            logger.info(f"市場ダイナミクス特徴量を追加: {added_cols}個")
            
            return new_df

        except Exception as e:
            logger.warning(f"市場ダイナミクス特徴量計算エラー: {e}")
            return features
    
    def _add_range_detection_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        RANGE検出に特化した特徴量
        """
        logger.info("🎯 RANGE検出特化特徴量を追加中...")
        
        try:
            new_features_dict = {}
            
            # 1. 価格レンジの狭さ（正規化）
            for window in [24, 72, 168]:
                high_max = data["high"].rolling(window=window).max()
                low_min = data["low"].rolling(window=window).min()
                price_range = high_max - low_min
                new_features_dict[f"Price_Range_Normalized_{window}h"] = price_range / (data["close"] + 1e-8)
            
            # 2. ボラティリティレジーム
            if "Realized_Vol_20" in data.columns:
                realized_vol = data["Realized_Vol_20"]
                vol_rank = realized_vol.rolling(window=720).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
                )
                new_features_dict["Volatility_Regime_Rank"] = vol_rank
                new_features_dict["Low_Volatility_Flag"] = (vol_rank < 0.25).astype(int)
            
            # 3. トレンド強度の欠如
            if "ADX" in data.columns:
                new_features_dict["Weak_Trend_Flag"] = (data["ADX"] < 25).astype(int)
                new_features_dict["ADX_Momentum"] = data["ADX"].pct_change(periods=24, fill_method=None)
            
            # 4. Choppiness指標
            if "CHOP" in data.columns:
                new_features_dict["Strong_Range_Flag"] = (data["CHOP"] > 61.8).astype(int)
                new_features_dict["CHOP_MA_24h"] = data["CHOP"].rolling(window=24).mean()
            
            # 5. 価格の往復運動
            price_direction = (data["close"].diff() > 0).astype(int)
            direction_changes = (price_direction != price_direction.shift(1)).astype(int)
            new_features_dict["Direction_Change_Count_24h"] = direction_changes.rolling(window=24).sum()
            
            # 6. 価格密度
            for window in [24, 72]:
                price_std = data["close"].rolling(window=window).std()
                price_mean = data["close"].rolling(window=window).mean()
                new_features_dict[f"Price_Density_{window}h"] = price_std / (price_mean + 1e-8)
            
            # 7. ボリンジャーバンド収縮度
            if "BBW" in data.columns:
                bbw_rank = data["BBW"].rolling(window=720).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
                )
                new_features_dict["BBW_Squeeze_Rank"] = bbw_rank
                new_features_dict["BB_Squeeze_Flag"] = (bbw_rank < 0.10).astype(int)
            
            # 8. 価格変化の絶対値平均
            abs_returns = data["close"].pct_change(fill_method=None).abs()
            new_features_dict["Abs_Returns_MA_24h"] = abs_returns.rolling(window=24).mean()
            
            abs_returns_rank = abs_returns.rolling(window=720).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
            )
            new_features_dict["Small_Movement_Flag"] = (abs_returns_rank < 0.25).astype(int)
            
            # 9. RANGE総合スコア
            # 一旦DataFrameにして計算
            temp_df = pd.DataFrame(new_features_dict)
            range_signals = []
            signal_cols = ["Low_Volatility_Flag", "Weak_Trend_Flag", "Strong_Range_Flag", "BB_Squeeze_Flag", "Small_Movement_Flag"]
            
            for col in signal_cols:
                if col in temp_df.columns:
                    range_signals.append(temp_df[col])
            
            if range_signals:
                new_features_dict["RANGE_Composite_Score"] = pd.concat(range_signals, axis=1).mean(axis=1)
            
            # 結合
            new_features_df = pd.DataFrame(new_features_dict, index=data.index)
            result = pd.concat([data, new_features_df], axis=1)
            
            added_cols = len(result.columns) - len(data.columns)
            logger.info(f"RANGE検出特化特徴量を追加: {added_cols}個")
            
            return result
            
        except Exception as e:
            logger.warning(f"RANGE検出特徴量計算エラー: {e}")
            return data


# グローバルインスタンス
advanced_feature_engineer = AdvancedFeatureEngineer()