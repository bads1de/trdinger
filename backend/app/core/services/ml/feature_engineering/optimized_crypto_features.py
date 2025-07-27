"""
最適化された暗号通貨特徴量エンジニアリング

深度分析の結果に基づいて最適化された特徴量を生成します。
- 高安定性特徴量の強化
- 不安定特徴量の改良
- 新しい複合特徴量の追加
- 時間遅れとスムージングの適用
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, List


logger = logging.getLogger(__name__)


class OptimizedCryptoFeatures:
    """最適化された暗号通貨特徴量エンジニアリング"""

    def __init__(self):
        """初期化"""
        self.feature_groups = {
            "stable_high_performers": [],
            "enhanced_technical": [],
            "robust_temporal": [],
            "advanced_composite": [],
            "smoothed_market_data": [],
            "regime_aware": [],
            "cross_timeframe": [],
            "volatility_adjusted": [],
        }

    def create_optimized_features(
        self, df: pd.DataFrame, lookback_periods: Optional[Dict[str, int]] = None
    ) -> pd.DataFrame:
        """
        最適化された特徴量を作成

        Args:
            df: 基本データ（OHLCV + OI + FR + FG）
            lookback_periods: 計算期間の設定

        Returns:
            最適化された特徴量が追加されたDataFrame
        """
        if lookback_periods is None:
            lookback_periods = {
                "short": 4,
                "medium": 24,
                "long": 168,
                "extra_long": 336,
            }

        logger.info("最適化された特徴量作成を開始")
        result_df = df.copy()

        # データ品質確保
        result_df = self._ensure_data_quality(result_df)

        # 各グループの最適化された特徴量を作成
        result_df = self._create_stable_high_performers(result_df, lookback_periods)
        result_df = self._create_enhanced_technical_features(
            result_df, lookback_periods
        )
        result_df = self._create_robust_temporal_features(result_df)
        result_df = self._create_advanced_composite_features(
            result_df, lookback_periods
        )
        result_df = self._create_smoothed_market_features(result_df, lookback_periods)
        result_df = self._create_regime_aware_features(result_df, lookback_periods)
        result_df = self._create_cross_timeframe_features(result_df, lookback_periods)
        result_df = self._create_volatility_adjusted_features(
            result_df, lookback_periods
        )

        # 最終クリーンアップ
        result_df = self._final_cleanup(result_df)

        logger.info(f"最適化特徴量作成完了: {len(result_df.columns)}個の特徴量")
        return result_df

    def _ensure_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """データ品質を確保（改良版）"""
        result_df = df.copy()

        # 必要なカラムの確認
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in result_df.columns:
                logger.warning(f"必須カラム {col} が見つかりません")
                return result_df

        # オプショナルカラムの補完（改良版）
        if "open_interest" not in result_df.columns:
            # 出来高ベースの疑似OI
            result_df["open_interest"] = result_df["Volume"].rolling(24).mean() * 10
        if "funding_rate" not in result_df.columns:
            # 価格勢いベースの疑似FR
            result_df["funding_rate"] = (
                result_df["Close"].pct_change(8).rolling(3).mean() * 0.1
            )
        if "fear_greed_value" not in result_df.columns:
            # ボラティリティベースの疑似FG
            volatility = result_df["Close"].pct_change().rolling(24).std()
            result_df["fear_greed_value"] = (
                50 - (volatility - volatility.median()) * 1000
            )

        # ロバストな前方補完
        for col in ["open_interest", "funding_rate", "fear_greed_value"]:
            # 外れ値の除去
            Q1 = result_df[col].quantile(0.25)
            Q3 = result_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            result_df[col] = result_df[col].clip(lower_bound, upper_bound)
            result_df[col] = result_df[col].ffill().fillna(result_df[col].median())

        return result_df

    def _create_stable_high_performers(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """安定性の高い高性能特徴量（分析結果に基づく）"""
        result_df = df.copy()

        # 1. 週末効果（最高安定性）
        result_df["is_weekend_enhanced"] = (df.index.dayofweek >= 5).astype(int)
        result_df["weekend_proximity"] = (
            np.minimum(df.index.dayofweek, 7 - df.index.dayofweek) / 7.0
        )

        # 2. VWAP系特徴量（高安定性）
        for period in [12, 24, 48, 72]:
            typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
            vwap = (typical_price * df["Volume"]).rolling(period).sum() / df[
                "Volume"
            ].rolling(period).sum()

            # スムージング適用
            vwap_smooth = vwap.rolling(3, center=True).mean().fillna(vwap)
            result_df[f"vwap_smooth_{period}h"] = vwap_smooth

            # 価格との関係（ロバスト版）
            price_vwap_ratio = df["Close"] / vwap_smooth
            result_df[f"price_vwap_ratio_robust_{period}h"] = price_vwap_ratio.rolling(
                3
            ).median()

        # 3. ボリンジャーバンド位置（改良版）
        for period in [20, 48]:
            ma = df["Close"].rolling(period).mean()
            # ロバストな標準偏差
            robust_std = df["Close"].rolling(period).apply(lambda x: np.std(x) * 1.4826)

            bb_upper = ma + (robust_std * 2)
            bb_lower = ma - (robust_std * 2)

            # 位置の計算（0-1に正規化）
            bb_position = (df["Close"] - bb_lower) / (bb_upper - bb_lower)
            result_df[f"bb_position_robust_{period}"] = bb_position.clip(0, 1)

        # 4. 曜日効果（改良版）
        result_df["day_of_week_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        result_df["day_of_week_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        self.feature_groups["stable_high_performers"].extend(
            [
                col
                for col in result_df.columns
                if col.startswith(
                    (
                        "is_weekend_",
                        "weekend_",
                        "vwap_smooth_",
                        "price_vwap_ratio_robust_",
                        "bb_position_robust_",
                        "day_of_week_",
                    )
                )
            ]
        )

        return result_df

    def _create_enhanced_technical_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """強化されたテクニカル指標"""
        result_df = df.copy()

        # 1. 適応的RSI
        for period in [14, 24]:
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

            # ゼロ除算対策
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            # スムージング
            rsi_smooth = rsi.rolling(3).mean()
            result_df[f"rsi_adaptive_{period}"] = rsi_smooth

            # RSIの勢い
            result_df[f"rsi_momentum_{period}"] = rsi_smooth.diff(4)

        # 2. 改良MACD
        ema12 = df["Close"].ewm(span=12).mean()
        ema26 = df["Close"].ewm(span=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9).mean()

        # MACDヒストグラムの改良
        macd_histogram = macd - macd_signal
        result_df["macd_histogram_smooth"] = macd_histogram.rolling(3).mean()
        result_df["macd_divergence"] = (macd / macd_signal - 1).rolling(5).mean()

        # 3. 多期間移動平均の関係
        ma_short = df["Close"].rolling(12).mean()
        ma_medium = df["Close"].rolling(24).mean()
        ma_long = df["Close"].rolling(48).mean()

        result_df["ma_alignment"] = (
            (ma_short > ma_medium).astype(int) + (ma_medium > ma_long).astype(int)
        ) / 2.0

        # 4. ボラティリティ調整済み指標
        volatility = df["Close"].pct_change().rolling(24).std()
        result_df["volatility_adjusted_return"] = df["Close"].pct_change() / (
            volatility + 1e-10
        )

        self.feature_groups["enhanced_technical"].extend(
            [
                col
                for col in result_df.columns
                if col.startswith(
                    (
                        "rsi_adaptive_",
                        "rsi_momentum_",
                        "macd_",
                        "ma_alignment",
                        "volatility_adjusted_",
                    )
                )
            ]
        )

        return result_df

    def _create_robust_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ロバストな時間関連特徴量"""
        result_df = df.copy()

        # 1. 時間帯の改良（連続値化）
        result_df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        result_df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

        # 2. 地域別取引時間（重み付き）
        asia_weight = np.exp(-(((df.index.hour - 4) % 24 - 12) ** 2) / 32)
        europe_weight = np.exp(-(((df.index.hour - 12) % 24 - 12) ** 2) / 32)
        us_weight = np.exp(-(((df.index.hour - 20) % 24 - 12) ** 2) / 32)

        result_df["asia_trading_intensity"] = asia_weight
        result_df["europe_trading_intensity"] = europe_weight
        result_df["us_trading_intensity"] = us_weight

        # 3. 月内効果（改良版）
        result_df["month_progress"] = df.index.day / 31.0
        result_df["month_end_proximity"] = (
            np.minimum(df.index.day, 32 - df.index.day) / 15.0
        )

        # 4. 季節性効果
        result_df["year_progress"] = df.index.dayofyear / 365.0
        result_df["quarter_sin"] = np.sin(2 * np.pi * df.index.quarter / 4)
        result_df["quarter_cos"] = np.cos(2 * np.pi * df.index.quarter / 4)

        self.feature_groups["robust_temporal"].extend(
            [
                col
                for col in result_df.columns
                if col.startswith(
                    ("hour_", "asia_", "europe_", "us_", "month_", "year_", "quarter_")
                )
            ]
        )

        return result_df

    def _create_advanced_composite_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """高度な複合特徴量"""
        result_df = df.copy()

        # 1. 多次元勢い指標
        price_momentum = df["Close"].pct_change(periods["medium"])
        volume_momentum = df["Volume"].pct_change(periods["medium"])
        oi_momentum = df["open_interest"].pct_change(periods["medium"])

        # 重み付き複合勢い
        result_df["weighted_momentum"] = (
            price_momentum * 0.5 + volume_momentum * 0.3 + oi_momentum * 0.2
        )

        # 2. 市場効率性指標
        returns = df["Close"].pct_change()
        autocorr_1 = returns.rolling(periods["medium"]).apply(
            lambda x: x.autocorr(lag=1)
        )
        result_df["market_efficiency"] = 1 - autocorr_1.abs()

        # 3. 流動性指標
        price_impact = (df["High"] - df["Low"]) / df["Volume"]
        result_df["liquidity_indicator"] = 1 / (
            price_impact.rolling(periods["short"]).mean() + 1e-10
        )

        # 4. 市場ストレス指標（改良版）
        price_vol = returns.rolling(periods["short"]).std()
        volume_vol = df["Volume"].pct_change().rolling(periods["short"]).std()
        fr_abs = df["funding_rate"].abs()

        result_df["market_stress_composite"] = (
            price_vol * 0.4 + volume_vol * 0.3 + fr_abs * 0.3
        )

        self.feature_groups["advanced_composite"].extend(
            [
                col
                for col in result_df.columns
                if col.startswith(
                    ("weighted_", "market_efficiency", "liquidity_", "market_stress_")
                )
            ]
        )

        return result_df

    def _create_smoothed_market_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """スムージングされた市場データ特徴量"""
        result_df = df.copy()

        # 1. 適応的スムージング
        for col in ["open_interest", "funding_rate"]:
            if col in df.columns:
                # 指数移動平均でスムージング
                smoothed = df[col].ewm(alpha=0.3).mean()
                result_df[f"{col}_smooth"] = smoothed

                # トレンド抽出
                trend = smoothed.rolling(periods["medium"]).apply(
                    lambda x: (
                        np.polyfit(range(len(x)), x, 1)[0]
                        if len(x) == periods["medium"]
                        else 0
                    )
                )
                result_df[f"{col}_trend"] = trend

        # 2. ノイズ除去された価格特徴量
        # Hodrick-Prescott フィルターの簡易版
        price_smooth = df["Close"].rolling(5, center=True).mean().fillna(df["Close"])
        result_df["price_denoised"] = price_smooth
        result_df["price_noise"] = df["Close"] - price_smooth

        # 3. ロバストな変動率
        for period in [periods["short"], periods["medium"]]:
            # 中央値ベースの変動率
            median_change = (
                df["Close"]
                .rolling(period)
                .apply(
                    lambda x: (
                        (x.iloc[-1] - x.median()) / x.median()
                        if len(x) == period
                        else 0
                    )
                )
            )
            result_df[f"robust_return_{period}h"] = median_change

        self.feature_groups["smoothed_market_data"].extend(
            [
                col
                for col in result_df.columns
                if col.startswith(
                    (
                        "open_interest_",
                        "funding_rate_",
                        "price_denoised",
                        "price_noise",
                        "robust_return_",
                    )
                )
            ]
        )

        return result_df

    def _create_regime_aware_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """市場レジーム認識特徴量"""
        result_df = df.copy()

        # 1. ボラティリティレジーム
        volatility = df["Close"].pct_change().rolling(periods["short"]).std()
        vol_q33 = volatility.rolling(periods["long"]).quantile(0.33)
        vol_q67 = volatility.rolling(periods["long"]).quantile(0.67)

        # ボラティリティレジームの分類
        vol_regime = np.zeros(len(volatility))
        vol_regime[volatility > vol_q67] = 2  # 高ボラティリティ
        vol_regime[(volatility >= vol_q33) & (volatility <= vol_q67)] = (
            1  # 中ボラティリティ
        )
        vol_regime[volatility < vol_q33] = 0  # 低ボラティリティ

        result_df["volatility_regime"] = vol_regime

        # 2. トレンドレジーム
        returns = df["Close"].pct_change()
        trend_strength = returns.rolling(periods["medium"]).mean() / (
            returns.rolling(periods["medium"]).std() + 1e-10
        )

        # トレンドレジームの分類
        trend_regime = np.ones(len(trend_strength))  # デフォルトは横ばい(1)
        trend_regime[trend_strength > 0.5] = 2  # 上昇トレンド
        trend_regime[trend_strength < -0.5] = 0  # 下降トレンド

        result_df["trend_regime"] = trend_regime

        # 3. 出来高レジーム
        volume_ma = df["Volume"].rolling(periods["medium"]).mean()
        volume_ratio = df["Volume"] / volume_ma

        result_df["volume_regime"] = (volume_ratio > 1.5).astype(int)

        self.feature_groups["regime_aware"].extend(
            [
                col
                for col in result_df.columns
                if col.startswith(
                    ("volatility_regime", "trend_regime", "volume_regime")
                )
            ]
        )

        return result_df

    def _create_cross_timeframe_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """クロスタイムフレーム特徴量"""
        result_df = df.copy()

        # 1. 多期間価格位置
        for short_period, long_period in [(4, 24), (24, 168)]:
            short_ma = df["Close"].rolling(short_period).mean()
            long_ma = df["Close"].rolling(long_period).mean()

            result_df[f"price_position_{short_period}_{long_period}"] = (
                (df["Close"] - long_ma) / (short_ma - long_ma + 1e-10)
            ).clip(-2, 2)

        # 2. 多期間勢いの一致度
        momentum_1h = df["Close"].pct_change(1)
        momentum_4h = df["Close"].pct_change(4)
        momentum_24h = df["Close"].pct_change(24)

        result_df["momentum_alignment"] = (
            (np.sign(momentum_1h) == np.sign(momentum_4h)).astype(int)
            + (np.sign(momentum_4h) == np.sign(momentum_24h)).astype(int)
        ) / 2.0

        # 3. 時間軸間の発散
        short_trend = (
            df["Close"]
            .rolling(periods["short"])
            .apply(
                lambda x: (
                    np.polyfit(range(len(x)), x, 1)[0]
                    if len(x) == periods["short"]
                    else 0
                )
            )
        )
        long_trend = (
            df["Close"]
            .rolling(periods["long"])
            .apply(
                lambda x: (
                    np.polyfit(range(len(x)), x, 1)[0]
                    if len(x) == periods["long"]
                    else 0
                )
            )
        )

        result_df["timeframe_divergence"] = (short_trend - long_trend).rolling(5).mean()

        self.feature_groups["cross_timeframe"].extend(
            [
                col
                for col in result_df.columns
                if col.startswith(
                    ("price_position_", "momentum_alignment", "timeframe_divergence")
                )
            ]
        )

        return result_df

    def _create_volatility_adjusted_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """ボラティリティ調整済み特徴量"""
        result_df = df.copy()

        # 1. 適応的ボラティリティ
        returns = df["Close"].pct_change()

        # GARCH風の適応的ボラティリティ
        adaptive_vol = returns.rolling(1).std()
        for i in range(1, len(returns)):
            if i > 0:
                adaptive_vol.iloc[i] = (
                    0.1 * returns.iloc[i] ** 2
                    + 0.85 * adaptive_vol.iloc[i - 1]
                    + 0.05 * returns.iloc[max(0, i - 5) : i].var()
                )

        result_df["adaptive_volatility"] = adaptive_vol

        # 2. ボラティリティ調整済みリターン
        result_df["vol_adjusted_return"] = returns / (adaptive_vol + 1e-10)

        # 3. ボラティリティ正規化された特徴量
        for col in ["funding_rate", "open_interest"]:
            if col in df.columns:
                col_vol = df[col].rolling(periods["medium"]).std()
                result_df[f"{col}_vol_normalized"] = (
                    df[col] - df[col].rolling(periods["medium"]).mean()
                ) / (col_vol + 1e-10)

        self.feature_groups["volatility_adjusted"].extend(
            [
                col
                for col in result_df.columns
                if col.startswith(
                    (
                        "adaptive_volatility",
                        "vol_adjusted_",
                        "funding_rate_vol_",
                        "open_interest_vol_",
                    )
                )
            ]
        )

        return result_df

    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """最終的なデータクリーンアップ"""
        result_df = df.copy()

        # 無限値とNaNの処理
        result_df = result_df.replace([np.inf, -np.inf], np.nan)

        # 数値カラムのNaN補完（ロバスト版）
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if result_df[col].isna().any():
                # 中央値で補完
                median_val = result_df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                result_df[col] = result_df[col].fillna(median_val)

        return result_df

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """特徴量グループを取得"""
        return self.feature_groups

    def get_top_features_by_stability(
        self, df: pd.DataFrame, target_col: str, top_n: int = 30
    ) -> List[str]:
        """
        安定性の高い特徴量を取得

        Args:
            df: 特徴量データ
            target_col: ターゲット変数
            top_n: 上位N個

        Returns:
            安定性の高い特徴量のリスト
        """
        if target_col not in df.columns:
            logger.warning(f"ターゲット変数 {target_col} が見つかりません")
            return []

        # 安定性の高い特徴量グループから優先的に選択
        priority_groups = [
            "stable_high_performers",
            "enhanced_technical",
            "robust_temporal",
        ]

        selected_features = []
        for group in priority_groups:
            group_features = self.feature_groups.get(group, [])
            available_features = [f for f in group_features if f in df.columns]
            selected_features.extend(available_features)

        # 他のグループからも追加
        for group, features in self.feature_groups.items():
            if group not in priority_groups:
                available_features = [
                    f
                    for f in features
                    if f in df.columns and f not in selected_features
                ]
                selected_features.extend(available_features)

        return selected_features[:top_n]
