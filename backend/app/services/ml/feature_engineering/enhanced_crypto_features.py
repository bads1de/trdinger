"""
暗号通貨特化の高度特徴量エンジニアリング

実際の取引データの特性を考慮した、効果的な特徴量を生成します。
OHLCV、OI、FR、FGデータの期間不一致を適切に処理し、
予測精度の向上に寄与する特徴量を作成します。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from ....utils.data_preprocessing import data_preprocessor


logger = logging.getLogger(__name__)


class EnhancedCryptoFeatures:
    """暗号通貨特化の高度特徴量エンジニアリング"""

    def __init__(self):
        """初期化"""
        self.feature_groups = {
            "price": [],
            "volume": [],
            "open_interest": [],
            "funding_rate": [],
            "fear_greed": [],
            "technical": [],
            "composite": [],
            "temporal": [],
        }

    def create_comprehensive_features(
        self, df: pd.DataFrame, lookback_periods: Optional[Dict[str, int]] = None
    ) -> pd.DataFrame:
        """
        包括的な特徴量を作成

        Args:
            df: 基本データ（OHLCV + OI + FR + FG）
            lookback_periods: 計算期間の設定

        Returns:
            特徴量が追加されたDataFrame
        """
        if lookback_periods is None:
            lookback_periods = {
                "short": 4,  # 4時間
                "medium": 24,  # 1日
                "long": 168,  # 1週間
            }

        logger.info("包括的な特徴量作成を開始")
        result_df = df.copy()

        # データ品質チェック
        result_df = self._ensure_data_quality(result_df)

        # 各グループの特徴量を作成
        result_df = self._create_price_features(result_df, lookback_periods)
        result_df = self._create_volume_features(result_df, lookback_periods)
        result_df = self._create_open_interest_features(result_df, lookback_periods)
        result_df = self._create_funding_rate_features(result_df, lookback_periods)
        result_df = self._create_fear_greed_features(result_df, lookback_periods)
        result_df = self._create_technical_features(result_df, lookback_periods)
        result_df = self._create_composite_features(result_df, lookback_periods)
        result_df = self._create_temporal_features(result_df)

        # 無限値とNaNの処理
        result_df = self._clean_features(result_df)

        logger.info(f"特徴量作成完了: {len(result_df.columns)}個の特徴量")
        return result_df

    def _ensure_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """データ品質を確保"""
        result_df = df.copy()

        # 必要なカラムの存在確認と補完
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in result_df.columns:
                logger.warning(f"必須カラム {col} が見つかりません")
                return result_df

        # オプショナルカラムの補完
        if "open_interest" not in result_df.columns:
            result_df["open_interest"] = 0
        if "funding_rate" not in result_df.columns:
            result_df["funding_rate"] = 0
        if "fear_greed_value" not in result_df.columns:
            result_df["fear_greed_value"] = 50  # 中立値

        # 統計的手法による補完
        optional_columns = ["open_interest", "funding_rate", "fear_greed_value"]
        result_df = data_preprocessor.transform_missing_values(
            result_df,
            strategy="median",
            columns=optional_columns,
            fit_if_needed=True
        )

        return result_df

    def _create_price_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """価格関連特徴量"""
        result_df = df.copy()

        # 基本価格特徴量
        result_df["price_range"] = (df["High"] - df["Low"]) / df["Close"]
        result_df["upper_shadow"] = (
            df["High"] - np.maximum(df["Open"], df["Close"])
        ) / df["Close"]
        result_df["lower_shadow"] = (
            np.minimum(df["Open"], df["Close"]) - df["Low"]
        ) / df["Close"]
        result_df["body_size"] = abs(df["Close"] - df["Open"]) / df["Close"]

        # 価格変動率（複数期間）
        for name, period in periods.items():
            result_df[f"price_return_{name}"] = df["Close"].pct_change(period)
            result_df[f"price_volatility_{name}"] = (
                df["Close"].rolling(period).std() / df["Close"].rolling(period).mean()
            )

            # 価格勢い
            result_df[f"price_momentum_{name}"] = (
                df["Close"] / df["Close"].shift(period) - 1
            )

        # 価格レベル特徴量
        for period in [24, 168]:  # 1日、1週間
            result_df[f"price_vs_high_{period}h"] = (
                df["Close"] / df["High"].rolling(period).max()
            )
            result_df[f"price_vs_low_{period}h"] = (
                df["Close"] / df["Low"].rolling(period).min()
            )

        self.feature_groups["price"].extend(
            [
                col
                for col in result_df.columns
                if col.startswith(("price_", "upper_", "lower_", "body_"))
            ]
        )

        return result_df

    def _create_volume_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """出来高関連特徴量"""
        result_df = df.copy()

        # 出来高変動率
        for name, period in periods.items():
            result_df[f"volume_change_{name}"] = df["Volume"].pct_change(period)
            result_df[f"volume_ma_{name}"] = df["Volume"].rolling(period).mean()
            result_df[f"volume_ratio_{name}"] = (
                df["Volume"] / df["Volume"].rolling(period).mean()
            )

        # VWAP（出来高加重平均価格）
        for period in [12, 24, 48]:
            typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
            vwap = (typical_price * df["Volume"]).rolling(period).sum() / df[
                "Volume"
            ].rolling(period).sum()
            result_df[f"vwap_{period}h"] = vwap
            result_df[f"price_vs_vwap_{period}h"] = (df["Close"] - vwap) / vwap

        # 出来高プロファイル
        result_df["volume_price_trend"] = (
            df["Volume"].rolling(24).corr(df["Close"].rolling(24).mean())
        )

        self.feature_groups["volume"].extend(
            [col for col in result_df.columns if col.startswith(("volume_", "vwap_"))]
        )

        return result_df

    def _create_open_interest_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """建玉残高関連特徴量"""
        result_df = df.copy()

        # OI変動率
        for name, period in periods.items():
            result_df[f"oi_change_{name}"] = df["open_interest"].pct_change(period)

        # OI vs 価格の関係
        result_df["oi_price_divergence"] = (
            df["open_interest"].pct_change() - df["Close"].pct_change()
        )

        # OI勢い
        for period in [24, 72, 168]:
            result_df[f"oi_momentum_{period}h"] = (
                df["open_interest"]
                .rolling(period)
                .apply(
                    lambda x: (
                        (x.iloc[-1] - x.iloc[0]) / x.iloc[0]
                        if len(x) > 0 and x.iloc[0] != 0
                        else 0
                    )
                )
            )

        # OI正規化
        result_df["oi_normalized"] = (
            df["open_interest"] / df["open_interest"].rolling(168).mean()
        )

        self.feature_groups["open_interest"].extend(
            [col for col in result_df.columns if col.startswith("oi_")]
        )

        return result_df

    def _create_funding_rate_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """ファンディングレート関連特徴量"""
        result_df = df.copy()

        # FR基本特徴量
        result_df["fr_abs"] = df["funding_rate"].abs()
        result_df["fr_change"] = df["funding_rate"].diff()

        # FR累積（トレンドの強さ）
        for period in [24, 72, 168]:
            result_df[f"fr_cumsum_{period}h"] = df["funding_rate"].rolling(period).sum()
            result_df[f"fr_mean_{period}h"] = df["funding_rate"].rolling(period).mean()

        # FR極値検出
        fr_quantiles = df["funding_rate"].quantile([0.05, 0.95])
        result_df["fr_extreme_positive"] = (
            df["funding_rate"] > fr_quantiles[0.95]
        ).astype(int)
        result_df["fr_extreme_negative"] = (
            df["funding_rate"] < fr_quantiles[0.05]
        ).astype(int)

        # FR vs 価格の関係
        result_df["fr_price_alignment"] = (
            np.sign(df["funding_rate"]) == np.sign(df["Close"].pct_change())
        ).astype(int)

        self.feature_groups["funding_rate"].extend(
            [col for col in result_df.columns if col.startswith("fr_")]
        )

        return result_df

    def _create_fear_greed_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """Fear & Greed関連特徴量"""
        result_df = df.copy()

        # FG変動
        result_df["fg_change"] = df["fear_greed_value"].diff()
        result_df["fg_change_24h"] = df["fear_greed_value"].diff(24)

        # FG極値
        result_df["fg_extreme_fear"] = (df["fear_greed_value"] <= 25).astype(int)
        result_df["fg_extreme_greed"] = (df["fear_greed_value"] >= 75).astype(int)
        result_df["fg_neutral"] = (
            (df["fear_greed_value"] > 40) & (df["fear_greed_value"] < 60)
        ).astype(int)

        # FG正規化
        result_df["fg_normalized"] = (df["fear_greed_value"] - 50) / 50

        # FG vs 価格の逆相関
        result_df["fg_price_contrarian"] = (
            (df["fear_greed_value"] < 30) & (df["Close"].pct_change() > 0)
        ).astype(int)

        self.feature_groups["fear_greed"].extend(
            [col for col in result_df.columns if col.startswith("fg_")]
        )

        return result_df

    def _create_technical_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """テクニカル指標"""
        result_df = df.copy()

        # RSI
        for period in [14, 24]:
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)  # ゼロ除算回避
            result_df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # ボリンジャーバンド
        for period in [20, 48]:
            ma = df["Close"].rolling(period).mean()
            std = df["Close"].rolling(period).std()
            result_df[f"bb_upper_{period}"] = ma + (std * 2)
            result_df[f"bb_lower_{period}"] = ma - (std * 2)
            result_df[f"bb_position_{period}"] = (
                (df["Close"] - (ma - std * 2)) / (std * 4)
            ).clip(0, 1)

        # MACD
        ema12 = df["Close"].ewm(span=12).mean()
        ema26 = df["Close"].ewm(span=26).mean()
        result_df["macd"] = ema12 - ema26
        result_df["macd_signal"] = result_df["macd"].ewm(span=9).mean()
        result_df["macd_histogram"] = result_df["macd"] - result_df["macd_signal"]

        self.feature_groups["technical"].extend(
            [
                col
                for col in result_df.columns
                if col.startswith(("rsi_", "bb_", "macd"))
            ]
        )

        return result_df

    def _create_composite_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """複合特徴量（相互作用）"""
        result_df = df.copy()

        # 価格 vs OI の関係
        result_df["price_oi_correlation"] = (
            df["Close"].rolling(24).corr(df["open_interest"])
        )

        # 出来高 vs 価格変動の関係
        result_df["volume_price_efficiency"] = df["Volume"] / (
            abs(df["Close"].pct_change()) + 1e-10
        )

        # マルチファクター勢い
        price_momentum = df["Close"].pct_change(24)
        oi_momentum = df["open_interest"].pct_change(24)
        result_df["multi_momentum"] = (price_momentum + oi_momentum) / 2

        # 市場ストレス指標
        result_df["market_stress"] = (
            result_df["price_volatility_short"]
            * result_df["fr_abs"]
            * (1 - result_df["fg_normalized"].abs())
        )

        self.feature_groups["composite"].extend(
            [
                col
                for col in result_df.columns
                if col.startswith(("price_oi_", "volume_price_", "multi_", "market_"))
            ]
        )

        return result_df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """時間関連特徴量"""
        result_df = df.copy()

        # 時間帯
        result_df["hour"] = result_df.index.hour
        result_df["day_of_week"] = result_df.index.dayofweek
        result_df["is_weekend"] = (result_df.index.dayofweek >= 5).astype(int)

        # 地域別取引時間
        result_df["asia_hours"] = (
            (result_df["hour"] >= 0) & (result_df["hour"] < 8)
        ).astype(int)
        result_df["europe_hours"] = (
            (result_df["hour"] >= 8) & (result_df["hour"] < 16)
        ).astype(int)
        result_df["us_hours"] = (
            (result_df["hour"] >= 16) & (result_df["hour"] < 24)
        ).astype(int)

        # 月末・月初効果
        result_df["month_end"] = (result_df.index.day >= 28).astype(int)
        result_df["month_start"] = (result_df.index.day <= 3).astype(int)

        self.feature_groups["temporal"].extend(
            [
                col
                for col in result_df.columns
                if col.startswith(
                    ("hour", "day_", "is_", "asia_", "europe_", "us_", "month_")
                )
            ]
        )

        return result_df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量のクリーニング"""
        result_df = df.copy()

        # 無限値をNaNに変換
        result_df = result_df.replace([np.inf, -np.inf], np.nan)

        # 統計的手法による数値カラムの補完
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
        result_df = data_preprocessor.transform_missing_values(
            result_df,
            strategy="median",
            columns=numeric_cols,
            fit_if_needed=True
        )

        return result_df

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """特徴量グループを取得"""
        return self.feature_groups

    def get_top_features_by_correlation(
        self, df: pd.DataFrame, target_col: str, top_n: int = 20
    ) -> List[str]:
        """
        相関の高い特徴量を取得

        Args:
            df: 特徴量データ
            target_col: ターゲット変数
            top_n: 上位N個

        Returns:
            上位特徴量のリスト
        """
        if target_col not in df.columns:
            logger.warning(f"ターゲット変数 {target_col} が見つかりません")
            return []

        # 数値カラムのみ
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_col]

        # 相関計算
        correlations = []
        for col in feature_cols:
            corr = df[col].corr(df[target_col])
            if not pd.isna(corr):
                correlations.append((col, abs(corr)))

        # 相関順にソート
        correlations.sort(key=lambda x: x[1], reverse=True)

        return [col for col, _ in correlations[:top_n]]
