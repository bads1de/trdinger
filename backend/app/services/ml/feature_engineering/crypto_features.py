"""
暗号通貨特化の特徴量エンジニアリング

実際の取引データの特性を考慮した、効果的な特徴量を生成します。
OHLCV、OI、FR、FGデータの期間不一致を適切に処理し、
予測精度の向上に寄与する特徴量を作成します。
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_feature_calculator import BaseFeatureCalculator
from ....utils.data_processing import data_processor as data_preprocessor

logger = logging.getLogger(__name__)


class CryptoFeatureCalculator(BaseFeatureCalculator):
    """暗号通貨特化の特徴量エンジニアリング（BaseFeatureCalculator継承）"""

    def __init__(self):
        """初期化"""
        super().__init__()
        self.feature_groups = {
            "price": [],
            "volume": [],
            "open_interest": [],
            "funding_rate": [],
            "technical": [],
            "composite": [],
            "temporal": [],
        }

    def calculate_features(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        暗号通貨特徴量を計算（BaseFeatureCalculatorの抽象メソッド実装）

        Args:
            df: OHLCV価格データ
            config: 計算設定（lookback_periodsを含む）

        Returns:
            暗号通貨特徴量が追加されたDataFrame
        """
        lookback_periods = config.get("lookback_periods", {})

        # create_crypto_featuresを呼び出し（後方互換性のため）
        result_df = self.create_crypto_features(df)

        return result_df

    def _ensure_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """データ品質を確保"""
        result_df = df.copy()

        # 必要なカラムの存在確認と補完
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in result_df.columns:
                logger.warning(f"必須カラム {col} が見つかりません")
                return result_df

        # オプショナルカラムの補完
        if "open_interest" not in result_df.columns:
            result_df["open_interest"] = 0
        if "funding_rate" not in result_df.columns:
            result_df["funding_rate"] = 0

        # 統計的手法による補完
        optional_columns = ["open_interest", "funding_rate"]
        for col in optional_columns:
            if col in result_df.columns and result_df[col].isna().any():
                result_df[col] = result_df[col].fillna(result_df[col].median())

        return result_df

    def _create_price_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """価格関連特徴量"""
        result_df = df.copy()

        # 新しい特徴量を辞書で収集（DataFrame断片化対策）
        new_features = {}

        # 基本価格特徴量
        new_features["price_range"] = (df["high"] - df["low"]) / df["close"]
        new_features["upper_shadow"] = (
            df["high"] - np.maximum(df["open"], df["close"])
        ) / df["close"]
        new_features["lower_shadow"] = (
            np.minimum(df["open"], df["close"]) - df["low"]
        ) / df["close"]
        new_features["body_size"] = abs(df["close"] - df["open"]) / df["close"]

        # 価格変動率（複数期間）
        for name, period in periods.items():
            new_features[f"price_return_{name}"] = df["close"].pct_change(period)
            new_features[f"price_volatility_{name}"] = (
                df["close"].rolling(period).std() / df["close"].rolling(period).mean()
            )

            # 価格勢い
            new_features[f"price_momentum_{name}"] = (
                df["close"] / df["close"].shift(period) - 1
            )

        # 価格レベル特徴量
        for period in [24, 168]:  # 1日、1週間
            new_features[f"price_vs_high_{period}h"] = (
                df["close"] / df["high"].rolling(period).max()
            )
            new_features[f"price_vs_low_{period}h"] = (
                df["close"] / df["low"].rolling(period).min()
            )

        # 一括で結合（DataFrame断片化回避）
        result_df = pd.concat([result_df, pd.DataFrame(new_features, index=result_df.index)], axis=1)

        # feature_groups更新
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

        # 新しい特徴量を辞書で収集（DataFrame断片化対策）
        new_features = {}

        # 出来高変動率
        for name, period in periods.items():
            new_features[f"volume_change_{name}"] = df["volume"].pct_change(period)
            new_features[f"volume_ma_{name}"] = df["volume"].rolling(period).mean()
            new_features[f"volume_ratio_{name}"] = (
                df["volume"] / df["volume"].rolling(period).mean()
            )

        # VWAP（出来高加重平均価格）
        for period in [12, 24, 48]:
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            vwap = (typical_price * df["volume"]).rolling(period).sum() / df[
                "volume"
            ].rolling(period).sum()
            new_features[f"vwap_{period}h"] = vwap
            new_features[f"price_vs_vwap_{period}h"] = (df["close"] - vwap) / vwap

        # 出来高プロファイル
        close_rolling_mean = df["close"].rolling(24).mean()
        volume_rolling = df["volume"].rolling(24)
        new_features["volume_price_trend"] = volume_rolling.corr(close_rolling_mean).fillna(0.0)  # type: ignore

        # 一括で結合（DataFrame断片化回避）
        result_df = pd.concat([result_df, pd.DataFrame(new_features, index=result_df.index)], axis=1)

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
            df["open_interest"].pct_change() - df["close"].pct_change()
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
            np.sign(df["funding_rate"]) == np.sign(df["close"].pct_change())
        ).astype(int)

        self.feature_groups["funding_rate"].extend(
            [col for col in result_df.columns if col.startswith("fr_")]
        )

        return result_df

    def _create_technical_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """テクニカル指標"""
        result_df = df.copy()

        # RSI（pandas-ta使用）
        import pandas_ta as ta

        for period in [14, 24]:
            rsi_result = ta.rsi(df["close"], length=period)
            if rsi_result is not None:
                result_df[f"rsi_{period}"] = rsi_result.fillna(50.0)
            else:
                result_df[f"rsi_{period}"] = 50.0

        # ボリンジャーバンド（pandas-ta使用）
        for period in [20, 48]:
            bb_result = ta.bbands(df["close"], length=period, std=2)
            if bb_result is not None:
                result_df[f"bb_upper_{period}"] = bb_result[f"BBU_{period}_2.0"].fillna(
                    df["close"]
                )
                result_df[f"bb_lower_{period}"] = bb_result[f"BBL_{period}_2.0"].fillna(
                    df["close"]
                )
                # BB Position計算
                bb_width = (
                    result_df[f"bb_upper_{period}"] - result_df[f"bb_lower_{period}"]
                )
                result_df[f"bb_position_{period}"] = (
                    (
                        (df["close"] - result_df[f"bb_lower_{period}"])
                        / (bb_width + 1e-10)
                    )
                    .clip(0, 1)
                    .fillna(0.5)
                )
            else:
                result_df[f"bb_upper_{period}"] = df["close"]
                result_df[f"bb_lower_{period}"] = df["close"]
                result_df[f"bb_position_{period}"] = 0.5

        # MACD（pandas-ta使用）
        macd_result = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd_result is not None:
            result_df["macd"] = macd_result["MACD_12_26_9"].fillna(0.0)
            result_df["macd_signal"] = macd_result["MACDs_12_26_9"].fillna(0.0)
            result_df["macd_histogram"] = macd_result["MACDh_12_26_9"].fillna(0.0)
        else:
            result_df["macd"] = 0.0
            result_df["macd_signal"] = 0.0
            result_df["macd_histogram"] = 0.0

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
            df["close"].rolling(24).corr(df["open_interest"])
        )

        # 出来高 vs 価格変動の関係
        result_df["volume_price_efficiency"] = df["volume"] / (
            abs(df["close"].pct_change()) + 1e-10
        )

        # マルチファクター勢い
        price_momentum = df["close"].pct_change(24)
        oi_momentum = df["open_interest"].pct_change(24)
        result_df["multi_momentum"] = (price_momentum + oi_momentum) / 2

        # 市場ストレス指標（FRデータがある場合のみ）
        if "fr_abs" in result_df.columns:
            result_df["market_stress"] = (
                result_df["price_volatility_short"] * result_df["fr_abs"]
            )
        else:
            # FRデータがない場合、出来高で代替
            result_df["market_stress"] = result_df["price_volatility_short"] * result_df["volume"].mean()

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

        # DatetimeIndexの確認
        if not isinstance(result_df.index, pd.DatetimeIndex):
            logger.warning(
                "インデックスがDatetimeIndexではありません。時間関連特徴量をスキップします。"
            )
            return result_df

        # 時間帯
        try:
            hour = getattr(result_df.index, "hour", 0)  # type: ignore
            dayofweek = getattr(result_df.index, "dayofweek", 0)  # type: ignore
            day = getattr(result_df.index, "day", 15)  # type: ignore

            result_df["hour"] = hour
            result_df["day_of_week"] = dayofweek
            result_df["is_weekend"] = (pd.Series(dayofweek) >= 5).astype(int)

            # 地域別取引時間
            result_df["asia_hours"] = (
                (pd.Series(hour) >= 0) & (pd.Series(hour) < 8)
            ).astype(int)
            result_df["europe_hours"] = (
                (pd.Series(hour) >= 8) & (pd.Series(hour) < 16)
            ).astype(int)
            result_df["us_hours"] = (
                (pd.Series(hour) >= 16) & (pd.Series(hour) < 24)
            ).astype(int)

            # 月末・月初効果
            result_df["month_end"] = (pd.Series(day) >= 28).astype(int)
            result_df["month_start"] = (pd.Series(day) <= 3).astype(int)
        except (AttributeError, TypeError) as e:
            logger.warning(f"時間関連特徴量の生成でエラー: {e}")
            result_df["hour"] = 0
            result_df["day_of_week"] = 0
            result_df["is_weekend"] = 0
            result_df["asia_hours"] = 0
            result_df["europe_hours"] = 0
            result_df["us_hours"] = 0
            result_df["month_end"] = 0
            result_df["month_start"] = 0

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
        for col in numeric_cols:
            try:
                # Seriesであることを確認
                if isinstance(result_df[col], pd.Series) and result_df[col].isna().sum() > 0:
                    result_df[col] = result_df[col].fillna(result_df[col].median())
            except (ValueError, TypeError) as e:
                # エラーが発生した列はスキップ
                logger.warning(f"カラム {col} の補完でエラー: {e}")

        return result_df

    def create_crypto_features(
        self,
        df: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """暗号通貨特化特徴量を生成するパブリックメソッド"""
        logger.info("暗号通貨特化特徴量を計算中...")

        # デフォルトの計算期間
        periods = {
            "short": 14,
            "medium": 24,
            "long": 72,
            "very_long": 168,
        }

        result_df = self._ensure_data_quality(df)

        # 各特徴量グループを計算
        result_df = self._create_price_features(result_df, periods)
        result_df = self._create_volume_features(result_df, periods)

        if open_interest_data is not None and not open_interest_data.empty:
            result_df = self._create_open_interest_features(result_df, periods)

        if funding_rate_data is not None and not funding_rate_data.empty:
            result_df = self._create_funding_rate_features(result_df, periods)

        result_df = self._create_technical_features(result_df, periods)

        # FRデータがある場合のみFR特徴量を計算
        if funding_rate_data is not None and not funding_rate_data.empty:
            result_df = self._create_funding_rate_features(result_df, periods)
        else:
            logger.debug("FRデータがないため、FR特徴量をスキップ")

        result_df = self._create_composite_features(result_df, periods)
        result_df = self._create_temporal_features(result_df)

        # 特徴量をクリーニング
        result_df = self._clean_features(result_df)

        # 統計情報を記録
        feature_count = len(result_df.columns) - len(df.columns)
        logger.info(f"暗号通貨特化特徴量を追加: {feature_count}個")

        return result_df
# 後方互換性のためのクラス別名
CryptoFeatures = CryptoFeatureCalculator
