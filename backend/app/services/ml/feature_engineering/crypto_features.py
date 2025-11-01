"""
暗号通貨特化の特徴量エンジニアリング

実際の取引データの特性を考慮した、効果的な特徴量を生成します。
OHLCV、OI、FR、FGデータの期間不一致を適切に処理し、
予測精度の向上に寄与する特徴量を作成します。
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from ....utils.data_processing import data_processor as data_preprocessor

logger = logging.getLogger(__name__)


class CryptoFeatures:
    """暗号通貨特化の特徴量エンジニアリング"""

    def __init__(self):
        """初期化"""
        self.feature_groups = {
            "price": [],
            "volume": [],
            "open_interest": [],
            "funding_rate": [],
            "technical": [],
            "composite": [],
            "temporal": [],
        }

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
        result_df = data_preprocessor.interpolate_columns(
            result_df, strategy="median", columns=optional_columns
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
        close_rolling_mean = df["Close"].rolling(24).mean()
        volume_rolling = df["Volume"].rolling(24)
        result_df["volume_price_trend"] = volume_rolling.corr(close_rolling_mean).fillna(0.0)  # type: ignore

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

    def _create_technical_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """テクニカル指標"""
        result_df = df.copy()

        # RSI（pandas-ta使用）
        import pandas_ta as ta

        for period in [14, 24]:
            rsi_result = ta.rsi(df["Close"], length=period)
            if rsi_result is not None:
                result_df[f"rsi_{period}"] = rsi_result.fillna(50.0)
            else:
                result_df[f"rsi_{period}"] = 50.0

        # ボリンジャーバンド（pandas-ta使用）
        for period in [20, 48]:
            bb_result = ta.bbands(df["Close"], length=period, std=2)
            if bb_result is not None:
                result_df[f"bb_upper_{period}"] = bb_result[f"BBU_{period}_2.0"].fillna(
                    df["Close"]
                )
                result_df[f"bb_lower_{period}"] = bb_result[f"BBL_{period}_2.0"].fillna(
                    df["Close"]
                )
                # BB Position計算
                bb_width = (
                    result_df[f"bb_upper_{period}"] - result_df[f"bb_lower_{period}"]
                )
                result_df[f"bb_position_{period}"] = (
                    (
                        (df["Close"] - result_df[f"bb_lower_{period}"])
                        / (bb_width + 1e-10)
                    )
                    .clip(0, 1)
                    .fillna(0.5)
                )
            else:
                result_df[f"bb_upper_{period}"] = df["Close"]
                result_df[f"bb_lower_{period}"] = df["Close"]
                result_df[f"bb_position_{period}"] = 0.5

        # MACD（pandas-ta使用）
        macd_result = ta.macd(df["Close"], fast=12, slow=26, signal=9)
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
            result_df["price_volatility_short"] * result_df["fr_abs"]
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
        result_df = data_preprocessor.interpolate_columns(
            result_df, strategy="median", columns=numeric_cols
        )

        return result_df