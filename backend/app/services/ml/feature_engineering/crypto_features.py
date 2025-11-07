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
        """価格関連特徴量（PriceFeaturesと重複しない暗号通貨特化特徴量のみ）"""
        result_df = df.copy()

        # 新しい特徴量を辞書で収集（DataFrame断片化対策）
        new_features = {}

        # 価格レベル特徴量（暗号通貨特化）- 24hのみに削減（1週間は冗長）
        new_features["price_vs_high_24h"] = df["close"] / df["high"].rolling(24).max()
        new_features["price_vs_low_24h"] = df["close"] / df["low"].rolling(24).min()

        # 一括で結合（DataFrame断片化回避）
        result_df = pd.concat(
            [result_df, pd.DataFrame(new_features, index=result_df.index)], axis=1
        )

        # feature_groups更新
        self.feature_groups["price"].extend(
            [col for col in result_df.columns if col.startswith("price_vs_")]
        )

        return result_df

    def _create_volume_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """出来高関連特徴量（主要な期間のみに削減）"""
        result_df = df.copy()

        # 新しい特徴量を辞書で収集（DataFrame断片化対策）
        new_features = {}

        # 出来高変動率（主要な期間のみ：short, medium）
        for name in ["short", "medium"]:
            if name in periods:
                period = periods[name]
                new_features[f"volume_change_{name}"] = df["volume"].pct_change(period)
                new_features[f"volume_ratio_{name}"] = (
                    df["volume"] / df["volume"].rolling(period).mean()
                )

        # VWAP（24hのみ - 最も重要）
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap_24 = (typical_price * df["volume"]).rolling(24).sum() / df[
            "volume"
        ].rolling(24).sum()
        new_features["vwap_24h"] = vwap_24
        new_features["price_vs_vwap_24h"] = (df["close"] - vwap_24) / vwap_24

        # 出来高プロファイル
        close_rolling_mean = df["close"].rolling(24).mean()
        volume_rolling = df["volume"].rolling(24)
        new_features["volume_price_trend"] = volume_rolling.corr(close_rolling_mean).fillna(0.0)  # type: ignore

        # 一括で結合（DataFrame断片化回避）
        result_df = pd.concat(
            [result_df, pd.DataFrame(new_features, index=result_df.index)], axis=1
        )

        self.feature_groups["volume"].extend(
            [col for col in result_df.columns if col.startswith(("volume_", "vwap_"))]
        )

        return result_df

    def _create_open_interest_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """建玉残高関連特徴量（重要な特徴のみ）"""
        result_df = df.copy()

        # 新しい特徴量を辞書で収集
        new_features = {}

        # OI変動率（medium期間のみ）
        if "medium" in periods:
            new_features["oi_change_medium"] = df["open_interest"].pct_change(
                periods["medium"]
            )

        # OI vs 価格の関係（最も重要）
        new_features["oi_price_divergence"] = (
            df["open_interest"].pct_change() - df["close"].pct_change()
        )

        # OI勢い（24hのみ - 最も有用）
        new_features["oi_momentum_24h"] = (
            df["open_interest"]
            .rolling(24)
            .apply(
                lambda x: (
                    (x.iloc[-1] - x.iloc[0]) / x.iloc[0]
                    if len(x) > 0 and x.iloc[0] != 0
                    else 0
                )
            )
        )

        # 一括で結合
        result_df = pd.concat(
            [result_df, pd.DataFrame(new_features, index=result_df.index)], axis=1
        )

        self.feature_groups["open_interest"].extend(
            [col for col in result_df.columns if col.startswith("oi_")]
        )

        return result_df

    def _create_funding_rate_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """
        ファンディングレート関連特徴量（新設計: Tier 1特徴量）

        新しいFundingRateFeatureCalculatorを使用してTier 1特徴量（15個）を生成：
        - 基本金利指標（4個）
        - 時間サイクル（3個）
        - モメンタム（3個）
        - レジーム（2個）
        - 価格相互作用（2個）
        """
        from .funding_rate_features import FundingRateFeatureCalculator

        # funding_rateカラムが存在しない場合はスキップ
        if "funding_rate" not in df.columns:
            logger.debug("funding_rateカラムが見つからないため、FR特徴量をスキップ")
            return df

        # timestampカラムの確認
        if "timestamp" not in df.columns:
            logger.warning("timestampカラムが見つからないため、FR特徴量をスキップ")
            return df

        # ファンディングレートデータを抽出
        funding_df = df[["timestamp", "funding_rate"]].copy()

        # FundingRateFeatureCalculatorを使用してTier 1特徴量を計算
        fr_calculator = FundingRateFeatureCalculator()
        result_df = fr_calculator.calculate_features(df, funding_df)

        # feature_groupsにFR特徴量を追加
        fr_features = [col for col in result_df.columns if col.startswith("fr_")]
        self.feature_groups["funding_rate"].extend(fr_features)

        logger.info(f"ファンディングレート特徴量を追加: {len(fr_features)}個")

        return result_df

    def _create_technical_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """テクニカル指標（標準期間のみ）"""
        result_df = df.copy()

        # 新しい特徴量を辞書で収集
        new_features = {}

        # RSI（14期間のみ - 標準）
        import pandas_ta as ta

        rsi_result = ta.rsi(df["close"], length=14)
        if rsi_result is not None:
            new_features["rsi_14"] = rsi_result.fillna(50.0)
        else:
            new_features["rsi_14"] = 50.0

        # ボリンジャーバンド（20期間のみ - 標準）
        bb_result = ta.bbands(df["close"], length=20, std=2)
        if bb_result is not None:
            new_features["bb_upper_20"] = bb_result["BBU_20_2.0"].fillna(df["close"])
            new_features["bb_lower_20"] = bb_result["BBL_20_2.0"].fillna(df["close"])
            # BB Position計算
            bb_width = new_features["bb_upper_20"] - new_features["bb_lower_20"]
            new_features["bb_position_20"] = (
                ((df["close"] - new_features["bb_lower_20"]) / (bb_width + 1e-10))
                .clip(0, 1)
                .fillna(0.5)
            )
        else:
            new_features["bb_upper_20"] = df["close"]
            new_features["bb_lower_20"] = df["close"]
            new_features["bb_position_20"] = 0.5

        # MACD（pandas-ta使用）
        macd_result = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd_result is not None:
            new_features["macd"] = macd_result["MACD_12_26_9"].fillna(0.0)
            new_features["macd_signal"] = macd_result["MACDs_12_26_9"].fillna(0.0)
            new_features["macd_histogram"] = macd_result["MACDh_12_26_9"].fillna(0.0)
        else:
            new_features["macd"] = 0.0
            new_features["macd_signal"] = 0.0
            new_features["macd_histogram"] = 0.0

        # 一括で結合
        result_df = pd.concat(
            [result_df, pd.DataFrame(new_features, index=result_df.index)], axis=1
        )

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
        """複合特徴量（相互作用）- 高寄与度のみ残す"""
        result_df = df.copy()

        # 出来高 vs 価格変動の関係（高寄与度: 0.0019）
        result_df["volume_price_efficiency"] = df["volume"] / (
            abs(df["close"].pct_change()) + 1e-10
        )

        # 削除された特徴量（低寄与度）:
        # - price_oi_correlation (スコア: 0.0)
        # - multi_momentum (スコア: -1.33e-17)
        # - market_stress (スコア: 0.0)

        self.feature_groups["composite"].extend(
            [col for col in result_df.columns if col.startswith("volume_price_")]
        )

        return result_df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """時間関連特徴量（削除: 暗号通貨市場では24時間取引で時間効果が弱い）"""
        # 分析結果: 時間・セッション関連特徴量は全て極めて低い寄与度のため削除
        # 削除された特徴量: hour, day_of_week, is_weekend, asia_hours, us_hours
        # 理由: 暗号通貨は24時間取引で地域別効果が弱い
        return df

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
                if (
                    isinstance(result_df[col], pd.Series)
                    and result_df[col].isna().sum() > 0
                ):
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

        # FRデータがある場合のみFR特徴量を計算（重複呼び出しを削除）
        if funding_rate_data is not None and not funding_rate_data.empty:
            result_df = self._create_funding_rate_features(result_df, periods)
        else:
            logger.debug("FRデータがないため、FR特徴量をスキップ")

        result_df = self._create_technical_features(result_df, periods)
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
