"""
テクニカル指標特徴量計算クラス

従来のテクニカル指標（RSI、MACD、ストキャスティクスなど）と
テクニカル特徴量を計算します。
"""

import logging
from typing import Any, Dict


import pandas as pd

from ...indicators.technical_indicators.momentum import MomentumIndicators
from ...indicators.technical_indicators.trend import TrendIndicators
from ...indicators.technical_indicators.volatility import VolatilityIndicators
from .base_feature_calculator import BaseFeatureCalculator

logger = logging.getLogger(__name__)


class TechnicalFeatureCalculator(BaseFeatureCalculator):
    """
    テクニカル指標特徴量計算クラス

    従来のテクニカル指標特徴量を計算します。
    """

    def __init__(self):
        """初期化"""
        super().__init__()

    def calculate_features(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        テクニカル特徴量を計算（BaseFeatureCalculatorの抽象メソッド実装）

        Args:
            df: OHLCV価格データ
            config: 計算設定（lookback_periodsを含む）

        Returns:
            テクニカル特徴量が追加されたDataFrame
        """
        lookback_periods = config.get("lookback_periods", {})

        # 複数のテクニカル特徴量を順次計算（全パターン生成）
        result_df = self.calculate_market_regime_features(df, lookback_periods)
        result_df = self.calculate_momentum_features(result_df, lookback_periods)
        result_df = self.calculate_pattern_features(result_df, lookback_periods)

        return result_df

    def calculate_market_regime_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int] = None
    ) -> pd.DataFrame:
        """
        市場レジーム特徴量を計算

        Args:
            df: OHLCV価格データ
            lookback_periods: 計算期間設定（オプション、旧API互換用）

        Returns:
            市場レジーム特徴量が追加されたDataFrame
        """
        # 旧API互換：lookback_periodsがNoneの場合はデフォルト値を設定
        if lookback_periods is None:
            lookback_periods = {"short_ma": 10, "long_ma": 50, "volatility": 20}

        try:
            if not self.validate_input_data(df, ["close", "high", "low"]):
                return df

            result_df = self.create_result_dataframe(df)

            # 新しい特徴量を辞書で収集（DataFrame断片化対策）
            new_features = {}

            # Removed: 低寄与度特徴量削除（LightGBM+XGBoost統合分析: 2025-01-05）
            # 削除された特徴量: Trend_Strength
            # 性能への影響: LightGBM -0.43%, XGBoost -0.43%（許容範囲内）

            # レンジ相場判定（pandas MAX/MIN使用）
            volatility_period = lookback_periods.get("volatility", 20)
            high_vals = result_df["high"]
            low_vals = result_df["low"]
            high_20 = high_vals.rolling(window=volatility_period).max()
            low_20 = low_vals.rolling(window=volatility_period).min()

            new_features["Range_Bound_Ratio"] = self.safe_ratio_calculation(
                result_df["close"] - low_20, high_20 - low_20, fill_value=0.5
            )

            # Removed: 低寄与度特徴量削除（LightGBM+XGBoost統合分析: 2025-01-05）
            # 削除された特徴量: Breakout_Strength
            # 性能への影響: LightGBM -0.43%, XGBoost -0.43%（許容範囲内）

            # 市場効率性（価格のランダムウォーク度）- 最適化版
            # Vectorized computation of price autocorrelation
            returns = result_df["close"].pct_change(fill_method=None).fillna(0)
            returns_lag1 = returns.shift(1)

            # Rolling correlation計算（pandas native使用、lambda回避）
            # Note: Series.rolling().corr()は効率的にCython実装
            new_features["Market_Efficiency"] = (
                returns.rolling(window=volatility_period, min_periods=3)
                .corr(
                    returns_lag1.rolling(window=volatility_period, min_periods=3).mean()
                )
                .fillna(0.0)
            )

            # 一括で結合（DataFrame断片化回避）
            result_df = pd.concat(
                [result_df, pd.DataFrame(new_features, index=result_df.index)], axis=1
            )

            return result_df

        except Exception as e:
            logger.error(f"市場レジーム特徴量計算エラー: {e}")
            return df

    def calculate_pattern_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int] = None
    ) -> pd.DataFrame:
        """
        パターン特徴量を計算（TDDで追加されたメソッド）

        Args:
            df: OHLCV価格データ
            lookback_periods: 計算期間設定（オプション、旧API互換用）

        Returns:
            パターン特徴量が追加されたDataFrame
        """
        # 旧API互換：lookback_periodsがNoneの場合はデフォルト値を設定
        if lookback_periods is None:
            lookback_periods = {"short_ma": 10, "long_ma": 50}

        try:
            if not self.validate_input_data(df, ["close", "high", "low", "open"]):
                return df

            result_df = self.create_result_dataframe(df)

            # カラム名を小文字に統一（大文字小文字の混在対応）
            result_df.columns = [col.lower() for col in result_df.columns]

            # 新しい特徴量を辞書で収集（DataFrame断片化対策）
            new_features = {}

            # ドージ・ストキャスティクス（過熱・過売判断）
            # MomentumIndicatorsを使用
            stoch_k, stoch_d = MomentumIndicators.stoch(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                k=14,
                d=3,
                smooth_k=3,
            )

            new_features["Stochastic_K"] = stoch_k.fillna(50.0)
            new_features["Stochastic_D"] = stoch_d.fillna(50.0)

            # ドージ・ストキャスティクス（KとDの乖離）
            new_features["Stochastic_Divergence"] = (
                new_features["Stochastic_K"] - new_features["Stochastic_D"]
            ).fillna(0.0)

            # ボリンジャーバンド（サポート・レジスタンス）
            # VolatilityIndicatorsを使用
            bb_upper, bb_middle, bb_lower = VolatilityIndicators.bbands(
                result_df["close"], length=20, std=2.0
            )

            new_features["BB_Upper"] = bb_upper.fillna(result_df["close"])
            new_features["BB_Middle"] = bb_middle.fillna(result_df["close"])
            new_features["BB_Lower"] = bb_lower.fillna(result_df["close"])

            # ボリンジャーバンドからの乖離率
            new_features["BB_Position"] = self.safe_ratio_calculation(
                result_df["close"] - new_features["BB_Lower"],
                new_features["BB_Upper"] - new_features["BB_Lower"],
                fill_value=0.5,
            )

            # Removed: MA_Short（重複特徴量削除: 2025-01-09）
            # 理由: price_features.pyのma_10と重複

            # 移動平均（トレンド判断）- MA_Longのみ保持
            # TrendIndicatorsを使用
            long_ma = lookback_periods.get("long_ma", 50)
            ma_long = TrendIndicators.sma(result_df["close"], length=long_ma)
            new_features["MA_Long"] = ma_long.fillna(result_df["close"])
            # 削除: MA_Cross - 理由: ほぼゼロの重要度（分析日: 2025-01-07）

            # ボラティリティパターン（ATRを使用）
            # VolatilityIndicatorsを使用
            atr_values = VolatilityIndicators.atr(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                length=14,
            )
            new_features["ATR"] = atr_values.fillna(0.0)

            # Removed: 低寄与度特徴量削除（LightGBM+XGBoost統合分析: 2025-01-05）
            # 削除された特徴量: Normalized_Volatility
            # 性能への影響: LightGBM -0.43%, XGBoost -0.43%（許容範囲内）

            # 一括で結合
            result_df = pd.concat(
                [result_df, pd.DataFrame(new_features, index=result_df.index)], axis=1
            )

            # 価格パターン（ダブルボトム、ヘッドアンドショルダー等の簡易検出）
            result_df = self._detect_price_patterns(result_df)

            return result_df

        except Exception as e:
            logger.error(f"パターン特徴量計算エラー: {e}")
            return df

    def _detect_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        簡易的な価格パターン検出

        Args:
            df: 価格データが含まれるDataFrame

        Returns:
            パターン特徴量が追加されたDataFrame
        """
        try:
            # 新しい特徴量を辞書で収集（DataFrame断片化対策）
            new_features = {}

            # Removed: Local_Min, Local_Max, Resistance_Level
            # (低寄与度特徴量削除: 2025-11-13)
            # これらの特徴量はモデルの予測精度向上に寄与しないため削除

            # 簡易的なサポート・レジスタントレベル（計算用に残す）
            window_size = 20
            support_level = df["close"].rolling(window=window_size, min_periods=1).min()
            resistance_level = (
                df["close"].rolling(window=window_size, min_periods=1).max()
            )

            # 価格がサポート/レジスタントに近いことを示す特徴量
            new_features["Near_Support"] = self.safe_ratio_calculation(
                df["close"] - support_level,
                resistance_level - support_level,
                fill_value=0.5,
            )
            new_features["Near_Resistance"] = self.safe_ratio_calculation(
                resistance_level - df["close"],
                resistance_level - support_level,
                fill_value=0.5,
            )

            # 一括で結合
            new_df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

            return new_df

        except Exception as e:
            logger.error(f"価格パターン検出エラー: {e}")
            # エラー時は元のDataFrameを返す
            return df

    def calculate_momentum_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int] = None
    ) -> pd.DataFrame:
        """
        モメンタム特徴量を計算

        Args:
            df: OHLCV価格データ
            lookback_periods: 計算期間設定（オプション、旧API互換用）

        Returns:
            モメンタム特徴量が追加されたDataFrame
        """
        # 旧API互換：lookback_periodsがNoneの場合はデフォルト値を設定
        if lookback_periods is None:
            lookback_periods = {"short_ma": 10, "long_ma": 50}

        try:
            if not self.validate_input_data(df, ["close", "high", "low"]):
                return df

            result_df = self.create_result_dataframe(df)

            # 新しい特徴量を辞書で収集（DataFrame断片化対策）
            new_features = {}

            # RSI（MomentumIndicators使用）
            new_features["RSI"] = MomentumIndicators.rsi(
                result_df["close"], period=14
            ).fillna(50.0)

            # MACD（MomentumIndicators使用）
            macd, signal, hist = MomentumIndicators.macd(
                result_df["close"], fast=12, slow=26, signal=9
            )
            new_features["MACD"] = macd.fillna(0.0)
            new_features["MACD_Signal"] = signal.fillna(0.0)
            new_features["MACD_Histogram"] = hist.fillna(0.0)

            # ウィリアムズ%R（MomentumIndicators使用）
            new_features["Williams_R"] = MomentumIndicators.willr(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                length=14,
            ).fillna(-50.0)

            # CCI（MomentumIndicators使用）
            new_features["CCI"] = MomentumIndicators.cci(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                length=20,
            ).fillna(0.0)

            # ROC（MomentumIndicators使用）
            new_features["ROC"] = MomentumIndicators.roc(
                result_df["close"], period=12
            ).fillna(0.0)

            # モメンタム（MomentumIndicators使用）
            new_features["Momentum"] = MomentumIndicators.mom(
                result_df["close"], length=10
            ).fillna(0.0)

            # 一括で結合
            result_df = pd.concat(
                [result_df, pd.DataFrame(new_features, index=result_df.index)], axis=1
            )

            return result_df

        except Exception as e:
            logger.error(f"モメンタム特征量計算エラー: {e}")
            return df

    def get_feature_names(self) -> list:
        """
        生成されるテクニカル特徴量名のリストを取得

        Returns:
            特徴量名のリスト
        """
        return [
            # 市場レジーム特徴量
            # Removed: "Trend_Strength" (低寄与度特徴量削除: 2025-01-05)
            "Range_Bound_Ratio",
            # Removed: "Breakout_Strength" (低寄与度特徴量削除: 2025-01-05)
            "Market_Efficiency",
            # モメンタム特徴量
            "RSI",
            "MACD",
            "MACD_Signal",
            "MACD_Histogram",
            "Stochastic_K",
            "Stochastic_D",
            "Williams_R",
            "CCI",
            "ROC",
            "Momentum",
        ]


# 互換性のための別名（旧名: TechnicalFeatureEngineer）
