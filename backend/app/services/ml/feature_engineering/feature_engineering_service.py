"""
特徴量エンジニアリングサービス

OHLCV、ファンディングレート（FR）、建玉残高（OI）データを受け取り、
市場の歪みや偏りを捉える高度な特徴量を計算します。

各特徴量計算クラスを統合し、単一責任原則に従って実装されています。
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.unified_config import unified_config

from .crypto_features import CryptoFeatures
from .data_frequency_manager import DataFrequencyManager
from .interaction_features import InteractionFeatureCalculator
from .market_data_features import MarketDataFeatureCalculator
from .price_features import PriceFeatureCalculator
from .technical_features import TechnicalFeatureCalculator
from app.services.ml.common.ml_utils import optimize_dtypes, generate_cache_key


logger = logging.getLogger(__name__)


# デフォルト特徴量リスト（最適化版: 27個）
# 2025-12-02 OI/FR統合評価に基づき更新
# 低重要度特徴量6個を削除: Williams_R, OI_RSI, Market_Stress,
# OI_Weighted_Price_Dev, OI_Trend_Efficiency, OI_Trend_Strength
DEFAULT_FEATURE_ALLOWLIST: Optional[List[str]] = [
    # === 出来高・需給 (Volume/Flow) - 最重要 ===
    "AD",  # Accumulation/Distribution
    "OBV",  # On Balance Volume
    "Volume_MA_20",
    "ADOSC",  # Chaikin A/D Oscillator
    # === モメンタム (Momentum) ===
    "RSI",  # Relative Strength Index
    # Williams_R は削除（評価31位/低重要度）
    "MACD_Histogram",  # MACD Histogram
    # === トレンド (Trend) ===
    "ADX",
    "MA_Long",
    "SMA_Cross_50_200",  # SMA Golden/Death Cross Distance
    "Trend_strength_20",
    # === ボラティリティ (Volatility) ===
    "NATR",  # Normalized ATR
    "Parkinson_Vol_20",
    "Close_range_20",
    "Historical_Volatility_20",
    # === 価格構造・その他 (Price Structure) ===
    "price_vs_low_24h",
    "VWAP_Deviation",  # VWAP Deviation
    "Price_Skewness_20",
    # === 市場データ (Market Data) - OI/FR最適化版 ===
    # OI（建玉）ベース特徴量 - 高重要度のみ採用
    # OI_RSI は削除（評価30位/低重要度）
    "Price_OI_Divergence",  # 評価14位
    "OI_Volume_Correlation",  # 評価6位
    "OI_Momentum_Ratio",  # 評価5位
    "OI_Liquidation_Risk",  # 評価24位
    # FR（資金調達率）ベース特徴量
    "FR_Extremity_Zscore",  # 評価9位
    "FR_Cumulative_Trend",  # 評価22位
    # 複合特徴量 - 高重要度のみ採用
    # Market_Stress は削除（評価31位/低重要度）
    "FR_OI_Sentiment",  # 評価12位
    "Liquidation_Risk",  # 評価19位
    # OI_Weighted_Price_Dev は削除（評価33位/低重要度）
    "FR_Volatility",  # 評価13位
    # OI_Trend_Efficiency は削除（評価34位/低重要度）
    "Volume_OI_Ratio",  # 評価4位 - 最重要OI特徴量
    # === Group B (OI/FR Technicals) ===
    "OI_MACD",
    "OI_MACD_Hist",
    "OI_BB_Position",
    "OI_BB_Width",
    "FR_MACD",
    # === Group C (Market Structure) ===
    "Amihud_Illiquidity",
    "Efficiency_Ratio",
    "Market_Impact",
]


class FeatureEngineeringService:
    """
    特徴量エンジニアリングサービス

    各特徴量計算クラスを統合し、高度な特徴量を生成します。
    単一責任原則に従い、各特徴量タイプの計算は専用クラスに委譲します。
    """

    def __init__(self):
        """初期化"""
        self.feature_cache = {}
        self.max_cache_size = 10  # 最大キャッシュサイズ
        self.cache_ttl = 3600  # キャッシュ有効期限（秒）

        # 特徴量計算クラスを初期化
        self.price_calculator = PriceFeatureCalculator()
        self.market_data_calculator = MarketDataFeatureCalculator()
        self.technical_calculator = TechnicalFeatureCalculator()
        self.interaction_calculator = InteractionFeatureCalculator()

        # データ頻度統一マネージャー
        self.frequency_manager = DataFrequencyManager()

        # 暗号通貨特化特徴量エンジニアリング（デフォルトで有効）
        self.crypto_features = CryptoFeatures()
        logger.debug("暗号通貨特化特徴量を有効化しました")

    def calculate_advanced_features(
        self,
        ohlcv_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        lookback_periods: Optional[Dict[str, int]] = None,
        profile: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        高度な特徴量を計算

        Args:
            ohlcv_data: OHLCV価格データ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）
            lookback_periods: 各特徴量の計算期間設定
            profile: 特徴量プロファイル ('research' または 'production')。
                    Noneの場合は設定から読み込み

        Returns:
            特徴量が追加されたDataFrame
        """
        try:
            if ohlcv_data.empty:
                raise ValueError("OHLCVデータが空です")

            # DataFrameのインデックスをDatetimeIndexに変換
            if not isinstance(ohlcv_data.index, pd.DatetimeIndex):
                if "timestamp" in ohlcv_data.columns:
                    ohlcv_data = ohlcv_data.set_index("timestamp")
                    logger.info("timestampカラムをインデックスに設定しました")
                else:
                    # timestampカラムがない場合は、現在の時刻から生成
                    logger.warning(
                        "timestampカラムが見つからないため、仮のDatetimeIndexを生成します"
                    )
                    ohlcv_data.index = pd.date_range(
                        start="2024-01-01", periods=len(ohlcv_data), freq="1H"
                    )

            # インデックスがDatetimeIndexであることを確認
            if not isinstance(ohlcv_data.index, pd.DatetimeIndex):
                raise ValueError(
                    "DataFrameのインデックスはDatetimeIndexである必要があります"
                )

            # メモリ使用量制限
            if len(ohlcv_data) > 200000:
                logger.warning(
                    f"大量のデータ（{len(ohlcv_data)}行）、最新200,000行に制限"
                )
                ohlcv_data = ohlcv_data.tail(200000)

            # キャッシュキーを生成
            cache_key = generate_cache_key(
                ohlcv_data,
                funding_rate_data,
                open_interest_data,
                extra_params=lookback_periods,
            )

            # キャッシュから結果を取得
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

            # データ頻度統一処理
            logger.info("データ頻度統一処理を開始")
            ohlcv_timeframe = self.frequency_manager.detect_ohlcv_timeframe(ohlcv_data)

            # データ整合性検証
            validation_result = self.frequency_manager.validate_data_alignment(
                ohlcv_data, funding_rate_data, open_interest_data
            )

            if not validation_result["is_valid"]:
                logger.warning("データ整合性に問題があります:")
                for error in validation_result["errors"]:
                    logger.warning(f"  エラー: {error}")

            # データ頻度を統一
            funding_rate_data, open_interest_data = (
                self.frequency_manager.align_data_frequencies(
                    ohlcv_data, funding_rate_data, open_interest_data, ohlcv_timeframe
                )
            )

            # デフォルトの計算期間
            if lookback_periods is None:
                lookback_periods = {
                    "short_ma": 10,
                    "long_ma": 50,
                    "volatility": 20,
                    "momentum": 14,
                    "volume": 20,
                }

            # 結果DataFrameを初期化
            result_df = ohlcv_data.copy()

            # データ型を最適化
            result_df = optimize_dtypes(result_df)

            # 価格特徴量
            result_df = self.price_calculator.calculate_features(
                result_df, {"lookback_periods": lookback_periods}
            )

            # テクニカル特徴量
            result_df = self.technical_calculator.calculate_features(
                result_df, {"lookback_periods": lookback_periods}
            )

            # 市場データ特徴量
            config = {
                "lookback_periods": lookback_periods,
                "funding_rate_data": funding_rate_data,
                "open_interest_data": open_interest_data,
            }
            result_df = self.market_data_calculator.calculate_features(
                result_df, config
            )

            # 暗号通貨特化特徴量
            if self.crypto_features is not None:
                logger.debug("暗号通貨特化特徴量を計算中...")
                result_df = self.crypto_features.create_crypto_features(
                    result_df, funding_rate_data, open_interest_data
                )

            # 相互作用特徴量
            result_df = self.interaction_calculator.calculate_interaction_features(
                result_df
            )

            # データ前処理
            logger.info("統計的手法による特徴量前処理を実行中...")
            try:
                numeric_columns = result_df.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    # 無限大値をNaNに変換
                    result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan)
                    # NaN値を前方補完
                    result_df[col] = result_df[col].ffill()
                    # 残りを0で埋める
                    result_df[col] = result_df[col].fillna(0.0)
                logger.info("データ前処理完了")
            except Exception as e:
                logger.warning(f"データ前処理エラー: {e}")
                result_df = result_df.fillna(0.0)

            # 重複カラムの削除
            has_duplicates = bool(result_df.columns.duplicated().any())
            if has_duplicates:
                duplicated_cols = result_df.columns[
                    result_df.columns.duplicated()
                ].tolist()
                logger.warning(
                    f"重複カラムを検出、最初のもののみ保持: {duplicated_cols}"
                )
                result_df = result_df.loc[
                    :, ~result_df.columns.duplicated(keep="first")
                ]

            logger.info(f"特徴量計算完了: {len(result_df.columns)}個の特徴量を生成")

            # クリップ
            numeric_columns = result_df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                result_df[col] = np.clip(result_df[col], -1e10, 1e10)

            # フィルタリング
            result_df = self._apply_feature_profile(result_df, profile)

            # キャッシュ保存
            self._save_to_cache(cache_key, result_df)

            return result_df

        except Exception as e:
            logger.error(f"特徴量計算エラー: {e}")
            raise

    def get_feature_names(self) -> List[str]:
        """
        生成される特徴量名のリストを取得

        Returns:
            特徴量名のリスト
        """
        feature_names = []
        feature_names.extend(self.price_calculator.get_feature_names())
        feature_names.extend(self.market_data_calculator.get_feature_names())
        feature_names.extend(self.technical_calculator.get_feature_names())
        feature_names.extend(self.interaction_calculator.get_feature_names())
        return feature_names

    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """キャッシュから結果を取得"""
        try:
            if cache_key in self.feature_cache:
                cached_data, timestamp = self.feature_cache[cache_key]
                if datetime.now().timestamp() - timestamp < self.cache_ttl:
                    return cached_data.copy()
                else:
                    del self.feature_cache[cache_key]
            return None
        except Exception as e:
            logger.warning(f"キャッシュ取得エラー: {e}")
            return None

    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """結果をキャッシュに保存"""
        try:
            if len(self.feature_cache) >= self.max_cache_size:
                oldest_key = min(
                    self.feature_cache.keys(), key=lambda k: self.feature_cache[k][1]
                )
                del self.feature_cache[oldest_key]
            self.feature_cache[cache_key] = (data.copy(), datetime.now().timestamp())
        except Exception as e:
            logger.warning(f"キャッシュ保存エラー: {e}")

    def _apply_feature_profile(
        self, df: pd.DataFrame, profile: Optional[str] = None
    ) -> pd.DataFrame:
        """特徴量フィルタリングを適用"""
        try:
            allowlist = unified_config.ml.feature_engineering.feature_allowlist
            if allowlist is None:
                allowlist = DEFAULT_FEATURE_ALLOWLIST
                if allowlist is None:
                    logger.info(f"全特徴量を使用: {len(df.columns)}個")
                    return df
                else:
                    logger.info(f"デフォルト特徴量リストを適用: {len(allowlist)}個")
            else:
                logger.info(f"カスタム特徴量リストを適用: {len(allowlist)}個")

            essential_columns = ["open", "high", "low", "close", "volume"]
            columns_to_keep = []
            for col in essential_columns:
                if col in df.columns:
                    columns_to_keep.append(col)

            missing_features = []
            for feature in allowlist:
                if feature in df.columns:
                    if feature not in columns_to_keep:
                        columns_to_keep.append(feature)
                else:
                    missing_features.append(feature)

            if missing_features:
                logger.warning(
                    f"allowlistに指定された{len(missing_features)}個の特徴量が "
                    f"見つかりません: {missing_features[:10]}"
                )

            original_count = len(df.columns)
            filtered_df = df[columns_to_keep]
            dropped_count = original_count - len(filtered_df.columns)

            logger.info(
                f"特徴量フィルタリング完了: "
                f"{original_count}個 → {len(filtered_df.columns)}個の特徴量 "
                f"({dropped_count}個をドロップ)"
            )
            return filtered_df

        except Exception as e:
            logger.error(f"特徴量プロファイル適用エラー: {e}")
            return df

    def clear_cache(self):
        """キャッシュをクリア"""
        self.feature_cache.clear()
        logger.info("特徴量キャッシュをクリアしました")

    def _generate_pseudo_open_interest_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """建玉残高疑似特徴量を生成"""
        try:
            return self.market_data_calculator.calculate_pseudo_open_interest_features(
                df, lookback_periods
            )
        except Exception as e:
            logger.error(f"建玉残高疑似特徴量生成エラー: {e}")
            return df

    def cleanup_resources(self):
        """リソースのクリーンアップ"""
        try:
            logger.info("FeatureEngineeringServiceのリソースクリーンアップを開始")
            self.clear_cache()
            logger.info("FeatureEngineeringServiceのリソースクリーンアップ完了")
        except Exception as e:
            logger.error(f"FeatureEngineeringServiceクリーンアップエラー: {e}")
