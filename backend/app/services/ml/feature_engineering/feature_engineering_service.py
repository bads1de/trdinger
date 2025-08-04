"""
特徴量エンジニアリングサービス

OHLCV、ファンディングレート（FR）、建玉残高（OI）データを受け取り、
市場の歪みや偏りを捉える高度な特徴量を計算します。

リファクタリング後：責任を分割し、各特徴量計算クラスを統合します。
AutoML機能も統合され、オプションで拡張特徴量計算が可能です。
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from ....utils.data_preprocessing import data_preprocessor
from ....utils.data_validation import DataValidator
from ....utils.unified_error_handler import safe_ml_operation
from .data_frequency_manager import DataFrequencyManager
from .fear_greed_features import FearGreedFeatureCalculator
from .interaction_features import InteractionFeatureCalculator
from .market_data_features import MarketDataFeatureCalculator
from .price_features import PriceFeatureCalculator
from .technical_features import TechnicalFeatureCalculator
from .temporal_features import TemporalFeatureCalculator

# AutoML関連のインポート（オプション）
try:
    from .automl_features.autofeat_calculator import AutoFeatCalculator
    from .automl_features.automl_config import AutoMLConfig
    from .automl_features.performance_optimizer import PerformanceOptimizer
    from .automl_features.tsfresh_calculator import TSFreshFeatureCalculator
    from .enhanced_crypto_features import EnhancedCryptoFeatures
    from .optimized_crypto_features import OptimizedCryptoFeatures

    AUTOML_AVAILABLE = True
except ImportError:
    AUTOML_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeatureEngineeringService:
    """
    特徴量エンジニアリングサービス

    各特徴量計算クラスを統合し、高度な特徴量を生成します。
    単一責任原則に従い、各特徴量タイプの計算は専用クラスに委譲します。
    AutoML機能もオプションで利用可能です。
    """

    def __init__(self, automl_config: Optional["AutoMLConfig"] = None):
        """
        初期化

        Args:
            automl_config: AutoML設定（オプション）
        """
        self.feature_cache = {}
        self.max_cache_size = 10  # 最大キャッシュサイズ
        self.cache_ttl = 3600  # キャッシュ有効期限（秒）

        # 特徴量計算クラスを初期化
        self.price_calculator = PriceFeatureCalculator()
        self.market_data_calculator = MarketDataFeatureCalculator()
        self.technical_calculator = TechnicalFeatureCalculator()
        self.temporal_calculator = TemporalFeatureCalculator()
        self.interaction_calculator = InteractionFeatureCalculator()
        self.fear_greed_calculator = FearGreedFeatureCalculator()

        # データ頻度統一マネージャー
        self.frequency_manager = DataFrequencyManager()

        # AutoML機能の初期化（オプション）
        self.automl_enabled = automl_config is not None and AUTOML_AVAILABLE
        if self.automl_enabled:
            self.automl_config = (
                automl_config or AutoMLConfig.get_financial_optimized_config()
            )

            # AutoML特徴量計算クラス
            self.tsfresh_calculator = TSFreshFeatureCalculator(
                self.automl_config.tsfresh
            )
            self.autofeat_calculator = AutoFeatCalculator(self.automl_config.autofeat)

            # パフォーマンス最適化クラス
            self.performance_optimizer = PerformanceOptimizer()

            # 暗号通貨特化特徴量エンジニアリング
            self.crypto_features = EnhancedCryptoFeatures()

            # 最適化された特徴量エンジニアリング
            self.optimized_features = OptimizedCryptoFeatures()

            # 統計情報
            self.last_enhancement_stats = {}

            logger.info("🤖 AutoML特徴量エンジニアリングが有効化されました")
        else:
            self.automl_config = None
            if automl_config is not None and not AUTOML_AVAILABLE:
                logger.warning(
                    "AutoML設定が指定されましたが、AutoMLモジュールが利用できません"
                )
            logger.info("📊 基本特徴量エンジニアリングを使用します")

    def calculate_advanced_features(
        self,
        ohlcv_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        fear_greed_data: Optional[pd.DataFrame] = None,
        lookback_periods: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """
        高度な特徴量を計算

        Args:
            ohlcv_data: OHLCV価格データ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）
            fear_greed_data: Fear & Greed Index データ（オプション）
            lookback_periods: 各特徴量の計算期間設定

        Returns:
            特徴量が追加されたDataFrame
        """
        try:
            if ohlcv_data.empty:
                raise ValueError("OHLCVデータが空です")

            # メモリ使用量制限
            if len(ohlcv_data) > 50000:
                logger.warning(
                    f"大量のデータ（{len(ohlcv_data)}行）、最新50,000行に制限"
                )
                ohlcv_data = ohlcv_data.tail(50000)

            # キャッシュキーを生成
            cache_key = self._generate_cache_key(
                ohlcv_data,
                funding_rate_data,
                open_interest_data,
                fear_greed_data,
                lookback_periods,
            )

            # キャッシュから結果を取得
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

            # データ頻度統一処理（最優先問題の解決）
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

            # 結果DataFrameを初期化（メモリ効率化）
            result_df = ohlcv_data.copy()

            # データ型を最適化
            result_df = self._optimize_dtypes(result_df)

            # 基本的な価格特徴量
            result_df = self.price_calculator.calculate_price_features(
                result_df, lookback_periods
            )

            # ボラティリティ特徴量
            result_df = self.price_calculator.calculate_volatility_features(
                result_df, lookback_periods
            )

            # 出来高特徴量
            result_df = self.price_calculator.calculate_volume_features(
                result_df, lookback_periods
            )

            # ファンディングレート特徴量（データがある場合）
            if funding_rate_data is not None and not funding_rate_data.empty:
                result_df = self.market_data_calculator.calculate_funding_rate_features(
                    result_df, funding_rate_data, lookback_periods
                )
                # 中間クリーニング
                fr_columns = [
                    "FR_MA_24",
                    "FR_MA_168",
                    "FR_Change",
                    "FR_Change_Rate",
                    "Price_FR_Divergence",
                    "FR_Normalized",
                    "FR_Trend",
                    "FR_Volatility",
                ]
                existing_fr_columns = [
                    col for col in fr_columns if col in result_df.columns
                ]
                if existing_fr_columns:
                    result_df = DataValidator.clean_dataframe(
                        result_df,
                        column_names=existing_fr_columns,
                        fill_method="median",
                    )

            # 建玉残高特徴量（データがある場合）
            if open_interest_data is not None and not open_interest_data.empty:
                result_df = (
                    self.market_data_calculator.calculate_open_interest_features(
                        result_df, open_interest_data, lookback_periods
                    )
                )
                # 中間クリーニング
                oi_columns = [
                    "OI_Change_Rate",
                    "OI_Change_Rate_24h",
                    "OI_Surge",
                    "Volatility_Adjusted_OI",
                    "OI_MA_24",
                    "OI_MA_168",
                    "OI_Trend",
                    "OI_Price_Correlation",
                    "OI_Normalized",
                ]
                existing_oi_columns = [
                    col for col in oi_columns if col in result_df.columns
                ]
                if existing_oi_columns:
                    result_df = DataValidator.clean_dataframe(
                        result_df,
                        column_names=existing_oi_columns,
                        fill_method="median",
                    )

            # 複合特徴量（FR + OI）
            if (
                funding_rate_data is not None
                and not funding_rate_data.empty
                and open_interest_data is not None
                and not open_interest_data.empty
            ):
                result_df = self.market_data_calculator.calculate_composite_features(
                    result_df, funding_rate_data, open_interest_data, lookback_periods
                )
                # 中間クリーニング
                composite_columns = [
                    "FR_OI_Ratio",
                    "Market_Heat_Index",
                    "Market_Stress",
                    "Market_Balance",
                ]
                existing_composite_columns = [
                    col for col in composite_columns if col in result_df.columns
                ]
                if existing_composite_columns:
                    result_df = DataValidator.clean_dataframe(
                        result_df,
                        column_names=existing_composite_columns,
                        fill_method="median",
                    )

            # Fear & Greed Index 特徴量（データがある場合）
            if fear_greed_data is not None and not fear_greed_data.empty:
                result_df = self.fear_greed_calculator.calculate_fear_greed_features(
                    result_df, fear_greed_data, lookback_periods
                )
                # 中間クリーニング
                fear_greed_columns = self.fear_greed_calculator.get_feature_names()
                existing_fg_columns = [
                    col for col in fear_greed_columns if col in result_df.columns
                ]
                if existing_fg_columns:
                    result_df = DataValidator.clean_dataframe(
                        result_df,
                        column_names=existing_fg_columns,
                        fill_method="median",
                    )

            # 市場レジーム特徴量
            result_df = self.technical_calculator.calculate_market_regime_features(
                result_df, lookback_periods
            )

            # モメンタム特徴量
            result_df = self.technical_calculator.calculate_momentum_features(
                result_df, lookback_periods
            )

            # パターン認識特徴量
            result_df = self.technical_calculator.calculate_pattern_features(
                result_df, lookback_periods
            )

            # 時間的特徴量
            result_df = self.temporal_calculator.calculate_temporal_features(result_df)

            # 相互作用特徴量（全ての基本特徴量が計算された後に実行）
            result_df = self.interaction_calculator.calculate_interaction_features(
                result_df
            )

            # データバリデーションとクリーンアップ
            # 主要な価格列を除外した特徴量列の一覧が必要な場合のみ、その場で計算すること
            # （未使用変数を避けるため、ここでは保持しません）

            # 高品質なデータ前処理を実行（スケーリング有効化、IQRベース外れ値検出）
            logger.info("統計的手法による特徴量前処理を実行中...")
            result_df = data_preprocessor.preprocess_features(
                result_df,
                imputation_strategy="median",
                scale_features=True,  # 特徴量スケーリングを有効化
                remove_outliers=True,
                outlier_threshold=3.0,
                scaling_method="robust",  # ロバストスケーリングを使用
                outlier_method="iqr",  # IQRベースの外れ値検出を使用
            )

            logger.info(f"特徴量計算完了: {len(result_df.columns)}個の特徴量を生成")

            # 結果をキャッシュに保存
            self._save_to_cache(cache_key, result_df)

            return result_df

        except Exception as e:
            logger.error(f"特徴量計算エラー: {e}")
            raise

    @safe_ml_operation(
        default_return=None, context="拡張特徴量計算でエラーが発生しました"
    )
    def calculate_enhanced_features(
        self,
        ohlcv_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        fear_greed_data: Optional[pd.DataFrame] = None,
        lookback_periods: Optional[Dict[str, int]] = None,
        automl_config: Optional[Dict] = None,
        target: Optional[pd.Series] = None,
        max_features_per_step: int = 100,
    ) -> pd.DataFrame:
        """
        拡張特徴量を計算（手動 + AutoML）- ステップ・バイ・ステップ方式

        Args:
            ohlcv_data: OHLCV価格データ
            funding_rate_data: ファンディングレートデータ
            open_interest_data: 建玉残高データ
            fear_greed_data: Fear & Greed Index データ
            lookback_periods: 計算期間設定
            automl_config: AutoML設定（辞書形式）
            target: ターゲット変数（特徴量選択用）
            max_features_per_step: 各ステップでの最大特徴量数

        Returns:
            拡張特徴量が追加されたDataFrame
        """
        if not self.automl_enabled:
            logger.warning("AutoML機能が無効です。基本特徴量計算を実行します")
            return self.calculate_advanced_features(
                ohlcv_data=ohlcv_data,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                fear_greed_data=fear_greed_data,
                lookback_periods=lookback_periods,
            )

        if ohlcv_data is None or ohlcv_data.empty:
            logger.warning("空のOHLCVデータが提供されました")
            return ohlcv_data

        start_time = time.time()

        try:
            # AutoML設定の更新
            if automl_config:
                self._update_automl_config(automl_config)

            logger.info("🔄 ステップ・バイ・ステップ特徴量生成を開始")

            # ステップ1: 手動特徴量を計算
            result_df = self._step1_manual_features(
                ohlcv_data,
                funding_rate_data,
                open_interest_data,
                fear_greed_data,
                lookback_periods,
            )

            # ステップ2: TSFresh特徴量を追加 + 特徴量選択
            if self.automl_config.tsfresh.enabled:
                result_df = self._step2_tsfresh_features(
                    result_df, target, max_features_per_step
                )

            # ステップ3: AutoFeat特徴量を追加 + 特徴量選択
            if self.automl_config.autofeat.enabled:
                result_df = self._step3_autofeat_features(
                    result_df, target, max_features_per_step
                )

            # 最終的な特徴量統計を記録
            final_feature_count = len(result_df.columns)
            logger.info(
                f"🎯 ステップ・バイ・ステップ特徴量生成完了: 最終特徴量数 {final_feature_count}個"
            )

            # 統計情報を更新
            total_time = time.time() - start_time
            self.last_enhancement_stats.update(
                {
                    "total_features": final_feature_count,
                    "total_time": total_time,
                    "data_rows": len(result_df),
                    "automl_config_used": self.automl_config.to_dict(),
                    "processing_method": "step_by_step",
                }
            )

            return result_df

        except Exception as e:
            logger.error(f"拡張特徴量計算エラー: {e}")
            raise

    def get_feature_names(self) -> List[str]:
        """
        生成される特徴量名のリストを取得

        Returns:
            特徴量名のリスト
        """
        feature_names = []

        # 各計算クラスから特徴量名を取得
        feature_names.extend(self.price_calculator.get_feature_names())
        feature_names.extend(self.market_data_calculator.get_feature_names())
        feature_names.extend(self.technical_calculator.get_feature_names())
        feature_names.extend(self.temporal_calculator.get_feature_names())
        feature_names.extend(self.interaction_calculator.get_feature_names())
        feature_names.extend(self.fear_greed_calculator.get_feature_names())

        return feature_names

    def _step1_manual_features(
        self,
        ohlcv_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        fear_greed_data: Optional[pd.DataFrame] = None,
        lookback_periods: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """ステップ1: 手動特徴量を計算"""
        logger.info("📊 ステップ1: 手動特徴量を計算中...")
        start_time = time.time()

        result_df = self.calculate_advanced_features(
            ohlcv_data=ohlcv_data,
            funding_rate_data=funding_rate_data,
            open_interest_data=open_interest_data,
            fear_greed_data=fear_greed_data,
            lookback_periods=lookback_periods,
        )

        manual_time = time.time() - start_time
        manual_feature_count = len(result_df.columns)

        # 統計情報を記録
        if hasattr(self, "last_enhancement_stats"):
            self.last_enhancement_stats.update(
                {
                    "manual_features": manual_feature_count,
                    "manual_time": manual_time,
                }
            )

        logger.info(
            f"✅ ステップ1完了: {manual_feature_count}個の手動特徴量 ({manual_time:.2f}秒)"
        )
        return result_df

    def _step2_tsfresh_features(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series],
        max_features: int = 100,
    ) -> pd.DataFrame:
        """ステップ2: TSFresh特徴量を追加 + 特徴量選択"""
        logger.info("🤖 ステップ2: TSFresh特徴量を計算中...")
        start_time = time.time()
        initial_feature_count = len(df.columns)

        # TSFresh特徴量を計算
        result_df = self.tsfresh_calculator.calculate_tsfresh_features(
            df=df,
            target=target,
            feature_selection=self.automl_config.tsfresh.feature_selection,
        )

        # 特徴量数が制限を超えている場合は選択を実行
        if len(result_df.columns) > max_features:
            logger.info(f"特徴量数が制限({max_features})を超過。特徴量選択を実行中...")
            result_df = self._select_top_features(result_df, target, max_features)

        tsfresh_time = time.time() - start_time
        added_features = len(result_df.columns) - initial_feature_count

        # 統計情報を記録
        if hasattr(self, "last_enhancement_stats"):
            self.last_enhancement_stats.update(
                {
                    "tsfresh_features": added_features,
                    "tsfresh_time": tsfresh_time,
                }
            )

        logger.info(
            f"✅ ステップ2完了: {added_features}個のTSFresh特徴量追加 ({tsfresh_time:.2f}秒)"
        )
        return result_df

    def _step3_autofeat_features(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series],
        max_features: int = 100,
    ) -> pd.DataFrame:
        """ステップ3: AutoFeat特徴量を追加 + 特徴量選択"""
        if target is None:
            logger.warning(
                "ターゲット変数がないため、AutoFeat特徴量生成をスキップします"
            )
            return df

        logger.info("🧬 ステップ3: AutoFeat特徴量を計算中...")
        start_time = time.time()
        initial_feature_count = len(df.columns)

        # AutoFeat特徴量を計算
        result_df, generation_info = self.autofeat_calculator.generate_features(
            df=df,
            target=target,
            task_type="regression",
            max_features=self.automl_config.autofeat.max_features,
        )

        # 特徴量数が制限を超えている場合は選択を実行
        if len(result_df.columns) > max_features:
            logger.info(f"特徴量数が制限({max_features})を超過。特徴量選択を実行中...")
            result_df = self._select_top_features(result_df, target, max_features)

        autofeat_time = time.time() - start_time
        added_features = len(result_df.columns) - initial_feature_count

        # 統計情報を記録
        if hasattr(self, "last_enhancement_stats"):
            self.last_enhancement_stats.update(
                {
                    "autofeat_features": added_features,
                    "autofeat_time": autofeat_time,
                }
            )

        logger.info(
            f"✅ ステップ3完了: {added_features}個のAutoFeat特徴量追加 ({autofeat_time:.2f}秒)"
        )
        return result_df

    def _generate_cache_key(
        self,
        ohlcv_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame],
        open_interest_data: Optional[pd.DataFrame],
        fear_greed_data: Optional[pd.DataFrame],
        lookback_periods: Optional[Dict[str, int]],
    ) -> str:
        """
        キャッシュキーを生成

        Args:
            ohlcv_data: OHLCV価格データ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）
            fear_greed_data: Fear & Greed Index データ（オプション）
            lookback_periods: 各特徴量の計算期間設定

        Returns:
            生成されたキャッシュキー文字列
        """
        import hashlib

        # データのハッシュを計算
        ohlcv_hash = hashlib.md5(str(ohlcv_data.shape).encode()).hexdigest()[:8]
        fr_hash = hashlib.md5(
            str(
                funding_rate_data.shape if funding_rate_data is not None else "None"
            ).encode()
        ).hexdigest()[:8]
        oi_hash = hashlib.md5(
            str(
                open_interest_data.shape if open_interest_data is not None else "None"
            ).encode()
        ).hexdigest()[:8]
        fg_hash = hashlib.md5(
            str(
                fear_greed_data.shape if fear_greed_data is not None else "None"
            ).encode()
        ).hexdigest()[:8]
        periods_hash = hashlib.md5(
            str(
                sorted(lookback_periods.items())
                if lookback_periods is not None
                else "None"
            ).encode()
        ).hexdigest()[:8]

        return f"features_{ohlcv_hash}_{fr_hash}_{oi_hash}_{fg_hash}_{periods_hash}"

    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        キャッシュから結果を取得

        Args:
            cache_key: キャッシュキー

        Returns:
            キャッシュされたDataFrame、またはNone
        """
        try:
            if cache_key in self.feature_cache:
                cached_data, timestamp = self.feature_cache[cache_key]

                # TTLチェック
                if datetime.now().timestamp() - timestamp < self.cache_ttl:
                    return cached_data.copy()
                else:
                    # 期限切れのキャッシュを削除
                    del self.feature_cache[cache_key]

            return None

        except Exception as e:
            logger.warning(f"キャッシュ取得エラー: {e}")
            return None

    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """
        結果をキャッシュに保存

        Args:
            cache_key: キャッシュキー
            data: 保存するDataFrame
        """
        try:
            # キャッシュサイズ制限
            if len(self.feature_cache) >= self.max_cache_size:
                # 最も古いキャッシュを削除
                oldest_key = min(
                    self.feature_cache.keys(), key=lambda k: self.feature_cache[k][1]
                )
                del self.feature_cache[oldest_key]

            # 新しいキャッシュを保存
            self.feature_cache[cache_key] = (data.copy(), datetime.now().timestamp())

        except Exception as e:
            logger.warning(f"キャッシュ保存エラー: {e}")

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データ型を最適化してメモリ使用量を削減

        Args:
            df: 最適化するDataFrame

        Returns:
            最適化されたDataFrame
        """
        try:
            optimized_df = df.copy()

            for col in optimized_df.columns:
                if col == "timestamp":
                    continue

                if optimized_df[col].dtype == "float64":
                    # float64をfloat32に変換（精度は十分）
                    optimized_df[col] = optimized_df[col].astype("float32")
                elif optimized_df[col].dtype == "int64":
                    # int64をint32に変換（範囲が十分な場合）
                    if (
                        optimized_df[col].min() >= -2147483648
                        and optimized_df[col].max() <= 2147483647
                    ):
                        optimized_df[col] = optimized_df[col].astype("int32")

            return optimized_df

        except Exception as e:
            logger.warning(f"データ型最適化エラー: {e}")
            return df

    def clear_cache(self):
        """
        キャッシュをクリア
        """
        self.feature_cache.clear()
        logger.info("特徴量キャッシュをクリアしました")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        キャッシュ情報を取得
        """
        return {
            "cache_size": len(self.feature_cache),
            "max_cache_size": self.max_cache_size,
            "cache_ttl": self.cache_ttl,
            "cache_keys": list(self.feature_cache.keys()),
        }

    def _select_top_features(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series],
        max_features: int,
    ) -> pd.DataFrame:
        """特徴量選択を実行して上位特徴量を選択"""
        if target is None or len(df.columns) <= max_features:
            return df

        try:
            from sklearn.feature_selection import SelectKBest, f_regression
            from sklearn.impute import SimpleImputer

            logger.info(f"特徴量選択を実行中: {len(df.columns)} → {max_features}個")

            # 欠損値を補完
            imputer = SimpleImputer(strategy="median")
            X_imputed = imputer.fit_transform(df)

            # 特徴量選択を実行
            selector = SelectKBest(score_func=f_regression, k=max_features)
            X_selected = selector.fit_transform(X_imputed, target)

            # 選択された特徴量のカラム名を取得
            selected_features = df.columns[selector.get_support()]
            result_df = pd.DataFrame(
                X_selected, columns=selected_features, index=df.index
            )

            logger.info(f"特徴量選択完了: {len(selected_features)}個の特徴量を選択")
            return result_df

        except Exception as e:
            logger.warning(f"特徴量選択でエラー: {e}. 元のDataFrameを返します")
            return df

    def _update_automl_config(self, config_dict: Dict[str, Any]):
        """AutoML設定を更新"""
        if not self.automl_enabled:
            logger.warning("AutoML機能が無効のため、設定更新をスキップします")
            return

        try:
            # TSFresh設定の更新
            if "tsfresh" in config_dict:
                tsfresh_config = config_dict["tsfresh"]
                if isinstance(tsfresh_config, dict):
                    for key, value in tsfresh_config.items():
                        if hasattr(self.automl_config.tsfresh, key):
                            setattr(self.automl_config.tsfresh, key, value)

                    # TSFreshCalculatorの設定も更新
                    self.tsfresh_calculator.config = self.automl_config.tsfresh

            # AutoFeat設定の更新
            if "autofeat" in config_dict:
                autofeat_config = config_dict["autofeat"]
                if isinstance(autofeat_config, dict):
                    for key, value in autofeat_config.items():
                        if hasattr(self.automl_config.autofeat, key):
                            setattr(self.automl_config.autofeat, key, value)

                    # AutoFeatCalculatorの設定も更新
                    self.autofeat_calculator.config = self.automl_config.autofeat

            logger.info("AutoML設定を更新しました")

        except Exception as e:
            logger.error(f"AutoML設定更新エラー: {e}")

    def get_enhancement_stats(self) -> Dict[str, Any]:
        """最後の拡張処理の統計情報を取得"""
        if not self.automl_enabled or not hasattr(self, "last_enhancement_stats"):
            return {}
        return self.last_enhancement_stats.copy()

    def get_automl_config(self) -> Dict[str, Any]:
        """現在のAutoML設定を取得"""
        if not self.automl_enabled:
            return {}
        return self.automl_config.to_dict()

    def set_automl_config(self, config: "AutoMLConfig"):
        """AutoML設定を設定"""
        if not self.automl_enabled:
            logger.warning("AutoML機能が無効のため、設定変更をスキップします")
            return

        self.automl_config = config
        self.tsfresh_calculator.config = config.tsfresh

    def get_available_automl_features(self) -> Dict[str, List[str]]:
        """利用可能なAutoML特徴量のリストを取得"""
        if not self.automl_enabled:
            return {}

        return {
            "tsfresh": self.tsfresh_calculator.get_feature_names(),
            "autofeat": self.autofeat_calculator.get_feature_names(),
        }

    def clear_automl_cache(self):
        """AutoML特徴量のキャッシュをクリア"""
        if not self.automl_enabled:
            return

        try:
            self.tsfresh_calculator.clear_cache()
            self.autofeat_calculator.clear_model()

            # 強制ガベージコレクション
            import gc

            collected = gc.collect()

            logger.info(
                f"AutoML特徴量キャッシュをクリアしました（{collected}オブジェクト回収）"
            )
        except Exception as e:
            logger.error(f"AutoMLキャッシュクリアエラー: {e}")

    def cleanup_resources(self):
        """リソースの完全クリーンアップ"""
        try:
            logger.info("FeatureEngineeringServiceのリソースクリーンアップを開始")

            # 基本キャッシュをクリア
            self.clear_cache()

            if self.automl_enabled:
                # AutoMLキャッシュをクリア
                self.clear_automl_cache()

                # 統計情報をクリア
                if hasattr(self, "last_enhancement_stats"):
                    self.last_enhancement_stats.clear()

                # 各計算機のリソースを個別にクリーンアップ
                if hasattr(self.tsfresh_calculator, "cleanup"):
                    self.tsfresh_calculator.cleanup()

                if hasattr(self.autofeat_calculator, "cleanup"):
                    self.autofeat_calculator.cleanup()

                # パフォーマンス最適化クラスのクリーンアップ
                if hasattr(self.performance_optimizer, "cleanup"):
                    self.performance_optimizer.cleanup()

            logger.info("FeatureEngineeringServiceのリソースクリーンアップ完了")

        except Exception as e:
            logger.error(f"FeatureEngineeringServiceクリーンアップエラー: {e}")
