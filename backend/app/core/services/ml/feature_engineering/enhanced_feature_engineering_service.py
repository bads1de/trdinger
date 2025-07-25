"""
拡張特徴量エンジニアリングサービス

既存の手動特徴量生成システムにAutoML特徴量を統合します。
"""

import logging
from typing import Dict, Optional, Any, List
import pandas as pd
import time

from .feature_engineering_service import FeatureEngineeringService
from .automl_features.tsfresh_calculator import TSFreshFeatureCalculator
from .automl_features.featuretools_calculator import FeaturetoolsCalculator
from .automl_features.autofeat_calculator import AutoFeatCalculator
from .automl_features.automl_config import AutoMLConfig, TSFreshConfig
from .automl_features.performance_optimizer import PerformanceOptimizer
from ....utils.unified_error_handler import safe_ml_operation

logger = logging.getLogger(__name__)


class EnhancedFeatureEngineeringService(FeatureEngineeringService):
    """
    拡張特徴量エンジニアリングサービス

    既存の手動特徴量にAutoML特徴量を追加します。
    """

    def __init__(self, automl_config: Optional[AutoMLConfig] = None):
        """
        初期化

        Args:
            automl_config: AutoML設定
        """
        super().__init__()

        # AutoML設定
        self.automl_config = (
            automl_config or AutoMLConfig.get_financial_optimized_config()
        )

        # AutoML特徴量計算クラス
        self.tsfresh_calculator = TSFreshFeatureCalculator(self.automl_config.tsfresh)
        self.featuretools_calculator = FeaturetoolsCalculator(
            self.automl_config.featuretools
        )
        self.autofeat_calculator = AutoFeatCalculator(self.automl_config.autofeat)

        # パフォーマンス最適化クラス
        self.performance_optimizer = PerformanceOptimizer()

        # 統計情報
        self.last_enhancement_stats = {}

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
    ) -> pd.DataFrame:
        """
        拡張特徴量を計算（手動 + AutoML）

        Args:
            ohlcv_data: OHLCV価格データ
            funding_rate_data: ファンディングレートデータ
            open_interest_data: 建玉残高データ
            fear_greed_data: Fear & Greed Index データ
            lookback_periods: 計算期間設定
            automl_config: AutoML設定（辞書形式）
            target: ターゲット変数（特徴量選択用）

        Returns:
            拡張特徴量が追加されたDataFrame
        """
        if ohlcv_data is None or ohlcv_data.empty:
            logger.warning("空のOHLCVデータが提供されました")
            return ohlcv_data

        start_time = time.time()

        try:
            # AutoML設定の更新
            if automl_config:
                self._update_automl_config(automl_config)

            # 1. 既存の手動特徴量を計算
            logger.info("手動特徴量を計算中...")
            manual_start_time = time.time()

            result_df = self.calculate_advanced_features(
                ohlcv_data=ohlcv_data,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                fear_greed_data=fear_greed_data,
                lookback_periods=lookback_periods,
            )

            manual_time = time.time() - manual_start_time
            manual_feature_count = len(result_df.columns)
            logger.info(
                f"手動特徴量生成完了: {manual_feature_count}個 ({manual_time:.2f}秒)"
            )

            # 2. TSFresh特徴量を追加
            tsfresh_feature_count = 0
            tsfresh_time = 0

            if self.automl_config.tsfresh.enabled:
                logger.info("TSFresh特徴量を計算中...")
                tsfresh_start_time = time.time()

                result_df = self.tsfresh_calculator.calculate_tsfresh_features(
                    df=result_df,
                    target=target,
                    feature_selection=self.automl_config.tsfresh.feature_selection,
                )

                tsfresh_time = time.time() - tsfresh_start_time
                tsfresh_feature_count = len(result_df.columns) - manual_feature_count
                logger.info(
                    f"TSFresh特徴量追加完了: {tsfresh_feature_count}個 ({tsfresh_time:.2f}秒)"
                )

            # 3. Featuretools特徴量を追加
            featuretools_feature_count = 0
            featuretools_time = 0

            if self.automl_config.featuretools.enabled:
                logger.info("Featuretools DFS特徴量を計算中...")
                featuretools_start_time = time.time()

                current_feature_count = len(result_df.columns)

                result_df = (
                    self.featuretools_calculator.calculate_featuretools_features(
                        df=result_df,
                        max_depth=self.automl_config.featuretools.max_depth,
                        max_features=self.automl_config.featuretools.max_features,
                    )
                )

                featuretools_time = time.time() - featuretools_start_time
                featuretools_feature_count = (
                    len(result_df.columns) - current_feature_count
                )
                logger.info(
                    f"Featuretools特徴量追加完了: {featuretools_feature_count}個 ({featuretools_time:.2f}秒)"
                )

            # 4. AutoFeat特徴量生成を実行
            autofeat_feature_count = 0
            autofeat_time = 0

            if self.automl_config.autofeat.enabled and target is not None:
                logger.info("AutoFeat自動特徴量生成を実行中...")
                autofeat_start_time = time.time()

                current_feature_count = len(result_df.columns)

                # データサイズに基づくメモリ推奨設定を取得
                data_size_mb = result_df.memory_usage(deep=True).sum() / 1024 / 1024
                memory_recommendations = (
                    self.performance_optimizer.get_memory_recommendations(
                        data_size_mb, len(result_df.columns)
                    )
                )

                logger.info(
                    f"データサイズ: {data_size_mb:.2f}MB, メモリ推奨設定: {memory_recommendations}"
                )

                # メモリ監視付きでAutoFeat特徴量生成を実行
                with self.performance_optimizer.monitor_memory_usage(
                    "AutoFeat特徴量生成"
                ):
                    # AutoFeatCalculatorをコンテキストマネージャーとして使用（メモリリーク防止）
                    with self.autofeat_calculator as calculator:
                        # AutoFeatで特徴量を生成
                        generated_df, generation_info = calculator.generate_features(
                            df=result_df,
                            target=target,
                            task_type="regression",  # デフォルトは回帰
                            max_features=self.automl_config.autofeat.max_features,
                        )

                        # 生成された特徴量で置き換え
                        if "error" not in generation_info:
                            result_df = generated_df
                            autofeat_feature_count = generation_info.get(
                                "generated_features", 0
                            )

                            # 生成情報を統計に追加
                            self.last_enhancement_stats["autofeat_generation_info"] = (
                                generation_info
                            )

                        # AutoFeatモデルのメモリクリーンアップ
                        self.performance_optimizer.cleanup_autofeat_memory(
                            calculator.autofeat_model
                        )

                autofeat_time = time.time() - autofeat_start_time
                logger.info(
                    f"AutoFeat特徴量生成完了: {autofeat_feature_count}個追加 ({autofeat_time:.2f}秒)"
                )

            # 5. 統計情報を保存
            total_time = time.time() - start_time
            total_features = len(result_df.columns)

            self.last_enhancement_stats.update(
                {
                    "manual_features": manual_feature_count,
                    "tsfresh_features": tsfresh_feature_count,
                    "featuretools_features": featuretools_feature_count,
                    "autofeat_features": autofeat_feature_count,
                    "total_features": total_features,
                    "manual_time": manual_time,
                    "tsfresh_time": tsfresh_time,
                    "featuretools_time": featuretools_time,
                    "autofeat_time": autofeat_time,
                    "total_time": total_time,
                    "data_rows": len(result_df),
                    "automl_config_used": self.automl_config.to_dict(),
                }
            )

            logger.info(
                f"拡張特徴量生成完了: 総計{total_features}個の特徴量 "
                f"(手動:{manual_feature_count}, TSFresh:{tsfresh_feature_count}, "
                f"Featuretools:{featuretools_feature_count}, AutoFeat:{autofeat_feature_count}) "
                f"処理時間:{total_time:.2f}秒"
            )

            # 最終的なメモリクリーンアップ
            self.performance_optimizer.force_garbage_collection()

            # メモリ統計をログ出力
            memory_stats = self.performance_optimizer._get_memory_usage()
            logger.info(f"特徴量生成完了時メモリ使用量: {memory_stats:.2f}MB")

            return result_df

        except Exception as e:
            logger.error(f"拡張特徴量計算エラー: {e}")
            # エラー時は手動特徴量のみ返す
            return self.calculate_advanced_features(
                ohlcv_data=ohlcv_data,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                fear_greed_data=fear_greed_data,
                lookback_periods=lookback_periods,
            )

    def _update_automl_config(self, config_dict: Dict[str, Any]):
        """AutoML設定を更新"""
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

            # Featuretools設定の更新
            if "featuretools" in config_dict:
                featuretools_config = config_dict["featuretools"]
                if isinstance(featuretools_config, dict):
                    for key, value in featuretools_config.items():
                        if hasattr(self.automl_config.featuretools, key):
                            setattr(self.automl_config.featuretools, key, value)

                    # FeaturetoolsCalculatorの設定も更新
                    self.featuretools_calculator.config = (
                        self.automl_config.featuretools
                    )

            # AutoFeat設定の更新
            if "autofeat" in config_dict:
                autofeat_config = config_dict["autofeat"]
                if isinstance(autofeat_config, dict):
                    for key, value in autofeat_config.items():
                        if hasattr(self.automl_config.autofeat, key):
                            setattr(self.automl_config.autofeat, key, value)

                    # AutoFeatCalculatorの設定も更新
                    self.autofeat_calculator.config = self.automl_config.autofeat

            logger.debug("AutoML設定を更新しました")

        except Exception as e:
            logger.error(f"AutoML設定更新エラー: {e}")

    def get_enhancement_stats(self) -> Dict[str, Any]:
        """最後の拡張処理の統計情報を取得"""
        return self.last_enhancement_stats.copy()

    def get_automl_config(self) -> Dict[str, Any]:
        """現在のAutoML設定を取得"""
        return self.automl_config.to_dict()

    def set_automl_config(self, config: AutoMLConfig):
        """AutoML設定を設定"""
        self.automl_config = config
        self.tsfresh_calculator.config = config.tsfresh

    def get_available_automl_features(self) -> Dict[str, List[str]]:
        """利用可能なAutoML特徴量のリストを取得"""
        return {
            "tsfresh": self.tsfresh_calculator.get_feature_names(),
            "featuretools": self.featuretools_calculator.get_feature_names(),
            "autofeat": self.autofeat_calculator.get_feature_names(),
        }

    def clear_automl_cache(self):
        """AutoML特徴量のキャッシュをクリア"""
        self.tsfresh_calculator.clear_cache()
        self.featuretools_calculator.clear_entityset()
        self.autofeat_calculator.clear_model()
        logger.info("AutoML特徴量キャッシュをクリアしました")

    def validate_automl_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """AutoML設定の妥当性を検証"""
        validation_result = {"valid": True, "errors": [], "warnings": []}

        try:
            # TSFresh設定の検証
            if "tsfresh" in config_dict:
                tsfresh_config = config_dict["tsfresh"]

                if "feature_count_limit" in tsfresh_config:
                    limit = tsfresh_config["feature_count_limit"]
                    if not isinstance(limit, int) or limit <= 0:
                        validation_result["errors"].append(
                            "feature_count_limitは正の整数である必要があります"
                        )
                    elif limit > 500:
                        validation_result["warnings"].append(
                            "feature_count_limitが大きすぎます。処理時間が長くなる可能性があります"
                        )

                if "fdr_level" in tsfresh_config:
                    fdr = tsfresh_config["fdr_level"]
                    if not isinstance(fdr, (int, float)) or not 0 < fdr < 1:
                        validation_result["errors"].append(
                            "fdr_levelは0と1の間の数値である必要があります"
                        )

                if "parallel_jobs" in tsfresh_config:
                    jobs = tsfresh_config["parallel_jobs"]
                    if not isinstance(jobs, int) or jobs <= 0:
                        validation_result["errors"].append(
                            "parallel_jobsは正の整数である必要があります"
                        )
                    elif jobs > 8:
                        validation_result["warnings"].append(
                            "parallel_jobsが大きすぎます。システムリソースを確認してください"
                        )

            # Featuretools設定の検証
            if "featuretools" in config_dict:
                featuretools_config = config_dict["featuretools"]

                if "max_depth" in featuretools_config:
                    depth = featuretools_config["max_depth"]
                    if not isinstance(depth, int) or depth <= 0:
                        validation_result["errors"].append(
                            "max_depthは正の整数である必要があります"
                        )
                    elif depth > 5:
                        validation_result["warnings"].append(
                            "max_depthが大きすぎます。計算時間が長くなる可能性があります"
                        )

                if "max_features" in featuretools_config:
                    features = featuretools_config["max_features"]
                    if not isinstance(features, int) or features <= 0:
                        validation_result["errors"].append(
                            "max_featuresは正の整数である必要があります"
                        )
                    elif features > 200:
                        validation_result["warnings"].append(
                            "max_featuresが大きすぎます。メモリ使用量が増加する可能性があります"
                        )

            validation_result["valid"] = len(validation_result["errors"]) == 0

        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"設定検証エラー: {e}")

        return validation_result
