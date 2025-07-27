"""
拡張特徴量エンジニアリングサービス

既存の手動特徴量生成システムにAutoML特徴量を統合します。
"""

import logging
from typing import Dict, Optional, Any, List, Tuple
import pandas as pd

import time

from .feature_engineering_service import FeatureEngineeringService
from .automl_features.tsfresh_calculator import TSFreshFeatureCalculator
from .automl_features.featuretools_calculator import FeaturetoolsCalculator
from .automl_features.autofeat_calculator import AutoFeatCalculator
from .automl_features.automl_config import AutoMLConfig
from .automl_features.performance_optimizer import PerformanceOptimizer
from .automl_features.memory_utils import (
    get_system_memory_info,
    optimize_dataframe_dtypes,
    memory_efficient_processing,
    check_memory_availability,
    log_memory_usage,
)
from .enhanced_crypto_features import EnhancedCryptoFeatures
from .optimized_crypto_features import OptimizedCryptoFeatures
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

        # 暗号通貨特化特徴量エンジニアリング
        self.crypto_features = EnhancedCryptoFeatures()

        # 最適化された特徴量エンジニアリング
        self.optimized_features = OptimizedCryptoFeatures()

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

            # 1.5. 暗号通貨特化特徴量を追加
            logger.info("暗号通貨特化特徴量を計算中...")
            crypto_start_time = time.time()

            try:
                result_df = self.crypto_features.create_comprehensive_features(
                    result_df, lookback_periods
                )
                crypto_time = time.time() - crypto_start_time
                crypto_feature_count = len(result_df.columns) - manual_feature_count
                logger.info(
                    f"暗号通貨特化特徴量生成完了: {crypto_feature_count}個 ({crypto_time:.2f}秒)"
                )
            except Exception as e:
                logger.warning(f"暗号通貨特化特徴量生成でエラー: {e}")
                crypto_time = 0
                crypto_feature_count = 0

            # 1.6. 最適化された特徴量を追加
            logger.info("最適化された特徴量を計算中...")
            optimized_start_time = time.time()

            try:
                result_df = self.optimized_features.create_optimized_features(
                    result_df, lookback_periods
                )
                optimized_time = time.time() - optimized_start_time
                optimized_feature_count = (
                    len(result_df.columns) - manual_feature_count - crypto_feature_count
                )
                logger.info(
                    f"最適化特徴量生成完了: {optimized_feature_count}個 ({optimized_time:.2f}秒)"
                )
            except Exception as e:
                logger.warning(f"最適化特徴量生成でエラー: {e}")
                optimized_time = 0
                optimized_feature_count = 0

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

                # システムメモリ情報を取得
                memory_info = get_system_memory_info()
                log_memory_usage("AutoFeat処理開始前")

                # メモリ可用性をチェック
                estimated_memory_mb = data_size_mb * 3  # 概算でデータサイズの3倍
                if not check_memory_availability(estimated_memory_mb):
                    logger.warning(
                        f"メモリ不足の可能性: 推定必要量 {estimated_memory_mb:.1f}MB"
                    )

                # DataFrameのメモリ最適化を事前に実行
                result_df = optimize_dataframe_dtypes(
                    result_df, aggressive=data_size_mb > 100
                )

                logger.info(
                    f"データサイズ: {data_size_mb:.2f}MB, メモリ推奨設定: {memory_recommendations}, "
                    f"システムメモリ使用率: {memory_info.get('used_percent', 0):.1f}%"
                )

                # メモリ効率的な処理でAutoFeat特徴量生成を実行
                with memory_efficient_processing("AutoFeat特徴量生成"):
                    with self.performance_optimizer.monitor_memory_usage(
                        "AutoFeat特徴量生成"
                    ):
                        # AutoFeatCalculatorをコンテキストマネージャーとして使用（メモリリーク防止）
                        with self.autofeat_calculator as calculator:
                            # AutoFeatで特徴量を生成
                            generated_df, generation_info = (
                                calculator.generate_features(
                                    df=result_df,
                                    target=target,
                                    task_type="regression",  # デフォルトは回帰
                                    max_features=self.automl_config.autofeat.max_features,
                                )
                            )

                            # 生成された特徴量で置き換え
                            if "error" not in generation_info:
                                # 生成されたDataFrameもメモリ最適化
                                result_df = optimize_dataframe_dtypes(
                                    generated_df, aggressive=data_size_mb > 100
                                )
                                autofeat_feature_count = generation_info.get(
                                    "generated_features", 0
                                )

                                # 生成情報を統計に追加
                                self.last_enhancement_stats[
                                    "autofeat_generation_info"
                                ] = generation_info

                            # AutoFeatモデルのメモリクリーンアップ
                            self.performance_optimizer.cleanup_autofeat_memory(
                                calculator.autofeat_model
                            )

                            # 処理後のメモリ状況をログ出力
                            log_memory_usage("AutoFeat処理完了後")

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
                f"(手動:{manual_feature_count}, 暗号通貨特化:{crypto_feature_count}, "
                f"最適化:{optimized_feature_count}, TSFresh:{tsfresh_feature_count}, "
                f"Featuretools:{featuretools_feature_count}, AutoFeat:{autofeat_feature_count}) "
                f"処理時間:{total_time:.2f}秒"
            )

            # 5. 統合特徴量選択（手動 + AutoML特徴量の冗長性除去）
            if target is not None and len(result_df.columns) > 0:
                logger.info("統合特徴量選択を実行中...")
                integrated_start_time = time.time()

                result_df, selection_info = self._perform_integrated_feature_selection(
                    result_df, target, manual_feature_count
                )

                integrated_time = time.time() - integrated_start_time
                final_feature_count = len(result_df.columns)

                logger.info(
                    f"統合特徴量選択完了: {total_features}個 → {final_feature_count}個 "
                    f"({integrated_time:.2f}秒)"
                )

                # 選択情報を統計に追加
                self.last_enhancement_stats["integrated_selection_info"] = (
                    selection_info
                )

                # 最終特徴量数を更新
                total_features = final_feature_count

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

    def _perform_integrated_feature_selection(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        manual_feature_count: int,
        max_features: int = 150,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        統合特徴量選択を実行

        手動生成特徴量とAutoML特徴量を統合して冗長性を除去し、
        最も有用な特徴量を選択します。

        Args:
            features_df: 全特徴量DataFrame
            target: ターゲット変数
            manual_feature_count: 手動特徴量の数
            max_features: 最大特徴量数

        Returns:
            選択された特徴量とメタデータ
        """
        try:
            from .automl_features.feature_selector import AdvancedFeatureSelector

            # 統合特徴量選択器を初期化
            integrated_selector = AdvancedFeatureSelector()

            # 特徴量を分類
            feature_categories = self._categorize_features(
                features_df, manual_feature_count
            )

            # 冗長性分析を実行
            redundancy_info = self._analyze_feature_redundancy(
                features_df, target, feature_categories
            )

            # 統合選択を実行
            selected_features, selection_info = (
                integrated_selector.select_features_comprehensive(
                    features_df,
                    target,
                    max_features=max_features,
                    selection_methods=[
                        "statistical_test",
                        "correlation_filter",  # 手動+AutoML間の冗長性除去
                        "mutual_information",
                        "importance_based",
                    ],
                )
            )

            # 選択結果に冗長性情報を追加
            selection_info.update(
                {
                    "feature_categories": feature_categories,
                    "redundancy_analysis": redundancy_info,
                    "integrated_selection": True,
                }
            )

            logger.info(
                f"統合特徴量選択結果: "
                f"手動:{feature_categories['manual_count']}個, "
                f"AutoML:{feature_categories['automl_count']}個 → "
                f"選択:{len(selected_features.columns)}個"
            )

            return selected_features, selection_info

        except Exception as e:
            logger.error(f"統合特徴量選択エラー: {e}")
            # エラー時は元の特徴量を返す
            return features_df, {"error": str(e), "integrated_selection": False}

    def _categorize_features(
        self, features_df: pd.DataFrame, manual_feature_count: int
    ) -> Dict[str, Any]:
        """
        特徴量を手動生成とAutoML生成に分類

        Args:
            features_df: 特徴量DataFrame
            manual_feature_count: 手動特徴量の数

        Returns:
            特徴量分類情報
        """
        all_columns = features_df.columns.tolist()

        # 手動特徴量（最初のN個）
        manual_features = all_columns[:manual_feature_count]

        # AutoML特徴量（残り）
        automl_features = all_columns[manual_feature_count:]

        # AutoML特徴量をツール別に分類
        tsfresh_features = [col for col in automl_features if col.startswith("TSF_")]
        featuretools_features = [
            col for col in automl_features if col.startswith("FT_")
        ]
        autofeat_features = [col for col in automl_features if col.startswith("AF_")]
        other_automl_features = [
            col
            for col in automl_features
            if not any(col.startswith(prefix) for prefix in ["TSF_", "FT_", "AF_"])
        ]

        return {
            "manual_features": manual_features,
            "manual_count": len(manual_features),
            "automl_features": automl_features,
            "automl_count": len(automl_features),
            "tsfresh_features": tsfresh_features,
            "tsfresh_count": len(tsfresh_features),
            "featuretools_features": featuretools_features,
            "featuretools_count": len(featuretools_features),
            "autofeat_features": autofeat_features,
            "autofeat_count": len(autofeat_features),
            "other_automl_features": other_automl_features,
            "other_automl_count": len(other_automl_features),
            "total_count": len(all_columns),
        }

    def _analyze_feature_redundancy(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        feature_categories: Dict[str, Any],
        correlation_threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """
        特徴量間の冗長性を分析

        特に手動特徴量とAutoML特徴量間の冗長性を重点的に分析します。
        例: manual_RSI と TSF_rsi_* の冗長性

        Args:
            features_df: 特徴量DataFrame
            target: ターゲット変数
            feature_categories: 特徴量分類情報
            correlation_threshold: 冗長性判定の相関閾値

        Returns:
            冗長性分析結果
        """
        try:
            # 相関行列を計算
            correlation_matrix = features_df.corr().abs()

            # 手動特徴量とAutoML特徴量間の高相関ペアを特定
            manual_automl_redundancy = []
            manual_features = feature_categories["manual_features"]
            automl_features = feature_categories["automl_features"]

            for manual_feat in manual_features:
                for automl_feat in automl_features:
                    if (
                        manual_feat in correlation_matrix.index
                        and automl_feat in correlation_matrix.columns
                    ):
                        corr_value = correlation_matrix.loc[manual_feat, automl_feat]
                        if corr_value > correlation_threshold:
                            # ターゲットとの相関も計算
                            manual_target_corr = abs(
                                features_df[manual_feat].corr(target)
                            )
                            automl_target_corr = abs(
                                features_df[automl_feat].corr(target)
                            )

                            manual_automl_redundancy.append(
                                {
                                    "manual_feature": manual_feat,
                                    "automl_feature": automl_feat,
                                    "correlation": corr_value,
                                    "manual_target_corr": manual_target_corr,
                                    "automl_target_corr": automl_target_corr,
                                    "recommended_removal": (
                                        manual_feat
                                        if manual_target_corr < automl_target_corr
                                        else automl_feat
                                    ),
                                }
                            )

            # AutoML特徴量間の冗長性も分析
            automl_automl_redundancy = []
            for i, feat1 in enumerate(automl_features):
                for feat2 in automl_features[i + 1 :]:
                    if (
                        feat1 in correlation_matrix.index
                        and feat2 in correlation_matrix.columns
                    ):
                        corr_value = correlation_matrix.loc[feat1, feat2]
                        if corr_value > correlation_threshold:
                            automl_automl_redundancy.append(
                                {
                                    "feature1": feat1,
                                    "feature2": feat2,
                                    "correlation": corr_value,
                                }
                            )

            redundancy_info = {
                "correlation_threshold": correlation_threshold,
                "manual_automl_redundancy": manual_automl_redundancy,
                "manual_automl_redundant_pairs": len(manual_automl_redundancy),
                "automl_automl_redundancy": automl_automl_redundancy,
                "automl_automl_redundant_pairs": len(automl_automl_redundancy),
                "total_redundant_pairs": len(manual_automl_redundancy)
                + len(automl_automl_redundancy),
            }

            logger.info(
                f"冗長性分析完了: "
                f"手動-AutoML冗長ペア:{len(manual_automl_redundancy)}個, "
                f"AutoML-AutoML冗長ペア:{len(automl_automl_redundancy)}個"
            )

            return redundancy_info

        except Exception as e:
            logger.error(f"冗長性分析エラー: {e}")
            return {"error": str(e), "analysis_completed": False}

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
