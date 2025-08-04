"""
拡張特徴量エンジニアリングサービス

既存の手動特徴量生成システムにAutoML特徴量を統合します。
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ....utils.unified_error_handler import safe_ml_operation
from .automl_features.autofeat_calculator import AutoFeatCalculator
from .automl_features.automl_config import AutoMLConfig
from .automl_features.performance_optimizer import PerformanceOptimizer
from .automl_features.tsfresh_calculator import TSFreshFeatureCalculator
from .enhanced_crypto_features import EnhancedCryptoFeatures
from .feature_engineering_service import FeatureEngineeringService
from .optimized_crypto_features import OptimizedCryptoFeatures

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
        tsfresh_features = [
            col
            for col in automl_features
            if col.startswith("TSF_") or col.startswith("TS_")
        ]
        autofeat_features = [col for col in automl_features if col.startswith("AF_")]
        other_automl_features = [
            col
            for col in automl_features
            if not any(col.startswith(prefix) for prefix in ["TSF_", "AF_"])
        ]

        return {
            "manual_features": manual_features,
            "manual_count": len(manual_features),
            "automl_features": automl_features,
            "automl_count": len(automl_features),
            "tsfresh_features": tsfresh_features,
            "tsfresh_count": len(tsfresh_features),
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
            "autofeat": self.autofeat_calculator.get_feature_names(),
        }

    def clear_automl_cache(self):
        """AutoML特徴量のキャッシュをクリア"""
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
            logger.info(
                "EnhancedFeatureEngineeringServiceのリソースクリーンアップを開始"
            )

            # AutoMLキャッシュをクリア
            self.clear_automl_cache()

            # 統計情報をクリア
            self.last_enhancement_stats.clear()

            # 各計算機のリソースを個別にクリーンアップ
            if hasattr(self.tsfresh_calculator, "cleanup"):
                self.tsfresh_calculator.cleanup()

            if hasattr(self.autofeat_calculator, "cleanup"):
                self.autofeat_calculator.cleanup()

            # パフォーマンス最適化クラスのクリーンアップ
            if hasattr(self.performance_optimizer, "cleanup"):
                self.performance_optimizer.cleanup()

            logger.info("EnhancedFeatureEngineeringServiceのリソースクリーンアップ完了")

        except Exception as e:
            logger.error(f"EnhancedFeatureEngineeringServiceクリーンアップエラー: {e}")

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

                if "performance_mode" in tsfresh_config:
                    mode = tsfresh_config["performance_mode"]
                    valid_modes = [
                        "fast",
                        "balanced",
                        "financial_optimized",
                        "comprehensive",
                    ]
                    if not isinstance(mode, str) or mode not in valid_modes:
                        validation_result["errors"].append(
                            f"performance_modeは{valid_modes}のいずれかである必要があります"
                        )

            # Featuretools設定キーはサポート外（完全削除済み）だが、互換性のため警告は出さないで無視
            if "featuretools" in config_dict:
                pass

            validation_result["valid"] = len(validation_result["errors"]) == 0

        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"設定検証エラー: {e}")

        return validation_result
