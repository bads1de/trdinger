"""
アンサンブル学習の基底クラス

全てのアンサンブル手法の共通インターフェースと基本機能を定義します。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ....utils.unified_error_handler import UnifiedModelError

logger = logging.getLogger(__name__)


class BaseEnsemble(ABC):
    """
    アンサンブル学習の基底クラス

    全てのアンサンブル手法（バギング、スタッキング等）の共通インターフェースを定義します。
    """

    def __init__(
        self, config: Dict[str, Any], automl_config: Optional[Dict[str, Any]] = None
    ):
        """
        初期化

        Args:
            config: アンサンブル設定
            automl_config: AutoML設定（オプション）
        """
        self.config = config
        self.automl_config = automl_config
        self.base_models: List[Any] = []
        self.meta_model: Optional[Any] = None
        self.is_fitted = False
        self.feature_columns: Optional[List[str]] = None
        self.scaler: Optional[Any] = None

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        アンサンブルモデルを学習

        Args:
            X_train: 学習用特徴量
            y_train: 学習用ターゲット
            X_test: テスト用特徴量（オプション）
            y_test: テスト用ターゲット（オプション）

        Returns:
            学習結果の辞書
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測を実行

        Args:
            X: 特徴量DataFrame

        Returns:
            予測結果
        """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測確率を取得

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率の配列
        """

    def _create_base_model(self, model_type: str) -> Any:
        """
        ベースモデルを作成

        Args:
            model_type: モデルタイプ（lightgbm, xgboost, catboost, tabnet, random_forest等）

        Returns:
            作成されたモデル
        """
        if model_type.lower() == "lightgbm":
            try:
                from ..models.lightgbm_wrapper import LightGBMModel

                return LightGBMModel(automl_config=self.automl_config)
            except ImportError:
                raise UnifiedModelError(
                    "LightGBMモデルラッパーのインポートに失敗しました"
                )
        elif model_type.lower() == "xgboost":
            try:
                from ..models.xgboost_wrapper import XGBoostModel

                return XGBoostModel(automl_config=self.automl_config)
            except ImportError:
                raise UnifiedModelError(
                    "XGBoostモデルラッパーのインポートに失敗しました"
                )
        elif model_type.lower() == "catboost":
            try:
                from ..models.catboost_wrapper import CatBoostModel

                return CatBoostModel(automl_config=self.automl_config)
            except ImportError:
                raise UnifiedModelError(
                    "CatBoostモデルラッパーのインポートに失敗しました"
                )
        elif model_type.lower() == "tabnet":
            try:
                from ..models.tabnet_wrapper import TabNetModel

                return TabNetModel(automl_config=self.automl_config)
            except ImportError:
                raise UnifiedModelError(
                    "TabNetモデルラッパーのインポートに失敗しました"
                )
        elif (
            model_type.lower() == "random_forest"
            or model_type.lower() == "randomforest"
        ):
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif (
            model_type.lower() == "gradient_boosting"
            or model_type.lower() == "gradientboosting"
        ):
            from sklearn.ensemble import GradientBoostingClassifier

            return GradientBoostingClassifier(
                n_estimators=50,  # 100→50に軽量化
                learning_rate=0.2,  # 0.1→0.2に高速化
                max_depth=3,
                subsample=0.8,  # サブサンプリングで高速化
                random_state=42,
                verbose=0,  # ログ抑制
            )
        elif model_type.lower() == "extratrees":
            try:
                from ..models.extratrees_wrapper import ExtraTreesModel

                return ExtraTreesModel(automl_config=self.automl_config)
            except ImportError:
                raise UnifiedModelError(
                    "ExtraTreesモデルラッパーのインポートに失敗しました"
                )
        elif model_type.lower() == "adaboost":
            try:
                from ..models.adaboost_wrapper import AdaBoostModel

                return AdaBoostModel(automl_config=self.automl_config)
            except ImportError:
                raise UnifiedModelError(
                    "AdaBoostモデルラッパーのインポートに失敗しました"
                )
        elif model_type.lower() == "ridge":
            try:
                from ..models.ridge_wrapper import RidgeModel

                return RidgeModel(automl_config=self.automl_config)
            except ImportError:
                raise UnifiedModelError("Ridgeモデルラッパーのインポートに失敗しました")
        elif model_type.lower() == "naivebayes":
            try:
                from ..models.naivebayes_wrapper import NaiveBayesModel

                return NaiveBayesModel(automl_config=self.automl_config)
            except ImportError:
                raise UnifiedModelError(
                    "NaiveBayesモデルラッパーのインポートに失敗しました"
                )
        elif model_type.lower() == "knn":
            try:
                from ..models.knn_wrapper import KNNModel

                return KNNModel(automl_config=self.automl_config)
            except ImportError:
                raise UnifiedModelError("KNNモデルラッパーのインポートに失敗しました")
        elif model_type.lower() == "logistic_regression":
            # scikit-learnのLogisticRegressionを直接使用
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                random_state=42,
                max_iter=1000,  # 収束を確実にするため
                solver="lbfgs",  # 多クラス分類に適したソルバー
                verbose=0,  # ログ抑制
            )
        else:
            raise UnifiedModelError(f"サポートされていないモデルタイプ: {model_type}")

    def _evaluate_predictions(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        予測結果を評価

        Args:
            y_true: 実際のターゲット値
            y_pred: 予測値
            y_pred_proba: 予測確率（オプション）

        Returns:
            評価指標の辞書
        """
        from ..evaluation.enhanced_metrics import (
            EnhancedMetricsCalculator,
            MetricsConfig,
        )

        # 統一された評価指標計算器を使用
        config = MetricsConfig(
            include_balanced_accuracy=True,
            include_pr_auc=True,
            include_roc_auc=True,
            include_confusion_matrix=True,
            include_classification_report=True,
            average_method="weighted",
            zero_division=0,
        )

        metrics_calculator = EnhancedMetricsCalculator(config)

        # numpy配列に変換（pandas Seriesの場合）
        y_true_array = y_true.values if hasattr(y_true, "values") else y_true

        # 包括的な評価指標を計算
        result = metrics_calculator.calculate_comprehensive_metrics(
            y_true_array, y_pred, y_pred_proba
        )

        logger.info("✅ アンサンブル評価指標計算完了（EnhancedMetricsCalculator使用）")

        return result

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        特徴量重要度を取得

        Returns:
            特徴量重要度の辞書（利用可能な場合）
        """
        if not self.is_fitted or not self.feature_columns:
            logger.warning(
                f"特徴量重要度取得不可: is_fitted={self.is_fitted}, feature_columns={len(self.feature_columns) if self.feature_columns else 0}"
            )
            return {}

        # ベースモデルから特徴量重要度を集約
        importance_dict = {}

        for i, model in enumerate(self.base_models):
            try:
                if hasattr(model, "feature_importances_"):
                    # scikit-learn系モデル
                    importances = model.feature_importances_
                    logger.info(
                        f"モデル{i}: scikit-learn系特徴量重要度を取得 ({len(importances)}個)"
                    )
                    for j, feature in enumerate(self.feature_columns):
                        if j < len(importances):
                            if feature not in importance_dict:
                                importance_dict[feature] = []
                            importance_dict[feature].append(importances[j])

                elif hasattr(model, "model") and hasattr(
                    model.model, "feature_importance"
                ):
                    # LightGBMModel (カスタムラッパー)
                    importances = model.model.feature_importance(importance_type="gain")
                    logger.info(
                        f"モデル{i}: LightGBM特徴量重要度を取得 ({len(importances)}個)"
                    )
                    for j, feature in enumerate(self.feature_columns):
                        if j < len(importances):
                            if feature not in importance_dict:
                                importance_dict[feature] = []
                            importance_dict[feature].append(importances[j])

                elif hasattr(model, "get_feature_importance"):
                    # カスタムget_feature_importanceメソッドを持つモデル
                    model_importance = model.get_feature_importance()
                    if model_importance:
                        logger.info(
                            f"モデル{i}: カスタムメソッドで特徴量重要度を取得 ({len(model_importance)}個)"
                        )
                        for feature, importance in model_importance.items():
                            if feature not in importance_dict:
                                importance_dict[feature] = []
                            importance_dict[feature].append(importance)

                else:
                    logger.warning(
                        f"モデル{i}: 特徴量重要度を取得できません (type: {type(model)})"
                    )

            except Exception as e:
                logger.error(f"モデル{i}の特徴量重要度取得エラー: {e}")

        # 平均を計算
        if importance_dict:
            avg_importance = {
                feature: float(np.mean(values))
                for feature, values in importance_dict.items()
            }
            logger.info(f"アンサンブル特徴量重要度を計算: {len(avg_importance)}個")
            return avg_importance

        logger.warning("特徴量重要度データが見つかりませんでした")
        return {}

    def save_models(self, base_path: str) -> List[str]:
        """
        アンサンブルモデルを保存（scikit-learn実装対応）

        Args:
            base_path: 保存先ベースパス

        Returns:
            保存されたファイルパスのリスト
        """
        from datetime import datetime
        import joblib

        saved_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # scikit-learn実装の場合
        if hasattr(self, "bagging_classifier") and self.bagging_classifier is not None:
            # BaggingClassifier保存
            model_path = f"{base_path}_bagging_classifier_{timestamp}.pkl"
            model_data = {
                "ensemble_classifier": self.bagging_classifier,
                "config": self.config,
                "automl_config": self.automl_config,
                "feature_columns": self.feature_columns,
                "is_fitted": self.is_fitted,
                "ensemble_type": "BaggingEnsemble",
                "sklearn_implementation": True,
            }
            joblib.dump(model_data, model_path)
            saved_paths.append(model_path)
            logger.info(f"BaggingClassifierを保存: {model_path}")

        elif (
            hasattr(self, "stacking_classifier")
            and self.stacking_classifier is not None
        ):
            # StackingClassifier保存
            model_path = f"{base_path}_stacking_classifier_{timestamp}.pkl"
            model_data = {
                "ensemble_classifier": self.stacking_classifier,
                "config": self.config,
                "automl_config": self.automl_config,
                "feature_columns": self.feature_columns,
                "is_fitted": self.is_fitted,
                "ensemble_type": "StackingEnsemble",
                "sklearn_implementation": True,
            }
            joblib.dump(model_data, model_path)
            saved_paths.append(model_path)
            logger.info(f"StackingClassifierを保存: {model_path}")

        else:
            # 従来の実装（後方互換性のため）
            logger.warning("従来のアンサンブル実装を保存")

            # 最高性能モデル1つのみを保存
            if len(self.base_models) == 1 and hasattr(self, "best_algorithm"):
                algorithm_name = getattr(self, "best_algorithm", "unknown")
                model_path = f"{base_path}_{algorithm_name}_{timestamp}.pkl"

                model_data = {
                    "model": self.base_models[0],
                    "config": self.config,
                    "automl_config": self.automl_config,
                    "feature_columns": self.feature_columns,
                    "is_fitted": self.is_fitted,
                    "best_algorithm": algorithm_name,
                    "best_model_score": getattr(self, "best_model_score", None),
                    "ensemble_type": self.__class__.__name__,
                    "selected_model_only": True,
                }
                joblib.dump(model_data, model_path)
                saved_paths.append(model_path)
                logger.info(f"最高性能モデルを保存: {model_path}")

            else:
                # 複数ファイル保存
                for i, model in enumerate(self.base_models):
                    model_path = f"{base_path}_base_model_{i}_{timestamp}.pkl"
                    joblib.dump(model, model_path)
                    saved_paths.append(model_path)

                if self.meta_model is not None:
                    meta_path = f"{base_path}_meta_model_{timestamp}.pkl"
                    joblib.dump(self.meta_model, meta_path)
                    saved_paths.append(meta_path)

                config_path = f"{base_path}_config_{timestamp}.pkl"
                config_data = {
                    "config": self.config,
                    "automl_config": self.automl_config,
                    "feature_columns": self.feature_columns,
                    "is_fitted": self.is_fitted,
                    "best_algorithm": getattr(self, "best_algorithm", None),
                    "best_model_score": getattr(self, "best_model_score", None),
                }
                joblib.dump(config_data, config_path)
                saved_paths.append(config_path)

        return saved_paths

    def load_models(self, base_path: str) -> bool:
        """
        アンサンブルモデルを読み込み（scikit-learn実装対応）

        Args:
            base_path: 読み込み元ベースパス

        Returns:
            読み込み成功フラグ
        """
        import glob
        import os
        import warnings
        import joblib
        from sklearn.exceptions import InconsistentVersionWarning

        try:
            # scikit-learn実装のファイルを検索
            sklearn_patterns = [
                f"{base_path}_bagging_classifier_*.pkl",
                f"{base_path}_stacking_classifier_*.pkl",
            ]

            sklearn_files = []
            for pattern in sklearn_patterns:
                sklearn_files.extend(glob.glob(pattern))

            if sklearn_files:
                # scikit-learn実装ファイルで読み込み
                sklearn_file = sorted(sklearn_files)[-1]  # 最新のファイルを選択
                logger.info(
                    f"scikit-learn実装ファイルでモデルを読み込み: {sklearn_file}"
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", InconsistentVersionWarning)
                    model_data = joblib.load(sklearn_file)

                # scikit-learn実装からデータを復元
                if isinstance(model_data, dict) and "ensemble_classifier" in model_data:
                    # BaggingClassifierまたはStackingClassifierを復元
                    if "bagging_classifier" in sklearn_file:
                        self.bagging_classifier = model_data["ensemble_classifier"]
                        logger.info("BaggingClassifierを復元")
                    elif "stacking_classifier" in sklearn_file:
                        self.stacking_classifier = model_data["ensemble_classifier"]
                        logger.info("StackingClassifierを復元")

                    self.config = model_data.get("config", {})
                    self.automl_config = model_data.get("automl_config", {})
                    self.feature_columns = model_data.get("feature_columns", [])
                    self.is_fitted = model_data.get("is_fitted", False)

                    logger.info("scikit-learn実装モデルの読み込み完了")
                    return True

            # 従来の統合ファイル形式を試す
            algorithm_pattern = f"{base_path}_*_*.pkl"
            unified_files = [
                f
                for f in glob.glob(algorithm_pattern)
                if not f.endswith("_config.pkl")
                and not f.endswith("_meta_model.pkl")
                and "bagging_classifier" not in f
                and "stacking_classifier" not in f
            ]

            if unified_files:
                # 統合ファイル形式で読み込み
                unified_file = sorted(unified_files)[-1]  # 最新のファイルを選択
                logger.info(f"統合ファイル形式でモデルを読み込み: {unified_file}")

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", InconsistentVersionWarning)
                    model_data = joblib.load(unified_file)

                # 統合ファイルからデータを復元
                if isinstance(model_data, dict) and "model" in model_data:
                    self.base_models = [model_data["model"]]
                    self.config = model_data.get("config", {})
                    self.automl_config = model_data.get("automl_config", {})
                    self.feature_columns = model_data.get("feature_columns", [])
                    self.is_fitted = model_data.get("is_fitted", False)
                    self.best_algorithm = model_data.get("best_algorithm", "unknown")
                    self.best_model_score = model_data.get("best_model_score", 0.0)
                    self.meta_model = None  # 統合ファイルではメタモデルは使用しない

                    logger.info(
                        f"統合ファイルから最高性能モデルを読み込み: {self.best_algorithm}"
                    )
                    return True
                else:
                    # 古い形式の単一モデルファイル
                    self.base_models = [model_data]
                    logger.info("古い形式の単一モデルファイルを読み込み")

            # 従来の分離ファイル形式で読み込み（後方互換性）
            logger.info("従来の分離ファイル形式で読み込みを試行")

            # 設定ファイルを検索
            config_patterns = [
                f"{base_path}_config_*.pkl",  # 新形式（タイムスタンプ付き）
                f"{base_path}_config.pkl",  # 旧形式
            ]

            config_path = None
            for pattern in config_patterns:
                files = glob.glob(pattern)
                if files:
                    config_path = sorted(files)[-1]  # 最新のファイルを選択
                    break

            if config_path and os.path.exists(config_path):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", InconsistentVersionWarning)
                    config_data = joblib.load(config_path)
                self.config = config_data["config"]
                self.automl_config = config_data["automl_config"]
                self.feature_columns = config_data["feature_columns"]
                self.is_fitted = config_data["is_fitted"]

                # 新しい属性を読み込み（存在する場合）
                if "best_algorithm" in config_data:
                    self.best_algorithm = config_data["best_algorithm"]
                if "best_model_score" in config_data:
                    self.best_model_score = config_data["best_model_score"]

            # ベースモデルを読み込み（従来形式）
            self.base_models = []

            # 旧形式のファイルを検索
            i = 0
            while True:
                model_path = f"{base_path}_base_model_{i}.pkl"
                if not os.path.exists(model_path):
                    break

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", InconsistentVersionWarning)
                    model = joblib.load(model_path)
                self.base_models.append(model)
                i += 1

            # メタモデルを読み込み（存在する場合）
            meta_patterns = [
                f"{base_path}_meta_model_*.pkl",  # 新形式
                f"{base_path}_meta_model.pkl",  # 旧形式
            ]

            for pattern in meta_patterns:
                files = glob.glob(pattern)
                if files:
                    meta_path = sorted(files)[-1]  # 最新のファイルを選択
                    if os.path.exists(meta_path):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", InconsistentVersionWarning)
                            self.meta_model = joblib.load(meta_path)
                    break

            logger.info(
                f"従来形式でモデルを読み込み: ベースモデル{len(self.base_models)}個"
            )
            return True

        except Exception as e:
            logger.error(f"アンサンブルモデルの読み込みに失敗: {e}")
            return False
