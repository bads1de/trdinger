"""
アンサンブル学習の基底クラス

全てのアンサンブル手法の共通インターフェースと基本機能を定義します。
"""

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


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
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測を実行

        Args:
            X: 特徴量DataFrame

        Returns:
            予測結果
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測確率を取得

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率の配列
        """
        pass

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
        elif model_type.lower() == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type.lower() == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier

            return GradientBoostingClassifier(
                n_estimators=50,  # 100→50に軽量化
                learning_rate=0.2,  # 0.1→0.2に高速化
                max_depth=3,
                subsample=0.8,  # サブサンプリングで高速化
                random_state=42,
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
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            classification_report,
            confusion_matrix,
        )

        # 基本的な評価指標
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # 分類レポート
        class_report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )

        # 混同行列
        conf_matrix = confusion_matrix(y_true, y_pred)

        result = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix.tolist(),
        }

        # ROC-AUCスコア（確率予測がある場合）
        if y_pred_proba is not None:
            try:
                from sklearn.metrics import roc_auc_score

                if len(np.unique(y_true)) == 2:  # 二値分類
                    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:  # 多クラス分類
                    roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class="ovr")
                result["roc_auc"] = roc_auc
            except Exception as e:
                logger.warning(f"ROC-AUCスコアの計算に失敗: {e}")
                result["roc_auc"] = None

        return result

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        特徴量重要度を取得

        Returns:
            特徴量重要度の辞書（利用可能な場合）
        """
        if not self.is_fitted or not self.feature_columns:
            return None

        # ベースモデルから特徴量重要度を集約
        importance_dict = {}

        for i, model in enumerate(self.base_models):
            if hasattr(model, "feature_importances_"):
                # scikit-learn系モデル
                importances = model.feature_importances_
                for j, feature in enumerate(self.feature_columns):
                    if feature not in importance_dict:
                        importance_dict[feature] = []
                    importance_dict[feature].append(importances[j])
            elif hasattr(model, "model") and hasattr(model.model, "feature_importance"):
                # LightGBMTrainer
                importances = model.model.feature_importance(importance_type="gain")
                for j, feature in enumerate(self.feature_columns):
                    if feature not in importance_dict:
                        importance_dict[feature] = []
                    importance_dict[feature].append(importances[j])

        # 平均を計算
        if importance_dict:
            avg_importance = {
                feature: np.mean(values) for feature, values in importance_dict.items()
            }
            return avg_importance

        return None

    def save_models(self, base_path: str) -> List[str]:
        """
        アンサンブルモデルを保存

        Args:
            base_path: 保存先ベースパス

        Returns:
            保存されたファイルパスのリスト
        """
        import joblib
        from datetime import datetime

        saved_paths = []

        # タイムスタンプを生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # アルゴリズム名を取得（BaggingEnsembleの場合）
        algorithm_name = getattr(self, "best_algorithm", "unknown")

        # ベースモデルを保存
        for i, model in enumerate(self.base_models):
            if len(self.base_models) == 1 and hasattr(self, "best_algorithm"):
                # 最高性能モデル1つのみの場合
                model_path = f"{base_path}_{algorithm_name}_{timestamp}.pkl"
            else:
                # 複数モデルの場合（従来の形式）
                model_path = f"{base_path}_base_model_{i}.pkl"

            # 全てのモデルをjoblibで保存（LightGBMModelも含む）
            joblib.dump(model, model_path)
            saved_paths.append(model_path)

        # メタモデルを保存（存在する場合）
        if self.meta_model is not None:
            meta_path = f"{base_path}_meta_model_{timestamp}.pkl"
            joblib.dump(self.meta_model, meta_path)
            saved_paths.append(meta_path)

        # 設定を保存
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
        アンサンブルモデルを読み込み

        Args:
            base_path: 読み込み元ベースパス

        Returns:
            読み込み成功フラグ
        """
        import joblib
        import os
        import glob

        try:
            # 設定ファイルを検索（タイムスタンプ付きファイルに対応）
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

            # ベースモデルを読み込み
            self.base_models = []

            # 新形式（アルゴリズム名付き）のファイルを検索
            algorithm_pattern = f"{base_path}_*_*.pkl"
            algorithm_files = [
                f
                for f in glob.glob(algorithm_pattern)
                if not f.endswith("_config.pkl") and not f.endswith("_meta_model.pkl")
            ]

            if algorithm_files:
                # 新形式のファイルが見つかった場合
                for model_path in sorted(algorithm_files):
                    if os.path.exists(model_path):
                        model = joblib.load(model_path)
                        self.base_models.append(model)
            else:
                # 旧形式のファイルを検索
                i = 0
                while True:
                    model_path = f"{base_path}_base_model_{i}.pkl"
                    if not os.path.exists(model_path):
                        break

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
                        self.meta_model = joblib.load(meta_path)
                    break

            return True

        except Exception as e:
            logger.error(f"アンサンブルモデルの読み込みに失敗: {e}")
            return False
