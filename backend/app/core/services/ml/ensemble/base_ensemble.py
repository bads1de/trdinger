"""
アンサンブル学習の基底クラス

全てのアンサンブル手法の共通インターフェースと基本機能を定義します。
"""

import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from ....utils.unified_error_handler import UnifiedModelError

logger = logging.getLogger(__name__)


class LightGBMModel:
    """
    アンサンブル内で使用するLightGBMモデルラッパー

    LightGBMTrainerの機能を簡略化してアンサンブル専用に最適化
    """

    def __init__(self, automl_config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            automl_config: AutoML設定（現在は未使用）
        """
        self.model = None
        self.is_trained = False
        self.feature_columns = None
        self.automl_config = automl_config

    def _train_model_impl(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """
        LightGBMモデルを学習

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ターゲット
            y_test: テスト用ターゲット

        Returns:
            学習結果
        """
        try:
            # 特徴量カラムを保存
            self.feature_columns = X_train.columns.tolist()

            # LightGBMデータセットを作成
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            # クラス数を判定
            num_classes = len(np.unique(y_train))

            # LightGBMパラメータ
            params = {
                "objective": "multiclass" if num_classes > 2 else "binary",
                "num_class": num_classes if num_classes > 2 else None,
                "metric": "multi_logloss" if num_classes > 2 else "binary_logloss",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.1,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "random_state": 42,
            }

            # モデル学習
            self.model = lgb.train(
                params,
                train_data,
                valid_sets=[train_data, valid_data],
                valid_names=["train", "valid"],
                num_boost_round=100,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(0),  # ログを抑制
                ],
            )

            # 予測と評価
            y_pred_proba = self.model.predict(
                X_test, num_iteration=self.model.best_iteration
            )

            if num_classes > 2:
                y_pred_class = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred_class = (y_pred_proba > 0.5).astype(int)

            # 評価指標を計算
            from sklearn.metrics import accuracy_score

            accuracy = accuracy_score(y_test, y_pred_class)

            self.is_trained = True

            logger.info(f"LightGBM学習開始: {num_classes}クラス分類")
            logger.info(f"クラス分布: {dict(y_train.value_counts())}")
            logger.info(f"LightGBMモデル学習完了: 精度={accuracy:.4f}")

            return {
                "accuracy": accuracy,
                "num_classes": num_classes,
                "best_iteration": self.model.best_iteration,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }

        except Exception as e:
            logger.error(f"LightGBMモデル学習エラー: {e}")
            raise UnifiedModelError(f"LightGBMモデル学習に失敗しました: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測を実行

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率
        """
        if not self.is_trained or self.model is None:
            raise UnifiedModelError("学習済みモデルがありません")

        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測確率を取得

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率
        """
        return self.predict(X)


class XGBoostModel:
    """
    アンサンブル内で使用するXGBoostモデルラッパー

    XGBoostを使用してアンサンブル専用に最適化されたモデル
    """

    def __init__(self, automl_config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            automl_config: AutoML設定（現在は未使用）
        """
        self.model = None
        self.is_trained = False
        self.feature_columns = None
        self.automl_config = automl_config

    def _train_model_impl(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """
        XGBoostモデルを学習

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ターゲット
            y_test: テスト用ターゲット

        Returns:
            学習結果
        """
        try:
            import xgboost as xgb

            # 特徴量カラムを保存
            self.feature_columns = X_train.columns.tolist()

            # クラス数を判定
            num_classes = len(np.unique(y_train))

            # XGBoostパラメータ
            params = {
                "objective": "multi:softprob" if num_classes > 2 else "binary:logistic",
                "num_class": num_classes if num_classes > 2 else None,
                "eval_metric": "mlogloss" if num_classes > 2 else "logloss",
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "verbosity": 0,
            }

            # XGBoostデータセットを作成
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            # モデル学習
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                evals=[(dtrain, "train"), (dtest, "eval")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )

            # 予測と評価
            y_pred_proba = self.model.predict(dtest)

            if num_classes > 2:
                y_pred_class = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred_class = (y_pred_proba > 0.5).astype(int)

            # 評価指標を計算
            from sklearn.metrics import accuracy_score

            accuracy = accuracy_score(y_test, y_pred_class)

            self.is_trained = True

            logger.info(f"XGBoost学習開始: {num_classes}クラス分類")
            logger.info(f"クラス分布: {dict(y_train.value_counts())}")
            logger.info(f"XGBoostモデル学習完了: 精度={accuracy:.4f}")

            return {
                "accuracy": accuracy,
                "num_classes": num_classes,
                "best_iteration": self.model.best_iteration,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }

        except ImportError:
            logger.error(
                "XGBoostがインストールされていません。pip install xgboostを実行してください。"
            )
            raise UnifiedModelError("XGBoostがインストールされていません")
        except Exception as e:
            logger.error(f"XGBoostモデル学習エラー: {e}")
            raise UnifiedModelError(f"XGBoostモデル学習に失敗しました: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測を実行

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率
        """
        if not self.is_trained or self.model is None:
            raise UnifiedModelError("学習済みモデルがありません")

        try:
            import xgboost as xgb

            dtest = xgb.DMatrix(X)
            predictions = self.model.predict(dtest)
            return predictions
        except ImportError:
            raise UnifiedModelError("XGBoostがインストールされていません")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測確率を取得

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率
        """
        return self.predict(X)


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
            model_type: モデルタイプ（lightgbm, random_forest等）

        Returns:
            作成されたモデル
        """
        if model_type.lower() == "lightgbm":
            return LightGBMModel(automl_config=self.automl_config)
        elif model_type.lower() == "xgboost":
            return XGBoostModel(automl_config=self.automl_config)
        elif model_type.lower() == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type.lower() == "logistic_regression":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(random_state=42, max_iter=1000)
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
        import os

        saved_paths = []

        # ベースモデルを保存
        for i, model in enumerate(self.base_models):
            model_path = f"{base_path}_base_model_{i}.pkl"
            # 全てのモデルをjoblibで保存（LightGBMModelも含む）
            joblib.dump(model, model_path)
            saved_paths.append(model_path)

        # メタモデルを保存（存在する場合）
        if self.meta_model is not None:
            meta_path = f"{base_path}_meta_model.pkl"
            joblib.dump(self.meta_model, meta_path)
            saved_paths.append(meta_path)

        # 設定を保存
        config_path = f"{base_path}_config.pkl"
        config_data = {
            "config": self.config,
            "automl_config": self.automl_config,
            "feature_columns": self.feature_columns,
            "is_fitted": self.is_fitted,
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

        try:
            # 設定を読み込み
            config_path = f"{base_path}_config.pkl"
            if os.path.exists(config_path):
                config_data = joblib.load(config_path)
                self.config = config_data["config"]
                self.automl_config = config_data["automl_config"]
                self.feature_columns = config_data["feature_columns"]
                self.is_fitted = config_data["is_fitted"]

            # ベースモデルを読み込み
            self.base_models = []
            i = 0
            while True:
                model_path = f"{base_path}_base_model_{i}.pkl"
                if not os.path.exists(model_path):
                    break

                # モデルタイプに応じて読み込み
                if "lightgbm" in str(self.config.get("base_models", [])):
                    model = LightGBMModel(automl_config=self.automl_config)
                    # LightGBMModelは直接joblibで読み込み
                    model = joblib.load(model_path)
                else:
                    model = joblib.load(model_path)

                self.base_models.append(model)
                i += 1

            # メタモデルを読み込み（存在する場合）
            meta_path = f"{base_path}_meta_model.pkl"
            if os.path.exists(meta_path):
                self.meta_model = joblib.load(meta_path)

            return True

        except Exception as e:
            logger.error(f"アンサンブルモデルの読み込みに失敗: {e}")
            return False
