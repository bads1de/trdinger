"""
アンサンブル学習の基底クラス

全てのアンサンブル手法の共通インターフェースと基本機能を定義します。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.unified_config import unified_config

from ..common.evaluation_utils import evaluate_model_predictions
from ..common.ml_utils import get_feature_importance_unified
from ..exceptions import MLModelError

logger = logging.getLogger(__name__)


class BaseEnsemble(ABC):
    """
    アンサンブル学習の基底クラス

    スタッキングアンサンブル手法の共通インターフェースを定義します。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初期化

        Args:
            config: アンサンブル設定
        """
        self.config = config
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
        ベースモデルを作成（Essential 2 Modelsのみサポート）

        Args:
            model_type: モデルタイプ（lightgbm, xgboost, logistic_regression, catboost）

        Returns:
            作成されたモデル
        """
        if model_type.lower() == "lightgbm":
            try:
                import lightgbm as lgb

                # デフォルトパラメータは後でset_paramsで上書きされるか、fitの引数で渡される
                return lgb.LGBMClassifier(
                    n_jobs=1,  # 並列処理はStackingClassifier側で制御
                    random_state=unified_config.ml.training.random_state,
                )
            except ImportError:
                raise MLModelError("LightGBMのインポートに失敗しました")
        elif model_type.lower() == "xgboost":
            try:
                import xgboost as xgb

                return xgb.XGBClassifier(
                    n_jobs=1,
                    random_state=unified_config.ml.training.random_state,
                    eval_metric="logloss",
                )
            except ImportError:
                raise MLModelError("XGBoostのインポートに失敗しました")
        elif model_type.lower() == "catboost":
            try:
                import catboost as cb

                return cb.CatBoostClassifier(
                    thread_count=1,
                    random_seed=unified_config.ml.training.random_state,
                    verbose=0,
                    allow_writing_files=False,
                )
            except ImportError:
                raise MLModelError("CatBoostのインポートに失敗しました")
        elif model_type.lower() == "logistic_regression":
            # scikit-learnのLogisticRegressionを直接使用
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                random_state=unified_config.ml.training.random_state,
                max_iter=unified_config.ml.training.lr_max_iter,
                solver="lbfgs",  # 多クラス分類に適したソルバー
                verbose=0,  # ログ抑制
            )
        else:
            raise MLModelError(
                f"サポートされていないモデルタイプ: {model_type}。サポートされているタイプ: lightgbm, xgboost, catboost, logistic_regression"
            )

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
        result = evaluate_model_predictions(y_true, y_pred, y_pred_proba)

        logger.info("✅ アンサンブル評価指標計算完了（共通評価関数使用）")

        return result

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        特徴量重要度を取得（ベースモデルから集約）

        Returns:
            特徴量重要度の辞書（利用可能な場合）
        """
        if not self.is_fitted or not self.feature_columns:
            logger.warning(
                f"特徴量重要度取得不可: is_fitted={self.is_fitted}, feature_columns={len(self.feature_columns) if self.feature_columns else 0}"
            )
            return {}

        # ベースモデルから特徴量重要度を集約
        all_importances: Dict[str, List[float]] = {}

        for i, model in enumerate(self.base_models):
            try:
                # 統一関数を使用して重要度を取得
                model_importance = get_feature_importance_unified(
                    model, self.feature_columns, top_n=len(self.feature_columns)
                )

                if model_importance:
                    logger.info(
                        f"モデル{i}: 特徴量重要度を取得 ({len(model_importance)}個)"
                    )
                    for feature, importance in model_importance.items():
                        if feature not in all_importances:
                            all_importances[feature] = []
                        all_importances[feature].append(importance)
                else:
                    logger.warning(f"モデル{i}: 特徴量重要度が取得できませんでした")

            except Exception as e:
                logger.error(f"モデル{i}の特徴量重要度取得エラー: {e}")

        # 平均を計算
        if all_importances:
            avg_importance = {
                feature: float(np.mean(values))
                for feature, values in all_importances.items()
            }
            logger.info(f"アンサンブル特徴量重要度を計算: {len(avg_importance)}個")
            return avg_importance

        logger.warning("特徴量重要度データが見つかりませんでした")
        return {}

    def save_models(self, base_path: str) -> List[str]:
        """
        アンサンブルモデルを保存（自前実装StackingEnsemble対応）

        Args:
            base_path: 保存先ベースパス

        Returns:
            保存されたファイルパスのリスト
        """
        import os
        from datetime import datetime

        import joblib

        from ..model_manager import model_manager

        saved_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # base_pathからモデル名を抽出
        model_name = os.path.basename(base_path)

        # 自前実装のStackingEnsembleの保存（優先）
        if hasattr(self, "_fitted_base_models") and hasattr(self, "_fitted_meta_model"):
            if self._fitted_base_models and self._fitted_meta_model is not None:
                # 自前実装形式で保存
                model_data = {
                    "fitted_base_models": self._fitted_base_models,
                    "fitted_meta_model": self._fitted_meta_model,
                    "base_model_types": getattr(self, "_base_model_types", []),
                    "meta_model_type": getattr(
                        self, "_meta_model_type", "logistic_regression"
                    ),
                    "config": self.config,
                    "feature_columns": self.feature_columns,
                    "is_fitted": self.is_fitted,
                    "passthrough": getattr(self, "passthrough", False),
                    "ensemble_type": "StackingEnsemble",
                    "sklearn_implementation": False,  # 自前実装
                }

                metadata = {
                    "ensemble_type": "StackingEnsemble",
                    "sklearn_implementation": False,
                    "feature_count": (
                        len(self.feature_columns) if self.feature_columns else 0
                    ),
                    "fitted_base_models": list(self._fitted_base_models.keys()),
                }

                try:
                    model_path = model_manager.save_model(
                        model=model_data,
                        model_name=model_name,
                        metadata=metadata,
                        feature_columns=self.feature_columns,
                    )
                    if model_path:
                        saved_paths.append(model_path)
                        logger.info(
                            f"StackingEnsemble（自前実装）をmodel_managerで保存: {model_path}"
                        )
                except Exception as e:
                    logger.error(f"model_managerによる保存に失敗: {e}")
                    # フォールバック: 従来のjoblib保存
                    model_path = f"{base_path}_stacking_ensemble_{timestamp}.pkl"
                    joblib.dump(model_data, model_path)
                    saved_paths.append(model_path)
                    logger.info(
                        f"StackingEnsembleをjoblibで保存(フォールバック): {model_path}"
                    )

                return saved_paths

        # 旧形式: scikit-learn StackingClassifierの保存（後方互換性）
        if (
            hasattr(self, "stacking_classifier")
            and self.stacking_classifier is not None
        ):
            # model_managerを使用して保存
            # StackingEnsembleの状態を含めた辞書を作成
            model_data = {
                "ensemble_classifier": self.stacking_classifier,
                "config": self.config,
                "feature_columns": self.feature_columns,
                "is_fitted": self.is_fitted,
                "ensemble_type": "StackingEnsemble",
                "sklearn_implementation": True,
            }

            metadata = {
                "ensemble_type": "StackingEnsemble",
                "sklearn_implementation": True,
                "feature_count": (
                    len(self.feature_columns) if self.feature_columns else 0
                ),
            }

            try:
                # model_manager.save_modelはパスを返す
                model_path = model_manager.save_model(
                    model=model_data,
                    model_name=model_name,
                    metadata=metadata,
                    feature_columns=self.feature_columns,
                )
                if model_path:
                    saved_paths.append(model_path)
                    logger.info(
                        f"StackingClassifierをmodel_managerで保存: {model_path}"
                    )
            except Exception as e:
                logger.error(f"model_managerによる保存に失敗: {e}")
                # フォールバック: 従来のjoblib保存
                model_path = f"{base_path}_stacking_classifier_{timestamp}.pkl"
                joblib.dump(model_data, model_path)
                saved_paths.append(model_path)
                logger.info(
                    f"StackingClassifierをjoblibで保存(フォールバック): {model_path}"
                )

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
        アンサンブルモデルを読み込み（scikit-learn StackingClassifier対応）

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
            # scikit-learn StackingClassifierファイルを検索
            sklearn_patterns = [
                f"{base_path}_stacking_classifier_*.pkl",
            ]

            sklearn_files = []
            for pattern in sklearn_patterns:
                sklearn_files.extend(glob.glob(pattern))

            if sklearn_files:
                # scikit-learn StackingClassifierファイルで読み込み
                sklearn_file = sorted(sklearn_files)[-1]  # 最新のファイルを選択
                logger.info(
                    f"scikit-learn StackingClassifierファイルでモデルを読み込み: {sklearn_file}"
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", InconsistentVersionWarning)
                    model_data = joblib.load(sklearn_file)

                # scikit-learn StackingClassifierからデータを復元
                if isinstance(model_data, dict) and "ensemble_classifier" in model_data:
                    # StackingClassifierを復元
                    if "stacking_classifier" in sklearn_file:
                        self.stacking_classifier = model_data["ensemble_classifier"]
                        logger.info("StackingClassifierを復元")

                    self.config = model_data.get("config", {})
                    self.feature_columns = model_data.get("feature_columns", [])
                    self.is_fitted = model_data.get("is_fitted", False)

                    logger.info("scikit-learn StackingClassifierモデルの読み込み完了")
                    return True

            # 従来の統合ファイル形式を試す
            algorithm_pattern = f"{base_path}_*_*.pkl"
            unified_files = [
                f
                for f in glob.glob(algorithm_pattern)
                if not f.endswith("_config.pkl")
                and not f.endswith("_meta_model.pkl")
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


