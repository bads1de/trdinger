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
        """ベースモデルを作成"""
        mt = model_type.lower()
        seed = unified_config.ml.training.random_state

        if mt == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(n_jobs=1, random_state=seed)
        if mt == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(n_jobs=1, random_state=seed, eval_metric="logloss")
        if mt == "catboost":
            import catboost as cb
            return cb.CatBoostClassifier(thread_count=1, random_seed=seed, verbose=0, allow_writing_files=False)
        if mt == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(random_state=seed, max_iter=unified_config.ml.training.lr_max_iter, solver="lbfgs")
        
        raise MLModelError(f"Unsupported model type: {model_type}")

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
        """アンサンブルモデルを保存"""
        import os
        from datetime import datetime
        import joblib
        from ..model_manager import model_manager

        m_name = os.path.basename(base_path)
        
        # 1. 自前実装 StackingEnsemble
        if hasattr(self, "_fitted_base_models") and self._fitted_base_models:
            data = {
                "fitted_base_models": self._fitted_base_models, "fitted_meta_model": self._fitted_meta_model,
                "base_model_types": getattr(self, "_base_model_types", []),
                "meta_model_type": getattr(self, "_meta_model_type", "logistic_regression"),
                "config": self.config, "feature_columns": self.feature_columns, "is_fitted": self.is_fitted,
                "passthrough": getattr(self, "passthrough", False), "ensemble_type": "StackingEnsemble"
            }
            meta = {
                "ensemble_type": "StackingEnsemble", "feature_count": len(self.feature_columns or []),
                "fitted_base_models": list(self._fitted_base_models.keys())
            }
            path = model_manager.save_model(model=data, model_name=m_name, metadata=meta, feature_columns=self.feature_columns)
            return [path] if path else []

        # 2. 旧形式 StackingClassifier
        if hasattr(self, "stacking_classifier") and self.stacking_classifier:
            data = {
                "ensemble_classifier": self.stacking_classifier, "config": self.config,
                "feature_columns": self.feature_columns, "is_fitted": self.is_fitted
            }
            path = model_manager.save_model(model=data, model_name=m_name, metadata={"ensemble_type": "StackingEnsemble"}, feature_columns=self.feature_columns)
            return [path] if path else []

        # 3. その他（従来形式）
        logger.warning("Falling back to legacy model saving")
        path = f"{base_path}_legacy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(self, path)
        return [path]

    def load_models(self, base_path: str) -> bool:
        """アンサンブルモデルを読み込み"""
        import glob
        import joblib
        from sklearn.exceptions import InconsistentVersionWarning
        import warnings

        try:
            # 1. StackingClassifier ファイル検索
            f_list = sorted(glob.glob(f"{base_path}_stacking_classifier_*.pkl"))
            if f_list:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", InconsistentVersionWarning)
                    data = joblib.load(f_list[-1])
                if isinstance(data, dict) and "ensemble_classifier" in data:
                    self.stacking_classifier = data["ensemble_classifier"]
                    self.config, self.feature_columns, self.is_fitted = data.get("config", {}), data.get("feature_columns", []), data.get("is_fitted", False)
                    return True

            # 2. 統合モデルファイル検索
            f_list = sorted([f for f in glob.glob(f"{base_path}_*_*.pkl") if not f.endswith(("_config.pkl", "_meta_model.pkl"))])
            if f_list:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", InconsistentVersionWarning)
                    data = joblib.load(f_list[-1])
                if isinstance(data, dict) and "model" in data:
                    self.base_models, self.config, self.is_fitted = [data["model"]], data.get("config", {}), data.get("is_fitted", False)
                    return True

            logger.warning(f"No suitable model found for {base_path}")
            return False
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False



