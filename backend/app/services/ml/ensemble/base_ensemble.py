"""
アンサンブル学習の基底クラス

全てのアンサンブル手法の共通インターフェースと基本機能を定義します。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..common.config import ml_config_manager
from ..common.exceptions import MLModelError
from ..common.utils import get_feature_importance_unified
from ..evaluation.metrics import metrics_collector

logger = logging.getLogger(__name__)


def evaluate_model_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """評価処理の共通ラッパー。

    旧テストや一部の呼び出し側がモジュール関数として patch しているため、
    互換性を保つ目的で公開しておく。
    """
    return metrics_collector.calculate_comprehensive_metrics(
        y_true, y_pred, y_pred_proba
    )


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
        # 後方互換のため、名前だけでなく実体モデルも保持できるよう Any にしておく
        self._base_models_list: List[Any] = []
        self._meta_model_ref: Optional[Any] = None
        self.is_fitted = False
        self.feature_columns: Optional[List[str]] = None
        self.scaler: Optional[Any] = None
        self._fitted_base_models: Dict[str, Any] = {}

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

    @property
    def base_models(self) -> List[Any]:
        """後方互換性のためのプロパティ"""
        return self._base_models_list

    @base_models.setter
    def base_models(self, value: List[Any]) -> None:
        """後方互換性のためのセッター"""
        self._base_models_list = value

    def _create_base_model(
        self, model_type: str, model_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """ベースモデルを作成"""
        mt = model_type.lower()
        params = {k: v for k, v in (model_params or {}).items() if v is not None}
        ml_training = ml_config_manager.config.training
        seed = params.pop(
            "random_state",
            params.pop(
                "random_seed",
                self.config.get("random_state", ml_training.random_state),
            ),
        )
        n_jobs = params.pop("n_jobs", self.config.get("n_jobs", 1))

        # 学習フロー側の制御パラメータはモデルコンストラクタに渡さない
        for key in (
            "early_stopping_rounds",
            "num_boost_round",
            "test_size",
            "cv_splits",
            "cross_validation_folds",
            "use_cross_validation",
            "horizon_n",
            "prediction_horizon",
            "threshold_method",
            "threshold",
            "threshold_up",
            "threshold_down",
            "quantile_threshold",
            "train_test_split",
            "validation_split",
            "optimization_settings",
            "save_model",
            "symbol",
            "timeframe",
            "start_date",
            "end_date",
            "ensemble_config",
            "single_model_config",
            "base_model_params",
            "meta_model_params",
            "models",
            "base_models",
            "model_type",
            "method",
        ):
            params.pop(key, None)

        if mt == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMClassifier(
                n_estimators=params.pop("n_estimators", 100),
                learning_rate=params.pop("learning_rate", 0.1),
                max_depth=params.pop("max_depth", -1),
                random_state=seed,
                n_jobs=n_jobs,
                **params,
            )
        if mt == "xgboost":
            import xgboost as xgb

            return xgb.XGBClassifier(
                n_estimators=params.pop("n_estimators", 100),
                learning_rate=params.pop("learning_rate", 0.1),
                max_depth=params.pop("max_depth", 6),
                subsample=params.pop("subsample", 0.8),
                colsample_bytree=params.pop("colsample_bytree", 0.8),
                random_state=seed,
                n_jobs=n_jobs,
                eval_metric=params.pop("eval_metric", "logloss"),
                verbosity=params.pop("verbosity", 0),
                **params,
            )
        if mt == "catboost":
            import catboost as cb

            return cb.CatBoostClassifier(
                iterations=params.pop("iterations", params.pop("n_estimators", 100)),
                learning_rate=params.pop("learning_rate", 0.1),
                depth=params.pop("depth", params.pop("max_depth", 6)),
                random_seed=seed,
                thread_count=params.pop(
                    "thread_count",
                    n_jobs if isinstance(n_jobs, int) and n_jobs > 0 else 1,
                ),
                verbose=params.pop("verbose", 0),
                allow_writing_files=params.pop("allow_writing_files", False),
                **params,
            )
        if mt == "logistic_regression":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                random_state=seed,
                max_iter=params.pop("max_iter", ml_training.lr_max_iter),
                solver=params.pop("solver", "lbfgs"),
                C=params.pop("C", 1.0),
                penalty=params.pop("penalty", "l2"),
                l1_ratio=params.pop("l1_ratio", None),
                n_jobs=params.pop("n_jobs", None),
                **params,
            )

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

        if self._fitted_base_models:
            candidate_models = list(self._fitted_base_models.values())
        else:
            candidate_models = list(self._base_models_list)

        for i, model in enumerate(candidate_models):
            if model is None:
                logger.warning(f"モデル{i}: モデル本体がありません")
                continue
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

        from ..models.model_manager import model_manager

        m_name = os.path.basename(base_path)

        # 1. 自前実装 StackingEnsemble
        if hasattr(self, "_fitted_base_models") and self._fitted_base_models:
            data = {
                "fitted_base_models": self._fitted_base_models,
                "fitted_meta_model": getattr(self, "_fitted_meta_model", None),
                "base_model_types": getattr(self, "_base_model_types", []),
                "meta_model_type": getattr(
                    self, "_meta_model_type", "logistic_regression"
                ),
                "config": self.config,
                "feature_columns": self.feature_columns,
                "is_fitted": self.is_fitted,
                "passthrough": getattr(self, "passthrough", False),
                "ensemble_type": "StackingEnsemble",
            }
            meta = {
                "ensemble_type": "StackingEnsemble",
                "best_algorithm": "stacking",
                "feature_count": len(self.feature_columns or []),
                "fitted_base_models": list(self._fitted_base_models.keys()),
            }
            path = model_manager.save_model(
                model=data,
                model_name=m_name,
                metadata=meta,
                feature_columns=self.feature_columns,
            )
            return [path] if path else []

        # 2. 旧形式 StackingClassifier
        if hasattr(self, "stacking_classifier") and self.stacking_classifier:
            data = {
                "ensemble_classifier": self.stacking_classifier,
                "config": self.config,
                "feature_columns": self.feature_columns,
                "is_fitted": self.is_fitted,
            }
            path = model_manager.save_model(
                model=data,
                model_name=m_name,
                metadata={"ensemble_type": "StackingEnsemble"},
                feature_columns=self.feature_columns,
            )
            return [path] if path else []

        # 3. その他（従来形式）
        logger.warning("Falling back to legacy model saving")
        path = f"{base_path}_legacy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(self, path)
        return [path]

    def load_models(self, base_path: str) -> bool:
        """アンサンブルモデルを読み込み"""
        import warnings

        import joblib
        from sklearn.exceptions import InconsistentVersionWarning

        from ..common.utils import collect_unique_files

        try:
            # 1. StackingClassifier ファイル検索
            f_list = collect_unique_files([f"{base_path}_stacking_classifier_*.pkl"])
            f_list.sort()
            if f_list:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", InconsistentVersionWarning)
                    data = joblib.load(f_list[-1])
                if isinstance(data, dict) and "ensemble_classifier" in data:
                    self.stacking_classifier = data["ensemble_classifier"]
                    self.config, self.feature_columns, self.is_fitted = (
                        data.get("config", {}),
                        data.get("feature_columns", []),
                        data.get("is_fitted", False),
                    )
                    return True

            # 2. 統合モデルファイル検索
            f_list = collect_unique_files([f"{base_path}_*_*.pkl"])
            f_list = [
                f for f in f_list if not f.endswith(("_config.pkl", "_meta_model.pkl"))
            ]
            f_list.sort()
            if f_list:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", InconsistentVersionWarning)
                    data = joblib.load(f_list[-1])
                if isinstance(data, dict) and "model" in data:
                    self._base_models_list, self.config, self.is_fitted = (
                        [data["model"]],
                        data.get("config", {}),
                        data.get("is_fitted", False),
                    )
                    return True

            logger.warning(f"No suitable model found for {base_path}")
            return False
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False
