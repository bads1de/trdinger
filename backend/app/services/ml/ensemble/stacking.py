"""
スタッキング（Stacking）アンサンブル手法の実装

自前実装によるスタッキング。OOF予測を生成し、メタモデルを直接学習することで
計算コストを最小化しつつ、各モデルの強みを活かした高精度な予測を実現します。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

from app.config.unified_config import unified_config

from ....utils.error_handler import ModelError
from ..common.utils import create_temporal_cv_splitter, validate_training_inputs
from .base_ensemble import BaseEnsemble

logger = logging.getLogger(__name__)


class StackingEnsemble(BaseEnsemble):
    """
    自前実装によるスタッキングアンサンブル

    sklearnのStackingClassifierを使用せず、以下の3ステップで実装:
    1. OOF予測生成: 各ベースモデルでcross_val_predictを実行
    2. メタモデル学習: OOF予測をメタ特徴量としてメタモデルを直接学習
    3. ベースモデル最終fit: 全データで各ベースモデルをフィット（推論用）

    これにより、sklearn.StackingClassifierを使用した場合と比較して
    ベースモデルの学習回数が約半分になり、計算コストを削減できます。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初期化

        Args:
            config: スタッキング設定
        """
        super().__init__(config)

        # スタッキング固有の設定
        self._base_model_types: List[str] = config.get(
            "base_models", ["lightgbm", "xgboost", "catboost"]
        )
        self._meta_model_type: str = (
            config.get("meta_model", "logistic_regression") or "logistic_regression"
        )
        self.cv_folds = config.get("cv_folds", 5)
        self.stack_method = config.get("stack_method", "predict_proba")
        self.random_state = config.get("random_state", 42)
        self.n_jobs = config.get("n_jobs", -1)  # 並列処理
        self.passthrough = config.get(
            "passthrough", False
        )  # 元特徴量をメタモデルに渡すか

        # 学習済みモデルを保持
        self._fitted_base_models: Dict[str, Any] = {}  # {name: fitted_model}
        self._fitted_meta_model: Optional[Any] = None
        self.oof_predictions: Optional[np.ndarray] = None  # OOF予測値を保持
        self.oof_base_model_predictions: Optional[pd.DataFrame] = (
            None  # 各ベースモデルのOOF予測確率を保持
        )
        self.X_train_original: Optional[pd.DataFrame] = (
            None  # メタモデルの特徴量として使うため
        )
        self.y_train_original: Optional[pd.Series] = None  # メタラベル生成のため

        self.y_train_original: Optional[pd.Series] = None  # メタラベル生成のため

        logger.info(
            f"StackingEnsemble初期化（自前実装）: base_models={self._base_model_types}, "
            f"meta_model={self._meta_model_type}, cv_folds={self.cv_folds}, "
            f"stack_method={self.stack_method}, n_jobs={self.n_jobs}, "
            f"passthrough={self.passthrough}"
        )

    @property
    def base_models(self) -> List[str]:
        """後方互換性のためのプロパティ"""
        return self._base_model_types

    @base_models.setter
    def base_models(self, value: List[str]) -> None:
        """後方互換性のためのセッター"""
        self._base_model_types = value

    @property
    def meta_model(self) -> str:
        """後方互換性のためのプロパティ"""
        return self._meta_model_type

    @meta_model.setter
    def meta_model(self, value: str) -> None:
        """後方互換性のためのセッター"""
        self._meta_model_type = value

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        base_model_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        自前実装のスタッキングアンサンブルモデルを学習
        """
        try:
            logger.info("スタッキング学習開始（計算コスト最適化版）")
            validate_training_inputs(X_train, y_train, X_test, y_test, log_info=True)
            self.feature_columns = X_train.columns.tolist()
            estimators = self._create_base_estimators(base_model_params)
            cv = self._create_cv_splitter(X_train)

            # ステップ1: OOF予測計算
            logger.info("[Step 1/3] ベースモデルのOOF予測を計算中...")
            oof_base_model_preds = self._calculate_oof_predictions(
                estimators, X_train, y_train, cv
            )

            # ステップ2: メタモデル学習
            logger.info("[Step 2/3] メタモデルを学習中...")
            meta_model_params = self.config.get("meta_model_params", {})
            meta_oof_pred = self._train_meta_model(
                oof_base_model_preds,
                X_train,
                y_train,
                meta_model_params=meta_model_params,
            )

            # ステップ3: ベースモデル最終フィット
            logger.info("[Step 3/3] ベースモデルを全データでフィット中...")
            self._fit_base_models_final(estimators, X_train, y_train)

            # 学習完了・結果保存
            self.is_fitted = True
            self.oof_predictions = meta_oof_pred[:, 1]
            self.oof_base_model_predictions = oof_base_model_preds
            self.X_train_original = X_train.copy()
            self.y_train_original = y_train.copy()

            # アンサンブル全体の評価
            ensemble_result = self._evaluate_ensemble(X_test, y_test)
            ensemble_result.update(
                {
                    "model_type": "StackingEnsemble",
                    "base_models": self._base_model_types,
                    "meta_model": self._meta_model_type,
                    "cv_folds": self.cv_folds,
                    "stack_method": self.stack_method,
                    "fitted_base_models": list(self._fitted_base_models.keys()),
                }
            )
            return ensemble_result

        except Exception as e:
            logger.error(f"スタッキングアンサンブル学習エラー: {e}")
            raise ModelError(f"スタッキングアンサンブル学習に失敗しました: {e}")

    def _calculate_oof_predictions(
        self,
        estimators: List[Tuple[str, Any]],
        X: pd.DataFrame,
        y: pd.Series,
        cv: Any,
    ) -> pd.DataFrame:
        """ベースモデルのOOF予測を計算"""
        oof_preds = pd.DataFrame(index=X.index)
        for name, estimator in estimators:
            try:
                pred = cross_val_predict(
                    estimator, X, y, cv=cv, method="predict_proba", n_jobs=self.n_jobs
                )
                oof_preds[name] = pred[:, 1]
            except Exception as e:
                logger.warning(f"  {name}のOOF予測計算エラー: {e}")
        if oof_preds.empty:
            raise ModelError("有効なOOF予測が生成できませんでした")
        return oof_preds

    def _train_meta_model(
        self,
        oof_preds: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        meta_model_params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """メタモデルを学習"""
        meta_features = (
            pd.concat([oof_preds, X], axis=1) if self.passthrough else oof_preds
        )
        self._fitted_meta_model = self._create_base_model(
            self._meta_model_type, meta_model_params
        )
        self._fitted_meta_model.fit(meta_features, y)

        meta_cv = self._create_cv_splitter(X)
        return cross_val_predict(
            clone(
                self._create_base_model(self._meta_model_type, meta_model_params)
            ),
            meta_features,
            y,
            cv=meta_cv,
            method="predict_proba",
            n_jobs=self.n_jobs,
        )

    def _fit_base_models_final(
        self,
        estimators: List[Tuple[str, Any]],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """全データでベースモデルをフィット"""
        self._fitted_base_models.clear()
        for name, estimator in estimators:
            try:
                model = clone(estimator)
                model.fit(X, y)
                self._fitted_base_models[name] = model
            except Exception as e:
                logger.warning(f"  {name}の最終フィットエラー: {e}")

    def _create_cv_splitter(self, X_train: pd.DataFrame) -> Any:
        """
        クロスバリデーション分割器を作成

        Args:
            X_train: 学習データ（インデックスから時系列情報を取得）

        Returns:
            CVスプリッター
        """
        cv_strategy = self.config.get("cv_strategy", "purged_kfold")
        t1_horizon_n = unified_config.ml.training.label_generation.horizon_n
        pct_embargo = getattr(unified_config.ml.training, "pct_embargo", 0.01)
        return create_temporal_cv_splitter(
            cv_strategy=cv_strategy,
            n_splits=self.cv_folds,
            index=X_train.index,
            pct_embargo=pct_embargo,
            horizon_n=t1_horizon_n,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        スタッキングアンサンブルで予測を実行

        Args:
            X: 特徴量DataFrame

        Returns:
            予測結果 (predict_probaの結果を返す、後方互換性のため)
        """
        return self.predict_proba(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        スタッキングアンサンブルで予測確率を取得
        """
        if (
            not self.is_fitted
            or not self._fitted_base_models
            or self._fitted_meta_model is None
        ):
            raise ModelError("モデルが学習されていません")

        # ベースモデルの予測確率
        base_preds = pd.DataFrame(
            {
                name: model.predict_proba(X)[:, 1]
                for name, model in self._fitted_base_models.items()
            },
            index=X.index,
        )

        # メタモデルで予測
        meta_features = (
            pd.concat([base_preds, X], axis=1) if self.passthrough else base_preds
        )
        return self._fitted_meta_model.predict_proba(meta_features)

    def _create_base_estimators(
        self, base_model_params: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[Tuple[str, Any]]:
        """
        ベースモデルのリストを作成（StackingClassifier用）

        Returns:
            (name, estimator)のタプルのリスト
        """
        estimators = []

        for model_type in self.base_models:
            try:
                params = {}
                if base_model_params:
                    params = base_model_params.get(model_type, base_model_params.get(model_type.lower(), {})) or {}
                model = self._create_base_model(model_type, params)
                estimators.append((model_type, model))
                logger.info(f"ベースモデル追加: {model_type}")
            except Exception as e:
                logger.warning(f"ベースモデル({model_type})の作成をスキップ: {e}")
                continue

        if not estimators:
            raise ModelError("有効なベースモデルが作成できませんでした")

        return estimators

    def _evaluate_ensemble(
        self,
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.Series],
    ) -> Dict[str, Any]:
        """
        スタッキングアンサンブル全体の評価を実行

        Args:
            X_test: テスト用特徴量
            y_test: テスト用ターゲット

        Returns:
            アンサンブル評価結果
        """
        result = {
            "model_type": "StackingEnsemble",
            "base_models": self._base_model_types,
            "meta_model": self._meta_model_type,
            "cv_folds": self.cv_folds,
            "stack_method": self.stack_method,
            "fitted_base_models": list(self._fitted_base_models.keys()),
        }

        # テストデータがある場合はアンサンブル予測を評価
        if X_test is not None and y_test is not None:
            try:
                y_pred = self.predict(X_test)
                y_pred_proba = self.predict_proba(X_test)

                ensemble_metrics = self._evaluate_predictions(
                    y_test, y_pred, y_pred_proba
                )
                result.update(ensemble_metrics)

            except Exception as e:
                logger.warning(f"アンサンブル評価でエラー: {e}")
                result["evaluation_error"] = str(e)

        return result

    def get_feature_importance(self) -> Dict[str, float]:
        """
        スタッキングアンサンブルの特徴量重要度を取得

        メタモデルの特徴量重要度を返します。
        ベースモデルの重要度は個別に取得する必要があります。

        Returns:
            特徴量重要度の辞書
        """
        if not self.is_fitted or self._fitted_meta_model is None:
            return {}

        try:
            # メタモデルの特徴量重要度を取得
            meta_model = self._fitted_meta_model

            # ベースモデル名を取得して特徴量名として使用
            estimator_names = list(self._fitted_base_models.keys())

            if hasattr(meta_model, "feature_importances_"):
                # Tree-based models
                importances = meta_model.feature_importances_

                if len(importances) == len(estimator_names):
                    feature_names = estimator_names
                else:
                    feature_names = [
                        f"meta_feature_{i}" for i in range(len(importances))
                    ]

                return dict(zip(feature_names, importances))
            elif hasattr(meta_model, "coef_"):
                # Linear models
                coef = meta_model.coef_
                # 2クラス分類のLogisticRegressionの場合、(1, n_features) の形状
                if coef.ndim > 1:
                    # 全クラスの平均絶対値、または単一要素ならその行を使用
                    if coef.shape[0] == 1:
                        coef = np.abs(coef[0])
                    else:
                        coef = np.abs(coef).mean(axis=0)
                else:
                    coef = np.abs(coef)

                if len(coef) == len(estimator_names):
                    return {
                        estimator_names[i]: float(coef[i]) for i in range(len(coef))
                    }
                else:
                    feature_names = [f"meta_feature_{i}" for i in range(len(coef))]
                    return {feature_names[i]: float(coef[i]) for i in range(len(coef))}
            else:
                logger.warning(
                    f"メタモデル({self._meta_model_type})は特徴量重要度をサポートしていません"
                )
                return {}

        except Exception as e:
            logger.warning(f"特徴量重要度の取得でエラー: {e}")
            return {}

    def load_models(self, base_path: str) -> bool:
        """
        スタッキングアンサンブルモデルを読み込み

        Args:
            base_path: モデル読み込みパス

        Returns:
            読み込み成功フラグ
        """
        try:
            import json
            import os

            import joblib
            from ..common.utils import collect_unique_files

            # base_path に紐づくファイルを優先し、見つからない場合のみ保存ディレクトリ全体を探す
            local_patterns = [
                f"{base_path}_stacking_ensemble_*.pkl",
                f"{base_path}_stacking_*.pkl",
                f"{base_path}_*.pkl",
            ]
            fallback_patterns = [
                os.path.join(
                    unified_config.ml.model.model_save_path, "stacking_*.pkl"
                ),
                os.path.join(
                    unified_config.ml.model.model_save_path,
                    "stacking_ensemble_*.pkl",
                ),
            ]

            model_files = collect_unique_files(local_patterns)
            if not model_files:
                model_files = collect_unique_files(fallback_patterns)

            if not model_files:
                logger.warning(
                    f"モデルファイルが見つかりません: {', '.join(local_patterns + fallback_patterns)}"
                )
                return False

            def _latest_model_key(path: str) -> tuple[float, str]:
                try:
                    return (os.path.getmtime(path), path)
                except OSError:
                    return (0.0, path)

            model_path = max(model_files, key=_latest_model_key)

            # モデルを読み込み
            model_data = joblib.load(model_path)

            payload = model_data
            sidecar_metadata = {}
            if isinstance(model_data, dict):
                sidecar_metadata = model_data.get("metadata", {}) or {}
                if "model" in model_data:
                    payload = model_data["model"]

            if isinstance(payload, dict):
                fitted_base_models = payload.get("fitted_base_models")
                fitted_meta_model = payload.get("fitted_meta_model")
                if not fitted_base_models or fitted_meta_model is None:
                    logger.warning("不明なモデル形式です")
                    return False

                self._fitted_base_models = dict(fitted_base_models)
                self._fitted_meta_model = fitted_meta_model
                self._base_model_types = list(
                    payload.get(
                        "base_model_types", list(self._fitted_base_models.keys())
                    )
                )
                self._meta_model_type = payload.get(
                    "meta_model_type", "logistic_regression"
                )
                self.feature_columns = payload.get(
                    "feature_columns", model_data.get("feature_columns", None)
                )
                self.config = payload.get("config", self.config)
                self.passthrough = payload.get("passthrough", False)
                self.cv_folds = payload.get("cv_folds", self.cv_folds)
                self.stack_method = payload.get("stack_method", self.stack_method)
                self.oof_predictions = payload.get("oof_predictions", None)
                self.oof_base_model_predictions = payload.get(
                    "oof_base_model_predictions", None
                )
                self.X_train_original = payload.get("X_train_original", None)
                self.y_train_original = payload.get("y_train_original", None)
                self.is_fitted = bool(payload.get("is_fitted", True))
                logger.info("自前実装のスタッキングモデルを読み込み完了")
            elif hasattr(payload, "_fitted_base_models"):
                self._fitted_base_models = dict(
                    getattr(payload, "_fitted_base_models", {})
                )
                self._fitted_meta_model = getattr(payload, "_fitted_meta_model", None)
                self._base_model_types = list(
                    getattr(
                        payload,
                        "_base_model_types",
                        list(self._fitted_base_models.keys()),
                    )
                )
                self._meta_model_type = getattr(
                    payload, "_meta_model_type", "logistic_regression"
                )
                self.feature_columns = getattr(
                    payload, "feature_columns", None
                ) or model_data.get("feature_columns", None)
                self.config = getattr(payload, "config", self.config)
                self.passthrough = getattr(payload, "passthrough", False)
                self.cv_folds = getattr(payload, "cv_folds", self.cv_folds)
                self.stack_method = getattr(payload, "stack_method", self.stack_method)
                self.oof_predictions = getattr(payload, "oof_predictions", None)
                self.oof_base_model_predictions = getattr(
                    payload, "oof_base_model_predictions", None
                )
                self.X_train_original = getattr(payload, "X_train_original", None)
                self.y_train_original = getattr(payload, "y_train_original", None)
                self.is_fitted = bool(getattr(payload, "is_fitted", True))
                logger.info("スタッキングモデルオブジェクトを読み込み完了")
            else:
                logger.warning("不明なモデル形式です")
                return False

            # メタデータを読み込み（存在する場合）
            metadata_candidates = []
            if model_path.endswith(".pkl"):
                metadata_candidates.extend(
                    [
                        model_path.replace(".pkl", ".meta.json"),
                        model_path.replace(".pkl", "_metadata.json"),
                    ]
                )
            else:
                metadata_candidates.extend(
                    [model_path + ".meta.json", model_path + "_metadata.json"]
                )
            metadata = None
            for metadata_path in metadata_candidates:
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    break

            if metadata:
                self._base_model_types = metadata.get(
                    "base_models", self._base_model_types
                )
                self._meta_model_type = metadata.get(
                    "meta_model", self._meta_model_type
                )
                self.cv_folds = metadata.get("cv_folds", self.cv_folds)
                self.stack_method = metadata.get("stack_method", self.stack_method)
                if "feature_columns" in metadata:
                    self.feature_columns = metadata["feature_columns"]
                if "passthrough" in metadata:
                    self.passthrough = metadata["passthrough"]
            elif sidecar_metadata:
                self._base_model_types = sidecar_metadata.get(
                    "base_models", self._base_model_types
                )
                self._meta_model_type = sidecar_metadata.get(
                    "meta_model", self._meta_model_type
                )
                self.cv_folds = sidecar_metadata.get("cv_folds", self.cv_folds)
                self.stack_method = sidecar_metadata.get(
                    "stack_method", self.stack_method
                )
                if "feature_columns" in sidecar_metadata:
                    self.feature_columns = sidecar_metadata["feature_columns"]
                if "passthrough" in sidecar_metadata:
                    self.passthrough = sidecar_metadata["passthrough"]

            self.is_fitted = bool(self._fitted_base_models) and self._fitted_meta_model is not None
            logger.info(f"スタッキングアンサンブルモデルを読み込みました: {model_path}")
            return True

        except Exception as e:
            logger.error(f"アンサンブルモデル読み込みエラー: {e}")
            return False

    def get_oof_predictions(self) -> Optional[np.ndarray]:
        """スタッキングモデルのOOF予測確率を返す"""
        return self.oof_predictions

    def get_oof_base_model_predictions(self) -> Optional[pd.DataFrame]:
        """各ベースモデルのOOF予測確率をDataFrameで返す"""
        return self.oof_base_model_predictions

    def get_X_train_original(self) -> Optional[pd.DataFrame]:
        """学習に使用した元の特徴量Xを返す"""
        return self.X_train_original

    def get_y_train_original(self) -> Optional[pd.Series]:
        """学習に使用した元のラベルyを返す"""
        return self.y_train_original

    def predict_base_models_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        各ベースモデルの予測確率を取得する

        Args:
            X: 特徴量DataFrame

        Returns:
            各ベースモデルの予測確率を含むDataFrame
        """
        if not self.is_fitted or not self._fitted_base_models:
            raise ModelError("モデルが学習されていません")

        try:
            base_preds = pd.DataFrame(index=X.index)
            for name, model in self._fitted_base_models.items():
                proba = model.predict_proba(X)
                base_preds[name] = proba[:, 1]  # ポジティブクラスの確率

            return base_preds

        except Exception as e:
            logger.error(f"ベースモデル予測確率取得エラー: {e}")
            raise ModelError(f"ベースモデル予測確率の取得に失敗しました: {e}")

    def clear_training_data(self) -> None:
        """
        学習用の一時データをクリアしてメモリを解放する

        推論（predict）には不要な、学習時の元データやOOF予測値を削除します。
        モデルのシリアライズ（保存）前に実行することで、ファイルサイズを削減できます。
        """
        self.X_train_original = None
        self.y_train_original = None
        self.oof_predictions = None
        self.oof_base_model_predictions = None
        logger.debug("StackingEnsembleの学習用一時データをクリアしました")

    def cleanup(self) -> None:
        """
        リソースをクリーンアップ

        大量のメモリを消費するモデルオブジェクトを明示的に解放します。
        """
        logger.info("StackingEnsembleのリソースをクリーンアップ中...")

        # ベースモデルのクリーンアップ
        if self._fitted_base_models:
            for name in list(self._fitted_base_models.keys()):
                self._fitted_base_models[name] = None
            self._fitted_base_models.clear()

        # メタモデルのクリーンアップ
        self._fitted_meta_model = None

        # OOF予測データのクリーンアップ
        self.oof_predictions = None
        self.oof_base_model_predictions = None
        self.X_train_original = None
        self.y_train_original = None

        self.is_fitted = False
        logger.info("StackingEnsembleのリソースクリーンアップ完了")
