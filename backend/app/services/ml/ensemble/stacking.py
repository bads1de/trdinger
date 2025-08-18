"""
スタッキング（Stacking）アンサンブル手法の実装

scikit-learnのStackingClassifierを使用した標準的なスタッキング実装。
複数の異なるベースモデルの予測をメタモデルで統合することで、
各モデルの強みを活かした高精度な予測を実現します。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold

from ....utils.error_handler import ModelError
from .base_ensemble import BaseEnsemble

logger = logging.getLogger(__name__)


class StackingEnsemble(BaseEnsemble):
    """
    scikit-learnのStackingClassifierを使用したスタッキングアンサンブル実装

    複数の異なるベースモデルの予測をメタモデルで統合し、
    各モデルの強みを活かした予測を行います。
    """

    def __init__(
        self, config: Dict[str, Any], automl_config: Optional[Dict[str, Any]] = None
    ):
        """
        初期化

        Args:
            config: スタッキング設定
            automl_config: AutoML設定（オプション）
        """
        super().__init__(config, automl_config)

        # スタッキング固有の設定
        self.base_models = config.get("base_models", ["lightgbm", "random_forest"])
        self.meta_model = config.get("meta_model", "logistic_regression")
        self.cv_folds = config.get("cv_folds", 5)
        self.stack_method = config.get("stack_method", "predict_proba")
        self.random_state = config.get("random_state", 42)
        self.n_jobs = config.get("n_jobs", -1)  # 並列処理
        self.passthrough = config.get(
            "passthrough", False
        )  # 元特徴量をメタモデルに渡すか

        # scikit-learn StackingClassifier
        self.stacking_classifier = None

        logger.info(
            f"StackingClassifier初期化: base_models={self.base_models}, "
            f"meta_model={self.meta_model}, cv_folds={self.cv_folds}, "
            f"stack_method={self.stack_method}, n_jobs={self.n_jobs}, "
            f"passthrough={self.passthrough}"
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        base_model_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        StackingClassifierを使用してスタッキングアンサンブルモデルを学習

        Args:
            X_train: 学習用特徴量
            y_train: 学習用ターゲット
            X_test: テスト用特徴量（オプション）
            y_test: テスト用ターゲット（オプション）
            base_model_params: 各ベースモデルの最適化されたパラメータ（オプション）

        Returns:
            学習結果の辞書
        """
        try:
            logger.info("StackingClassifierによるスタッキングアンサンブル学習を開始")

            # 入力データの検証
            if X_train is None or X_train.empty:
                raise ValueError("学習用特徴量データが空です")
            if y_train is None or len(y_train) == 0:
                raise ValueError("学習用ターゲットデータが空です")
            if len(X_train) != len(y_train):
                raise ValueError("特徴量とターゲットの長さが一致しません")

            logger.info(
                f"学習データサイズ: {len(X_train)}行, {len(X_train.columns)}特徴量"
            )
            logger.info(f"ターゲット分布: {y_train.value_counts().to_dict()}")

            self.feature_columns = X_train.columns.tolist()

            # ベースモデルのリストを作成
            try:
                estimators = self._create_base_estimators()
                logger.info(f"ベースモデル作成完了: {[name for name, _ in estimators]}")
            except Exception as e:
                logger.error(f"ベースモデル作成エラー: {e}")
                raise ModelError(f"ベースモデルの作成に失敗しました: {e}")

            # メタモデルを作成
            try:
                final_estimator = self._create_base_model(self.meta_model)
                logger.info(f"メタモデル作成完了: {self.meta_model}")
            except Exception as e:
                logger.error(f"メタモデル作成エラー: {e}")
                raise ModelError(
                    f"メタモデル({self.meta_model})の作成に失敗しました: {e}"
                )

            # クロスバリデーション設定
            cv = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )

            # StackingClassifierを初期化
            self.stacking_classifier = StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                cv=cv,
                stack_method=self.stack_method,
                n_jobs=self.n_jobs,
                passthrough=self.passthrough,
                verbose=0,  # ログ抑制
            )

            logger.info("StackingClassifier学習開始")

            # 学習実行
            self.stacking_classifier.fit(X_train, y_train)
            self.is_fitted = True

            logger.info("StackingClassifier学習完了")

            # アンサンブル全体の評価
            ensemble_result = self._evaluate_ensemble(X_test, y_test)

            # 学習結果情報を追加
            ensemble_result.update(
                {
                    "model_type": "StackingClassifier",
                    "base_models": self.base_models,
                    "meta_model": self.meta_model,
                    "cv_folds": self.cv_folds,
                    "stack_method": self.stack_method,
                    "n_jobs": self.n_jobs,
                    "passthrough": self.passthrough,
                    "sklearn_implementation": True,
                    "training_samples": len(X_train),
                    "test_samples": len(X_test) if X_test is not None else 0,
                }
            )

            logger.info("スタッキングアンサンブル学習完了")
            return ensemble_result

        except Exception as e:
            logger.error(f"スタッキングアンサンブル学習エラー: {e}")
            raise ModelError(f"スタッキングアンサンブル学習に失敗しました: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        StackingClassifierで予測を実行

        Args:
            X: 特徴量DataFrame

        Returns:
            予測結果
        """
        if not self.is_fitted or self.stacking_classifier is None:
            raise ModelError("モデルが学習されていません")

        return self.stacking_classifier.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        StackingClassifierで予測確率を取得

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率の配列
        """
        if not self.is_fitted or self.stacking_classifier is None:
            raise ModelError("モデルが学習されていません")

        return self.stacking_classifier.predict_proba(X)

    def _create_base_estimators(self) -> List[Tuple[str, Any]]:
        """
        ベースモデルのリストを作成（StackingClassifier用）

        Returns:
            (name, estimator)のタプルのリスト
        """
        estimators = []

        for model_type in self.base_models:
            try:
                model = self._create_base_model(model_type)
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
        StackingClassifierアンサンブル全体の評価を実行

        Args:
            X_test: テスト用特徴量
            y_test: テスト用ターゲット

        Returns:
            アンサンブル評価結果
        """
        result = {
            "model_type": "StackingClassifier",
            "base_models": self.base_models,
            "meta_model": self.meta_model,
            "cv_folds": self.cv_folds,
            "stack_method": self.stack_method,
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
        if not self.is_fitted or self.stacking_classifier is None:
            return {}

        try:
            # メタモデルの特徴量重要度を取得
            final_estimator = self.stacking_classifier.final_estimator_

            if hasattr(final_estimator, "feature_importances_"):
                # Tree-based models
                importances = final_estimator.feature_importances_
                feature_names = [f"meta_feature_{i}" for i in range(len(importances))]
                return dict(zip(feature_names, importances))
            elif hasattr(final_estimator, "coef_"):
                # Linear models
                coef = final_estimator.coef_
                if coef.ndim > 1:
                    coef = np.abs(coef).mean(axis=0)
                else:
                    coef = np.abs(coef)
                feature_names = [f"meta_feature_{i}" for i in range(len(coef))]
                return dict(zip(feature_names, coef))
            else:
                logger.warning(
                    f"メタモデル({self.meta_model})は特徴量重要度をサポートしていません"
                )
                return {}

        except Exception as e:
            logger.warning(f"特徴量重要度の取得でエラー: {e}")
            return {}

    def save_models(self, model_path: str) -> bool:
        """
        スタッキングアンサンブルモデルを保存

        Args:
            model_path: モデル保存パス

        Returns:
            保存成功フラグ
        """
        if not self.is_fitted or self.stacking_classifier is None:
            logger.warning("学習済みモデルがないため保存をスキップします")
            return False

        try:
            import joblib
            import os

            # ディレクトリを作成
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # StackingClassifierを保存
            joblib.dump(self.stacking_classifier, model_path)

            # メタデータを保存
            metadata = {
                "model_type": "StackingClassifier",
                "base_models": self.base_models,
                "meta_model": self.meta_model,
                "cv_folds": self.cv_folds,
                "stack_method": self.stack_method,
                "feature_columns": self.feature_columns,
                "is_fitted": self.is_fitted,
            }

            metadata_path = model_path.replace(".pkl", "_metadata.json")
            import json

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"スタッキングアンサンブルモデルを保存しました: {model_path}")
            return True

        except Exception as e:
            logger.error(f"モデル保存エラー: {e}")
            return False

    def load_models(self, model_path: str) -> bool:
        """
        スタッキングアンサンブルモデルを読み込み

        Args:
            model_path: モデル読み込みパス

        Returns:
            読み込み成功フラグ
        """
        try:
            import joblib
            import os
            import json

            if not os.path.exists(model_path):
                logger.warning(f"モデルファイルが見つかりません: {model_path}")
                return False

            # StackingClassifierを読み込み
            self.stacking_classifier = joblib.load(model_path)

            # メタデータを読み込み
            metadata_path = model_path.replace(".pkl", "_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                self.base_models = metadata.get("base_models", self.base_models)
                self.meta_model = metadata.get("meta_model", self.meta_model)
                self.cv_folds = metadata.get("cv_folds", self.cv_folds)
                self.stack_method = metadata.get("stack_method", self.stack_method)
                self.feature_columns = metadata.get("feature_columns", None)

            self.is_fitted = True
            logger.info(f"スタッキングアンサンブルモデルを読み込みました: {model_path}")
            return True

        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            return False
