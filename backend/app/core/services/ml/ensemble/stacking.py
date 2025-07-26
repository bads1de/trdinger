"""
スタッキング（Stacked Generalization）アンサンブル手法の実装

複数の異なるアルゴリズムの予測をメタモデルの入力として使用し、
最終的な予測を行うことで予測精度を向上させます。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .base_ensemble import BaseEnsemble
from ....utils.unified_error_handler import UnifiedModelError

logger = logging.getLogger(__name__)


class StackingEnsemble(BaseEnsemble):
    """
    スタッキングアンサンブル手法の実装

    複数の異なるベースモデルの予測をメタ特徴量として使用し、
    メタモデルで最終予測を行います。
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
        self.base_model_types = config.get("base_models", ["lightgbm", "random_forest"])
        self.meta_model_type = config.get("meta_model", "logistic_regression")
        self.cv_folds = config.get("cv_folds", 5)
        self.use_probas = config.get("use_probas", True)
        self.random_state = config.get("random_state", 42)

        logger.info(
            f"スタッキングアンサンブル初期化: base_models={self.base_model_types}, "
            f"meta_model={self.meta_model_type}, cv_folds={self.cv_folds}"
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        スタッキングアンサンブルモデルを学習

        Args:
            X_train: 学習用特徴量
            y_train: 学習用ターゲット
            X_test: テスト用特徴量（オプション）
            y_test: テスト用ターゲット（オプション）

        Returns:
            学習結果の辞書
        """
        try:
            logger.info("スタッキングアンサンブル学習を開始")

            self.feature_columns = X_train.columns.tolist()
            self.base_models = []

            # Step 1: ベースモデルを学習してメタ特徴量を生成
            meta_features_train = self._generate_meta_features(X_train, y_train)

            # Step 2: ベースモデルを全データで再学習
            base_model_results = []
            for i, model_type in enumerate(self.base_model_types):
                logger.info(f"ベースモデル {model_type} を全データで学習中...")

                base_model = self._create_base_model(model_type)

                # モデルを学習
                if hasattr(base_model, "_train_model_impl"):
                    # LightGBMModel等のカスタムモデル
                    if X_test is not None and y_test is not None:
                        result = base_model._train_model_impl(
                            X_train, X_test, y_train, y_test
                        )
                    else:
                        # テストデータがない場合は学習データから分割
                        from sklearn.model_selection import train_test_split

                        X_train_split, X_val_split, y_train_split, y_val_split = (
                            train_test_split(
                                X_train,
                                y_train,
                                test_size=0.2,
                                random_state=self.random_state,
                            )
                        )
                        result = base_model._train_model_impl(
                            X_train_split, X_val_split, y_train_split, y_val_split
                        )

                    base_model.is_trained = True
                    base_model.feature_columns = self.feature_columns
                    base_model_results.append(result)

                else:
                    # scikit-learn系モデル
                    base_model.fit(X_train, y_train)

                    # 評価（テストデータがある場合）
                    if X_test is not None and y_test is not None:
                        y_pred = base_model.predict(X_test)
                        y_pred_proba = None
                        if hasattr(base_model, "predict_proba"):
                            y_pred_proba = base_model.predict_proba(X_test)

                        result = self._evaluate_predictions(
                            y_test, y_pred, y_pred_proba
                        )
                        base_model_results.append(result)

                self.base_models.append(base_model)

            # Step 3: メタモデルを学習
            logger.info(f"メタモデル {self.meta_model_type} を学習中...")
            self.meta_model = self._create_base_model(self.meta_model_type)

            if hasattr(self.meta_model, "_train_model_impl"):
                # LightGBMModel等の場合
                meta_features_df = pd.DataFrame(meta_features_train)
                from sklearn.model_selection import train_test_split

                X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
                    meta_features_df,
                    y_train,
                    test_size=0.2,
                    random_state=self.random_state,
                )
                meta_result = self.meta_model._train_model_impl(
                    X_meta_train, X_meta_val, y_meta_train, y_meta_val
                )
                self.meta_model.is_trained = True
                self.meta_model.feature_columns = [
                    f"meta_feature_{i}" for i in range(meta_features_train.shape[1])
                ]
            else:
                # scikit-learn系の場合
                self.meta_model.fit(meta_features_train, y_train)
                meta_result = {"meta_model_type": self.meta_model_type}

            self.is_fitted = True

            # アンサンブル全体の評価
            ensemble_result = self._evaluate_ensemble(
                X_test, y_test, base_model_results, meta_result
            )

            logger.info("スタッキングアンサンブル学習完了")
            return ensemble_result

        except Exception as e:
            logger.error(f"スタッキングアンサンブル学習エラー: {e}")
            raise UnifiedModelError(f"スタッキングアンサンブル学習に失敗しました: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        スタッキングアンサンブルで予測を実行

        Args:
            X: 特徴量DataFrame

        Returns:
            予測結果
        """
        if not self.is_fitted:
            raise UnifiedModelError("モデルが学習されていません")

        # ベースモデルの予測からメタ特徴量を生成
        meta_features = self._generate_meta_features_predict(X)

        # メタモデルで最終予測
        if hasattr(self.meta_model, "predict") and hasattr(
            self.meta_model, "is_trained"
        ):
            # LightGBMModel等
            meta_features_df = pd.DataFrame(meta_features)
            predictions = self.meta_model.predict(meta_features_df)
            if predictions.ndim > 1:
                predictions = np.argmax(predictions, axis=1)
        else:
            # scikit-learn系
            predictions = self.meta_model.predict(meta_features)

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        スタッキングアンサンブルで予測確率を取得

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率の配列
        """
        if not self.is_fitted:
            raise UnifiedModelError("モデルが学習されていません")

        # ベースモデルの予測からメタ特徴量を生成
        meta_features = self._generate_meta_features_predict(X)

        # メタモデルで予測確率を取得
        if hasattr(self.meta_model, "predict") and hasattr(
            self.meta_model, "is_trained"
        ):
            # LightGBMModel等のカスタムモデル
            meta_features_df = pd.DataFrame(meta_features)
            probabilities = self.meta_model.predict(meta_features_df)
            # カスタムモデルは既に適切な形状で確率を返すことを期待
        else:
            # scikit-learn系
            if hasattr(self.meta_model, "predict_proba"):
                probabilities = self.meta_model.predict_proba(meta_features)
            else:
                # predict_probaがない場合はpredictの結果をワンホット化
                pred = self.meta_model.predict(meta_features)
                n_classes = len(np.unique(pred))
                probabilities = np.eye(n_classes)[pred]

        # 3クラス分類であることを確認
        if probabilities.ndim == 2 and probabilities.shape[1] == 3:
            return probabilities
        else:
            raise UnifiedModelError(
                f"メタモデルの予測確率が3クラス分類ではありません: {probabilities.shape}"
            )

    def _generate_meta_features(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        クロスバリデーションを使用してメタ特徴量を生成

        Args:
            X: 特徴量DataFrame
            y: ターゲットSeries

        Returns:
            メタ特徴量の配列
        """
        meta_features = []

        # クロスバリデーション設定
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        for model_type in self.base_model_types:
            logger.info(f"メタ特徴量生成: {model_type}")

            # ベースモデルを作成
            base_model = self._create_base_model(model_type)

            if hasattr(base_model, "_train_model_impl"):
                # LightGBMModel等の場合、手動でクロスバリデーション
                cv_predictions = np.zeros(len(X))

                for train_idx, val_idx in cv.split(X, y):
                    X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

                    fold_model = self._create_base_model(model_type)
                    fold_model._train_model_impl(
                        X_fold_train, X_fold_val, y_fold_train, y_fold_val
                    )
                    fold_model.is_trained = True
                    fold_model.feature_columns = X.columns.tolist()

                    if self.use_probas:
                        fold_pred = fold_model.predict(X_fold_val)
                        if fold_pred.ndim > 1:
                            # 多クラス分類の場合、最大確率クラスの確率を使用
                            fold_pred = np.max(fold_pred, axis=1)
                    else:
                        fold_pred = fold_model.predict(X_fold_val)
                        if fold_pred.ndim > 1:
                            fold_pred = np.argmax(fold_pred, axis=1)

                    cv_predictions[val_idx] = fold_pred.flatten()

                meta_features.append(cv_predictions)

            else:
                # scikit-learn系の場合
                if self.use_probas and hasattr(base_model, "predict_proba"):
                    cv_predictions = cross_val_predict(
                        base_model, X, y, cv=cv, method="predict_proba"
                    )
                    # 多クラス分類の場合、最大確率クラスの確率を使用
                    if cv_predictions.ndim > 1:
                        cv_predictions = np.max(cv_predictions, axis=1)
                else:
                    cv_predictions = cross_val_predict(base_model, X, y, cv=cv)

                meta_features.append(cv_predictions)

        return np.column_stack(meta_features)

    def _generate_meta_features_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測時のメタ特徴量を生成

        Args:
            X: 特徴量DataFrame

        Returns:
            メタ特徴量の配列
        """
        meta_features = []

        for model in self.base_models:
            if hasattr(model, "predict") and hasattr(model, "is_trained"):
                # LightGBMModel等
                if self.use_probas:
                    pred = model.predict(X)
                    if pred.ndim > 1:
                        # 多クラス分類の場合、最大確率クラスの確率を使用
                        pred = np.max(pred, axis=1)
                else:
                    pred = model.predict(X)
                    if pred.ndim > 1:
                        pred = np.argmax(pred, axis=1)
            else:
                # scikit-learn系
                if self.use_probas and hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X)
                    if pred.ndim > 1:
                        pred = np.max(pred, axis=1)
                else:
                    pred = model.predict(X)

            meta_features.append(pred.flatten())

        return np.column_stack(meta_features)

    def _evaluate_ensemble(
        self,
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.Series],
        base_model_results: List[Dict[str, Any]],
        meta_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        アンサンブル全体の評価を実行

        Args:
            X_test: テスト用特徴量
            y_test: テスト用ターゲット
            base_model_results: 各ベースモデルの結果
            meta_result: メタモデルの結果

        Returns:
            アンサンブル評価結果
        """
        result = {
            "model_type": "StackingEnsemble",
            "base_model_types": self.base_model_types,
            "meta_model_type": self.meta_model_type,
            "cv_folds": self.cv_folds,
            "use_probas": self.use_probas,
            "base_model_results": base_model_results,
            "meta_model_result": meta_result,
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

                # 個別モデルとの比較
                if base_model_results:
                    base_accuracies = [
                        r.get("accuracy", 0)
                        for r in base_model_results
                        if "accuracy" in r
                    ]
                    if base_accuracies:
                        result["base_model_avg_accuracy"] = np.mean(base_accuracies)
                        result["ensemble_improvement"] = result.get(
                            "accuracy", 0
                        ) - np.mean(base_accuracies)

            except Exception as e:
                logger.warning(f"アンサンブル評価でエラー: {e}")
                result["evaluation_error"] = str(e)

        return result
