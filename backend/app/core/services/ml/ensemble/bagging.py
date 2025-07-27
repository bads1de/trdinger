"""
バギング（Bootstrap Aggregating）アンサンブル手法の実装

同じアルゴリズムのモデルを異なるデータサブセットで学習させ、
予測を平均化することで予測精度と頑健性を向上させます。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.utils import resample

from .base_ensemble import BaseEnsemble
from ....utils.unified_error_handler import UnifiedModelError

logger = logging.getLogger(__name__)


class BaggingEnsemble(BaseEnsemble):
    """
    バギングアンサンブル手法の実装

    複数のベースモデルを異なるブートストラップサンプルで学習させ、
    予測を平均化します。
    """

    def __init__(
        self, config: Dict[str, Any], automl_config: Optional[Dict[str, Any]] = None
    ):
        """
        初期化

        Args:
            config: バギング設定
            automl_config: AutoML設定（オプション）
        """
        super().__init__(config, automl_config)

        # バギング固有の設定
        self.n_estimators = config.get("n_estimators", 5)
        self.bootstrap_fraction = config.get("bootstrap_fraction", 0.8)
        self.random_state = config.get("random_state", 42)
        self.base_model_type = config.get("base_model_type", "lightgbm")

        # 混合バギング設定（多様性確保）
        self.mixed_models = config.get("mixed_models", None)

        # 最高性能モデル情報を保存するための属性
        self.best_algorithm = None
        self.best_model_score = -1
        if self.mixed_models:
            # 混合モデルが指定されている場合、n_estimatorsをモデル数に調整
            self.n_estimators = len(self.mixed_models)
            logger.info(
                f"混合バギングアンサンブル初期化: models={self.mixed_models}, "
                f"bootstrap_fraction={self.bootstrap_fraction}"
            )
        else:
            logger.info(
                f"バギングアンサンブル初期化: n_estimators={self.n_estimators}, "
                f"base_model={self.base_model_type}, bootstrap_fraction={self.bootstrap_fraction}"
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
        バギングアンサンブルモデルを学習

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
            logger.info("バギングアンサンブル学習を開始")

            self.feature_columns = X_train.columns.tolist()
            self.base_models = []

            # 各ベースモデルの学習結果を保存
            model_results = []

            # 最高性能モデル追跡用
            best_model = None
            best_score = -1
            best_algorithm = None

            # 複数のベースモデルを学習
            for i in range(self.n_estimators):
                logger.info(f"ベースモデル {i+1}/{self.n_estimators} を学習中...")

                # ブートストラップサンプリング
                X_bootstrap, y_bootstrap = self._bootstrap_sample(X_train, y_train)

                # ベースモデルを作成（混合モデル対応）
                if self.mixed_models:
                    # 混合バギング：各インデックスに対応するモデルタイプを使用
                    model_type = self.mixed_models[i % len(self.mixed_models)]
                    base_model = self._create_base_model(model_type)
                    logger.info(f"混合バギング: モデル{i+1}={model_type}")
                else:
                    # 通常のバギング：同じモデルタイプを使用
                    model_type = self.base_model_type
                    base_model = self._create_base_model(self.base_model_type)

                # モデルを学習
                if hasattr(base_model, "_train_model_impl"):
                    # LightGBMModel等のカスタムモデル
                    if X_test is not None and y_test is not None:
                        result = base_model._train_model_impl(
                            X_bootstrap, X_test, y_bootstrap, y_test
                        )
                    else:
                        # テストデータがない場合は学習データから分割
                        from sklearn.model_selection import train_test_split

                        X_train_split, X_val_split, y_train_split, y_val_split = (
                            train_test_split(
                                X_bootstrap,
                                y_bootstrap,
                                test_size=0.2,
                                random_state=self.random_state + i,
                            )
                        )
                        result = base_model._train_model_impl(
                            X_train_split, X_val_split, y_train_split, y_val_split
                        )

                    base_model.is_trained = True
                    base_model.feature_columns = self.feature_columns
                    model_results.append(result)

                    # 性能評価（accuracyを基準とする）
                    current_score = result.get("accuracy", 0.0)

                else:
                    # scikit-learn系モデル
                    base_model.fit(X_bootstrap, y_bootstrap)

                    # 評価（テストデータがある場合）
                    if X_test is not None and y_test is not None:
                        y_pred = base_model.predict(X_test)
                        y_pred_proba = None
                        if hasattr(base_model, "predict_proba"):
                            y_pred_proba = base_model.predict_proba(X_test)

                        result = self._evaluate_predictions(
                            y_test, y_pred, y_pred_proba
                        )
                        model_results.append(result)

                        # 性能評価（accuracyを基準とする）
                        current_score = result.get("accuracy", 0.0)
                    else:
                        current_score = 0.0
                        result = {}

                # 最高性能モデルの更新
                if current_score > best_score:
                    best_score = current_score
                    best_model = base_model
                    best_algorithm = model_type
                    logger.info(
                        f"新しい最高性能モデル: {model_type} (accuracy: {current_score:.4f})"
                    )

                self.base_models.append(base_model)
                logger.info(
                    f"ベースモデル {i+1} の学習完了: {model_type} (accuracy: {current_score:.4f})"
                )

            # 最高性能モデルのみを保持
            if best_model is not None:
                self.base_models = [best_model]
                self.best_algorithm = best_algorithm
                self.best_model_score = best_score
                logger.info(
                    f"最高性能モデルを選択: {best_algorithm} (accuracy: {best_score:.4f})"
                )
            else:
                logger.warning("有効なモデルが見つかりませんでした")

            self.is_fitted = True

            # アンサンブル全体の評価（最高性能モデルベース）
            ensemble_result = self._evaluate_ensemble(X_test, y_test, model_results)

            # 最高性能モデル情報を結果に追加
            ensemble_result.update(
                {
                    "best_algorithm": best_algorithm,
                    "best_model_score": best_score,
                    "selected_model_only": True,
                    "total_models_trained": self.n_estimators,
                }
            )

            logger.info("バギングアンサンブル学習完了")
            return ensemble_result

        except Exception as e:
            logger.error(f"バギングアンサンブル学習エラー: {e}")
            raise UnifiedModelError(f"バギングアンサンブル学習に失敗しました: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        バギングアンサンブルで予測を実行

        Args:
            X: 特徴量DataFrame

        Returns:
            予測結果（多数決）
        """
        if not self.is_fitted:
            raise UnifiedModelError("モデルが学習されていません")

        # 各ベースモデルの予測を取得
        predictions = []
        for model in self.base_models:
            if hasattr(model, "predict"):
                if hasattr(model, "is_trained") and model.is_trained:
                    # LightGBMTrainer等
                    pred = model.predict(X)
                    if pred.ndim > 1:
                        pred = np.argmax(pred, axis=1)
                else:
                    # scikit-learn系
                    pred = model.predict(X)
                predictions.append(pred)

        # 多数決で最終予測を決定
        predictions_array = np.array(predictions)
        final_predictions = []

        for i in range(predictions_array.shape[1]):
            votes = predictions_array[:, i]
            # 最頻値を取得
            unique_values, counts = np.unique(votes, return_counts=True)
            final_predictions.append(unique_values[np.argmax(counts)])

        return np.array(final_predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        バギングアンサンブルで予測確率を取得

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率の配列（平均）
        """
        if not self.is_fitted:
            raise UnifiedModelError("モデルが学習されていません")

        # 各ベースモデルの予測確率を取得
        probabilities = []
        for model in self.base_models:
            if hasattr(model, "predict"):
                if hasattr(model, "is_trained") and model.is_trained:
                    # LightGBMTrainer等
                    proba = model.predict(X)
                    if proba.ndim == 1:
                        # 二値分類の場合、確率形式に変換
                        proba = np.column_stack([1 - proba, proba])
                else:
                    # scikit-learn系
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X)
                    else:
                        # predict_probaがない場合はpredictの結果をワンホット化
                        pred = model.predict(X)
                        n_classes = len(np.unique(pred))
                        proba = np.eye(n_classes)[pred]

                probabilities.append(proba)

        # 確率の平均を計算
        if probabilities:
            avg_probabilities = np.mean(probabilities, axis=0)
            return avg_probabilities
        else:
            raise UnifiedModelError("予測確率を取得できませんでした")

    def _bootstrap_sample(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        ブートストラップサンプリングを実行

        Args:
            X: 特徴量DataFrame
            y: ターゲットSeries

        Returns:
            サンプリングされた特徴量とターゲット
        """
        n_samples = int(len(X) * self.bootstrap_fraction)

        # ブートストラップサンプリング（復元抽出）
        indices = resample(
            range(len(X)),
            n_samples=n_samples,
            random_state=self.random_state,
            replace=True,
        )

        X_bootstrap = X.iloc[indices].reset_index(drop=True)
        y_bootstrap = y.iloc[indices].reset_index(drop=True)

        return X_bootstrap, y_bootstrap

    def _evaluate_ensemble(
        self,
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.Series],
        model_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        アンサンブル全体の評価を実行

        Args:
            X_test: テスト用特徴量
            y_test: テスト用ターゲット
            model_results: 各ベースモデルの結果

        Returns:
            アンサンブル評価結果
        """
        result = {
            "model_type": "BaggingEnsemble",
            "n_estimators": self.n_estimators,
            "bootstrap_fraction": self.bootstrap_fraction,
            "base_model_type": self.base_model_type,
            "base_model_results": model_results,
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
                if model_results:
                    base_accuracies = [
                        r.get("accuracy", 0) for r in model_results if "accuracy" in r
                    ]
                    if base_accuracies:
                        result["base_model_avg_accuracy"] = np.mean(base_accuracies)
                        result["ensemble_improvement"] = result.get(
                            "accuracy", 0
                        ) - np.mean(base_accuracies)

            except Exception as e:
                logger.warning(f"アンサンブル評価でエラー: {e}")
                result["evaluation_error"] = str(e)

        # 特徴量重要度
        feature_importance = self.get_feature_importance()
        if feature_importance:
            result["feature_importance"] = feature_importance

        return result
