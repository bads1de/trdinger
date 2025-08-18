"""
バギング（Bootstrap Aggregating）アンサンブル手法の実装

scikit-learnのBaggingClassifierを使用した標準的なバギング実装。
複数のベースモデルを異なるブートストラップサンプルで学習させ、
予測を平均化することで予測精度と頑健性を向上させます。
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier


from ....utils.error_handler import ModelError
from .base_ensemble import BaseEnsemble

logger = logging.getLogger(__name__)


class BaggingEnsemble(BaseEnsemble):
    """
    scikit-learnのBaggingClassifierを使用したバギングアンサンブル実装

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
        self.n_estimators = config.get("n_estimators", 10)
        self.max_samples = config.get("bootstrap_fraction", 0.8)
        self.random_state = config.get("random_state", 42)
        self.base_model_type = config.get("base_model_type", "lightgbm")
        self.n_jobs = config.get("n_jobs", -1)  # 並列処理

        # scikit-learn BaggingClassifier
        self.bagging_classifier = None

        logger.info(
            f"BaggingClassifier初期化: n_estimators={self.n_estimators}, "
            f"base_model={self.base_model_type}, max_samples={self.max_samples}, "
            f"n_jobs={self.n_jobs}"
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
        BaggingClassifierを使用してバギングアンサンブルモデルを学習

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
            logger.info("BaggingClassifierによるバギングアンサンブル学習を開始")

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

            # ベースモデルを作成
            try:
                base_estimator = self._create_base_model(self.base_model_type)
                logger.info(f"ベースモデル作成完了: {self.base_model_type}")
            except Exception as e:
                logger.error(f"ベースモデル作成エラー: {e}")
                raise ModelError(
                    f"ベースモデル({self.base_model_type})の作成に失敗しました: {e}"
                )

            # BaggingClassifierを初期化
            try:
                self.bagging_classifier = BaggingClassifier(
                    estimator=base_estimator,
                    n_estimators=self.n_estimators,
                    max_samples=self.max_samples,
                    bootstrap=True,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbose=0,
                )
                logger.info("BaggingClassifier初期化完了")
            except Exception as e:
                logger.error(f"BaggingClassifier初期化エラー: {e}")
                raise ModelError(f"BaggingClassifierの初期化に失敗しました: {e}")

            # モデルを学習
            logger.info(
                f"BaggingClassifier学習開始: {self.n_estimators}個のベースモデル"
            )
            try:
                self.bagging_classifier.fit(X_train, y_train)
                self.is_fitted = True

                # base_modelsリストを設定（EnsembleTrainerとの互換性のため）
                self.base_models = [self.bagging_classifier]
                self.best_algorithm = "bagging_" + self.base_model_type

                logger.info("BaggingClassifier学習完了")
            except Exception as e:
                logger.error(f"BaggingClassifier学習エラー: {e}")
                raise ModelError(f"BaggingClassifierの学習に失敗しました: {e}")

            # アンサンブル全体の評価
            ensemble_result = self._evaluate_ensemble(X_test, y_test)

            # 学習結果情報を追加
            ensemble_result.update(
                {
                    "model_type": "BaggingClassifier",
                    "n_estimators": self.n_estimators,
                    "max_samples": self.max_samples,
                    "base_model_type": self.base_model_type,
                    "n_jobs": self.n_jobs,
                    "sklearn_implementation": True,
                    "training_samples": len(X_train),
                    "test_samples": len(X_test) if X_test is not None else 0,
                }
            )

            logger.info("バギングアンサンブル学習完了")
            return ensemble_result

        except Exception as e:
            logger.error(f"バギングアンサンブル学習エラー: {e}")
            raise ModelError(f"バギングアンサンブル学習に失敗しました: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        BaggingClassifierで予測を実行

        Args:
            X: 特徴量DataFrame

        Returns:
            予測結果
        """
        if not self.is_fitted or self.bagging_classifier is None:
            raise ModelError("モデルが学習されていません")

        return self.bagging_classifier.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        BaggingClassifierで予測確率を取得

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率の配列
        """
        if not self.is_fitted or self.bagging_classifier is None:
            raise ModelError("モデルが学習されていません")

        return self.bagging_classifier.predict_proba(X)

    def _evaluate_ensemble(
        self,
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.Series],
    ) -> Dict[str, Any]:
        """
        BaggingClassifierアンサンブル全体の評価を実行

        Args:
            X_test: テスト用特徴量
            y_test: テスト用ターゲット

        Returns:
            アンサンブル評価結果
        """
        result = {
            "model_type": "BaggingClassifier",
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "base_model_type": self.base_model_type,
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

        # 特徴量重要度（BaggingClassifierから取得）
        feature_importance = self.get_feature_importance()
        if feature_importance:
            result["feature_importance"] = feature_importance

        return result

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        BaggingClassifierから特徴量重要度を取得

        Returns:
            特徴量重要度の辞書（特徴量名: 重要度）
        """
        if not self.is_fitted or self.bagging_classifier is None:
            return None

        try:
            # ベースモデルが特徴量重要度を持つ場合
            if hasattr(self.bagging_classifier.estimators_[0], "feature_importances_"):
                # 各ベースモデルの特徴量重要度を平均
                importances = []
                for estimator in self.bagging_classifier.estimators_:
                    if hasattr(estimator, "feature_importances_"):
                        importances.append(estimator.feature_importances_)

                if importances and self.feature_columns:
                    avg_importance = np.mean(importances, axis=0)
                    return dict(zip(self.feature_columns, avg_importance))

            return None
        except Exception as e:
            logger.warning(f"特徴量重要度の取得でエラー: {e}")
            return None

    def save_models(self, base_path: str) -> list:
        """
        BaggingClassifierモデルを保存

        Args:
            base_path: 保存先ベースパス

        Returns:
            保存されたファイルパスのリスト
        """
        return super().save_models(base_path)

    def load_models(self, base_path: str) -> bool:
        """
        BaggingClassifierモデルを読み込み

        Args:
            base_path: 読み込み元ベースパス

        Returns:
            読み込み成功フラグ
        """
        return super().load_models(base_path)
