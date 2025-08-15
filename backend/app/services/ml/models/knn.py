"""
K-Nearest Neighborsモデルラッパー

アンサンブル学習で使用するKNNモデルのラッパークラスを提供します。
scikit-learnのKNeighborsClassifierを使用してアンサンブル専用に最適化されたモデルです。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


from ....utils.unified_error_handler import UnifiedModelError

logger = logging.getLogger(__name__)


class KNNModel:
    """
    アンサンブル内で使用するKNNモデルラッパー

    scikit-learnのKNeighborsClassifierを使用してアンサンブル専用に最適化されたモデル
    """

    # アルゴリズム名（AlgorithmRegistryから取得）
    ALGORITHM_NAME = "knn"

    def __init__(
        self,
        automl_config: Optional[Dict[str, Any]] = None,
        n_neighbors: int = 5,
        weights: str = "distance",
        algorithm: str = "auto",
        metric: str = "minkowski",
        p: int = 2,
        n_jobs: int = -1,
        **kwargs,
    ):
        """
        初期化

        Args:
            automl_config: AutoML設定（現在は未使用）
            n_neighbors: 近傍数
            weights: 重み付け方法
            algorithm: アルゴリズム
            metric: 距離メトリック
            p: ミンコフスキー距離のパラメータ
            n_jobs: 並列処理数
            **kwargs: その他のパラメータ
        """
        self.model = None
        self.is_trained = False
        self.feature_columns = None
        self.automl_config = automl_config
        self.classes_ = None  # sklearn互換性のため

        # sklearn互換性のためのパラメータ
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.metric = metric
        self.p = p
        self.n_jobs = n_jobs

        # デフォルトパラメータ（最適化された設定）
        self.default_params = {
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "metric": self.metric,
            "p": self.p,
            "n_jobs": self.n_jobs,
            "leaf_size": 30,  # 効率的な検索のためのリーフサイズ
        }

        # その他のパラメータを設定
        for key, value in kwargs.items():
            setattr(self, key, value)

    def fit(self, X, y) -> "KNNModel":
        """
        sklearn互換のfitメソッド

        Args:
            X: 学習用特徴量（DataFrame or numpy array）
            y: 学習用ターゲット（Series or numpy array）

        Returns:
            self: 学習済みモデル
        """
        try:
            # numpy配列をDataFrameに変換
            if not isinstance(X, pd.DataFrame):
                if hasattr(self, "feature_columns") and self.feature_columns:
                    X = pd.DataFrame(X, columns=self.feature_columns)
                else:
                    X = pd.DataFrame(
                        X, columns=[f"feature_{i}" for i in range(X.shape[1])]
                    )

            if not isinstance(y, pd.Series):
                y = pd.Series(y)

            # 特徴量カラムを保存
            self.feature_columns = list(X.columns)

            # モデル初期化
            self.model = KNeighborsClassifier(**self.default_params)

            # 学習実行（KNNは遅延学習なので実際はデータを保存するだけ）
            self.model.fit(X, y)

            # classes_属性を設定（sklearn互換性のため）
            self.classes_ = np.unique(y)
            self.is_trained = True

            return self

        except Exception as e:
            logger.error(f"sklearn互換fit実行エラー: {e}")
            raise UnifiedModelError(f"KNNモデルのfit実行に失敗しました: {e}")

    def _train_model_impl(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        **training_params,
    ) -> Dict[str, Any]:
        """
        KNNモデルを学習

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ターゲット
            y_test: テスト用ターゲット

        Returns:
            学習結果の辞書
        """
        try:
            logger.info("🔍 KNNモデルの学習を開始")

            # 特徴量カラムを保存
            self.feature_columns = list(X_train.columns)

            # モデル初期化
            self.model = KNeighborsClassifier(**self.default_params)

            # 学習実行（KNNは遅延学習なので実際はデータを保存するだけ）
            self.model.fit(X_train, y_train)

            # 予測
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)

            # 確率予測（KNNは確率予測をサポート）
            y_pred_proba_train = self.model.predict_proba(X_train)
            y_pred_proba_test = self.model.predict_proba(X_test)

            # 統一された評価指標計算器を使用
            from ..evaluation.enhanced_metrics import (
                EnhancedMetricsCalculator,
                MetricsConfig,
            )

            config = MetricsConfig(
                include_balanced_accuracy=True,
                include_pr_auc=True,
                include_roc_auc=True,
                include_confusion_matrix=True,
                include_classification_report=True,
                average_method="weighted",
                zero_division=0,
            )

            metrics_calculator = EnhancedMetricsCalculator(config)

            # 包括的な評価指標を計算（テストデータ）
            test_metrics = metrics_calculator.calculate_comprehensive_metrics(
                y_test, y_pred_test, y_pred_proba_test
            )

            # 包括的な評価指標を計算（学習データ）
            train_metrics = metrics_calculator.calculate_comprehensive_metrics(
                y_train, y_pred_train, y_pred_proba_train
            )

            # 特徴量重要度（KNNでは直接的な重要度なし）
            # 各特徴量の分散を重要度として近似
            feature_variance = X_train.var()
            total_variance = feature_variance.sum()
            feature_importance = {}
            if total_variance > 0:
                feature_importance = dict(
                    zip(self.feature_columns, feature_variance / total_variance)
                )

            self.is_trained = True

            results = {
                "algorithm": self.ALGORITHM_NAME,
                "n_neighbors": self.model.n_neighbors,
                "weights": self.model.weights,
                "algorithm_type": self.model.algorithm,
                "metric": self.model.metric,
                "feature_count": len(self.feature_columns),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "num_classes": (
                    len(self.model.classes_) if hasattr(self.model, "classes_") else 0
                ),
                "feature_importance": feature_importance,
            }

            # テストデータの評価指標を追加（プレフィックス付き）
            for key, value in test_metrics.items():
                if key not in ["error"]:  # エラー情報は除外
                    results[f"test_{key}"] = value
                    # フロントエンド用の統一キー（test_なしのキー）
                    if key in [
                        "accuracy",
                        "balanced_accuracy",
                        "f1_score",
                        "matthews_corrcoef",
                    ]:
                        results[key] = value
                    elif key == "roc_auc" or key == "roc_auc_ovr":
                        results["auc_roc"] = value
                    elif key == "pr_auc" or key == "pr_auc_macro":
                        results["auc_pr"] = value

            # 学習データの評価指標を追加（プレフィックス付き）
            for key, value in train_metrics.items():
                if key not in ["error"]:  # エラー情報は除外
                    results[f"train_{key}"] = value

            logger.info(
                f"✅ KNN学習完了 - テスト精度: {test_metrics.get('accuracy', 0.0):.4f}"
            )
            return results

        except Exception as e:
            logger.error(f"❌ KNN学習エラー: {e}")
            raise UnifiedModelError(f"KNN学習に失敗しました: {e}")

    def predict(self, X) -> np.ndarray:
        """
        sklearn互換の予測メソッド

        Args:
            X: 特徴量（DataFrame or numpy array）

        Returns:
            予測結果
        """
        if not self.is_trained or self.model is None:
            raise UnifiedModelError("モデルが学習されていません")

        try:
            # numpy配列をDataFrameに変換
            if not isinstance(X, pd.DataFrame):
                if hasattr(self, "feature_columns") and self.feature_columns:
                    X = pd.DataFrame(X, columns=self.feature_columns)
                else:
                    X = pd.DataFrame(
                        X, columns=[f"feature_{i}" for i in range(X.shape[1])]
                    )

            # 特徴量の順序を確認
            if self.feature_columns:
                X = X[self.feature_columns]

            predictions = self.model.predict(X)
            return predictions

        except Exception as e:
            logger.error(f"KNN予測エラー: {e}")
            raise UnifiedModelError(f"予測に失敗しました: {e}")

    def predict_proba(self, X) -> np.ndarray:
        """
        sklearn互換の予測確率メソッド

        Args:
            X: 特徴量（DataFrame or numpy array）

        Returns:
            予測確率の配列
        """
        if not self.is_trained or self.model is None:
            raise UnifiedModelError("モデルが学習されていません")

        try:
            # numpy配列をDataFrameに変換
            if not isinstance(X, pd.DataFrame):
                if hasattr(self, "feature_columns") and self.feature_columns:
                    X = pd.DataFrame(X, columns=self.feature_columns)
                else:
                    X = pd.DataFrame(
                        X, columns=[f"feature_{i}" for i in range(X.shape[1])]
                    )

            # 特徴量の順序を確認
            if self.feature_columns:
                X = X[self.feature_columns]

            probabilities = self.model.predict_proba(X)
            return probabilities

        except Exception as e:
            logger.error(f"KNN確率予測エラー: {e}")
            raise UnifiedModelError(f"確率予測に失敗しました: {e}")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        sklearn互換のパラメータ取得メソッド

        Args:
            deep: 深いコピーを行うかどうか（未使用）

        Returns:
            パラメータの辞書
        """
        _ = deep  # 未使用パラメータ
        # 基本パラメータ
        params = {
            "automl_config": self.automl_config,
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "metric": self.metric,
            "p": self.p,
            "n_jobs": self.n_jobs,
        }

        # 動的に追加されたパラメータも含める
        for attr_name in dir(self):
            if (
                not attr_name.startswith("_")
                and attr_name not in params
                and attr_name
                not in [
                    "model",
                    "is_trained",
                    "feature_columns",
                    "classes_",
                    "default_params",
                    "fit",
                    "predict",
                    "predict_proba",
                    "get_params",
                    "set_params",
                ]
            ):
                try:
                    attr_value = getattr(self, attr_name)
                    if not callable(attr_value):
                        params[attr_name] = attr_value
                except Exception:
                    # 属性取得に失敗した場合は無視
                    pass

        return params

    def set_params(self, **params) -> "KNNModel":
        """
        sklearn互換のパラメータ設定メソッド

        Args:
            **params: 設定するパラメータ

        Returns:
            self: 設定後のモデル
        """
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
                # default_paramsも更新
                if param in self.default_params:
                    self.default_params[param] = value
            else:
                logger.warning(f"未知のパラメータ: {param}")
        return self

    @property
    def feature_columns(self) -> List[str]:
        """特徴量カラム名のリストを取得"""
        return self._feature_columns

    @feature_columns.setter
    def feature_columns(self, columns: List[str]):
        """特徴量カラム名のリストを設定"""
        self._feature_columns = columns

    def get_feature_importance(self) -> Dict[str, float]:
        """
        特徴量重要度を取得（距離ベースの重要度計算）

        Returns:
            特徴量重要度の辞書
        """
        if not self.is_trained or self.model is None:
            raise UnifiedModelError("モデルが学習されていません")

        if not self.feature_columns:
            raise UnifiedModelError("特徴量カラムが設定されていません")

        # 学習データから距離ベースの重要度を計算
        if hasattr(self.model, "_fit_X"):
            X_train = pd.DataFrame(self.model._fit_X, columns=self.feature_columns)

            # 各特徴量の分散を重要度として使用（改良版）
            feature_variance = X_train.var()

            # 特徴量間の相関を考慮した重要度計算
            try:
                # 各特徴量の近傍距離への寄与度を計算
                nn = NearestNeighbors(
                    n_neighbors=min(5, len(X_train)), metric=self.metric, p=self.p
                )
                nn.fit(X_train)
                distances, _ = nn.kneighbors(X_train)

                # 各特徴量の距離への寄与度を計算
                feature_importance = {}
                for i, col in enumerate(self.feature_columns):
                    # 特徴量を除いた場合の距離変化を計算
                    X_without_feature = X_train.drop(columns=[col])
                    if X_without_feature.shape[1] > 0:
                        nn_without = NearestNeighbors(
                            n_neighbors=min(5, len(X_train)),
                            metric=self.metric,
                            p=self.p,
                        )
                        nn_without.fit(X_without_feature)
                        distances_without, _ = nn_without.kneighbors(X_without_feature)

                        # 距離の変化を重要度として使用
                        importance = np.mean(
                            np.abs(
                                distances.mean(axis=1) - distances_without.mean(axis=1)
                            )
                        )
                        feature_importance[col] = importance
                    else:
                        feature_importance[col] = feature_variance.iloc[i]

                # 正規化
                total_importance = sum(feature_importance.values())
                if total_importance > 0:
                    return {
                        k: v / total_importance for k, v in feature_importance.items()
                    }

            except Exception as e:
                logger.warning(f"距離ベース重要度計算でエラー: {e}")
                # フォールバック：分散ベースの重要度
                total_variance = feature_variance.sum()
                if total_variance > 0:
                    return dict(
                        zip(self.feature_columns, feature_variance / total_variance)
                    )

        return {}

    def get_model_info(self) -> Dict[str, Any]:
        """
        モデル情報を取得

        Returns:
            モデル情報の辞書
        """
        if not self.is_trained or self.model is None:
            return {"status": "not_trained"}

        return {
            "algorithm": self.ALGORITHM_NAME,
            "n_neighbors": self.model.n_neighbors,
            "weights": self.model.weights,
            "algorithm_type": self.model.algorithm,
            "metric": self.model.metric,
            "p": self.model.p,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "status": "trained",
        }
