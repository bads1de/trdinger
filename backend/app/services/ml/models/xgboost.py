"""
XGBoostモデルラッパー

アンサンブル学習で使用するXGBoostモデルのラッパークラスを提供します。
XGBoostを使用してアンサンブル専用に最適化されたモデルです。
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import xgboost as xgb

from ....utils.error_handler import ModelError

logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    アンサンブル内で使用するXGBoostモデルラッパー

    XGBoostを使用してアンサンブル専用に最適化されたモデル
    """

    # アルゴリズム名（AlgorithmRegistryから取得）
    ALGORITHM_NAME = "xgboost"

    def __init__(self):
        """
        初期化
        """
        self.model = None
        self.is_trained = False
        self.feature_columns = None
        self.scaler = None

    def _train_model_impl(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        **kwargs,
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

            # XGBoostデータセットを作成（特徴量名を設定）
            self.feature_names = self.feature_columns.copy()
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)  # type: ignore
            dtest = xgb.DMatrix(X_test, label=y_test, feature_names=self.feature_names)  # type: ignore

            # モデル学習
            self.model = xgb.train(  # type: ignore
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

            # 共通の評価関数を使用
            from ..common.evaluation_utils import evaluate_model_predictions

            detailed_metrics = evaluate_model_predictions(
                y_test, y_pred_class, y_pred_proba
            )

            # 学習開始ログ
            logger.info(f"XGBoost学習開始: {num_classes}クラス分類")
            logger.info(f"クラス分布: {dict(y_train.value_counts())}")

            # 特徴量重要度を計算（修正版）
            feature_importance = self._calculate_feature_importance()

            self.is_trained = True

            logger.info(
                f"XGBoostモデル学習完了: 精度={detailed_metrics.get('accuracy', 0.0):.4f}"
            )

            # 詳細な評価指標を含む結果を返す
            result = {
                "algorithm": self.ALGORITHM_NAME,  # アルゴリズム名を追加
                "num_classes": num_classes,
                "best_iteration": self.model.best_iteration,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": len(self.feature_columns),
                "feature_importance": feature_importance,  # 特徴量重要度を追加
                **detailed_metrics,  # 詳細な評価指標を追加
            }

            return result

        except ImportError:
            logger.error(
                "XGBoostがインストールされていません。pip install xgboostを実行してください。"
            )
            raise ModelError("XGBoostがインストールされていません")
        except Exception as e:
            logger.error(f"XGBoostモデル学習エラー: {e}")
            raise ModelError(f"XGBoostモデル学習に失敗しました: {e}")

    def _calculate_feature_importance(self) -> Dict[str, float]:
        """特徴量重要度を計算（修正版）"""
        try:
            if not self.model or not hasattr(self.model, "get_score"):
                logger.warning("モデルまたはget_score()メソッドがありません")
                return {col: 0.0 for col in self.feature_columns}

            # get_score()で重要度を取得
            importance_scores = self.model.get_score(importance_type="gain")

            logger.info(f"XGBoost get_score() result: {importance_scores}")

            # 特徴量名を正しくマッピング
            feature_importance = {}

            # feature_namesが存在する場合
            if hasattr(self, "feature_names") and self.feature_names:
                for feature_name in self.feature_names:
                    feature_importance[feature_name] = importance_scores.get(
                        feature_name, 0.0
                    )
            else:
                # フォールバック: インデックスを使用
                for i, col in enumerate(self.feature_columns):
                    feature_key = f"f{i}"
                    feature_importance[col] = importance_scores.get(feature_key, 0.0)

            logger.info(f"計算された特徴量重要度: {len(feature_importance)}個")

            # デバッグログ: 0でない重要度の個数
            non_zero_count = sum(
                1 for score in feature_importance.values() if score > 0
            )
            logger.info(
                f"重要度が0でない特徴量数: {non_zero_count}/{len(feature_importance)}"
            )

            return feature_importance

        except Exception as e:
            logger.error(f"特徴量重要度計算エラー: {e}")
            import traceback

            traceback.print_exc()
            # フォールバック: すべて0とする
            return {col: 0.0 for col in self.feature_columns}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測を実行

        Args:
            X: 特徴量DataFrame

        Returns:
            予測クラスラベル
        """
        if not self.is_trained or self.model is None:
            raise ModelError("学習済みモデルがありません")

        dtest = xgb.DMatrix(X)  # type: ignore
        predictions = self.model.predict(dtest)

        # クラスラベルに変換
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            # 多クラス分類
            return np.argmax(predictions, axis=1)
        else:
            # 二値分類
            return (predictions > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測確率を取得

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率
        """
        if not self.is_trained or self.model is None:
            raise ModelError("学習済みモデルがありません")

        dtest = xgb.DMatrix(X)  # type: ignore
        probabilities = self.model.predict(dtest)

        # 二値分類の場合は2次元に変換
        if probabilities.ndim == 1:
            probabilities = np.column_stack([1 - probabilities, probabilities])

        return probabilities

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        特徴量重要度を取得（修正版）

        Args:
            top_n: 上位N個の特徴量

        Returns:
            特徴量重要度の辞書
        """
        if not self.is_trained or not self.model:
            logger.warning("学習済みモデルがありません")
            return {}

        try:
            # 修正された特徴量重要度計算を使用
            feature_importance = self._calculate_feature_importance()

            if not feature_importance:
                logger.warning("特徴量重要度の計算に失敗")
                return {}

            # 重要度でソートして上位N個を取得
            sorted_importance = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:top_n]

            result = dict(sorted_importance)
            logger.info(f"XGBoost特徴量重要度を取得: {len(result)}個")
            return result

        except Exception as e:
            logger.error(f"XGBoost特徴量重要度取得エラー: {e}")
            return {}
