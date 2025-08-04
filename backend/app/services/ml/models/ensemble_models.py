"""
アンサンブル学習モデル

複数のアルゴリズムを組み合わせて予測精度を向上させます。
RandomForestの40.55%からXGBoost+アンサンブルで55%以上を目指します。
"""

import logging
import warnings
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


class EnsembleModelManager:
    """アンサンブル学習モデル管理クラス"""

    def __init__(self):
        """初期化"""
        self.models = {}
        self.ensemble_model = None
        self.feature_importance = None

    def create_base_models(self) -> Dict[str, Any]:
        """ベースモデルを作成"""
        logger.info("🤖 ベースモデルを作成中...")

        models = {}

        # 1. Random Forest（改良版）
        models["rf"] = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        # 2. XGBoost
        models["xgb"] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            eval_metric="mlogloss",
            use_label_encoder=False,
        )

        # 3. LightGBM
        models["lgb"] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )

        # 4. Logistic Regression
        models["lr"] = LogisticRegression(
            C=1.0, class_weight="balanced", random_state=42, max_iter=1000
        )

        # 5. SVM（小さなデータセット用）
        models["svm"] = SVC(
            C=1.0,
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )

        # 6. Extra Trees（ランダム性強化）
        models["extra_trees"] = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        # 7. Gradient Boosting
        models["gb"] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )

        # 8. AdaBoost
        models["ada"] = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, class_weight="balanced"),
            n_estimators=100,
            learning_rate=1.0,
            random_state=42,
        )

        # 9. Ridge Classifier（線形）
        models["ridge"] = RidgeClassifier(
            alpha=1.0, class_weight="balanced", random_state=42
        )

        # 10. Naive Bayes
        models["nb"] = GaussianNB()

        # 11. K-Nearest Neighbors
        models["knn"] = KNeighborsClassifier(
            n_neighbors=5, weights="distance", metric="minkowski"
        )

        logger.info(f"✅ {len(models)}個のベースモデルを作成")
        return models

    def create_voting_ensemble(self, base_models: Dict[str, Any]) -> VotingClassifier:
        """投票アンサンブルを作成"""
        logger.info("🗳️ 投票アンサンブルを作成中...")

        # モデルリストを作成
        estimators = [(name, model) for name, model in base_models.items()]

        # ソフト投票（確率ベース）
        voting_ensemble = VotingClassifier(estimators=estimators, voting="soft")

        return voting_ensemble

    def create_stacking_ensemble(
        self, base_models: Dict[str, Any]
    ) -> StackingClassifier:
        """スタッキングアンサンブルを作成"""
        logger.info("📚 スタッキングアンサンブルを作成中...")

        # モデルリストを作成
        estimators = [(name, model) for name, model in base_models.items()]

        # メタ学習器（Logistic Regression）
        meta_learner = LogisticRegression(
            C=1.0, class_weight="balanced", random_state=42
        )

        # スタッキング分類器
        stacking_ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=3,  # 時系列データなので少なめ
            stack_method="predict_proba",
        )

        return stacking_ensemble

    def train_and_evaluate_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Dict[str, Dict[str, float]]:
        """モデルを学習・評価"""
        logger.info("🏋️ モデル学習・評価を開始...")

        results = {}

        # ベースモデルを作成
        base_models = self.create_base_models()

        # 1. 個別モデルの評価
        for name, model in base_models.items():
            logger.info(f"📊 {name}モデルを学習中...")

            try:
                # 学習
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_train)

                # 予測
                y_pred = model.predict(X_test)
                # 評価
                results[name] = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred, average="weighted"),
                }

                logger.info(
                    f"  {name}: 精度={results[name]['accuracy']:.4f}, "
                    f"バランス精度={results[name]['balanced_accuracy']:.4f}"
                )

            except Exception as e:
                logger.error(f"{name}モデルでエラー: {e}")
                continue

        # 2. 投票アンサンブルの評価
        try:
            logger.info("🗳️ 投票アンサンブルを学習中...")
            voting_ensemble = self.create_voting_ensemble(base_models)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                voting_ensemble.fit(X_train, y_train)

            y_pred_voting = voting_ensemble.predict(X_test)

            results["voting_ensemble"] = {
                "accuracy": accuracy_score(y_test, y_pred_voting),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_voting),
                "f1_score": f1_score(y_test, y_pred_voting, average="weighted"),
            }

            logger.info(
                f"  投票アンサンブル: 精度={results['voting_ensemble']['accuracy']:.4f}"
            )

        except Exception as e:
            logger.error(f"投票アンサンブルでエラー: {e}")

        # 3. スタッキングアンサンブルの評価
        try:
            logger.info("📚 スタッキングアンサンブルを学習中...")
            stacking_ensemble = self.create_stacking_ensemble(base_models)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stacking_ensemble.fit(X_train, y_train)

            y_pred_stacking = stacking_ensemble.predict(X_test)

            results["stacking_ensemble"] = {
                "accuracy": accuracy_score(y_test, y_pred_stacking),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_stacking),
                "f1_score": f1_score(y_test, y_pred_stacking, average="weighted"),
            }

            logger.info(
                f"  スタッキングアンサンブル: 精度={results['stacking_ensemble']['accuracy']:.4f}"
            )

            # 最高性能モデルとして保存
            self.ensemble_model = stacking_ensemble

        except Exception as e:
            logger.error(f"スタッキングアンサンブルでエラー: {e}")

        return results

    def get_feature_importance(self, X: pd.DataFrame) -> Optional[pd.Series]:
        """特徴量重要度を取得"""
        if self.ensemble_model is None:
            return None

        try:
            # スタッキングアンサンブルの場合、ベースモデルの重要度を平均
            if hasattr(self.ensemble_model, "estimators_"):
                importances = []

                for name, model in self.ensemble_model.estimators_:
                    if hasattr(model, "feature_importances_"):
                        importances.append(model.feature_importances_)
                    elif hasattr(model, "coef_"):
                        # 線形モデルの場合は係数の絶対値
                        importances.append(np.abs(model.coef_[0]))

                if importances:
                    avg_importance = np.mean(importances, axis=0)
                    return pd.Series(avg_importance, index=X.columns).sort_values(
                        ascending=False
                    )

        except Exception as e:
            logger.error(f"特徴量重要度取得エラー: {e}")

        return None

    def optimize_hyperparameters(
        self, X_train: pd.DataFrame, y_train: pd.Series, model_type: str = "xgb"
    ) -> Dict[str, Any]:
        """ハイパーパラメータ最適化（簡易版）"""
        logger.info(f"🔧 {model_type}のハイパーパラメータ最適化中...")

        if model_type == "xgb":
            # XGBoostのグリッドサーチ（簡易版）
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [6, 8, 10],
                "learning_rate": [0.05, 0.1, 0.15],
            }

            best_score = 0
            best_params = {}

            for n_est in param_grid["n_estimators"]:
                for depth in param_grid["max_depth"]:
                    for lr in param_grid["learning_rate"]:
                        model = xgb.XGBClassifier(
                            n_estimators=n_est,
                            max_depth=depth,
                            learning_rate=lr,
                            random_state=42,
                            eval_metric="mlogloss",
                            use_label_encoder=False,
                        )

                        # 3-fold CV
                        scores = cross_val_score(
                            model, X_train, y_train, cv=3, scoring="balanced_accuracy"
                        )
                        avg_score = scores.mean()

                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = {
                                "n_estimators": n_est,
                                "max_depth": depth,
                                "learning_rate": lr,
                            }

            logger.info(f"最適パラメータ: {best_params}, スコア: {best_score:.4f}")
            return best_params

        return {}

    def create_optimized_ensemble(
        self, X_train: pd.DataFrame, y_train: pd.Series, optimize: bool = True
    ) -> Any:
        """最適化されたアンサンブルモデルを作成"""
        logger.info("🚀 最適化されたアンサンブルモデルを作成中...")

        models = {}

        # 1. 最適化されたXGBoost
        if optimize:
            best_params = self.optimize_hyperparameters(X_train, y_train, "xgb")
            models["xgb_optimized"] = xgb.XGBClassifier(
                **best_params,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                eval_metric="mlogloss",
                use_label_encoder=False,
            )
        else:
            models["xgb"] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                eval_metric="mlogloss",
                use_label_encoder=False,
            )

        # 2. 改良されたRandom Forest
        models["rf_improved"] = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        # 3. LightGBM
        models["lgb"] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )

        # スタッキングアンサンブル
        estimators = [(name, model) for name, model in models.items()]

        meta_learner = LogisticRegression(
            C=1.0, class_weight="balanced", random_state=42, max_iter=1000
        )

        optimized_ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=3,
            stack_method="predict_proba",
        )

        logger.info(f"✅ 最適化アンサンブル作成完了: {len(models)}個のベースモデル")

        return optimized_ensemble


# グローバルインスタンス
ensemble_manager = EnsembleModelManager()
