"""
アンサンブル学習用ハイパーパラメータ空間定義

アンサンブル学習で使用される各ベースモデルとアンサンブル手法固有の
パラメータ空間を定義します。Optunaの高度な最適化機能を活用。
"""

from typing import Dict, List
import optuna

from .optuna_optimizer import ParameterSpace


class EnsembleParameterSpace:
    """アンサンブル学習用パラメータ空間管理クラス"""

    @staticmethod
    def get_lightgbm_parameter_space() -> Dict[str, ParameterSpace]:
        """LightGBMのパラメータ空間"""
        return {
            "lgb_num_leaves": ParameterSpace(type="integer", low=10, high=100),
            "lgb_learning_rate": ParameterSpace(type="real", low=0.01, high=0.3),
            "lgb_feature_fraction": ParameterSpace(type="real", low=0.5, high=1.0),
            "lgb_bagging_fraction": ParameterSpace(type="real", low=0.5, high=1.0),
            "lgb_min_data_in_leaf": ParameterSpace(type="integer", low=5, high=50),
            "lgb_max_depth": ParameterSpace(type="integer", low=3, high=15),
            "lgb_reg_alpha": ParameterSpace(type="real", low=0.0, high=1.0),
            "lgb_reg_lambda": ParameterSpace(type="real", low=0.0, high=1.0),
        }

    @staticmethod
    def get_xgboost_parameter_space() -> Dict[str, ParameterSpace]:
        """XGBoostのパラメータ空間"""
        return {
            "xgb_max_depth": ParameterSpace(type="integer", low=3, high=15),
            "xgb_learning_rate": ParameterSpace(type="real", low=0.01, high=0.3),
            "xgb_subsample": ParameterSpace(type="real", low=0.5, high=1.0),
            "xgb_colsample_bytree": ParameterSpace(type="real", low=0.5, high=1.0),
            "xgb_min_child_weight": ParameterSpace(type="integer", low=1, high=10),
            "xgb_reg_alpha": ParameterSpace(type="real", low=0.0, high=1.0),
            "xgb_reg_lambda": ParameterSpace(type="real", low=0.0, high=1.0),
            "xgb_gamma": ParameterSpace(type="real", low=0.0, high=0.5),
        }

    @staticmethod
    def get_randomforest_parameter_space() -> Dict[str, ParameterSpace]:
        """RandomForestのパラメータ空間"""
        return {
            "rf_n_estimators": ParameterSpace(type="integer", low=50, high=300),
            "rf_max_depth": ParameterSpace(type="integer", low=3, high=20),
            "rf_min_samples_split": ParameterSpace(type="integer", low=2, high=20),
            "rf_min_samples_leaf": ParameterSpace(type="integer", low=1, high=10),
            "rf_max_features": ParameterSpace(
                type="categorical", categories=["sqrt", "log2", "auto"]
            ),
            "rf_bootstrap": ParameterSpace(
                type="categorical", categories=[True, False]
            ),
        }

    @staticmethod
    def get_catboost_parameter_space() -> Dict[str, ParameterSpace]:
        """CatBoostのパラメータ空間"""
        return {
            "cat_iterations": ParameterSpace(type="integer", low=100, high=1000),
            "cat_learning_rate": ParameterSpace(type="real", low=0.01, high=0.3),
            "cat_depth": ParameterSpace(type="integer", low=3, high=10),
            "cat_l2_leaf_reg": ParameterSpace(type="real", low=1.0, high=10.0),
            "cat_border_count": ParameterSpace(type="integer", low=32, high=255),
            "cat_bagging_temperature": ParameterSpace(type="real", low=0.0, high=1.0),
            "cat_random_strength": ParameterSpace(type="real", low=0.0, high=10.0),
            "cat_subsample": ParameterSpace(type="real", low=0.5, high=1.0),
            "cat_colsample_bylevel": ParameterSpace(type="real", low=0.5, high=1.0),
        }

    @staticmethod
    def get_tabnet_parameter_space() -> Dict[str, ParameterSpace]:
        """TabNetのパラメータ空間"""
        return {
            "tab_n_d": ParameterSpace(type="integer", low=8, high=64),
            "tab_n_a": ParameterSpace(type="integer", low=8, high=64),
            "tab_n_steps": ParameterSpace(type="integer", low=3, high=10),
            "tab_gamma": ParameterSpace(type="real", low=1.0, high=2.0),
            "tab_lambda_sparse": ParameterSpace(type="real", low=1e-6, high=1e-3),
            "tab_optimizer_lr": ParameterSpace(type="real", low=0.005, high=0.05),
            "tab_scheduler_step_size": ParameterSpace(type="integer", low=10, high=50),
            "tab_scheduler_gamma": ParameterSpace(type="real", low=0.8, high=0.99),
            "tab_n_independent": ParameterSpace(type="integer", low=1, high=5),
            "tab_n_shared": ParameterSpace(type="integer", low=1, high=5),
            "tab_momentum": ParameterSpace(type="real", low=0.01, high=0.4),
        }

    @staticmethod
    def get_adaboost_parameter_space() -> Dict[str, ParameterSpace]:
        """AdaBoostのパラメータ空間"""
        return {
            "ada_n_estimators": ParameterSpace(type="integer", low=50, high=300),
            "ada_learning_rate": ParameterSpace(type="real", low=0.01, high=2.0),
            "ada_algorithm": ParameterSpace(
                type="categorical", categories=["SAMME", "SAMME.R"]
            ),
        }

    @staticmethod
    def get_extratrees_parameter_space() -> Dict[str, ParameterSpace]:
        """ExtraTreesのパラメータ空間"""
        return {
            "et_n_estimators": ParameterSpace(type="integer", low=50, high=300),
            "et_max_depth": ParameterSpace(type="integer", low=3, high=20),
            "et_min_samples_split": ParameterSpace(type="integer", low=2, high=20),
            "et_min_samples_leaf": ParameterSpace(type="integer", low=1, high=10),
            "et_max_features": ParameterSpace(
                type="categorical", categories=["sqrt", "log2", "auto"]
            ),
            "et_bootstrap": ParameterSpace(
                type="categorical", categories=[True, False]
            ),
        }

    @staticmethod
    def get_gradientboosting_parameter_space() -> Dict[str, ParameterSpace]:
        """GradientBoostingのパラメータ空間"""
        return {
            "gb_n_estimators": ParameterSpace(type="integer", low=50, high=300),
            "gb_learning_rate": ParameterSpace(type="real", low=0.01, high=0.3),
            "gb_max_depth": ParameterSpace(type="integer", low=3, high=15),
            "gb_min_samples_split": ParameterSpace(type="integer", low=2, high=20),
            "gb_min_samples_leaf": ParameterSpace(type="integer", low=1, high=10),
            "gb_subsample": ParameterSpace(type="real", low=0.5, high=1.0),
            "gb_max_features": ParameterSpace(
                type="categorical", categories=["sqrt", "log2", "auto"]
            ),
        }

    @staticmethod
    def get_knn_parameter_space() -> Dict[str, ParameterSpace]:
        """KNNのパラメータ空間（最適化された設定）"""
        return {
            "knn_n_neighbors": ParameterSpace(type="integer", low=3, high=20),
            "knn_weights": ParameterSpace(
                type="categorical", categories=["uniform", "distance"]
            ),
            "knn_algorithm": ParameterSpace(
                type="categorical", categories=["auto", "ball_tree", "kd_tree", "brute"]
            ),
            "knn_leaf_size": ParameterSpace(type="integer", low=10, high=50),
            "knn_p": ParameterSpace(type="integer", low=1, high=2),
            "knn_metric": ParameterSpace(
                type="categorical",
                categories=["minkowski", "euclidean", "manhattan", "chebyshev"],
            ),
        }

    @staticmethod
    def get_ridge_parameter_space() -> Dict[str, ParameterSpace]:
        """Ridgeのパラメータ空間"""
        return {
            "ridge_alpha": ParameterSpace(type="real", low=0.01, high=100.0),
            "ridge_solver": ParameterSpace(
                type="categorical",
                categories=[
                    "auto",
                    "svd",
                    "cholesky",
                    "lsqr",
                    "sparse_cg",
                    "sag",
                    "saga",
                ],
            ),
            "ridge_max_iter": ParameterSpace(type="integer", low=100, high=2000),
        }

    @staticmethod
    def get_naivebayes_parameter_space() -> Dict[str, ParameterSpace]:
        """NaiveBayesのパラメータ空間"""
        return {
            "nb_alpha": ParameterSpace(type="real", low=0.01, high=10.0),
            "nb_fit_prior": ParameterSpace(
                type="categorical", categories=[True, False]
            ),
        }

    @staticmethod
    def get_bagging_parameter_space() -> Dict[str, ParameterSpace]:
        """バギングアンサンブル固有のパラメータ空間"""
        return {
            "bagging_n_estimators": ParameterSpace(type="integer", low=3, high=10),
            "bagging_max_samples": ParameterSpace(type="real", low=0.5, high=1.0),
            "bagging_max_features": ParameterSpace(type="real", low=0.5, high=1.0),
        }

    @staticmethod
    def get_stacking_parameter_space() -> Dict[str, ParameterSpace]:
        """スタッキングアンサンブル固有のパラメータ空間"""
        return {
            # メタモデル（LogisticRegression）のパラメータ
            "stacking_meta_C": ParameterSpace(type="real", low=0.01, high=10.0),
            "stacking_meta_penalty": ParameterSpace(
                type="categorical", categories=["l1", "l2", "elasticnet"]
            ),
            "stacking_meta_solver": ParameterSpace(
                type="categorical", categories=["liblinear", "saga"]
            ),
            # CV分割数
            "stacking_cv_folds": ParameterSpace(type="integer", low=3, high=10),
        }

    @classmethod
    def get_ensemble_parameter_space(
        self, ensemble_method: str, enabled_models: list
    ) -> Dict[str, ParameterSpace]:
        """
        アンサンブル手法と有効なモデルに基づいてパラメータ空間を構築

        Args:
            ensemble_method: アンサンブル手法 ("bagging" or "stacking")
            enabled_models: 有効なベースモデルのリスト

        Returns:
            統合されたパラメータ空間
        """
        parameter_space = {}

        # ベースモデルのパラメータ空間を追加
        if "lightgbm" in enabled_models:
            parameter_space.update(self.get_lightgbm_parameter_space())

        if "xgboost" in enabled_models:
            parameter_space.update(self.get_xgboost_parameter_space())

        if "randomforest" in enabled_models:
            parameter_space.update(self.get_randomforest_parameter_space())

        if "catboost" in enabled_models:
            parameter_space.update(self.get_catboost_parameter_space())

        if "tabnet" in enabled_models:
            parameter_space.update(self.get_tabnet_parameter_space())

        if "adaboost" in enabled_models:
            parameter_space.update(self.get_adaboost_parameter_space())

        if "extratrees" in enabled_models:
            parameter_space.update(self.get_extratrees_parameter_space())

        if "gradientboosting" in enabled_models:
            parameter_space.update(self.get_gradientboosting_parameter_space())

        if "knn" in enabled_models:
            parameter_space.update(self.get_knn_parameter_space())

        if "ridge" in enabled_models:
            parameter_space.update(self.get_ridge_parameter_space())

        if "naivebayes" in enabled_models:
            parameter_space.update(self.get_naivebayes_parameter_space())

        # アンサンブル手法固有のパラメータを追加
        if ensemble_method == "bagging":
            parameter_space.update(self.get_bagging_parameter_space())
        elif ensemble_method == "stacking":
            parameter_space.update(self.get_stacking_parameter_space())

        return parameter_space



    @staticmethod
    def _suggest_lightgbm_params(trial: optuna.Trial) -> Dict[str, any]:
        """LightGBMパラメータの提案"""
        return {
            "lgb_num_leaves": trial.suggest_int("lgb_num_leaves", 10, 100),
            "lgb_learning_rate": trial.suggest_float(
                "lgb_learning_rate", 0.01, 0.3, log=True
            ),
            "lgb_feature_fraction": trial.suggest_float(
                "lgb_feature_fraction", 0.5, 1.0
            ),
            "lgb_bagging_fraction": trial.suggest_float(
                "lgb_bagging_fraction", 0.5, 1.0
            ),
            "lgb_min_data_in_leaf": trial.suggest_int("lgb_min_data_in_leaf", 5, 50),
            "lgb_max_depth": trial.suggest_int("lgb_max_depth", 3, 15),
            "lgb_reg_alpha": trial.suggest_float("lgb_reg_alpha", 0.0, 1.0),
            "lgb_reg_lambda": trial.suggest_float("lgb_reg_lambda", 0.0, 1.0),
        }

    @staticmethod
    def _suggest_xgboost_params(trial: optuna.Trial) -> Dict[str, any]:
        """XGBoostパラメータの提案"""
        return {
            "xgb_max_depth": trial.suggest_int("xgb_max_depth", 3, 15),
            "xgb_learning_rate": trial.suggest_float(
                "xgb_learning_rate", 0.01, 0.3, log=True
            ),
            "xgb_subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
            "xgb_colsample_bytree": trial.suggest_float(
                "xgb_colsample_bytree", 0.5, 1.0
            ),
            "xgb_min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 10),
            "xgb_reg_alpha": trial.suggest_float("xgb_reg_alpha", 0.0, 1.0),
            "xgb_reg_lambda": trial.suggest_float("xgb_reg_lambda", 0.0, 1.0),
            "xgb_gamma": trial.suggest_float("xgb_gamma", 0.0, 0.5),
        }

    @staticmethod
    def _suggest_randomforest_params(trial: optuna.Trial) -> Dict[str, any]:
        """RandomForestパラメータの提案"""
        return {
            "rf_n_estimators": trial.suggest_int("rf_n_estimators", 50, 300),
            "rf_max_depth": trial.suggest_int("rf_max_depth", 3, 20),
            "rf_min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
            "rf_min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
            "rf_max_features": trial.suggest_categorical(
                "rf_max_features", ["sqrt", "log2", "auto"]
            ),
            "rf_bootstrap": trial.suggest_categorical("rf_bootstrap", [True, False]),
        }

    @staticmethod
    def _suggest_knn_params(trial: optuna.Trial) -> Dict[str, any]:
        """KNNパラメータの提案（最適化された距離計算）"""
        return {
            "knn_n_neighbors": trial.suggest_int("knn_n_neighbors", 3, 20),
            "knn_weights": trial.suggest_categorical(
                "knn_weights", ["uniform", "distance"]
            ),
            "knn_algorithm": trial.suggest_categorical(
                "knn_algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
            ),
            "knn_leaf_size": trial.suggest_int("knn_leaf_size", 10, 50),
            "knn_p": trial.suggest_int("knn_p", 1, 2),
            "knn_metric": trial.suggest_categorical(
                "knn_metric", ["minkowski", "euclidean", "manhattan", "chebyshev"]
            ),
        }

    @staticmethod
    def _suggest_bagging_params(trial: optuna.Trial) -> Dict[str, any]:
        """バギングパラメータの提案"""
        return {
            "bagging_n_estimators": trial.suggest_int("bagging_n_estimators", 3, 10),
            "bagging_max_samples": trial.suggest_float("bagging_max_samples", 0.5, 1.0),
            "bagging_max_features": trial.suggest_float(
                "bagging_max_features", 0.5, 1.0
            ),
        }

    @staticmethod
    def _suggest_stacking_params(trial: optuna.Trial) -> Dict[str, any]:
        """スタッキングパラメータの提案"""
        return {
            "stacking_meta_C": trial.suggest_float(
                "stacking_meta_C", 0.01, 10.0, log=True
            ),
            "stacking_meta_penalty": trial.suggest_categorical(
                "stacking_meta_penalty", ["l1", "l2", "elasticnet"]
            ),
            "stacking_meta_solver": trial.suggest_categorical(
                "stacking_meta_solver", ["liblinear", "saga"]
            ),
            "stacking_cv_folds": trial.suggest_int("stacking_cv_folds", 3, 10),
        }
