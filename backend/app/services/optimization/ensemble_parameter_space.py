"""
アンサンブル学習用ハイパーパラメータ空間定義

アンサンブル学習で使用される各ベースモデルとアンサンブル手法固有の
パラメータ空間を定義します。
"""

from typing import Dict
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

        # アンサンブル手法固有のパラメータを追加
        if ensemble_method == "bagging":
            parameter_space.update(self.get_bagging_parameter_space())
        elif ensemble_method == "stacking":
            parameter_space.update(self.get_stacking_parameter_space())

        return parameter_space
