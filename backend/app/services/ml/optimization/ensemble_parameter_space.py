"""
アンサンブル学習用ハイパーパラメータ空間定義

アンサンブル学習で使用される各ベースモデルとアンサンブル手法固有の
パラメータ空間を定義します。Optunaの高度な最適化機能を活用。
"""

from typing import Any, Dict

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
        cls, ensemble_method: str, enabled_models: list
    ) -> Dict[str, ParameterSpace]:
        """
        アンサンブル手法と有効なモデルに基づいてパラメータ空間を構築

        Args:
            ensemble_method: アンサンブル手法 ("stacking")
            enabled_models: 有効なベースモデルのリスト

        Returns:
            統合されたパラメータ空間
        """
        parameter_space = {}

        # ベースモデルのパラメータ空間を追加
        if "lightgbm" in enabled_models:
            parameter_space.update(cls.get_lightgbm_parameter_space())

        if "xgboost" in enabled_models:
            parameter_space.update(cls.get_xgboost_parameter_space())

        # アンサンブル手法固有のパラメータを追加（スタッキング）
        if ensemble_method == "stacking":
            parameter_space.update(cls.get_stacking_parameter_space())

        return parameter_space

    @staticmethod
    def _suggest_lightgbm_params(trial: optuna.Trial) -> Dict[str, Any]:
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
    def _suggest_xgboost_params(trial: optuna.Trial) -> Dict[str, Any]:
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
    def _suggest_stacking_params(trial: optuna.Trial) -> Dict[str, Any]:
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



