"""
アンサンブル学習用ハイパーパラメータ空間定義

アンサンブル学習で使用される各ベースモデルとアンサンブル手法固有の
パラメータ空間を定義します。Optunaの高度な最適化機能を活用。
"""

from typing import Dict

from .optuna_optimizer import ParameterSpace, build_lightgbm_parameter_space


class EnsembleParameterSpace:
    """アンサンブル学習用パラメータ空間管理クラス

    各ベースモデル（LightGBM、XGBoostなど）とアンサンブル手法（スタッキングなど）
    に特化したハイパーパラメータの探索空間を定義します。
    Optunaによるベイズ最適化で使用するパラメータ範囲を指定します。
    """

    @staticmethod
    def get_lightgbm_parameter_space() -> Dict[str, ParameterSpace]:
        """LightGBM用のハイパーパラメータ探索空間を返す。

        Returns:
            Dict[str, ParameterSpace]: LightGBM固有のパラメータ空間。
                正則化パラメータ（reg_alpha、reg_lambda）を含む。
        """
        params = build_lightgbm_parameter_space(prefix="lgb_")
        params.update(
            {
                "lgb_reg_alpha": ParameterSpace(type="real", low=0.0, high=1.0),
                "lgb_reg_lambda": ParameterSpace(type="real", low=0.0, high=1.0),
            }
        )
        return params

    @staticmethod
    def get_xgboost_parameter_space() -> Dict[str, ParameterSpace]:
        """XGBoost用のハイパーパラメータ探索空間を返す。

        Returns:
            Dict[str, ParameterSpace]: XGBoost固有のパラメータ空間。
                木の深さ、学習率、サブサンプリング率、正則化パラメータなどを含む。
        """
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
        """スタッキングアンサンブル固有のハイパーパラメータ探索空間を返す。

        メタモデル（LogisticRegression）の正則化パラメータ、
        CV分割数など、スタッキング特有のパラメータを含みます。

        Returns:
            Dict[str, ParameterSpace]: スタッキング固有のパラメータ空間。
        """
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
        """統合されたハイパーパラメータ探索空間を構築する。

        指定されたアンサンブル手法と有効なベースモデルに応じて、
        必要なパラメータ空間を結合して返します。

        Args:
            ensemble_method: アンサンブル手法（"stacking"など）。
            enabled_models: 有効にするベースモデルのリスト
                （"lightgbm"、"xgboost"など）。

        Returns:
            Dict[str, ParameterSpace]: 統合されたパラメータ空間。
                指定されたモデルと手法に対応するパラメータのみを含む。
        """
        ps = {}
        # ベースモデル
        mappings = {
            "lightgbm": cls.get_lightgbm_parameter_space,
            "xgboost": cls.get_xgboost_parameter_space,
        }
        for m in enabled_models:
            if m in mappings:
                ps.update(mappings[m]())

        # アンサンブル手法
        if ensemble_method == "stacking":
            ps.update(cls.get_stacking_parameter_space())

        return ps
