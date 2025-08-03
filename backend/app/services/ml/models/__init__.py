"""
MLモデルラッパーモジュール

アンサンブル学習で使用する各種機械学習モデルのラッパークラスを提供します。
全てのモデルラッパーは統一されたインターフェースを実装し、
アンサンブル内で一貫した方法で使用できます。

統一インターフェース:
- __init__(automl_config: Optional[Dict[str, Any]] = None)
- _train_model_impl(X_train, X_test, y_train, y_test, **training_params) -> Dict[str, Any]
- predict(X: pd.DataFrame) -> np.ndarray
- predict_proba(X: pd.DataFrame) -> np.ndarray
- is_trained: bool プロパティ
- feature_columns: List[str] プロパティ
"""

# モデルラッパーのインポート（遅延インポートでImportErrorを回避）
__all__ = [
    "LightGBMModel",
    "XGBoostModel",
    "CatBoostModel",
    "TabNetModel",
    "RandomForestModel",
    "ExtraTreesModel",
    "GradientBoostingModel",
    "AdaBoostModel",
    "RidgeModel",
    "NaiveBayesModel",
    "KNNModel",
    "algorithm_registry",
]


def get_available_models():
    """
    利用可能なモデルラッパーのリストを取得

    Returns:
        利用可能なモデルタイプのリスト
    """
    available = []

    try:
        from .lightgbm_wrapper import LightGBMModel

        available.append("lightgbm")
    except ImportError:
        pass

    try:
        from .xgboost_wrapper import XGBoostModel

        available.append("xgboost")
    except ImportError:
        pass

    try:
        from .catboost_wrapper import CatBoostModel

        available.append("catboost")
    except ImportError:
        pass

    try:
        from .tabnet_wrapper import TabNetModel

        available.append("tabnet")
    except ImportError:
        pass

    try:
        from .randomforest_wrapper import RandomForestModel

        available.append("randomforest")
    except ImportError:
        pass

    try:
        from .extratrees_wrapper import ExtraTreesModel

        available.append("extratrees")
    except ImportError:
        pass

    try:
        from .gradientboosting_wrapper import GradientBoostingModel

        available.append("gradientboosting")
    except ImportError:
        pass

    try:
        from .adaboost_wrapper import AdaBoostModel

        available.append("adaboost")
    except ImportError:
        pass

    try:
        from .ridge_wrapper import RidgeModel

        available.append("ridge")
    except ImportError:
        pass

    try:
        from .naivebayes_wrapper import NaiveBayesModel

        available.append("naivebayes")
    except ImportError:
        pass

    return available
