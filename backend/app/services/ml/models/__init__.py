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
        pass

        available.append("lightgbm")
    except ImportError:
        pass

    try:
        pass

        available.append("xgboost")
    except ImportError:
        pass

    try:
        pass

        available.append("catboost")
    except ImportError:
        pass

    try:
        pass

        available.append("tabnet")
    except ImportError:
        pass

    try:
        pass

        available.append("randomforest")
    except ImportError:
        pass

    try:
        pass

        available.append("extratrees")
    except ImportError:
        pass

    try:
        pass

        available.append("gradientboosting")
    except ImportError:
        pass

    try:
        pass

        available.append("adaboost")
    except ImportError:
        pass

    try:
        pass

        available.append("ridge")
    except ImportError:
        pass

    try:
        pass

        available.append("naivebayes")
    except ImportError:
        pass

    return available
