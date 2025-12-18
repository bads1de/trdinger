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

from .catboost import CatBoostModel
from .lightgbm import LightGBMModel
from .xgboost import XGBoostModel
from ..common.algorithm_registry import algorithm_registry

__all__ = [
    "LightGBMModel",
    "XGBoostModel",
    "CatBoostModel",
    "algorithm_registry",
]


def get_available_models():
    """
    利用可能なモデルラッパーのリストを取得（Essential 3 Models）

    Returns:
        利用可能なモデルタイプのリスト
    """
    import importlib.util

    available = []

    # Essential 3 Modelsをチェック
    essential_models = ["lightgbm", "xgboost", "catboost"]

    for model in essential_models:
        if importlib.util.find_spec(model):
            available.append(model)

    return available



