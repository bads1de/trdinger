"""
ML関連のデフォルト設定定数

トレーニング設定などのハードコードされた値を統一管理します。
"""

from typing import Any, Dict


def get_default_ensemble_config() -> Dict[str, Any]:
    """デフォルトのアンサンブル設定を取得"""
    return {
        "enabled": True, "method": "stacking",
        "stacking_params": {
            "base_models": ["lightgbm", "xgboost"],
            "meta_model": "lightgbm", "cv_folds": 5,
            "use_probas": True, "random_state": 42,
        },
    }


def get_default_single_model_config() -> Dict[str, Any]:
    """デフォルトの単一モデル設定を取得"""
    return {"model_type": "lightgbm"}



