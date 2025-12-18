"""
アルゴリズム名レジストリ

MLモデルのアルゴリズム名を一元管理するクラスを提供します。
モデルクラス名と標準化されたアルゴリズム名のマッピングを管理し、
一貫性のあるアルゴリズム名の使用を保証します。
"""

import logging

logger = logging.getLogger(__name__)


class AlgorithmRegistry:
    """
    アルゴリズム名レジストリクラス

    モデルクラス名と標準化されたアルゴリズム名のマッピングを一元管理し、
    アルゴリズム名の正規化・検証・取得機能を提供します。
    """

    # クラス名からアルゴリズム名へのマッピング（Essential 2 Modelsのみ）
    _CLASS_TO_ALGORITHM_MAPPING = {
        # LightGBM
        "lightgbmmodel": "lightgbm",
        "lgbmclassifier": "lightgbm",
        "lgbmregressor": "lightgbm",
        # XGBoost
        "xgbclassifier": "xgboost",
        "xgbregressor": "xgboost",
        # Ensemble
        "ensembletrainer": "ensemble",
        "stackingensemble": "stacking",
        # CatBoost
        "catboostmodel": "catboost",
        "catboostclassifier": "catboost",
        "catboostregressor": "catboost",
    }

    # サポートされているアルゴリズム名のセット
    _SUPPORTED_ALGORITHMS = set(_CLASS_TO_ALGORITHM_MAPPING.values())

    @classmethod
    def get_algorithm_name(cls, model_class_name: str) -> str:
        """モデルクラス名から標準化されたアルゴリズム名を取得"""
        if not model_class_name:
            return "unknown"
        name = model_class_name.lower()

        # 1. 直接マッピング
        for key, val in cls._CLASS_TO_ALGORITHM_MAPPING.items():
            if key in name:
                return val

        # 2. 接尾辞除去による推測
        base = name
        for s in ["trainer", "model", "classifier", "regressor", "wrapper"]:
            if base.endswith(s):
                base = base[:-len(s)]
                break
        
        return base if base in cls._SUPPORTED_ALGORITHMS else "unknown"


# グローバルインスタンスを作成
algorithm_registry = AlgorithmRegistry()



