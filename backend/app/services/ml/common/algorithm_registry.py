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
        """
        モデルクラス名から標準化されたアルゴリズム名を取得

        Args:
            model_class_name: モデルクラス名（小文字）

        Returns:
            標準化されたアルゴリズム名
        """
        if not model_class_name:
            logger.warning("モデルクラス名が空です")
            return "unknown"

        # 小文字に変換して処理
        class_name_lower = model_class_name.lower()

        # マッピングから検索
        for key, value in cls._CLASS_TO_ALGORITHM_MAPPING.items():
            if key in class_name_lower:
                logger.debug(
                    f"アルゴリズム名をマッピング: {class_name_lower} -> {value}"
                )
                return value

        # マッピングが見つからない場合はクラス名から推測
        # "trainer", "model", "classifier", "regressor" などの接尾辞を除去
        base_name = class_name_lower
        for suffix in ["trainer", "model", "classifier", "regressor", "wrapper"]:
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]
                break

        if base_name in cls._SUPPORTED_ALGORITHMS:
            logger.debug(
                f"クラス名からアルゴリズム名を推測: {class_name_lower} -> {base_name}"
            )
            return base_name

        logger.warning(f"未知のモデルクラス名: {model_class_name}")
        return "unknown"


# グローバルインスタンスを作成
algorithm_registry = AlgorithmRegistry()
