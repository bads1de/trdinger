"""
アルゴリズム名レジストリ

MLモデルのアルゴリズム名を一元管理するクラスを提供します。
モデルクラス名と標準化されたアルゴリズム名のマッピングを管理し、
一貫性のあるアルゴリズム名の使用を保証します。
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class AlgorithmRegistry:
    """
    アルゴリズム名レジストリクラス

    モデルクラス名と標準化されたアルゴリズム名のマッピングを一元管理し、
    アルゴリズム名の正規化・検証・取得機能を提供します。
    """

    # クラス名からアルゴリズム名へのマッピング
    _CLASS_TO_ALGORITHM_MAPPING = {
        # LightGBM
        "lightgbmmodel": "lightgbm",
        "lgbmclassifier": "lightgbm",
        "lgbmregressor": "lightgbm",
        # XGBoost
        "xgbclassifier": "xgboost",
        "xgbregressor": "xgboost",
        # CatBoost
        "catboostclassifier": "catboost",
        "catboostregressor": "catboost",
        # RandomForest
        "randomforestclassifier": "randomforest",
        "randomforestregressor": "randomforest",
        "randomforestmodel": "randomforest",
        # ExtraTrees
        "extratreesclassifier": "extratrees",
        "extratreesregressor": "extratrees",
        "extratreesmodel": "extratrees",
        # GradientBoosting
        "gradientboostingclassifier": "gradientboosting",
        "gradientboostingregressor": "gradientboosting",
        "gradientboostingmodel": "gradientboosting",
        # AdaBoost
        "adaboostclassifier": "adaboost",
        "adaboostregressor": "adaboost",
        "adaboostmodel": "adaboost",
        # Ridge
        "ridgeclassifier": "ridge",
        "ridgeregressor": "ridge",
        "ridgemodel": "ridge",
        # NaiveBayes
        "gaussiannb": "naivebayes",
        "naivebayesmodel": "naivebayes",
        # KNN
        "kneighborsclassifier": "knn",
        "kneighborsregressor": "knn",
        "knnmodel": "knn",
        # TabNet
        "tabnetclassifier": "tabnet",
        "tabnetregressor": "tabnet",
        # Ensemble
        "ensembletrainer": "ensemble",
        "baggingensemble": "bagging",
        "stackingensemble": "stacking",
        # Single Model
        "singlemodeltrainer": "single",
    }

    # アルゴリズム名から表示名へのマッピング（オプション）
    _ALGORITHM_TO_DISPLAY_NAME = {
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "catboost": "CatBoost",
        "randomforest": "Random Forest",
        "extratrees": "Extra Trees",
        "gradientboosting": "Gradient Boosting",
        "adaboost": "AdaBoost",
        "ridge": "Ridge Classifier",
        "naivebayes": "Naive Bayes",
        "knn": "K-Nearest Neighbors",
        "tabnet": "TabNet",
        "ensemble": "Ensemble",
        "bagging": "Bagging",
        "stacking": "Stacking",
        "single": "Single Model",
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

    @classmethod
    def get_display_name(cls, algorithm_name: str) -> str:
        """
        アルゴリズム名から表示名を取得

        Args:
            algorithm_name: 標準化されたアルゴリズム名

        Returns:
            表示名
        """
        return cls._ALGORITHM_TO_DISPLAY_NAME.get(
            algorithm_name, algorithm_name.title()
        )

    @classmethod
    def is_supported_algorithm(cls, algorithm_name: str) -> bool:
        """
        アルゴリズムがサポートされているかを判定

        Args:
            algorithm_name: アルゴリズム名

        Returns:
            サポートされている場合はTrue
        """
        return algorithm_name in cls._SUPPORTED_ALGORITHMS

    @classmethod
    def get_supported_algorithms(cls) -> list:
        """
        サポートされているアルゴリズム名のリストを取得

        Returns:
            サポートされているアルゴリズム名のリスト
        """
        return sorted(cls._SUPPORTED_ALGORITHMS)

    @classmethod
    def get_algorithm_mapping(cls) -> Dict[str, str]:
        """
        クラス名からアルゴリズム名へのマッピング辞書を取得

        Returns:
            マッピング辞書のコピー
        """
        return cls._CLASS_TO_ALGORITHM_MAPPING.copy()

    @classmethod
    def add_algorithm_mapping(cls, class_name: str, algorithm_name: str) -> None:
        """
        新しいアルゴリズムマッピングを追加

        Args:
            class_name: クラス名
            algorithm_name: アルゴリズム名
        """
        cls._CLASS_TO_ALGORITHM_MAPPING[class_name.lower()] = algorithm_name.lower()
        cls._SUPPORTED_ALGORITHMS.add(algorithm_name.lower())
        logger.info(
            f"新しいアルゴリズムマッピングを追加: {class_name} -> {algorithm_name}"
        )

    @classmethod
    def validate_algorithm_name(cls, algorithm_name: str) -> bool:
        """
        アルゴリズム名を検証

        Args:
            algorithm_name: 検証するアルゴリズム名

        Returns:
            有効なアルゴリズム名の場合はTrue
        """
        if not algorithm_name:
            return False

        return algorithm_name.lower() in cls._SUPPORTED_ALGORITHMS


# グローバルインスタンスを作成
algorithm_registry = AlgorithmRegistry()
