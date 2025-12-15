"""
ML関連のデフォルト設定定数

トレーニング設定などのハードコードされた値を統一管理します。
"""

from typing import Any, Dict


class DefaultTrainingConfigs:
    """デフォルトのトレーニング設定を提供するクラス"""

    @staticmethod
    def get_default_ensemble_config() -> Dict[str, Any]:
        """
        デフォルトのアンサンブル設定を取得

        Returns:
            Dict[str, Any]: デフォルトのアンサンブル設定辞書
        """
        return {
            "enabled": True,
            "method": "stacking",
            "stacking_params": {
                "base_models": [
                    "lightgbm",
                    "xgboost",
                ],
                "meta_model": "lightgbm",
                "cv_folds": 5,
                "use_probas": True,
                "random_state": 42,
            },
        }

    @staticmethod
    def get_default_single_model_config() -> Dict[str, Any]:
        """
        デフォルトの単一モデル設定を取得

        Returns:
            Dict[str, Any]: デフォルトの単一モデル設定辞書
        """
        return {"model_type": "lightgbm"}


# モジュールレベルの便利関数
def get_default_ensemble_config() -> Dict[str, Any]:
    """
    デフォルトのアンサンブル設定を取得

    Returns:
        Dict[str, Any]: デフォルトのアンサンブル設定辞書

    Example:
        >>> config = get_default_ensemble_config()
        >>> config["method"]
        'stacking'
    """
    return DefaultTrainingConfigs.get_default_ensemble_config()


def get_default_single_model_config() -> Dict[str, Any]:
    """
    デフォルトの単一モデル設定を取得

    Returns:
        Dict[str, Any]: デフォルトの単一モデル設定辞書

    Example:
        >>> config = get_default_single_model_config()
        >>> config["model_type"]
        'lightgbm'
    """
    return DefaultTrainingConfigs.get_default_single_model_config()


