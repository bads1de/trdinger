"""
オプティマイザーファクトリー

最適化手法に応じて適切なオプティマイザーインスタンスを生成します。
"""

import logging
from typing import Dict, Any, Optional, List

from .base_optimizer import BaseOptimizer
from .bayesian_optimizer import BayesianOptimizer
from .grid_search_optimizer import GridSearchOptimizer
from .random_search_optimizer import RandomSearchOptimizer

logger = logging.getLogger(__name__)


class OptimizerFactory:
    """
    オプティマイザーファクトリー

    最適化手法名に基づいて適切なオプティマイザーインスタンスを生成します。
    """

    # サポートされている最適化手法
    SUPPORTED_METHODS = {
        "bayesian": BayesianOptimizer,
        "grid": GridSearchOptimizer,
        "random": RandomSearchOptimizer,
    }

    # 手法の別名マッピング
    METHOD_ALIASES = {
        "bayes": "bayesian",
        "bayesian_optimization": "bayesian",
        "gp": "bayesian",
        "grid_search": "grid",
        "gridsearch": "grid",
        "random_search": "random",
        "randomsearch": "random",
        "random_sampling": "random",
    }

    @classmethod
    def create_optimizer(
        cls, method: str, config: Optional[Dict[str, Any]] = None
    ) -> BaseOptimizer:
        """
        最適化手法に応じてオプティマイザーを作成

        Args:
            method: 最適化手法名 ("bayesian", "grid", "random")
            config: オプティマイザー固有の設定（オプション）

        Returns:
            オプティマイザーインスタンス

        Raises:
            ValueError: 未対応の最適化手法が指定された場合
            RuntimeError: オプティマイザーの作成に失敗した場合
        """
        try:
            # 手法名を正規化
            normalized_method = cls._normalize_method_name(method)

            # オプティマイザークラスを取得
            optimizer_class = cls.SUPPORTED_METHODS.get(normalized_method)
            if optimizer_class is None:
                raise ValueError(
                    f"未対応の最適化手法: '{method}'. "
                    f"サポートされている手法: {list(cls.SUPPORTED_METHODS.keys())}"
                )

            # オプティマイザーインスタンスを作成
            optimizer = optimizer_class()

            # 設定を適用
            if config:
                cls._apply_config(optimizer, config)

            logger.info(
                f"オプティマイザーを作成しました: {optimizer.__class__.__name__}"
            )
            return optimizer

        except Exception as e:
            logger.error(f"オプティマイザー作成中にエラーが発生しました: {e}")
            raise RuntimeError(f"オプティマイザーの作成に失敗しました: {e}") from e

    @classmethod
    def _normalize_method_name(cls, method: str) -> str:
        """
        手法名を正規化

        Args:
            method: 手法名

        Returns:
            正規化された手法名
        """
        if not method or not isinstance(method, str):
            raise ValueError("手法名は空でない文字列である必要があります")

        # 小文字に変換し、空白を除去
        normalized = method.lower().strip()

        # 別名を正規名に変換
        if normalized in cls.METHOD_ALIASES:
            normalized = cls.METHOD_ALIASES[normalized]

        return normalized

    @classmethod
    def _apply_config(cls, optimizer: BaseOptimizer, config: Dict[str, Any]) -> None:
        """
        オプティマイザーに設定を適用

        Args:
            optimizer: オプティマイザーインスタンス
            config: 設定辞書
        """
        try:
            if hasattr(optimizer, "config") and isinstance(optimizer.config, dict):
                # 既存の設定を更新
                for key, value in config.items():
                    if key in optimizer.config:
                        optimizer.config[key] = value
                        logger.debug(f"設定を更新: {key} = {value}")
                    else:
                        logger.warning(f"未知の設定項目をスキップ: {key}")
            else:
                logger.warning("オプティマイザーに設定を適用できませんでした")

        except Exception as e:
            logger.warning(f"設定適用中にエラーが発生しました: {e}")

    @classmethod
    def get_supported_methods(cls) -> List[str]:
        """
        サポートされている最適化手法のリストを取得

        Returns:
            最適化手法名のリスト
        """
        return list(cls.SUPPORTED_METHODS.keys())

    @classmethod
    def get_method_aliases(cls) -> Dict[str, str]:
        """
        手法の別名マッピングを取得

        Returns:
            別名から正規名へのマッピング
        """
        return cls.METHOD_ALIASES.copy()

    @classmethod
    def is_supported_method(cls, method: str) -> bool:
        """
        指定された手法がサポートされているかチェック

        Args:
            method: 手法名

        Returns:
            サポートされている場合True
        """
        try:
            normalized_method = cls._normalize_method_name(method)
            return normalized_method in cls.SUPPORTED_METHODS
        except ValueError:
            return False

    @classmethod
    def get_method_description(cls, method: str) -> str:
        """
        最適化手法の説明を取得

        Args:
            method: 手法名

        Returns:
            手法の説明
        """
        descriptions = {
            "bayesian": "ベイジアン最適化 - ガウス過程を用いた効率的な最適化",
            "grid": "グリッドサーチ - パラメータ空間を網羅的に探索",
            "random": "ランダムサーチ - パラメータ空間をランダムに探索",
        }

        try:
            normalized_method = cls._normalize_method_name(method)
            return descriptions.get(normalized_method, "説明なし")
        except ValueError:
            return "未対応の手法"

    @classmethod
    def create_optimizer_with_defaults(
        cls, method: str, model_type: str = "lightgbm"
    ) -> BaseOptimizer:
        """
        デフォルト設定でオプティマイザーを作成

        Args:
            method: 最適化手法名
            model_type: モデルタイプ（デフォルトパラメータ空間の決定に使用）

        Returns:
            オプティマイザーインスタンス
        """
        try:
            optimizer = cls.create_optimizer(method)

            # モデルタイプに応じたデフォルト設定を適用
            if hasattr(optimizer, "get_default_parameter_space"):
                default_space = optimizer.get_default_parameter_space(model_type)
                logger.info(f"デフォルトパラメータ空間を設定: {model_type}")
                logger.debug(f"パラメータ空間: {list(default_space.keys())}")

            return optimizer

        except Exception as e:
            logger.error(f"デフォルトオプティマイザー作成中にエラーが発生しました: {e}")
            raise


# 便利関数
def create_bayesian_optimizer(
    config: Optional[Dict[str, Any]] = None,
) -> BayesianOptimizer:
    """ベイジアン最適化オプティマイザーを作成"""
    return OptimizerFactory.create_optimizer("bayesian", config)


def create_grid_search_optimizer(
    config: Optional[Dict[str, Any]] = None,
) -> GridSearchOptimizer:
    """グリッドサーチオプティマイザーを作成"""
    return OptimizerFactory.create_optimizer("grid", config)


def create_random_search_optimizer(
    config: Optional[Dict[str, Any]] = None,
) -> RandomSearchOptimizer:
    """ランダムサーチオプティマイザーを作成"""
    return OptimizerFactory.create_optimizer("random", config)
