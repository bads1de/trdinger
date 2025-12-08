"""
戦略クラスファクトリー

"""

import logging
from typing import Any, Dict, Type

from backtesting import Strategy

from .auto_strategy_loader import AutoStrategyLoader, AutoStrategyLoaderError

logger = logging.getLogger(__name__)


class StrategyClassCreationError(Exception):
    """戦略クラス生成エラー"""


class StrategyClassFactory:
    """
    戦略クラスファクトリー

    設定から適切な戦略クラスを生成します。
    """

    def __init__(self) -> None:
        self._auto_strategy_loader = AutoStrategyLoader()

    def create_strategy_class(self, strategy_config: Dict[str, Any]) -> Type[Strategy]:
        """
        戦略設定から戦略クラスを生成

        Args:
            strategy_config: 戦略設定

        Returns:
            生成された戦略クラス

        Raises:
            StrategyClassCreationError: 戦略クラス生成に失敗した場合
        """
        try:
            # オートストラテジーの場合
            if self._is_auto_strategy(strategy_config):
                try:
                    return self._auto_strategy_loader.create_auto_strategy_class(
                        strategy_config
                    )
                except AutoStrategyLoaderError as e:
                    raise StrategyClassCreationError(
                        f"オートストラテジーの生成に失敗しました: {e}"
                    )
            else:
                # 将来的に他の戦略タイプをサポートする場合はここに追加
                raise StrategyClassCreationError(
                    f"サポートされていない戦略タイプです: {strategy_config.get('strategy_type')}"
                )

        except StrategyClassCreationError:
            raise
        except Exception as e:
            logger.error(f"戦略クラス生成エラー: {e}")
            raise StrategyClassCreationError(f"戦略クラスの生成に失敗しました: {e}")

    def _is_auto_strategy(self, strategy_config: Dict[str, Any]) -> bool:
        """オートストラテジーかどうかを判定"""
        return "strategy_gene" in strategy_config or (
            "parameters" in strategy_config
            and "strategy_gene" in strategy_config["parameters"]
        )

    def get_strategy_parameters(
        self, strategy_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        戦略設定からパラメータを抽出

        Args:
            strategy_config: 戦略設定

        Returns:
            戦略パラメータ
        """
        if self._is_auto_strategy(strategy_config):
            # オートストラテジーの場合、パラメータは戦略クラス生成時に設定済み
            return {}
        else:
            # 通常の戦略の場合、parametersを返す
            return strategy_config.get("parameters", {})
