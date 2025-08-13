"""
戦略クラスファクトリー

バックテスト用の戦略クラス生成を担当します。
"""

import logging
from typing import Any, Dict, Type

from backtesting import Strategy

logger = logging.getLogger(__name__)


class StrategyClassCreationError(Exception):
    """戦略クラス生成エラー"""


class StrategyClassFactory:
    """
    戦略クラスファクトリー

    設定から適切な戦略クラスを生成します。
    """

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
                return self._create_auto_strategy_class(strategy_config)
            else:
                # 将来的に他の戦略タイプをサポートする場合はここに追加
                raise StrategyClassCreationError(
                    f"サポートされていない戦略タイプです: {strategy_config.get('strategy_type')}"
                )

        except Exception as e:
            logger.error(f"戦略クラス生成エラー: {e}")
            raise StrategyClassCreationError(f"戦略クラスの生成に失敗しました: {e}")

    def _is_auto_strategy(self, strategy_config: Dict[str, Any]) -> bool:
        """オートストラテジーかどうかを判定"""
        return "strategy_gene" in strategy_config or (
            "parameters" in strategy_config
            and "strategy_gene" in strategy_config["parameters"]
        )

    def _create_auto_strategy_class(
        self, strategy_config: Dict[str, Any]
    ) -> Type[Strategy]:
        """
        オートストラテジーのクラスを生成

        Args:
            strategy_config: 戦略設定

        Returns:
            生成された戦略クラス

        Raises:
            StrategyClassCreationError: 戦略遺伝子が見つからない場合
        """
        # 遅延インポートで循環参照を回避
        try:
            from app.services.auto_strategy.generators.strategy_factory import (
                StrategyFactory,
            )
            from app.services.auto_strategy.models.gene_serialization import (
                GeneSerializer,
            )
        except ImportError as e:
            raise StrategyClassCreationError(
                f"オートストラテジーモジュールのインポートに失敗しました: {e}"
            )

        # 戦略遺伝子を取得
        gene_data = self._extract_strategy_gene(strategy_config)
        if not gene_data:
            raise StrategyClassCreationError(
                "オートストラテジーの実行には、戦略遺伝子 (strategy_gene) が必要です。"
            )

        try:
            # 戦略遺伝子を復元
            from app.services.auto_strategy.models.gene_strategy import StrategyGene

            serializer = GeneSerializer()
            gene = serializer.dict_to_strategy_gene(gene_data, StrategyGene)

            # 戦略ファクトリーで戦略クラスを生成
            strategy_factory = StrategyFactory()
            strategy_class = strategy_factory.create_strategy_class(gene)

            return strategy_class

        except Exception as e:
            raise StrategyClassCreationError(f"戦略遺伝子の復元に失敗しました: {e}")

    def _extract_strategy_gene(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """戦略設定から戦略遺伝子を抽出"""
        # 直接strategy_geneがある場合
        if "strategy_gene" in strategy_config:
            return strategy_config["strategy_gene"]

        # parametersの中にある場合
        parameters = strategy_config.get("parameters", {})
        if "strategy_gene" in parameters:
            return parameters["strategy_gene"]

        return None

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
