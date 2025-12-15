"""
オートストラテジーローダー

循環参照を回避しつつ、オートストラテジー関連のモジュールを動的にロードして戦略クラスを生成するためのヘルパークラスです。
"""

import logging
from typing import Any, Dict, Type

from backtesting import Strategy

logger = logging.getLogger(__name__)


class AutoStrategyLoaderError(Exception):
    """オートストラテジー読み込みエラー"""


class AutoStrategyLoader:
    """
    オートストラテジーローダー

    app.services.auto_strategyパッケージへの依存を一箇所に集約し、
    遅延インポートを用いて循環参照を回避します。
    """

    def load_strategy_gene(self, strategy_config: Dict[str, Any]) -> Any:
        """
        戦略設定から戦略遺伝子オブジェクトをロード・復元

        Args:
            strategy_config: 戦略設定

        Returns:
            StrategyGeneオブジェクト

        Raises:
            AutoStrategyLoaderError: ロードに失敗した場合
        """
        try:
            # 遅延インポート
            from app.services.auto_strategy.models import StrategyGene
            from app.services.auto_strategy.serializers.gene_serialization import (
                GeneSerializer,
            )
        except ImportError as e:
            raise AutoStrategyLoaderError(
                f"オートストラテジーモジュールのインポートに失敗しました: {e}"
            )

        # 戦略遺伝子データを抽出
        gene_data = self._extract_strategy_gene(strategy_config)
        if not gene_data:
            raise AutoStrategyLoaderError(
                "オートストラテジーの設定に戦略遺伝子 (strategy_gene) が含まれていません。"
            )

        try:
            # 戦略遺伝子を復元
            serializer = GeneSerializer()
            gene = serializer.dict_to_strategy_gene(gene_data, StrategyGene)
            return gene
        except Exception as e:
            raise AutoStrategyLoaderError(f"戦略遺伝子の復元に失敗しました: {e}")

    def create_auto_strategy_class(
        self, strategy_config: Dict[str, Any]
    ) -> Type[Strategy]:
        """
        オートストラテジーのクラスを生成

        Args:
            strategy_config: 戦略設定

        Returns:
            生成された戦略クラス (UniversalStrategy)

        Raises:
            AutoStrategyLoaderError: クラス生成に失敗した場合
        """
        # 遅延インポートで循環参照を回避
        try:
            from app.services.auto_strategy.generators.strategy_factory import (
                StrategyFactory,
            )
        except ImportError as e:
            raise AutoStrategyLoaderError(
                f"オートストラテジーモジュールのインポートに失敗しました: {e}"
            )

        # 遺伝子をロードして検証（整合性チェックのため）
        gene = self.load_strategy_gene(strategy_config)

        # 戦略ファクトリーで戦略クラスを取得（UniversalStrategyが返される）
        strategy_factory = StrategyFactory()
        strategy_class = strategy_factory.create_strategy_class(gene)

        return strategy_class

    def _extract_strategy_gene(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """戦略設定から戦略遺伝子を抽出"""
        # 直接strategy_geneがある場合
        if "strategy_gene" in strategy_config:
            return strategy_config["strategy_gene"]

        # parametersの中にある場合
        parameters = strategy_config.get("parameters", {})
        if "strategy_gene" in parameters:
            return parameters["strategy_gene"]

        return {}
