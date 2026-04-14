"""
オートストラテジーローダー

循環参照を回避しつつ、オートストラテジー関連のモジュールを動的にロードして戦略クラスを生成するためのヘルパークラスです。
"""

import logging
from typing import Dict, Protocol, Type, cast

from app.types import SerializableValue
from backtesting import Strategy

logger = logging.getLogger(__name__)


class _ValidatableGene(Protocol):
    """validate メソッドを持つ遺伝子オブジェクトのプロトコル"""

    def validate(self) -> tuple[bool, list[str]]: ...


class AutoStrategyLoaderError(Exception):
    """オートストラテジー読み込みエラー"""


class AutoStrategyLoader:
    """
    オートストラテジーローダー

    app.services.auto_strategyパッケージへの依存を一箇所に集約し、
    遅延インポートを用いて循環参照を回避します。
    """

    def _import_strategy_gene_components(self) -> tuple[type, type]:
        """StrategyGene と GeneSerializer を遅延インポートする"""
        try:
            from app.services.auto_strategy.genes import StrategyGene
            from app.services.auto_strategy.serializers.serialization import (
                GeneSerializer,
            )

            return StrategyGene, GeneSerializer
        except ImportError as e:
            raise AutoStrategyLoaderError(
                f"オートストラテジーモジュールのインポートに失敗しました: {e}"
            )

    def _import_universal_strategy(self) -> Type[Strategy]:
        """UniversalStrategy を遅延インポートする"""
        try:
            from app.services.auto_strategy.strategies.universal_strategy import (
                UniversalStrategy,
            )

            return UniversalStrategy
        except ImportError as e:
            raise AutoStrategyLoaderError(
                f"オートストラテジーモジュールのインポートに失敗しました: {e}"
            )

    def load_strategy_gene(
        self, strategy_config: Dict[str, SerializableValue]
    ) -> object:
        """
        戦略設定から戦略遺伝子オブジェクトをロード・復元

        Args:
            strategy_config: 戦略設定

        Returns:
            StrategyGeneオブジェクト

        Raises:
            AutoStrategyLoaderError: ロードに失敗した場合
        """
        StrategyGene, GeneSerializer = self._import_strategy_gene_components()

        # 戦略遺伝子データを抽出
        gene_data = self._extract_strategy_gene(strategy_config)
        if gene_data is None or (isinstance(gene_data, dict) and not gene_data):
            raise AutoStrategyLoaderError(
                "オートストラテジーの設定に戦略遺伝子 (strategy_gene) が含まれていません。"
            )

        if isinstance(gene_data, StrategyGene):
            return gene_data

        # DEAP の Individual は StrategyGene を継承していることがある
        # 属性ベースで判定して対応
        if hasattr(gene_data, "indicators") and hasattr(
            gene_data, "long_entry_conditions"
        ):
            return gene_data

        try:
            # 戦略遺伝子を復元
            serializer = GeneSerializer()
            gene = serializer.dict_to_strategy_gene(gene_data, StrategyGene)
            return gene
        except Exception as e:
            raise AutoStrategyLoaderError(f"戦略遺伝子の復元に失敗しました: {e}")

    def create_auto_strategy_class(
        self, strategy_config: Dict[str, SerializableValue]
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
        UniversalStrategy = self._import_universal_strategy()

        # 遺伝子をロードして検証（整合性チェックのため）
        gene = cast(_ValidatableGene, self.load_strategy_gene(strategy_config))

        # 遺伝子のバリデーション実行
        is_valid, errors = gene.validate()
        if not is_valid:
            raise AutoStrategyLoaderError(f"無効な戦略遺伝子です: {', '.join(errors)}")

        # UniversalStrategyクラスを返す
        return UniversalStrategy

    def _extract_strategy_gene(
        self, strategy_config: Dict[str, SerializableValue]
    ):
        """戦略設定から戦略遺伝子を抽出"""
        # 直接strategy_geneがある場合
        gene_data = strategy_config.get("strategy_gene")
        
        # StrategyGene オブジェクトが直接渡された場合
        if gene_data is not None and hasattr(gene_data, "indicators") and hasattr(
            gene_data, "long_entry_conditions"
        ):
            return gene_data
        
        if isinstance(gene_data, dict):
            return gene_data

        # StrategyGene オブジェクトが直接渡された場合（to_dictメソッドがある場合）
        if gene_data is not None:
            if hasattr(gene_data, "to_dict"):
                return gene_data.to_dict()  # type: ignore[return-value]
            raise AutoStrategyLoaderError(
                f"戦略遺伝子の形式が不正です。dict または to_dict() メソッドを持つオブジェクトが必要です: {type(gene_data)}"
            )

        # parametersの中にある場合
        parameters = strategy_config.get("parameters", {})
        if isinstance(parameters, dict):
            gene_data = parameters.get("strategy_gene")
            
            # StrategyGene オブジェクトがparametersの中にある場合
            if gene_data is not None and hasattr(gene_data, "indicators") and hasattr(
                gene_data, "long_entry_conditions"
            ):
                return gene_data
            
            if isinstance(gene_data, dict):
                return gene_data

        return {}
