"""
反復改善ループ用シード戦略プロバイダ

自動検証に合格した過去の生成戦略を DB から取得し、
次の GA 実験の初期集団へ注入するためのシード戦略として提供します。
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, cast

from database.repositories.generated_strategy_repository import (
    GeneratedStrategyRepository,
)

from ..config.ga import GAConfig
from ..genes.strategy import StrategyGene
from ..serializers.serialization import GeneSerializer

logger = logging.getLogger(__name__)


class PreviousStrategySeedProvider:
    """
    過去に合格した戦略をシード戦略として提供するプロバイダ。

    自動検証パイプラインで合格（validation.passed=True）した戦略を
    gene_data から StrategyGene へ復元し、次回実験のシードとして返します。
    """

    def __init__(
        self,
        db_session_factory: Any,
        serializer: GeneSerializer | None = None,
    ) -> None:
        """
        初期化

        Args:
            db_session_factory: DBセッションファクトリ
            serializer: 遺伝子シリアライザ（省略時は内部生成）
        """
        self._db_session_factory = db_session_factory
        self._serializer = serializer or GeneSerializer()

    def get_seed_strategies(self, ga_config: GAConfig) -> list[StrategyGene]:
        """
        反復改善用のシード戦略を取得する。

        Args:
            ga_config: 現在の GA 実行設定（iterative_improvement_config を参照）

        Returns:
            StrategyGene のリスト。有効でない場合は空リスト。
        """
        iterative_config = ga_config.iterative_improvement_config
        if not iterative_config.enabled:
            return []

        try:
            with self._db_session_factory() as db:
                repo = GeneratedStrategyRepository(db)
                records = repo.get_strategies_by_fitness(
                    limit=iterative_config.max_seed_strategies,
                    min_fitness=iterative_config.min_fitness,
                    validation_passed_only=iterative_config.validation_passed_only,
                )
        except Exception as exc:
            logger.warning(
                "過去の合格戦略の取得に失敗したため反復改善シードをスキップします: %s",
                exc,
            )
            return []

        seeds: list[StrategyGene] = []
        for record in records:
            try:
                gene_data: dict[str, Any] = cast(dict[str, Any], record.gene_data or {})
                gene = self._serializer.dict_to_strategy_gene(
                    gene_data,
                    StrategyGene,
                )
                # 同一IDの重複注入を避けるため、シード毎に新しいIDを付与
                if hasattr(gene, "id"):
                    gene.id = str(uuid.uuid4())
                seeds.append(gene)
            except Exception as exc:
                logger.warning(
                    "過去戦略の復元に失敗したためシードから除外します: %s", exc
                )

        logger.info(
            "反復改善: 過去の合格戦略 %d 件をシードとして取得しました",
            len(seeds),
        )
        return seeds
