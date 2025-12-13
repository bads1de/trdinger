"""
生成戦略リポジトリ

GAによって生成された戦略の永続化処理を管理します。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import Float, cast, desc
from sqlalchemy.orm import Session

from database.models import GeneratedStrategy

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class GeneratedStrategyRepository(BaseRepository):
    """生成戦略のリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, GeneratedStrategy)

    def to_dict(self, model_instance: GeneratedStrategy) -> dict:
        """生成戦略を辞書に変換

        Args:
            model_instance: 変換するモデルインスタンス

        Returns:
            変換された辞書
        """
        return super().to_dict(model_instance)

    def save_strategy(
        self,
        experiment_id: int,
        gene_data: Dict[str, Any],
        generation: int,
        fitness_score: Optional[float] = None,
        fitness_values: Optional[List[float]] = None,
        parent_ids: Optional[List[int]] = None,
        backtest_result_id: Optional[int] = None,
    ) -> GeneratedStrategy:
        """
        戦略を保存

        Args:
            experiment_id: 実験ID
            gene_data: 戦略遺伝子データ
            generation: 世代数
            fitness_score: フィットネススコア（単一目的用）
            fitness_values: フィットネス値リスト（多目的最適化用）
            parent_ids: 親戦略のIDリスト
            backtest_result_id: バックテスト結果ID

        Returns:
            保存された戦略
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context="戦略保存", is_api_call=False)
        def _save_strategy():
            # 遺伝子データの整合性を確保
            validated_gene_data = self._validate_gene_data(gene_data)

            strategy = GeneratedStrategy(
                experiment_id=experiment_id,
                gene_data=validated_gene_data,
                generation=generation,
                fitness_score=fitness_score,
                fitness_values=fitness_values,
                parent_ids=parent_ids,
                backtest_result_id=backtest_result_id,
            )

            self.db.add(strategy)
            self.db.commit()
            self.db.refresh(strategy)

            return strategy

        return _save_strategy()

    def save_strategies_batch(
        self, strategies_data: List[Dict[str, Any]]
    ) -> List[GeneratedStrategy]:
        """
        戦略を一括保存

        Args:
            strategies_data: 戦略データのリスト

        Returns:
            保存された戦略のリスト
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context="戦略一括保存", is_api_call=False)
        def _save_strategies_batch():
            strategies = []
            for data in strategies_data:
                # 遺伝子データの整合性を確保
                validated_gene_data = self._validate_gene_data(data["gene_data"])

                strategy = GeneratedStrategy(
                    experiment_id=data["experiment_id"],
                    gene_data=validated_gene_data,
                    generation=data["generation"],
                    fitness_score=data.get("fitness_score"),
                    fitness_values=data.get("fitness_values"),
                    parent_ids=data.get("parent_ids"),
                    backtest_result_id=data.get("backtest_result_id"),
                )
                strategies.append(strategy)

            self.db.add_all(strategies)
            self.db.commit()

            for strategy in strategies:
                self.db.refresh(strategy)

            logger.info(f"戦略を一括保存しました: {len(strategies)} 件")
            return strategies

        return _save_strategies_batch()

    def get_strategies_by_experiment(
        self,
        experiment_id: int,
        generation: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[GeneratedStrategy]:
        """
        実験別で戦略を取得

        Args:
            experiment_id: 実験ID
            generation: 世代数（指定時はその世代のみ）
            limit: 取得件数制限

        Returns:
            戦略のリスト
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context="実験別戦略取得", is_api_call=False, default_return=[])
        def _get_strategies_by_experiment():
            filters = {"experiment_id": experiment_id}
            if generation is not None:
                filters["generation"] = generation

            # BaseRepositoryの汎用メソッドを使用
            return self.get_filtered_data(
                filters=filters,
                order_by_column="fitness_score",
                order_asc=False,
                limit=limit,
            )

        return _get_strategies_by_experiment()

    def get_strategies_by_generation(
        self, experiment_id: int, generation: int
    ) -> List[GeneratedStrategy]:
        """
        世代別で戦略を取得

        Args:
            experiment_id: 実験ID
            generation: 世代数

        Returns:
            戦略のリスト
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context="世代別戦略取得", is_api_call=False, default_return=[])
        def _get_strategies_by_generation():
            # BaseRepositoryの汎用メソッドを使用
            return self.get_filtered_data(
                filters={"experiment_id": experiment_id, "generation": generation},
                order_by_column="fitness_score",
                order_asc=False,
            )

        return _get_strategies_by_generation()

    def get_strategies_with_backtest_results(
        self, limit: int = 50, offset: int = 0, experiment_id: Optional[int] = None
    ) -> List[GeneratedStrategy]:
        """
        バックテスト結果を持つ戦略を取得

        Args:
            limit: 取得件数制限
            offset: オフセット
            experiment_id: 実験IDフィルター

        Returns:
            バックテスト結果を持つ戦略のリスト
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="バックテスト結果付き戦略取得", is_api_call=False, default_return=[]
        )
        def _get_strategies_with_backtest_results():
            query = self.db.query(GeneratedStrategy).join(
                GeneratedStrategy.backtest_result
            )
            query = query.filter(
                GeneratedStrategy.fitness_score.isnot(None),
                GeneratedStrategy.fitness_score > 0.0,
            )

            if experiment_id is not None:
                query = query.filter(GeneratedStrategy.experiment_id == experiment_id)

            return query.offset(offset).limit(limit).all()

        return _get_strategies_with_backtest_results()

    def get_filtered_and_sorted_strategies(
        self,
        limit: int = 50,
        offset: int = 0,
        risk_level: Optional[str] = None,
        experiment_id: Optional[int] = None,
        min_fitness: Optional[float] = None,
        sort_by: str = "fitness_score",
        sort_order: str = "desc",
    ) -> Tuple[int, List[GeneratedStrategy]]:
        """
        フィルタリングとソートを適用して戦略を取得
        """
        from app.utils.error_handler import safe_operation
        from database.models import BacktestResult

        @safe_operation(
            context="フィルタリング戦略取得", is_api_call=False, default_return=(0, [])
        )
        def _get_filtered_and_sorted_strategies():
            query = self.db.query(GeneratedStrategy).join(
                GeneratedStrategy.backtest_result
            )

            # フィルタリング
            if experiment_id is not None:
                query = query.filter(GeneratedStrategy.experiment_id == experiment_id)
            if min_fitness is not None:
                query = query.filter(GeneratedStrategy.fitness_score >= min_fitness)

            # リスクレベルのフィルタリング (DBレベルでは難しいので後処理)
            # ただし、max_drawdownで事前にある程度絞り込むことは可能
            if risk_level:
                if risk_level == "low":
                    query = query.filter(
                        cast(BacktestResult.performance_metrics["max_drawdown"], Float)
                        <= 0.05
                    )
                elif risk_level == "medium":
                    query = query.filter(
                        cast(BacktestResult.performance_metrics["max_drawdown"], Float)
                        > 0.05,
                        cast(BacktestResult.performance_metrics["max_drawdown"], Float)
                        <= 0.15,
                    )
                elif risk_level == "high":
                    query = query.filter(
                        cast(BacktestResult.performance_metrics["max_drawdown"], Float)
                        > 0.15
                    )

            # ソート
            sort_column = getattr(GeneratedStrategy, sort_by, None)
            if sort_column is None:
                # BacktestResultのメトリクスでソートする場合
                sort_column = cast(BacktestResult.performance_metrics[sort_by], Float)

            if sort_order.lower() == "desc":
                query = query.order_by(desc(sort_column))
            else:
                query = query.order_by(sort_column)

            total_count = query.count()
            strategies = query.offset(offset).limit(limit).all()

            return total_count, strategies

        return _get_filtered_and_sorted_strategies()

    def delete_all_strategies(self) -> int:
        """
        すべての生成された戦略を削除

        Returns:
            削除された件数
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context="全戦略削除", is_api_call=False)
        def _delete_all_strategies():
            deleted_count = self.db.query(GeneratedStrategy).delete()
            self.db.commit()
            return deleted_count

        return _delete_all_strategies()

    def _validate_gene_data(self, gene_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        遺伝子データの整合性を確保

        Args:
            gene_data: 遺伝子データ

        Returns:
            検証済み遺伝子データ
        """
        validated_data = gene_data.copy()

        # 必須フィールドの確保
        required_fields = {
            "id": "",
            "indicators": [],
            "entry_conditions": [],
            "exit_conditions": [],
            "risk_management": {},
            "metadata": {},
        }

        for field, default_value in required_fields.items():
            if field not in validated_data:
                validated_data[field] = default_value
                logger.debug(f"遺伝子データに欠損フィールド '{field}' を補完しました。")

        # データ型の確認
        if not isinstance(validated_data["indicators"], list):
            validated_data["indicators"] = []
        if not isinstance(validated_data["entry_conditions"], list):
            validated_data["entry_conditions"] = []
        if not isinstance(validated_data["exit_conditions"], list):
            validated_data["exit_conditions"] = []
        if not isinstance(validated_data["risk_management"], dict):
            validated_data["risk_management"] = {}
        if not isinstance(validated_data["metadata"], dict):
            validated_data["metadata"] = {}

        return validated_data

    def unlink_backtest_result(self, backtest_result_id: int) -> int:
        """
        指定されたバックテスト結果に関連する戦略のbacktest_result_idをNoneに更新

        Args:
            backtest_result_id: バックテスト結果ID

        Returns:
            更新された戦略の数
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context="戦略バックテストリンク解除", is_api_call=False)
        def _unlink_backtest_result():
            count = (
                self.db.query(GeneratedStrategy)
                .filter(GeneratedStrategy.backtest_result_id == backtest_result_id)
                .update({GeneratedStrategy.backtest_result_id: None})
            )
            self.db.commit()
            return count

        return _unlink_backtest_result()
