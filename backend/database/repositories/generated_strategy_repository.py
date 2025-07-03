"""
生成戦略リポジトリ

GAによって生成された戦略の永続化処理を管理します。
"""

from typing import List, Optional, Dict, Any, cast as typing_cast, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc, cast, Float
import logging

from .base_repository import BaseRepository
from database.models import GeneratedStrategy
from app.core.utils.database_utils import DatabaseQueryHelper

logger = logging.getLogger(__name__)


class GeneratedStrategyRepository(BaseRepository):
    """生成戦略のリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, GeneratedStrategy)

    def save_strategy(
        self,
        experiment_id: int,
        gene_data: Dict[str, Any],
        generation: int,
        fitness_score: Optional[float] = None,
        parent_ids: Optional[List[int]] = None,
        backtest_result_id: Optional[int] = None,
    ) -> GeneratedStrategy:
        """
        戦略を保存

        Args:
            experiment_id: 実験ID
            gene_data: 戦略遺伝子データ
            generation: 世代数
            fitness_score: フィットネススコア
            parent_ids: 親戦略のIDリスト
            backtest_result_id: バックテスト結果ID

        Returns:
            保存された戦略
        """
        try:
            # 遺伝子データの整合性を確保
            validated_gene_data = self._validate_gene_data(gene_data)

            strategy = GeneratedStrategy(
                experiment_id=experiment_id,
                gene_data=validated_gene_data,
                generation=generation,
                fitness_score=fitness_score,
                parent_ids=parent_ids,
                backtest_result_id=backtest_result_id,
            )

            self.db.add(strategy)
            self.db.commit()
            self.db.refresh(strategy)

            logger.debug(
                f"戦略を保存しました: 実験{experiment_id}, 世代{generation}, ID{strategy.id}"
            )
            return strategy

        except Exception as e:
            self.db.rollback()
            logger.error(f"戦略の保存中にエラーが発生しました: {e}")
            raise

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
        try:
            strategies = []
            for data in strategies_data:
                # 遺伝子データの整合性を確保
                validated_gene_data = self._validate_gene_data(data["gene_data"])

                strategy = GeneratedStrategy(
                    experiment_id=data["experiment_id"],
                    gene_data=validated_gene_data,
                    generation=data["generation"],
                    fitness_score=data.get("fitness_score"),
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

        except Exception as e:
            self.db.rollback()
            logger.error(f"戦略の一括保存中にエラーが発生しました: {e}")
            raise

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
        try:
            filters = {"experiment_id": experiment_id}
            if generation is not None:
                filters["generation"] = generation

            return DatabaseQueryHelper.get_filtered_records(
                db=self.db,
                model_class=GeneratedStrategy,
                filters=filters,
                order_by_column="fitness_score",
                order_asc=False,
                limit=limit,
            )

        except Exception as e:
            logger.error(f"実験IDによる戦略の取得中にエラーが発生しました: {e}")
            return []

    def get_best_strategies(
        self, experiment_id: Optional[int] = None, limit: int = 10
    ) -> List[GeneratedStrategy]:
        """
        最高フィットネスの戦略を取得

        Args:
            experiment_id: 実験ID（指定時はその実験内のみ）
            limit: 取得件数

        Returns:
            戦略のリスト
        """
        try:
            query = self.db.query(GeneratedStrategy).filter(
                GeneratedStrategy.fitness_score.isnot(None)
            )

            if experiment_id is not None:
                query = query.filter(GeneratedStrategy.experiment_id == experiment_id)

            return (
                query.order_by(desc(GeneratedStrategy.fitness_score)).limit(limit).all()
            )

        except Exception as e:
            logger.error(f"最高フィットネス戦略の取得中にエラーが発生しました: {e}")
            return []

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
        try:
            filters = {"experiment_id": experiment_id, "generation": generation}
            return DatabaseQueryHelper.get_filtered_records(
                db=self.db,
                model_class=GeneratedStrategy,
                filters=filters,
                order_by_column="fitness_score",
                order_asc=False,
            )

        except Exception as e:
            logger.error(f"世代による戦略の取得中にエラーが発生しました: {e}")
            return []

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
        try:
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

        except Exception as e:
            logger.error(
                f"バックテスト結果を持つ戦略の取得中にエラーが発生しました: {e}"
            )
            return []

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
        from database.models import BacktestResult

        try:
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

        except Exception as e:
            logger.error(f"フィルタリング済み戦略の取得中にエラーが発生しました: {e}")
            return 0, []

    def update_fitness_score(self, strategy_id: int, fitness_score: float) -> bool:
        """
        フィットネススコアを更新

        Args:
            strategy_id: 戦略ID
            fitness_score: フィットネススコア

        Returns:
            更新成功フラグ
        """
        try:
            strategy = (
                self.db.query(GeneratedStrategy)
                .filter(GeneratedStrategy.id == strategy_id)
                .first()
            )

            if not strategy:
                logger.warning(f"指定されたIDの戦略が見つかりません: {strategy_id}")
                return False

            strategy.fitness_score = fitness_score  # type: ignore
            self.db.commit()

            logger.debug(
                f"フィットネススコアを更新: {strategy_id} -> {fitness_score:.4f}"
            )
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"フィットネススコアの更新中にエラーが発生しました: {e}")
            return False

    def update_backtest_result(self, strategy_id: int, backtest_result_id: int) -> bool:
        """
        バックテスト結果IDを更新

        Args:
            strategy_id: 戦略ID
            backtest_result_id: バックテスト結果ID

        Returns:
            更新成功フラグ
        """
        try:
            strategy = (
                self.db.query(GeneratedStrategy)
                .filter(GeneratedStrategy.id == strategy_id)
                .first()
            )

            if not strategy:
                logger.warning(f"指定されたIDの戦略が見つかりません: {strategy_id}")
                return False

            strategy.backtest_result_id = backtest_result_id  # type: ignore
            self.db.commit()

            logger.debug(
                f"バックテスト結果IDを更新: {strategy_id} -> {backtest_result_id}"
            )
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"バックテスト結果IDの更新中にエラーが発生しました: {e}")
            return False

    def get_strategy_by_id(self, strategy_id: int) -> Optional[GeneratedStrategy]:
        """
        IDで戦略を取得

        Args:
            strategy_id: 戦略ID

        Returns:
            戦略（存在しない場合はNone）
        """
        try:
            return (
                self.db.query(GeneratedStrategy)
                .filter(GeneratedStrategy.id == strategy_id)
                .first()
            )

        except Exception as e:
            logger.error(f"戦略の取得中にエラーが発生しました: {e}")
            return None

    def get_generation_statistics(
        self, experiment_id: int, generation: int
    ) -> Dict[str, Any]:
        """
        世代の統計情報を取得

        Args:
            experiment_id: 実験ID
            generation: 世代数

        Returns:
            統計情報の辞書
        """
        try:
            strategies = self.get_strategies_by_generation(experiment_id, generation)

            # 世代が存在しない場合は空の辞書を返す
            if not strategies:
                return {}

            fitness_scores = typing_cast(
                List[float],
                [s.fitness_score for s in strategies if s.fitness_score is not None],
            )

            if not fitness_scores:
                return {"strategy_count": len(strategies)}

            return {
                "strategy_count": len(strategies),
                "best_fitness": max(fitness_scores),
                "worst_fitness": min(fitness_scores),
                "average_fitness": sum(fitness_scores) / len(fitness_scores),
                "fitness_scores": fitness_scores,
            }

        except Exception as e:
            logger.error(f"世代統計の取得中にエラーが発生しました: {e}")
            return {}

    def delete_strategies_by_experiment(self, experiment_id: int) -> int:
        """
        実験に関連する戦略を削除

        Args:
            experiment_id: 実験ID

        Returns:
            削除された件数
        """
        try:
            deleted_count = (
                self.db.query(GeneratedStrategy)
                .filter(GeneratedStrategy.experiment_id == experiment_id)
                .delete()
            )

            self.db.commit()
            logger.info(f"実験{experiment_id}の戦略を削除しました: {deleted_count} 件")
            return deleted_count

        except Exception as e:
            self.db.rollback()
            logger.error(f"戦略の削除中にエラーが発生しました: {e}")
            return 0

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
