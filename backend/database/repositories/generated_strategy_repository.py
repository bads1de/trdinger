"""
生成戦略リポジトリ

GAによって生成された戦略の永続化処理を管理します。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, cast as t_cast

from sqlalchemy import Float, cast, desc
from sqlalchemy.orm import Session, selectinload, defer

from database.models import GeneratedStrategy

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class GeneratedStrategyRepository(BaseRepository):
    """生成戦略のリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, GeneratedStrategy)

    def _query_with_backtest_result(self):
        """
        backtest_result を eager load 済みのクエリを返す。

        戦略とバックテスト結果のリレーションを事前にロードしたクエリを返します。
        N+1クエリ問題を回避するために、selectinloadを使用して関連データを
        一度のクエリで取得します。

        重いJSONカラム（equity_curve, trade_history）はdeferして
        必要になるまで読み込みを遅延させ、メモリ使用量とクエリ時間を削減します。

        outerjoin を使用して、バックテスト結果を持たない戦略も含めます。
        backtest_result_id が NULL の戦略も取得対象になります。

        Returns:
            Query: backtest_resultをeager load済みのSQLAlchemy Queryオブジェクト
        """
        return (
            self.db.query(GeneratedStrategy)
            .outerjoin(GeneratedStrategy.backtest_result)
            .options(
                selectinload(GeneratedStrategy.backtest_result).options(
                    defer(t_cast(Any, "equity_curve")),
                    defer(t_cast(Any, "trade_history")),
                )
            )
        )

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

            # 高速化のためrefreshはスキップ（IDはcommit時点で設定されている）
            # for strategy in strategies:
            #     self.db.refresh(strategy)

            logger.info(f"戦略を一括保存しました: {len(strategies)} 件")
            return strategies

        return _save_strategies_batch()

    def get_strategies_by_experiment(
        self,
        experiment_id: int,
        generation: Optional[int] = None,
        limit: Optional[int] = None,
        eager_load_backtest: bool = True,
    ) -> List[GeneratedStrategy]:
        """
        実験別で戦略を取得

        Args:
            experiment_id: 実験ID
            generation: 世代数（指定時はその世代のみ）
            limit: 取得件数制限
            eager_load_backtest: バックテスト結果を事前に読み込むか（デフォルト: True）

        Returns:
            戦略のリスト
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context="実験別戦略取得", is_api_call=False, default_return=[])
        def _get_strategies_by_experiment():
            query = self.db.query(GeneratedStrategy).filter(
                GeneratedStrategy.experiment_id == experiment_id
            )
            
            if generation is not None:
                query = query.filter(GeneratedStrategy.generation == generation)

            if eager_load_backtest:
                query = query.options(
                    selectinload(GeneratedStrategy.backtest_result).options(
                        defer(t_cast(Any, "equity_curve")),
                        defer(t_cast(Any, "trade_history")),
                    )
                )

            query = query.order_by(desc(GeneratedStrategy.fitness_score))
            
            if limit is not None:
                query = query.limit(limit)

            return query.all()

        return _get_strategies_by_experiment()

    def get_strategies_by_generation(
        self, experiment_id: int, generation: int, eager_load_backtest: bool = True
    ) -> List[GeneratedStrategy]:
        """
        世代別で戦略を取得

        Args:
            experiment_id: 実験ID
            generation: 世代数
            eager_load_backtest: バックテスト結果を事前に読み込むか（デフォルト: True）

        Returns:
            戦略のリスト
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context="世代別戦略取得", is_api_call=False, default_return=[])
        def _get_strategies_by_generation():
            query = self.db.query(GeneratedStrategy).filter(
                GeneratedStrategy.experiment_id == experiment_id,
                GeneratedStrategy.generation == generation,
            )
            
            if eager_load_backtest:
                query = query.options(
                    selectinload(GeneratedStrategy.backtest_result).options(
                        defer(t_cast(Any, "equity_curve")),
                        defer(t_cast(Any, "trade_history")),
                    )
                )
            
            query = query.order_by(desc(GeneratedStrategy.fitness_score))

            return query.all()

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
            query = self._query_with_backtest_result()
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

        複数のフィルター条件とソート条件を適用して戦略を取得します。
        リスクレベルによるフィルタリングもサポートします。

        Args:
            limit: 取得件数（デフォルト: 50）
            offset: オフセット（デフォルト: 0）
            risk_level: リスクレベル（'low', 'medium', 'high'のいずれか、オプション）
            experiment_id: 実験IDフィルター（オプション）
            min_fitness: 最小フィットネススコア（オプション）
            sort_by: ソート対象カラム名（デフォルト: 'fitness_score'）
            sort_order: ソート順（'desc'または'asc'、デフォルト: 'desc'）

        Returns:
            Tuple[int, List[GeneratedStrategy]]: (総数, 戦略リスト)のタプル

        リスクレベルフィルター:
            - low: max_drawdown <= 0.05
            - medium: 0.05 < max_drawdown <= 0.15
            - high: max_drawdown > 0.15
        """
        from app.utils.error_handler import safe_operation
        from database.models import BacktestResult

        @safe_operation(
            context="フィルタリング戦略取得", is_api_call=False, default_return=(0, [])
        )
        def _get_filtered_and_sorted_strategies():
            from sqlalchemy import func
            
            # カウント用の軽量クエリ（eager loadingなし）
            count_query = (
                self.db.query(func.count(GeneratedStrategy.id))
                .outerjoin(GeneratedStrategy.backtest_result)
            )
            
            # フィルタリング（カウントクエリ）
            if experiment_id is not None:
                count_query = count_query.filter(GeneratedStrategy.experiment_id == experiment_id)
            if min_fitness is not None:
                count_query = count_query.filter(GeneratedStrategy.fitness_score >= min_fitness)
            if risk_level:
                if risk_level == "low":
                    count_query = count_query.filter(
                        cast(BacktestResult.performance_metrics["max_drawdown"], Float)
                        <= 0.05
                    )
                elif risk_level == "medium":
                    count_query = count_query.filter(
                        cast(BacktestResult.performance_metrics["max_drawdown"], Float)
                        > 0.05,
                        cast(BacktestResult.performance_metrics["max_drawdown"], Float)
                        <= 0.15,
                    )
                elif risk_level == "high":
                    count_query = count_query.filter(
                        cast(BacktestResult.performance_metrics["max_drawdown"], Float)
                        > 0.15
                    )
            
            total_count = count_query.scalar()

            # データ取得用のクエリ（eager loadingあり）
            query = self._query_with_backtest_result()

            # フィルタリング（データ取得クエリ）
            if experiment_id is not None:
                query = query.filter(GeneratedStrategy.experiment_id == experiment_id)
            if min_fitness is not None:
                query = query.filter(GeneratedStrategy.fitness_score >= min_fitness)
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
            "long_entry_conditions": [],
            "short_entry_conditions": [],
            "long_exit_conditions": [],
            "short_exit_conditions": [],
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
        if not isinstance(validated_data["long_entry_conditions"], list):
            validated_data["long_entry_conditions"] = []
        if not isinstance(validated_data["short_entry_conditions"], list):
            validated_data["short_entry_conditions"] = []
        if not isinstance(validated_data["long_exit_conditions"], list):
            validated_data["long_exit_conditions"] = []
        if not isinstance(validated_data["short_exit_conditions"], list):
            validated_data["short_exit_conditions"] = []
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


