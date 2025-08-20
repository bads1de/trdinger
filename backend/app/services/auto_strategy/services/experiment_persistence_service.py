"""
実験永続化サービス
GA実験に関連するデータのデータベースへの保存、更新、取得を管理します。
"""

import logging
import re
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.services.auto_strategy.models.gene_serialization import GeneSerializer
from app.services.backtest.backtest_service import BacktestService
from database.repositories.backtest_result_repository import BacktestResultRepository
from database.repositories.ga_experiment_repository import GAExperimentRepository
from database.repositories.generated_strategy_repository import (
    GeneratedStrategyRepository,
)

from ..models.ga_config import GAConfig
from ..models.gene_strategy import StrategyGene

logger = logging.getLogger(__name__)


class ExperimentPersistenceService:
    """
    実験永続化サービス
    """

    def __init__(self, db_session_factory, backtest_service: BacktestService):
        """
        初期化

        Args:
            db_session_factory: DBセッションファクトリ
            backtest_service: バックテストサービス
        """
        self.db_session_factory = db_session_factory
        self.backtest_service = backtest_service

    def create_experiment(
        self,
        experiment_id: str,
        experiment_name: str,
        ga_config: GAConfig,
        backtest_config: Dict[str, Any],
    ) -> str:
        """
        実験を作成

        Args:
            experiment_id: フロントエンドで生成された実験ID（UUID）
            experiment_name: 実験名
            ga_config: GA設定
            backtest_config: バックテスト設定

        Returns:
            実験ID（入力されたものと同じ）
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context="実験作成", is_api_call=False)
        def _create_experiment():
            with self.db_session_factory() as db:
                ga_experiment_repo = GAExperimentRepository(db)
                config_data = {
                    "ga_config": ga_config.to_dict(),
                    "backtest_config": backtest_config,
                    "experiment_id": experiment_id,  # フロントエンドで生成されたUUIDを保存
                }
                db_experiment = ga_experiment_repo.create_experiment(
                    name=experiment_name,
                    config=config_data,
                    total_generations=ga_config.generations,
                    status="running",
                )
                logger.info(
                    f"実験を作成しました: {experiment_name} (DB ID: {db_experiment.id}, UUID: {experiment_id})"
                )
                # フロントエンドで生成されたUUIDを返す
                return experiment_id

        return _create_experiment()

    def save_experiment_result(
        self,
        experiment_id: str,
        result: Dict[str, Any],
        ga_config: GAConfig,
        backtest_config: Dict[str, Any],
    ):
        """実験結果をデータベースに保存"""
        from app.utils.error_handler import safe_operation

        @safe_operation(context="実験結果保存", is_api_call=False)
        def _save_experiment_result():
            experiment_info = self.get_experiment_info(experiment_id)
            if not experiment_info:
                logger.error(f"実験情報が見つかりません: {experiment_id}")
                return

            logger.info(f"実験結果保存開始: {experiment_id}")

            with self.db_session_factory() as db:
                self._save_best_strategy_and_run_detailed_backtest(
                    db,
                    experiment_id,
                    experiment_info,
                    result,
                    ga_config,
                    backtest_config,
                )
                self._save_other_strategies(db, experiment_info, result, ga_config)

                # 多目的最適化の場合、パレート最適解も保存
                if ga_config.enable_multi_objective and "pareto_front" in result:
                    self._save_pareto_front(db, experiment_info, result, ga_config)

            logger.info(f"実験結果保存完了: {experiment_id}")

        _save_experiment_result()

    def _save_best_strategy_and_run_detailed_backtest(
        self,
        db: Session,
        experiment_id: str,
        experiment_info: Dict[str, Any],
        result: Dict[str, Any],
        ga_config: GAConfig,
        backtest_config: Dict[str, Any],
    ):
        """最良戦略を保存し、詳細なバックテストを実行して結果を保存する"""
        generated_strategy_repo = GeneratedStrategyRepository(db)
        backtest_result_repo = BacktestResultRepository(db)

        db_experiment_id = experiment_info["db_id"]
        best_strategy = result["best_strategy"]
        best_fitness = result["best_fitness"]

        # 多目的最適化の場合、fitness_valuesも保存
        fitness_values = None
        if ga_config.enable_multi_objective and isinstance(best_fitness, (list, tuple)):
            fitness_values = list(best_fitness)
            fitness_score = (
                best_fitness[0] if best_fitness else 0.0
            )  # 後方互換性のため最初の値を使用
        else:
            fitness_score = (
                best_fitness if isinstance(best_fitness, (int, float)) else 0.0
            )

        serializer = GeneSerializer()
        best_strategy_record = generated_strategy_repo.save_strategy(
            experiment_id=db_experiment_id,
            gene_data=serializer.strategy_gene_to_dict(best_strategy),
            generation=ga_config.generations,
            fitness_score=fitness_score,
            fitness_values=fitness_values,
        )

        logger.info(f"最良戦略を保存しました: DB ID {best_strategy_record.id}")

        from app.utils.error_handler import safe_operation

        @safe_operation(context="最良戦略の詳細バックテスト", is_api_call=False)
        def _save_detailed_backtest():
            logger.info("最良戦略の詳細バックテストと結果保存を開始...")
            detailed_backtest_config = self._prepare_detailed_backtest_config(
                best_strategy, experiment_info, backtest_config
            )
            detailed_result = self.backtest_service.run_backtest(
                detailed_backtest_config
            )

            backtest_result_data = self._prepare_backtest_result_data(
                detailed_result,
                detailed_backtest_config,
                experiment_id,
                db_experiment_id,
                best_fitness,
            )
            saved_backtest_result = backtest_result_repo.save_backtest_result(
                backtest_result_data
            )
            logger.info(
                f"最良戦略のバックテスト結果を保存しました: ID {saved_backtest_result.get('id')}"
            )

        _save_detailed_backtest()

    def _prepare_detailed_backtest_config(
        self,
        best_strategy: StrategyGene,
        experiment_info: Dict[str, Any],
        backtest_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """詳細バックテスト用の設定を準備する"""
        serializer = GeneSerializer()
        config = backtest_config.copy()

        # 元の実験名から不要な部分を削除し、日付を整形
        original_name = experiment_info.get("name", "")
        # 'AUTO_STRATEGY_GA_' プレフィックスを削除
        cleaned_name = re.sub(r"^AUTO_STRATEGY_GA_", "GA_", original_name)
        # 日付部分を 'YYMMDD' 形式に変換
        cleaned_name = re.sub(
            r"(\d{4})-(\d{2})-(\d{2})",
            lambda m: f"{m.group(1)[2:]}{m.group(2)}{m.group(3)}",
            cleaned_name,
        )
        # シンボル名 (例: _BTC_USDT_) を削除
        cleaned_name = re.sub(r"_[A-Z]+_[A-Z]+_", "_", cleaned_name)
        # 末尾のアンダースコアを削除
        cleaned_name = cleaned_name.rstrip("_")

        # 新しい戦略名を生成
        strategy_name = f"AS_{cleaned_name}_{best_strategy.id[:6]}"

        config["strategy_name"] = strategy_name
        config["strategy_config"] = {
            "strategy_type": "GENERATED_AUTO",
            "parameters": {
                "strategy_gene": serializer.strategy_gene_to_dict(best_strategy)
            },
        }
        return config

    def _prepare_backtest_result_data(
        self,
        detailed_result: Dict[str, Any],
        config: Dict[str, Any],
        experiment_id: str,
        db_experiment_id: int,
        best_fitness: float,
    ) -> Dict[str, Any]:
        """backtest_resultsテーブルに保存するためのデータを構築する"""
        return {
            "strategy_name": config["strategy_name"],
            "symbol": config["symbol"],
            "timeframe": config["timeframe"],
            "start_date": config["start_date"],
            "end_date": config["end_date"],
            "initial_capital": config["initial_capital"],
            "commission_rate": config.get("commission_rate", 0.001),
            "config_json": {
                "strategy_config": config["strategy_config"],
                "experiment_id": experiment_id,
                "db_experiment_id": db_experiment_id,
                "fitness_score": best_fitness,
            },
            "performance_metrics": detailed_result.get("performance_metrics", {}),
            "equity_curve": detailed_result.get("equity_curve", []),
            "trade_history": detailed_result.get("trade_history", []),
            "execution_time": detailed_result.get("execution_time"),
            "status": "completed",
        }

    def _save_other_strategies(
        self,
        db: Session,
        experiment_info: Dict[str, Any],
        result: Dict[str, Any],
        ga_config: GAConfig,
    ):
        """最良戦略以外の戦略をバッチ保存する"""
        serializer = GeneSerializer()
        all_strategies = result.get("all_strategies", [])
        if not all_strategies or len(all_strategies) <= 1:
            return

        best_strategy = result["best_strategy"]
        db_experiment_id = experiment_info["db_id"]
        generated_strategy_repo = GeneratedStrategyRepository(db)

        strategies_data = []
        for i, strategy in enumerate(all_strategies[:100]):  # 上位100件に制限
            if strategy.id != best_strategy.id:
                strategies_data.append(
                    {
                        "experiment_id": db_experiment_id,
                        "gene_data": serializer.strategy_gene_to_dict(strategy),
                        "generation": ga_config.generations,
                        "fitness_score": result.get(
                            "fitness_scores", [0.0] * len(all_strategies)
                        )[i],
                    }
                )

        if strategies_data:
            saved_count = generated_strategy_repo.save_strategies_batch(strategies_data)
            logger.info(f"追加戦略を一括保存しました: {saved_count} 件")

    def _save_pareto_front(
        self,
        db: Session,
        experiment_info: Dict[str, Any],
        result: Dict[str, Any],
        ga_config: GAConfig,
    ):
        """パレート最適解を保存する"""
        serializer = GeneSerializer()
        pareto_front = result.get("pareto_front", [])
        if not pareto_front:
            return

        db_experiment_id = experiment_info["db_id"]
        generated_strategy_repo = GeneratedStrategyRepository(db)

        strategies_data = []
        for solution in pareto_front:
            strategy = solution.get("strategy")
            fitness_values = solution.get("fitness_values")

            if strategy and fitness_values:
                strategies_data.append(
                    {
                        "experiment_id": db_experiment_id,
                        "gene_data": serializer.strategy_gene_to_dict(strategy),
                        "generation": ga_config.generations,
                        "fitness_score": (
                            fitness_values[0] if fitness_values else 0.0
                        ),  # 後方互換性
                        "fitness_values": fitness_values,
                    }
                )

        if strategies_data:
            saved_count = generated_strategy_repo.save_strategies_batch(strategies_data)
            logger.info(f"パレート最適解を一括保存しました: {saved_count} 件")

    def complete_experiment(self, experiment_id: str):
        """実験を完了状態にする"""
        from app.utils.error_handler import safe_operation

        @safe_operation(context="実験完了処理", is_api_call=False)
        def _complete_experiment():
            experiment_info = self.get_experiment_info(experiment_id)
            if not experiment_info:
                logger.error(f"実験情報が見つかりません: {experiment_id}")
                return
            with self.db_session_factory() as db:
                ga_experiment_repo = GAExperimentRepository(db)
                db_experiment_id = experiment_info["db_id"]
                ga_experiment_repo.update_experiment_status(
                    db_experiment_id, "completed"
                )

        _complete_experiment()

    def fail_experiment(self, experiment_id: str):
        """実験を失敗状態にする"""
        from app.utils.error_handler import safe_operation

        @safe_operation(context="実験失敗処理", is_api_call=False)
        def _fail_experiment():
            experiment_info = self.get_experiment_info(experiment_id)
            if not experiment_info:
                logger.error(f"実験情報が見つかりません: {experiment_id}")
                return
            with self.db_session_factory() as db:
                ga_experiment_repo = GAExperimentRepository(db)
                db_experiment_id = experiment_info["db_id"]
                ga_experiment_repo.update_experiment_status(db_experiment_id, "failed")

        _fail_experiment()

    def stop_experiment(self, experiment_id: str):
        """実験を停止状態にする"""
        from app.utils.error_handler import safe_operation

        @safe_operation(context="実験停止処理", is_api_call=False)
        def _stop_experiment():
            experiment_info = self.get_experiment_info(experiment_id)
            if not experiment_info:
                logger.error(f"実験情報が見つかりません: {experiment_id}")
                return
            with self.db_session_factory() as db:
                ga_experiment_repo = GAExperimentRepository(db)
                db_experiment_id = experiment_info["db_id"]
                ga_experiment_repo.update_experiment_status(db_experiment_id, "stopped")

        _stop_experiment()


    def list_experiments(self) -> List[Dict[str, Any]]:
        """実験一覧を取得"""
        from app.utils.error_handler import safe_operation

        @safe_operation(context="実験一覧取得", is_api_call=False, default_return=[])
        def _list_experiments():
            with self.db_session_factory() as db:
                ga_experiment_repo = GAExperimentRepository(db)
                experiments = ga_experiment_repo.get_recent_experiments(limit=100)

                return [
                    {
                        "id": exp.id,
                        "experiment_name": exp.name,
                        "status": exp.status,
                        "created_at": (
                            exp.created_at.isoformat()
                            if exp.created_at is not None
                            else None
                        ),
                        "completed_at": (
                            exp.completed_at.isoformat()
                            if exp.completed_at is not None
                            else None
                        ),
                    }
                    for exp in experiments
                ]

        return _list_experiments()

    def get_experiment_info(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """実験情報を取得"""
        from app.utils.error_handler import safe_operation

        @safe_operation(context="実験情報取得", is_api_call=False, default_return=None)
        def _get_experiment_info():
            with self.db_session_factory() as db:
                ga_experiment_repo = GAExperimentRepository(db)
                experiments = ga_experiment_repo.get_recent_experiments(limit=100)
                for exp in experiments:
                    # UUID一致（configに保存したexperiment_id）や名称/ID一致で照合
                    try:
                        cfg = exp.config or {}
                    except Exception:
                        cfg = {}
                    if (
                        cfg.get("experiment_id") == experiment_id
                        or (
                            hasattr(exp, "name")
                            and exp.name is not None
                            and exp.name == experiment_id
                        )
                        or str(exp.id) == experiment_id
                    ):
                        return {
                            "db_id": exp.id,
                            "name": exp.name,
                            "status": exp.status,
                            "config": exp.config,
                            "created_at": exp.created_at,
                            "completed_at": exp.completed_at,
                        }
                logger.warning(f"実験が見つかりません: {experiment_id}")
                return None

        return _get_experiment_info()
