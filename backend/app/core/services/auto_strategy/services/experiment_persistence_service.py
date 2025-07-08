"""
実験永続化サービス
GA実験に関連するデータのデータベースへの保存、更新、取得を管理します。
"""

import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from ..models.ga_config import GAConfig
from ..models.gene_strategy import StrategyGene
from database.repositories.generated_strategy_repository import (
    GeneratedStrategyRepository,
)
from database.repositories.ga_experiment_repository import GAExperimentRepository
from database.repositories.backtest_result_repository import BacktestResultRepository
from app.core.services.backtest_service import BacktestService

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
        self, experiment_name: str, ga_config: GAConfig, backtest_config: Dict[str, Any]
    ) -> str:
        """
        実験を作成
        """
        try:
            with self.db_session_factory() as db:
                ga_experiment_repo = GAExperimentRepository(db)
                config_data = {
                    "ga_config": ga_config.to_dict(),
                    "backtest_config": backtest_config,
                }
                db_experiment = ga_experiment_repo.create_experiment(
                    name=experiment_name,
                    config=config_data,
                    total_generations=ga_config.generations,
                    status="running",
                )
                logger.info(
                    f"実験を作成しました: {experiment_name} (DB ID: {db_experiment.id})"
                )
                return str(db_experiment.id)
        except Exception as e:
            logger.error(f"実験作成エラー: {e}")
            raise

    def save_experiment_result(
        self,
        experiment_id: str,
        result: Dict[str, Any],
        ga_config: GAConfig,
        backtest_config: Dict[str, Any],
    ):
        """実験結果をデータベースに保存"""
        try:
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

            logger.info(f"実験結果保存完了: {experiment_id}")

        except Exception as e:
            logger.error(
                f"GA実験結果の保存中にエラーが発生しました: {e}", exc_info=True
            )
            pass

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

        best_strategy_record = generated_strategy_repo.save_strategy(
            experiment_id=db_experiment_id,
            gene_data=best_strategy.to_dict(),
            generation=ga_config.generations,
            fitness_score=best_fitness,
        )
        logger.info(f"最良戦略を保存しました: DB ID {best_strategy_record.id}")

        try:
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

        except Exception as e:
            logger.error(
                f"最良戦略の詳細バックテスト結果の保存中にエラー: {e}", exc_info=True
            )

    def _prepare_detailed_backtest_config(
        self,
        best_strategy: StrategyGene,
        experiment_info: Dict[str, Any],
        backtest_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """詳細バックテスト用の設定を準備する"""
        config = backtest_config.copy()
        config["strategy_name"] = (
            f"AUTO_STRATEGY_{experiment_info['name']}_{best_strategy.id[:8]}"
        )
        config["strategy_config"] = {
            "strategy_type": "GENERATED_AUTO",
            "parameters": {"strategy_gene": best_strategy.to_dict()},
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
                        "gene_data": strategy.to_dict(),
                        "generation": ga_config.generations,
                        "fitness_score": result.get(
                            "fitness_scores", [0.0] * len(all_strategies)
                        )[i],
                    }
                )

        if strategies_data:
            saved_count = generated_strategy_repo.save_strategies_batch(strategies_data)
            logger.info(f"追加戦略を一括保存しました: {saved_count} 件")

    def complete_experiment(self, experiment_id: str):
        """実験を完了状態にする"""
        try:
            with self.db_session_factory() as db:
                ga_experiment_repo = GAExperimentRepository(db)
                db_experiment_id = int(experiment_id)
                ga_experiment_repo.update_experiment_status(
                    db_experiment_id, "completed"
                )
        except Exception as e:
            logger.error(f"実験完了処理エラー: {e}")
            pass

    def fail_experiment(self, experiment_id: str):
        """実験を失敗状態にする"""
        try:
            with self.db_session_factory() as db:
                ga_experiment_repo = GAExperimentRepository(db)
                db_experiment_id = int(experiment_id)
                ga_experiment_repo.update_experiment_status(db_experiment_id, "failed")
        except Exception as e:
            logger.error(f"実験失敗処理エラー: {e}")
            pass

    def stop_experiment(self, experiment_id: str):
        """実験を停止状態にする"""
        try:
            with self.db_session_factory() as db:
                ga_experiment_repo = GAExperimentRepository(db)
                db_experiment_id = int(experiment_id)
                ga_experiment_repo.update_experiment_status(db_experiment_id, "stopped")
        except Exception as e:
            logger.error(f"実験停止処理エラー: {e}")
            pass

    def get_experiment_result(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """実験結果を取得"""
        try:
            with self.db_session_factory() as db:
                ga_experiment_repo = GAExperimentRepository(db)
                experiments = ga_experiment_repo.get_experiments_by_status("completed")

                for exp in experiments:
                    if exp.experiment_name == experiment_id:
                        return {
                            "id": exp.id,
                            "experiment_name": exp.experiment_name,
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
                return None
        except Exception as e:
            logger.error(f"実験結果取得エラー: {e}")
            return None

    def list_experiments(self) -> List[Dict[str, Any]]:
        """実験一覧を取得"""
        try:
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
        except Exception as e:
            logger.error(f"実験一覧取得エラー: {e}")
            return []

    def get_experiment_info(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """実験情報を取得"""
        try:
            with self.db_session_factory() as db:
                ga_experiment_repo = GAExperimentRepository(db)
                experiments = ga_experiment_repo.get_recent_experiments(limit=100)
                for exp in experiments:
                    if (
                        hasattr(exp, "name")
                        and exp.name is not None
                        and experiment_id in exp.name
                    ) or str(exp.id) == experiment_id:
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
        except Exception as e:
            logger.error(f"実験情報取得エラー: {e}")
            return None
