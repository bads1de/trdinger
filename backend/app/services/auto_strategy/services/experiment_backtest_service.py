"""
実験バックテストサービス

GA実験で最良個体に対する詳細バックテストの構築と実行を担当します。
バックテストの実行ロジックを永続化処理から切り離すためのサービスです。
"""

import logging
import re
from copy import deepcopy
from typing import Any, Dict, Optional

from app.services.auto_strategy.core.evaluation.report_persistence import (
    attach_backtest_evaluation_summary,
)
from app.services.auto_strategy.serializers.serialization import GeneSerializer
from app.services.backtest.config.builders import build_execution_config
from app.services.backtest.services.backtest_service import BacktestService

from ..config import GAConfig
from ..genes import StrategyGene

logger = logging.getLogger(__name__)


class ExperimentBacktestService:
    """
    実験バックテストサービス

    最良戦略に対する詳細バックテストの設定生成、実行、レコード整形を担当します。
    """

    def __init__(
        self,
        backtest_service: BacktestService,
        serializer: Optional[GeneSerializer] = None,
    ):
        """初期化"""
        self.backtest_service = backtest_service
        self.serializer = serializer or GeneSerializer()

    def create_detailed_backtest_result_data(
        self,
        result: Dict[str, Any],
        ga_config: GAConfig,
        backtest_config: Dict[str, Any],
        experiment_id: str,
        experiment_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        最良個体に対する詳細バックテストを実行し、保存用レコードを作成する

        Args:
            result: GA実行結果
            ga_config: GA設定
            backtest_config: ベースバックテスト設定
            experiment_id: フロントエンド向け実験ID
            experiment_info: 実験情報（DB ID, 実験名など）

        Returns:
            backtest_results テーブルへ保存するためのレコード辞書
        """
        if not isinstance(result, dict):
            raise TypeError("GA実行結果は辞書である必要があります")

        best_strategy = result["best_strategy"]
        best_fitness = result["best_fitness"]
        experiment_data = experiment_info if isinstance(experiment_info, dict) else {}

        experiment_name = str(experiment_data.get("name", experiment_id))
        db_experiment_id = int(experiment_data.get("db_id", 0) or 0)
        fitness_score = self._normalize_fitness_score(best_fitness, ga_config)
        evaluation_summary = result.get("best_evaluation_summary")

        detailed_config = self._prepare_detailed_backtest_config(
            best_strategy=best_strategy,
            experiment_name=experiment_name,
            backtest_config=backtest_config,
        )

        logger.info("詳細バックテストを開始します: %s", experiment_id)
        detailed_result = self.backtest_service.run_backtest(detailed_config)

        return self._prepare_backtest_result_data(
            detailed_result=detailed_result,
            config=detailed_config,
            experiment_id=experiment_id,
            db_experiment_id=db_experiment_id,
            best_fitness=fitness_score,
            evaluation_summary=evaluation_summary,
        )

    def _normalize_fitness_score(self, best_fitness: Any, ga_config: GAConfig) -> float:
        """GA結果から単一のフィットネススコアを抽出する"""
        if ga_config.enable_multi_objective and isinstance(best_fitness, (list, tuple)):
            return float(best_fitness[0]) if best_fitness else 0.0

        if isinstance(best_fitness, (int, float)):
            return float(best_fitness)

        return 0.0

    def _prepare_detailed_backtest_config(
        self,
        best_strategy: StrategyGene,
        experiment_name: str,
        backtest_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        詳細バックテスト用の最終的な設定を構築

        実験名を読みやすい形に整形し、戦略の遺伝子データを
        バックテスト設定へ組み込みます。
        """
        config = backtest_config.copy()

        cleaned_name = re.sub(r"^AUTO_STRATEGY_GA_", "GA_", experiment_name)
        cleaned_name = re.sub(
            r"(\d{4})-(\d{2})-(\d{2})",
            lambda m: f"{m.group(1)[2:]}{m.group(2)}{m.group(3)}",
            cleaned_name,
        )
        cleaned_name = re.sub(r"_[A-Z]+_[A-Z]+_", "_", cleaned_name)
        cleaned_name = cleaned_name.rstrip("_")

        strategy_id = str(getattr(best_strategy, "id", ""))[:6] or "unknown"
        return build_execution_config(
            config,
            strategy_name=f"AS_{cleaned_name}_{strategy_id}",
            strategy_config={
                "strategy_type": "GENERATED_GA",
                "parameters": {
                    "strategy_gene": self.serializer.strategy_gene_to_dict(
                        best_strategy
                    )
                },
            },
        )

    def _prepare_backtest_result_data(
        self,
        detailed_result: Dict[str, Any],
        config: Dict[str, Any],
        experiment_id: str,
        db_experiment_id: int,
        best_fitness: float,
        evaluation_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        backtest_results テーブルへ保存するためのレコードデータを構築
        """
        config_json = {
            "strategy_config": deepcopy(config["strategy_config"]),
            "experiment_id": experiment_id,
            "db_experiment_id": db_experiment_id,
            "fitness_score": best_fitness,
        }
        config_json = attach_backtest_evaluation_summary(
            config_json,
            evaluation_summary,
        )
        return {
            "strategy_name": config["strategy_name"],
            "symbol": config["symbol"],
            "timeframe": config["timeframe"],
            "start_date": config["start_date"],
            "end_date": config["end_date"],
            "initial_capital": config["initial_capital"],
            "commission_rate": config.get("commission_rate", 0.001),
            "config_json": config_json,
            "performance_metrics": detailed_result.get("performance_metrics", {}),
            "equity_curve": detailed_result.get("equity_curve", []),
            "trade_history": detailed_result.get("trade_history", []),
            "execution_time": detailed_result.get("execution_time"),
            "status": "completed",
        }
