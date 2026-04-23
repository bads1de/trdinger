"""
戦略統合サービス

オートストラテジーで生成された戦略を
フロントエンド用の統一フォーマットで提供します。
"""

import logging
from typing import Any, Dict, List, Optional, cast

from sqlalchemy.orm import Session

from app.services.auto_strategy.core.evaluation.report_persistence import (
    extract_evaluation_summary,
)
from app.services.auto_strategy.genes import StrategyGene
from app.utils import response as response_utils
from database.models import BacktestResult, GeneratedStrategy
from database.repositories.generated_strategy_repository import (
    GeneratedStrategyRepository,
)

logger = logging.getLogger(__name__)


class GeneratedStrategyService:
    """
    生成済み戦略サービス

    オートストラテジーで生成された戦略を、フロントエンド向けの
    統一フォーマットに変換・整形して提供します。
    """

    def __init__(self, db: Session):
        """初期化"""
        self.db = db
        self.generated_strategy_repo = GeneratedStrategyRepository(db)

    def get_strategies(
        self,
        limit: int = 50,
        offset: int = 0,
        risk_level: Optional[str] = None,
        experiment_id: Optional[int] = None,
        min_fitness: Optional[float] = None,
        sort_by: str = "fitness_score",
        sort_order: str = "desc",
    ) -> Dict[str, Any]:
        """
        生成済み戦略の一覧を取得

        DBから指定された条件（リスクレベル、実験ID、フィットネス等）で
        戦略を抽出し、フロントエンドでの表示に適した形式に変換して返します。

        Args:
            limit: 取得件数
            offset: 取得開始位置
            risk_level: リスクフィルター（'low', 'medium', 'high'）
            experiment_id: 特定の実験に紐づくもののみ取得
            min_fitness: 最小適応度
            sort_by: ソート対象フィールド
            sort_order: ソート順（'asc', 'desc'）

        Returns:
            {'strategies': List, 'total_count': int, 'has_more': bool}
        """
        try:
            # フィルタリングとソートをリポジトリ層で実行
            total_count, strategies_from_db = (
                self.generated_strategy_repo.get_filtered_and_sorted_strategies(
                    limit=limit,
                    offset=offset,
                    risk_level=risk_level,
                    experiment_id=experiment_id,
                    min_fitness=min_fitness,
                    sort_by=sort_by,
                    sort_order=sort_order,
                )
            )

            # フロントエンド向け形式に変換
            paginated_strategies = [
                s
                for s in (
                    self._convert_generated_strategy_to_display_format(
                        strategy
                    )
                    for strategy in strategies_from_db
                )
                if s is not None
            ]

            return {
                "strategies": paginated_strategies,
                "total_count": total_count,
                "has_more": offset + limit < total_count,
            }

        except Exception as e:
            logger.error(
                f"生成済み戦略の取得中にエラーが発生しました: {e}",
                exc_info=True,
            )
            raise

    def get_strategies_with_response(
        self,
        limit: int = 50,
        offset: int = 0,
        risk_level: Optional[str] = None,
        experiment_id: Optional[int] = None,
        min_fitness: Optional[float] = None,
        sort_by: str = "fitness_score",
        sort_order: str = "desc",
    ) -> Dict[str, Any]:
        """
        生成済み戦略を API レスポンス形式で返す互換メソッド。

        旧呼び出し元が `api_response` 前提で残っているため、`get_strategies`
        の結果をそのまま包んで返します。
        """
        result = self.get_strategies(
            limit=limit,
            offset=offset,
            risk_level=risk_level,
            experiment_id=experiment_id,
            min_fitness=min_fitness,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return response_utils.api_response(success=True, data=result)

    def _convert_generated_strategy_to_display_format(
        self, strategy: GeneratedStrategy
    ) -> Optional[Dict[str, Any]]:
        """
        GeneratedStrategyを表示用の形式に変換します。

        Args:
            strategy: 生成された戦略

        Returns:
            表示形式の戦略データ
        """
        try:
            gene_data = cast(Dict[str, Any], strategy.gene_data)
            backtest_result = strategy.backtest_result

            # 基本情報の抽出
            strategy_name = self._extract_strategy_name(gene_data)
            description = self._generate_strategy_description(gene_data)
            indicators = self._extract_indicators(gene_data)
            parameters = self._extract_parameters(gene_data)

            # パフォーマンス指標の抽出
            performance_metrics = self._extract_performance_metrics(
                backtest_result
            )

            # リスクレベルの計算
            risk_level = self._calculate_risk_level(performance_metrics)
            evaluation_summary = self._extract_evaluation_summary(gene_data)

            return {
                "id": f"auto_{strategy.id}",
                "name": strategy_name,
                "description": description,
                "category": "auto_generated",
                "indicators": indicators,
                "parameters": parameters,
                "expected_return": performance_metrics.get(
                    "total_return", 0.0
                ),
                "sharpe_ratio": performance_metrics.get("sharpe_ratio", 0.0),
                "max_drawdown": performance_metrics.get("max_drawdown", 0.0),
                "win_rate": performance_metrics.get("win_rate", 0.0),
                "risk_level": risk_level,
                "recommended_timeframe": gene_data.get("timeframe", "1h"),
                "source": "auto_strategy",
                "experiment_id": strategy.experiment_id,
                "generation": strategy.generation,
                "fitness_score": strategy.fitness_score,
                "evaluation_summary": evaluation_summary,
                "robustness_pass_rate": (
                    evaluation_summary.get("pass_rate")
                    if isinstance(evaluation_summary, dict)
                    else None
                ),
                "selection_rank": (
                    evaluation_summary.get("selection_rank")
                    if isinstance(evaluation_summary, dict)
                    else None
                ),
                "created_at": (
                    strategy.created_at.isoformat()
                    if strategy.created_at is not None
                    else None
                ),
                "updated_at": (
                    strategy.updated_at.isoformat()
                    if hasattr(strategy, "updated_at")
                    and strategy.updated_at is not None
                    else None
                ),
            }

        except Exception as e:
            logger.error(f"戦略の変換中にエラーが発生しました: {e}")
            return None

    def _extract_strategy_name(self, gene_data: Dict[str, Any]) -> str:
        """戦略名を抽出"""
        indicator_names = self._extract_enabled_indicator_names(gene_data)

        if indicator_names:
            return f"GA生成戦略_{'+'.join(indicator_names[:3])}"
        else:
            return "GA生成戦略"

    def _generate_strategy_description(self, gene_data: Dict[str, Any]) -> str:
        """戦略の説明を生成"""
        enabled_indicators = self._extract_enabled_indicator_names(gene_data)

        if enabled_indicators:
            return f"遺伝的アルゴリズムで生成された{'+'.join(enabled_indicators)}複合戦略"
        else:
            return "遺伝的アルゴリズムで生成された戦略"

    def _extract_indicators(self, gene_data: Dict[str, Any]) -> List[str]:
        """使用指標を抽出"""
        return self._extract_enabled_indicator_names(gene_data)

    def _extract_enabled_indicator_names(
        self, gene_data: Dict[str, Any]
    ) -> List[str]:
        """有効な指標名を抽出"""
        indicators = gene_data.get("indicators", [])
        return [
            indicator.get("type", "").upper()
            for indicator in indicators
            if indicator.get("enabled", False)
        ]

    def _extract_parameters(self, gene_data: Dict[str, Any]) -> Dict[str, Any]:
        """パラメータを抽出"""
        parameters = {
            "indicators": gene_data.get("indicators", []),
            "risk_management": gene_data.get("risk_management", {}),
            "long_entry_conditions": gene_data.get(
                "long_entry_conditions", {}
            ),
            "short_entry_conditions": gene_data.get(
                "short_entry_conditions", {}
            ),
        }
        for field_name in StrategyGene.sub_gene_field_names():
            parameters[field_name] = gene_data.get(field_name)
        return parameters

    def _extract_evaluation_summary(
        self, gene_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """保存済みの評価 summary を取得する。"""
        return extract_evaluation_summary(gene_data)

    def _extract_performance_metrics(
        self, backtest_result: Optional[BacktestResult]
    ) -> Dict[str, float]:
        """パフォーマンス指標を抽出"""
        if (
            backtest_result is None
            or backtest_result.performance_metrics is None
        ):
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
            }

        metrics = backtest_result.performance_metrics
        return {
            "total_return": metrics.get("total_return", 0.0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
            "max_drawdown": abs(metrics.get("max_drawdown", 0.0)),
            "win_rate": metrics.get("win_rate", 0.0),
            "profit_factor": metrics.get("profit_factor", 0.0),
            "total_trades": metrics.get("total_trades", 0),
        }

    def _calculate_risk_level(
        self, performance_metrics: Dict[str, float]
    ) -> str:
        """リスクレベルを計算"""
        max_drawdown = performance_metrics.get("max_drawdown", 0.0)

        if max_drawdown <= 0.05:  # 5%以下
            return "low"
        elif max_drawdown <= 0.15:  # 15%以下
            return "medium"
        else:
            return "high"
