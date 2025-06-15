"""
戦略統合サービス

オートストラテジーで生成された戦略とショーケース戦略を統合して
フロントエンド用の統一フォーマットで提供します。
"""

import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from database.models import GeneratedStrategy, BacktestResult
from database.repositories.generated_strategy_repository import (
    GeneratedStrategyRepository,
)
from database.repositories.backtest_result_repository import BacktestResultRepository

logger = logging.getLogger(__name__)


class StrategyIntegrationService:
    """
    戦略統合サービス

    オートストラテジー由来の戦略とショーケース戦略を統合して
    統一されたフォーマットで提供します。
    """

    def __init__(self, db: Session):
        """初期化"""
        self.db = db
        self.generated_strategy_repo = GeneratedStrategyRepository(db)
        self.backtest_result_repo = BacktestResultRepository(db)

    def get_unified_strategies(
        self,
        limit: int = 50,
        offset: int = 0,
        category: Optional[str] = None,
        risk_level: Optional[str] = None,
        experiment_id: Optional[int] = None,
        min_fitness: Optional[float] = None,
        sort_by: str = "fitness_score",
        sort_order: str = "desc",
    ) -> Dict[str, Any]:
        """
        統合された戦略一覧を取得

        Args:
            limit: 取得件数制限
            offset: オフセット
            category: カテゴリフィルター
            risk_level: リスクレベルフィルター
            experiment_id: 実験IDフィルター
            min_fitness: 最小フィットネススコア
            sort_by: ソート項目
            sort_order: ソート順序

        Returns:
            統合された戦略データ
        """
        try:
            # ショーケース戦略は現在サポートされていないため、空のリストを返す
            showcase_strategies = []

            # オートストラテジー由来の戦略を取得
            auto_strategies = self._get_auto_generated_strategies(
                limit, offset, sort_by, sort_order
            )

            # 統合
            all_strategies = showcase_strategies + auto_strategies

            # フィルタリング適用
            all_strategies = self._apply_filters(
                all_strategies, category, risk_level, experiment_id, min_fitness
            )

            # ソート
            all_strategies = self._sort_strategies(all_strategies, sort_by, sort_order)

            # ページネーション適用
            total_count = len(all_strategies)
            paginated_strategies = all_strategies[offset : offset + limit]

            return {
                "strategies": paginated_strategies,
                "total_count": total_count,
                "has_more": offset + limit < total_count,
            }

        except Exception as e:
            logger.error(f"統合戦略取得エラー: {e}")
            raise

    def _get_auto_generated_strategies(
        self, limit: int, offset: int, sort_by: str, sort_order: str
    ) -> List[Dict[str, Any]]:
        """オートストラテジー由来の戦略を取得"""
        try:
            # 有効なフィットネススコアとバックテスト結果を持つ戦略のみを取得
            strategies = (
                self.generated_strategy_repo.get_strategies_with_backtest_results(
                    limit=limit
                    * 3,  # フィルタリング後のページネーションのため多めに取得
                    offset=0,
                )
            )

            converted_strategies = []
            for strategy in strategies:
                # フィットネススコアが有効で、バックテスト結果がある戦略のみ処理
                if (
                    strategy.fitness_score is not None
                    and strategy.fitness_score > 0.0
                    and strategy.backtest_result is not None
                ):

                    converted = self._convert_generated_strategy_to_showcase_format(
                        strategy
                    )
                    if converted:
                        converted_strategies.append(converted)

            logger.info(
                f"有効な戦略数: {len(converted_strategies)} / {len(strategies)}"
            )
            return converted_strategies

        except Exception as e:
            logger.error(f"オートストラテジー取得エラー: {e}")
            return []

    def _convert_generated_strategy_to_showcase_format(
        self, strategy: GeneratedStrategy
    ) -> Optional[Dict[str, Any]]:
        """
        GeneratedStrategyをショーケース形式に変換

        Args:
            strategy: 生成された戦略

        Returns:
            ショーケース形式の戦略データ
        """
        try:
            gene_data = strategy.gene_data
            backtest_result = strategy.backtest_result

            # 基本情報の抽出
            strategy_name = self._extract_strategy_name(gene_data)
            description = self._generate_strategy_description(gene_data)
            indicators = self._extract_indicators(gene_data)
            parameters = self._extract_parameters(gene_data)

            # パフォーマンス指標の抽出
            performance_metrics = self._extract_performance_metrics(backtest_result)

            # リスクレベルの計算
            risk_level = self._calculate_risk_level(performance_metrics)

            return {
                "id": f"auto_{strategy.id}",
                "name": strategy_name,
                "description": description,
                "category": "auto_generated",
                "indicators": indicators,
                "parameters": parameters,
                "expected_return": performance_metrics.get("total_return", 0.0),
                "sharpe_ratio": performance_metrics.get("sharpe_ratio", 0.0),
                "max_drawdown": performance_metrics.get("max_drawdown", 0.0),
                "win_rate": performance_metrics.get("win_rate", 0.0),
                "risk_level": risk_level,
                "recommended_timeframe": gene_data.get("timeframe", "1h"),
                "source": "auto_strategy",
                "experiment_id": strategy.experiment_id,
                "generation": strategy.generation,
                "fitness_score": strategy.fitness_score,
                "created_at": (
                    strategy.created_at.isoformat() if strategy.created_at else None
                ),
                "updated_at": (
                    strategy.created_at.isoformat() if strategy.created_at else None
                ),
            }

        except Exception as e:
            logger.error(f"戦略変換エラー: {e}")
            return None

    def _extract_strategy_name(self, gene_data: Dict[str, Any]) -> str:
        """戦略名を抽出"""
        indicators = gene_data.get("indicators", [])
        indicator_names = []

        # リスト形式の場合
        if isinstance(indicators, list):
            for indicator in indicators:
                if indicator.get("enabled", False):
                    indicator_names.append(indicator.get("type", "").upper())

        # 辞書形式の場合（後方互換性）
        elif isinstance(indicators, dict):
            for indicator_type, indicator_config in indicators.items():
                if indicator_config.get("enabled", False):
                    indicator_names.append(indicator_type.upper())

        if indicator_names:
            return f"GA生成戦略_{'+'.join(indicator_names[:3])}"
        else:
            return "GA生成戦略"

    def _generate_strategy_description(self, gene_data: Dict[str, Any]) -> str:
        """戦略の説明を生成"""
        indicators = gene_data.get("indicators", [])
        enabled_indicators = []

        # リスト形式の場合
        if isinstance(indicators, list):
            enabled_indicators = [
                indicator.get("type", "").upper()
                for indicator in indicators
                if indicator.get("enabled", False)
            ]

        # 辞書形式の場合（後方互換性）
        elif isinstance(indicators, dict):
            enabled_indicators = [
                name.upper()
                for name, config in indicators.items()
                if config.get("enabled", False)
            ]

        if enabled_indicators:
            return (
                f"遺伝的アルゴリズムで生成された{'+'.join(enabled_indicators)}複合戦略"
            )
        else:
            return "遺伝的アルゴリズムで生成された戦略"

    def _extract_indicators(self, gene_data: Dict[str, Any]) -> List[str]:
        """使用指標を抽出"""
        indicators = gene_data.get("indicators", [])

        # リスト形式の場合
        if isinstance(indicators, list):
            return [
                indicator.get("type", "").upper()
                for indicator in indicators
                if indicator.get("enabled", False)
            ]

        # 辞書形式の場合（後方互換性）
        elif isinstance(indicators, dict):
            return [
                name.upper()
                for name, config in indicators.items()
                if config.get("enabled", False)
            ]

        return []

    def _extract_parameters(self, gene_data: Dict[str, Any]) -> Dict[str, Any]:
        """パラメータを抽出"""
        return {
            "indicators": gene_data.get("indicators", {}),
            "risk_management": gene_data.get("risk_management", {}),
            "entry_conditions": gene_data.get("entry_conditions", {}),
            "exit_conditions": gene_data.get("exit_conditions", {}),
        }

    def _extract_performance_metrics(
        self, backtest_result: Optional[BacktestResult]
    ) -> Dict[str, float]:
        """パフォーマンス指標を抽出"""
        if not backtest_result or not backtest_result.performance_metrics:
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
            "max_drawdown": abs(metrics.get("max_drawdown", 0.0)),  # 正の値に変換
            "win_rate": metrics.get("win_rate", 0.0),
            "profit_factor": metrics.get("profit_factor", 0.0),
            "total_trades": metrics.get("total_trades", 0),
        }

    def _calculate_risk_level(self, performance_metrics: Dict[str, float]) -> str:
        """リスクレベルを計算"""
        max_drawdown = performance_metrics.get("max_drawdown", 0.0)

        if max_drawdown <= 0.05:  # 5%以下
            return "low"
        elif max_drawdown <= 0.15:  # 15%以下
            return "medium"
        else:
            return "high"

    def _sort_strategies(
        self, strategies: List[Dict[str, Any]], sort_by: str, sort_order: str
    ) -> List[Dict[str, Any]]:
        """戦略をソート"""
        try:
            reverse = sort_order.lower() == "desc"

            if sort_by == "expected_return":
                strategies.sort(
                    key=lambda x: x.get("expected_return", 0), reverse=reverse
                )
            elif sort_by == "sharpe_ratio":
                strategies.sort(key=lambda x: x.get("sharpe_ratio", 0), reverse=reverse)
            elif sort_by == "max_drawdown":
                strategies.sort(
                    key=lambda x: x.get("max_drawdown", 0), reverse=not reverse
                )  # 小さい方が良い
            elif sort_by == "win_rate":
                strategies.sort(key=lambda x: x.get("win_rate", 0), reverse=reverse)
            elif sort_by == "fitness_score":
                strategies.sort(
                    key=lambda x: x.get("fitness_score", 0), reverse=reverse
                )
            elif sort_by == "created_at":
                strategies.sort(key=lambda x: x.get("created_at", ""), reverse=reverse)
            else:
                # デフォルトはフィットネススコア順
                strategies.sort(key=lambda x: x.get("fitness_score", 0), reverse=True)

            return strategies

        except Exception as e:
            logger.error(f"ソートエラー: {e}")
            return strategies

    def _apply_filters(
        self,
        strategies: List[Dict[str, Any]],
        category: Optional[str] = None,
        risk_level: Optional[str] = None,
        experiment_id: Optional[int] = None,
        min_fitness: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """戦略にフィルターを適用"""
        try:
            filtered_strategies = strategies

            # カテゴリフィルター
            if category:
                filtered_strategies = [
                    s for s in filtered_strategies if s.get("category") == category
                ]

            # リスクレベルフィルター
            if risk_level:
                filtered_strategies = [
                    s for s in filtered_strategies if s.get("risk_level") == risk_level
                ]

            # 実験IDフィルター
            if experiment_id is not None:
                filtered_strategies = [
                    s
                    for s in filtered_strategies
                    if s.get("experiment_id") == experiment_id
                ]

            # 最小フィットネススコアフィルター
            if min_fitness is not None:
                filtered_strategies = [
                    s
                    for s in filtered_strategies
                    if s.get("fitness_score", 0) >= min_fitness
                ]

            logger.info(
                f"フィルター適用: {len(strategies)} -> {len(filtered_strategies)} 戦略"
            )
            return filtered_strategies

        except Exception as e:
            logger.error(f"フィルター適用エラー: {e}")
            return strategies
