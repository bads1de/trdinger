"""
オートストラテジー統合管理サービス

APIルーター内に散在していたオートストラテジー関連のビジネスロジックを統合管理します。
責務の分離とSOLID原則に基づいた設計を実現します。
"""

import logging
from typing import Dict, Any

from app.services.auto_strategy.models.gene_strategy import StrategyGene
from app.services.auto_strategy.services.auto_strategy_service import (
    AutoStrategyService,
)

logger = logging.getLogger(__name__)


class AutoStrategyOrchestrationService:
    """
    オートストラテジー統合管理サービス

    戦略テスト、戦略遺伝子復元等の
    統一的な処理を担当します。
    """

    def __init__(self):
        """初期化"""
        pass

    async def test_strategy(
        self, request, auto_strategy_service: AutoStrategyService
    ) -> Dict[str, Any]:
        """
        単一戦略のテスト実行

        指定された戦略遺伝子から戦略を生成し、バックテストを実行します。

        Args:
            request: StrategyTestRequest
            auto_strategy_service: AutoStrategyService

        Returns:
            戦略テスト結果
        """
        try:
            # 戦略遺伝子の復元
            from app.services.auto_strategy.models.gene_serialization import (
                GeneSerializer,
            )

            serializer = GeneSerializer()
            strategy_gene = serializer.dict_to_strategy_gene(
                request.strategy_gene, StrategyGene
            )

            # テスト実行
            result = auto_strategy_service.test_strategy_generation(
                strategy_gene, request.backtest_config
            )

            if result["success"]:
                return {
                    "success": True,
                    "result": result,
                    "message": "戦略テストが完了しました",
                }
            else:
                return {
                    "success": False,
                    "result": None,
                    "errors": result.get("errors"),
                    "message": "戦略テストに失敗しました",
                }

        except Exception as e:
            logger.error(f"戦略テストエラー: {e}")
            return {
                "success": False,
                "result": None,
                "errors": [str(e)],
                "message": f"戦略テスト中にエラーが発生しました: {str(e)}",
            }

    def validate_experiment_stop(
        self, experiment_id: str, auto_strategy_service: AutoStrategyService
    ) -> bool:
        """
        実験停止のバリデーション

        Args:
            experiment_id: 実験ID
            auto_strategy_service: AutoStrategyService

        Returns:
            停止可能かどうか

        Raises:
            ValueError: 停止できない場合
        """
        success = auto_strategy_service.stop_experiment(experiment_id)

        if not success:
            logger.warning(
                f"実験 {experiment_id} を停止できませんでした（存在しないか、既に完了している可能性があります）"
            )
            raise ValueError(
                "実験を停止できませんでした（存在しないか、既に完了している可能性があります）"
            )

        return success

    from typing import Optional

    def format_experiment_result(self, result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        実験結果のフォーマット

        Args:
            result: 実験結果（None の可能性あり）

        Returns:
            フォーマット済み結果（必ず Dict を返す）
        """
        if result is None:
            return {
                "success": False,
                "message": "実験結果が見つかりませんでした",
                "data": {"result": None},
            }

        # 多目的最適化の結果かどうかを判定
        if "pareto_front" in result and "objectives" in result:
            return {
                "success": True,
                "message": "多目的最適化実験結果を取得しました",
                "data": {
                    "result": result,
                    "pareto_front": result.get("pareto_front"),
                    "objectives": result.get("objectives"),
                },
            }
        else:
            return {
                "success": True,
                "message": "実験結果を取得しました",
                "data": {"result": result},
            }
