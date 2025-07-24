"""
バックテスト統合管理サービス

APIルーター内に散在していたバックテスト関連のビジネスロジックを統合管理します。
責務の分離とSOLID原則に基づいた設計を実現します。
"""

import logging
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session

from database.repositories.backtest_result_repository import BacktestResultRepository
from database.repositories.ga_experiment_repository import GAExperimentRepository
from database.repositories.generated_strategy_repository import GeneratedStrategyRepository
from app.core.services.backtest_service import BacktestService
from app.core.utils.api_utils import APIResponseHelper

logger = logging.getLogger(__name__)


class BacktestOrchestrationService:
    """
    バックテスト統合管理サービス

    バックテスト結果の取得、削除、戦略一覧取得等の
    統一的な処理を担当します。
    """

    def __init__(self):
        """初期化"""
        pass

    async def get_backtest_results(
        self,
        db: Session,
        limit: int = 50,
        offset: int = 0,
        symbol: Optional[str] = None,
        strategy_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        バックテスト結果一覧を取得

        Args:
            db: データベースセッション
            limit: 取得件数
            offset: オフセット
            symbol: 取引ペアフィルター
            strategy_name: 戦略名フィルター

        Returns:
            バックテスト結果一覧
        """
        try:
            backtest_repo = BacktestResultRepository(db)

            results = backtest_repo.get_backtest_results(
                limit=limit, offset=offset, symbol=symbol, strategy_name=strategy_name
            )

            total = backtest_repo.count_backtest_results(
                symbol=symbol, strategy_name=strategy_name
            )

            return {"success": True, "results": results, "total": total}

        except Exception as e:
            logger.error(f"バックテスト結果取得エラー: {e}")
            raise

    async def get_backtest_result_by_id(
        self, db: Session, result_id: int
    ) -> Dict[str, Any]:
        """
        ID指定でバックテスト結果を取得

        Args:
            db: データベースセッション
            result_id: バックテスト結果ID

        Returns:
            バックテスト結果
        """
        try:
            backtest_repo = BacktestResultRepository(db)
            result = backtest_repo.get_backtest_result_by_id(result_id)

            if result is None:
                return {
                    "success": False,
                    "error": "Backtest result not found",
                    "status_code": 404,
                }

            return {"success": True, "result": result}

        except Exception as e:
            logger.error(f"バックテスト結果取得エラー (ID: {result_id}): {e}")
            raise

    async def delete_backtest_result(
        self, db: Session, result_id: int
    ) -> Dict[str, Any]:
        """
        バックテスト結果を削除

        Args:
            db: データベースセッション
            result_id: バックテスト結果ID

        Returns:
            削除結果
        """
        try:
            backtest_repo = BacktestResultRepository(db)
            success = backtest_repo.delete_backtest_result(result_id)

            if not success:
                return {
                    "success": False,
                    "error": "Backtest result not found",
                    "status_code": 404,
                }

            return APIResponseHelper.api_response(
                success=True, message="バックテスト結果を削除しました"
            )

        except Exception as e:
            logger.error(f"バックテスト結果削除エラー (ID: {result_id}): {e}")
            raise

    async def delete_all_backtest_results(self, db: Session) -> Dict[str, Any]:
        """
        すべてのバックテスト結果を削除

        Args:
            db: データベースセッション

        Returns:
            削除結果
        """
        try:
            backtest_repo = BacktestResultRepository(db)
            ga_experiment_repo = GAExperimentRepository(db)
            generated_strategy_repo = GeneratedStrategyRepository(db)

            # 関連テーブルのデータをすべて削除
            # 1. 生成された戦略を削除
            generated_strategy_count = generated_strategy_repo.delete_all_strategies()

            # 2. GA実験を削除
            ga_experiment_count = ga_experiment_repo.delete_all_experiments()

            # 3. バックテスト結果を削除
            backtest_count = backtest_repo.delete_all_backtest_results()

            return APIResponseHelper.api_response(
                success=True,
                message="すべてのバックテスト関連データを削除しました",
                data={
                    "deleted_backtest_results": backtest_count,
                    "deleted_ga_experiments": ga_experiment_count,
                    "deleted_generated_strategies": generated_strategy_count,
                },
            )

        except Exception as e:
            logger.error(f"全バックテスト結果削除エラー: {e}")
            raise

    async def get_supported_strategies(self) -> Dict[str, Any]:
        """
        サポートされている戦略一覧を取得

        Returns:
            戦略一覧
        """
        try:
            backtest_service = BacktestService()
            strategies = backtest_service.get_supported_strategies()
            return {"success": True, "strategies": strategies}

        except Exception as e:
            logger.error(f"サポート戦略取得エラー: {e}")
            raise

    async def execute_backtest(
        self, request, db: Session
    ) -> Dict[str, Any]:
        """
        バックテストを実行し、結果をデータベースに保存

        Args:
            request: BacktestRequestオブジェクト
            db: データベースセッション

        Returns:
            実行結果
        """
        try:
            backtest_service = BacktestService()
            result = backtest_service.execute_and_save_backtest(request, db)
            return result

        except Exception as e:
            logger.error(f"バックテスト実行エラー: {e}")
            raise
