"""
バックテスト統合管理サービス

"""

import logging
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.services.backtest.backtest_service import BacktestService
from app.utils.response import api_response
from database.repositories.backtest_result_repository import BacktestResultRepository
from database.repositories.ga_experiment_repository import GAExperimentRepository
from database.repositories.generated_strategy_repository import (
    GeneratedStrategyRepository,
)

logger = logging.getLogger(__name__)


class BacktestOrchestrationService:
    """
    バックテスト統合管理サービス

    バックテスト結果の取得、削除、戦略一覧取得等の
    統一的な処理を担当します。
    """

    def __init__(self):
        """初期化"""

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
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="バックテスト結果取得",
            is_api_call=True,
            default_return=api_response(
                success=False,
                error="バックテスト結果取得でエラーが発生しました",
                status_code=500,
            ),
        )
        def _get_backtest_results():
            backtest_repo = BacktestResultRepository(db)

            results = backtest_repo.get_backtest_results(
                limit=limit, offset=offset, symbol=symbol, strategy_name=strategy_name
            )

            total = backtest_repo.count_backtest_results(
                symbol=symbol, strategy_name=strategy_name
            )

            return {
                "success": True,
                "results": results,
                "total": total,
            }

        return _get_backtest_results()

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
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context=f"バックテスト結果取得 (ID: {result_id})",
            is_api_call=True,
            default_return=api_response(
                success=False,
                error="バックテスト結果取得でエラーが発生しました",
                status_code=500,
            ),
        )
        def _get_backtest_result_by_id():
            backtest_repo = BacktestResultRepository(db)
            result = backtest_repo.get_backtest_result_by_id(result_id)

            if result is None:
                return api_response(
                    success=False, error="Backtest result not found", status_code=404
                )

            return api_response(success=True, data=result)

        return _get_backtest_result_by_id()

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
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context=f"バックテスト結果削除 (ID: {result_id})",
            is_api_call=True,
            default_return=api_response(
                success=False,
                error="バックテスト結果削除でエラーが発生しました",
                status_code=500,
            ),
        )
        def _delete_backtest_result():
            # 関連する戦略のリンクを解除
            strategy_repo = GeneratedStrategyRepository(db)
            strategy_repo.unlink_backtest_result(result_id)

            backtest_repo = BacktestResultRepository(db)
            success = backtest_repo.delete_backtest_result(result_id)

            if not success:
                return api_response(
                    success=False, error="Backtest result not found", status_code=404
                )

            return api_response(success=True, message="バックテスト結果を削除しました")

        return _delete_backtest_result()

    async def delete_all_backtest_results(self, db: Session) -> Dict[str, Any]:
        """
        すべてのバックテスト結果を削除

        Args:
            db: データベースセッション

        Returns:
            削除結果
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="全バックテスト結果削除",
            is_api_call=True,
            default_return=api_response(
                success=False,
                error="全バックテスト結果削除でエラーが発生しました",
                status_code=500,
            ),
        )
        def _delete_all_backtest_results():
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

            return api_response(
                success=True,
                message="すべてのバックテスト関連データを削除しました",
                data={
                    "deleted_backtest_results": backtest_count,
                    "deleted_ga_experiments": ga_experiment_count,
                    "deleted_generated_strategies": generated_strategy_count,
                },
            )

        return _delete_all_backtest_results()

    async def get_supported_strategies(self) -> Dict[str, Any]:
        """
        サポートされている戦略一覧を取得

        Returns:
            戦略一覧
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="サポート戦略取得",
            is_api_call=True,
            default_return=api_response(
                success=False,
                error="サポート戦略取得でエラーが発生しました",
                status_code=500,
            ),
        )
        def _get_supported_strategies():
            backtest_service = BacktestService()
            try:
                strategies = backtest_service.get_supported_strategies()
                return api_response(success=True, data={"strategies": strategies})
            finally:
                backtest_service.cleanup()

        return _get_supported_strategies()



