"""
バックテスト統合管理サービス

バックテスト結果のCRUD操作、戦略管理、関連データの統合処理を担当する
オーケストレーションサービスです。API層からのリクエストを受け、
データベースリポジトリと連携して一貫性のあるデータ操作を提供します。
"""

import logging
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.utils.error_handler import safe_operation
from app.utils.response import api_response
from database.repositories.backtest_result_repository import BacktestResultRepository
from database.repositories.ga_experiment_repository import GAExperimentRepository
from database.repositories.generated_strategy_repository import (
    GeneratedStrategyRepository,
)

logger = logging.getLogger(__name__)


class BacktestOrchestrationService:
    """
    バックテストのデータ操作と戦略管理を統括するオーケストレーションサービスです。

    主な責務:
    - 結果の取得とフィルタリング: データベースから特定の銘柄や戦略に基づいたバックテスト結果を抽出します。
    - 安全な削除: バックテスト結果を削除する際、関連する `GeneratedStrategy` とのリンク解除を適切に行い、参照整合性を保ちます。
    - 実験データの統合: GAの実行結果（実験）と、個別のバックテスト実行結果を紐付けて管理します。
    """

    def __init__(self):
        """
        BacktestOrchestrationServiceを初期化

        バックテスト結果の取得、削除、戦略管理等の統一的な処理を
        提供するサービスを初期化します。

        Note:
            このサービスはステートレスであり、各操作で必要な `Session` はメソッド引数として受け取ります。
        """

    async def get_backtest_results(
        self,
        db: Session,
        limit: int = 50,
        offset: int = 0,
        symbol: Optional[str] = None,
        strategy_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        条件に合致するバックテスト結果の一覧を取得します。

        Args:
            db (Session): データベースセッション。
            limit (int): 1ページあたりの取得最大件数。デフォルトは50。
            offset (int): 読み飛ばす件数（ページネーション用）。デフォルトは0。
            symbol (Optional[str]): 特定の通貨ペアで絞り込む場合の検索文字列。
            strategy_name (Optional[str]): 特定の戦略名で絞り込む場合の検索文字列。

        Returns:
            Dict[str, Any]: 結果一覧とメタデータを含む辞書。
                - "success" (bool): 取得の成否。
                - "results" (List[dict]): 取得された結果レコードのリスト。
                - "total" (int): フィルター条件に合致する全レコード数。
        """

        @safe_operation(context="バックテスト結果取得", is_api_call=True)
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

        @safe_operation(
            context=f"バックテスト結果取得 (ID: {result_id})", is_api_call=True
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

        @safe_operation(
            context=f"バックテスト結果削除 (ID: {result_id})", is_api_call=True
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

        @safe_operation(context="全バックテスト結果削除", is_api_call=True)
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

        @safe_operation(context="サポート戦略取得", is_api_call=True)
        def _get_supported_strategies():
            from ..config.constants import SUPPORTED_STRATEGIES

            return api_response(success=True, data={"strategies": SUPPORTED_STRATEGIES})

        return _get_supported_strategies()
