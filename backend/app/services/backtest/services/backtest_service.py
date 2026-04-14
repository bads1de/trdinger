"""
バックテスト実行サービス

"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
from sqlalchemy.orm import Session

from database.connection import get_db
from database.repositories.backtest_result_repository import BacktestResultRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository

from ..config.builders import build_execution_config
from ..execution.backtest_executor import BacktestExecutionError
from ..execution.backtest_orchestrator import BacktestOrchestrator
from .backtest_data_service import BacktestDataService

logger = logging.getLogger(__name__)


class BacktestService:
    """
    バックテスト実行サービス

    責務:
    1. DBセッションとデータサービスの初期化・管理
    2. バックテストオーケストレーターへの処理委譲
    3. 結果のデータベースへの保存
    """

    def __init__(self, data_service: Optional[BacktestDataService] = None):
        """
        初期化

        Args:
            data_service: データ変換サービス（テスト時にモックを注入可能）
        """
        self.data_service: Optional[BacktestDataService] = data_service
        self._db_session: Optional[Session] = None  # DBセッション保持用
        self._orchestrator: Optional[BacktestOrchestrator] = None  # 遅延初期化

    def run_backtest(
        self, config: Dict[str, Any], preloaded_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        指定された設定と市場データでバックテスト・シミュレーションを実行します。

        このメソッドは、バックテスト実行の統合窓口として以下の手順を管理します：
        1. データサービス（`BacktestDataService`）およびオーケストレーター（`BacktestOrchestrator`）の初期化。
        2. `preloaded_data` が提供されていない場合、設定された期間のデータをDBから取得。
        3. `BacktestOrchestrator` に処理を委譲し、実際のシミュレーションを実行。

        Args:
            config (Dict[str, Any]): バックテストの実行設定。
                以下のキーを含む必要があります：
                - "symbol" (str): 取引ペア。
                - "timeframe" (str): 時間軸。
                - "strategy_name" (str): 実行する戦略名。
                - "strategy_config" (Dict): 戦略固有のパラメータ。
                - "initial_capital" (float): 初期資産。
                - "commission_rate" (float): 手数料率。
            preloaded_data (Optional[pd.DataFrame]): メモリ上にロード済みのOHLCVデータ。提供された場合、DBへのデータリクエストをスキップします。

        Returns:
            Dict[str, Any]: シミュレーション結果を含む辞書。
                主なキー：
                - "success" (bool): 実行の成否。
                - "performance_metrics" (Dict): シャープレシオ、ドローダウン、勝率等の統計。
                - "equity_curve" (List[Dict]): 資産推移データ（チャート表示用）。
                - "trades" (List[Dict]): 全トレード履歴。

        Raises:
            BacktestExecutionError: 設定の不備、データの欠如、または実行中の致命的なエラーが発生した場合。
        """
        try:
            # 1. データサービスの初期化
            self.ensure_data_service_initialized()

            # 2. オーケストレーターの初期化
            self._ensure_orchestrator_initialized()

            # 3. オーケストレーターに委譲
            if self._orchestrator is None:
                raise BacktestExecutionError("オーケストレーターが初期化されていません")

            return self._orchestrator.run(config, preloaded_data)

        except Exception as e:
            logger.error(f"バックテスト実行エラー: {e}")
            raise

    def ensure_data_service_initialized(self) -> None:
        """データサービスの初期化を確保"""
        if self.data_service is None:
            try:
                # 外部からセッションが注入されていない場合のみ新規作成
                if self._db_session is None:
                    self._db_session = next(get_db())

                ohlcv_repo = OHLCVRepository(self._db_session)
                oi_repo = OpenInterestRepository(self._db_session)
                fr_repo = FundingRateRepository(self._db_session)
                self.data_service = BacktestDataService(
                    ohlcv_repo=ohlcv_repo, oi_repo=oi_repo, fr_repo=fr_repo
                )
                logger.debug("バックテストデータサービスを初期化しました")
            except Exception as e:
                logger.error(f"バックテストデータサービスの初期化に失敗しました: {e}")
                raise BacktestExecutionError(
                    f"データサービスの初期化に失敗しました: {e}"
                )

    def _ensure_orchestrator_initialized(self) -> None:
        """オーケストレーターの初期化を確保"""
        if self._orchestrator is None:
            if self.data_service is None:
                raise BacktestExecutionError(
                    "データサービスが初期化されていません。バックテストを実行できません。"
                )

            try:
                self._orchestrator = BacktestOrchestrator(self.data_service)
                logger.debug("バックテストオーケストレーターを初期化しました")
            except Exception as e:
                logger.error(
                    f"バックテストオーケストレーターの初期化に失敗しました: {e}"
                )
                raise BacktestExecutionError(
                    f"オーケストレーターの初期化に失敗しました: {e}"
                )

    def get_supported_strategies(self) -> Dict[str, Any]:
        """
        サポートされている戦略一覧を取得

        Returns:
            戦略一覧
        """
        self.ensure_data_service_initialized()
        self._ensure_orchestrator_initialized()
        if self._orchestrator is None:
            raise BacktestExecutionError("オーケストレーターが初期化されていません")
        return self._orchestrator.get_supported_strategies()

    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        if self._db_session is not None:
            try:
                self._db_session.close()
            except Exception:
                pass
            finally:
                self._db_session = None
            logger.debug("DBセッションをクリーンアップしました")

        self.data_service = None
        self._orchestrator = None

    def _build_execution_config(self, request: Any) -> Dict[str, Any]:
        """dict / Pydantic モデルどちらからでもバックテスト設定を組み立てる。"""
        return build_execution_config(request)

    def execute_and_save_backtest(self, request, db_session: Session) -> Dict[str, Any]:
        """
        バックテストを実行し、その結果をデータベースに永続化します（Web API向け）。

        このメソッドは、APIからのリクエストを受け取り、以下の手順を実行します：
        1. リクエストオブジェクト（Pydanticモデル等）から実行用設定を構築。
        2. `run_backtest` を呼び出してシミュレーションを実行。
        3. 実行成功時、`BacktestResultRepository` を使用して結果を `backtest_results` テーブルに保存。
        4. 保存されたレコードのIDを含む最終的なレスポンスを生成。

        Args:
            request (Any): APIリクエストオブジェクト（`BacktestRequest` 等）、または設定辞書。
            db_session (Session): データベースセッション。結果の保存に使用されます。

        Returns:
            Dict[str, Any]: APIレスポンス形式の辞書。
                保存された `backtest_id` や、計算されたパフォーマンス要約を含みます。

        Note:
            - トランザクション: 提供された `db_session` を使用して、データの読み込みと結果の保存を同一セッション内で行います。
            - エラーハンドリング: 実行失敗時はエラーメッセージを含む辞書を返し、ステータスコード 500 を示唆します。
        """
        try:
            # 外部セッションを内部にも設定してトランザクションを統一
            self._db_session = db_session

            config = self._build_execution_config(request)

            # バックテストを実行
            result = self.run_backtest(config)

            # 結果をデータベースに保存（同じセッションを使用）
            backtest_repo = BacktestResultRepository(db_session)
            saved_result = backtest_repo.save_backtest_result(result)

            return {"success": True, "result": saved_result}

        except Exception as e:
            logger.error(f"バックテスト実行・保存エラー: {e}", exc_info=True)
            return {"success": False, "error": str(e), "status_code": 500}
        finally:
            # 内部状態のみリセット（外部セッションのライフサイクルは呼び出し側が管理）
            self._db_session = None
            self.data_service = None
            self._orchestrator = None
