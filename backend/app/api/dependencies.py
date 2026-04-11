"""
依存性注入用のファクトリ関数

FastAPIの依存性注入システムで使用するサービスファクトリ関数を提供します。
Orchestration Serviceパターンに基づいて、API層とサービス層の結合を解消します。
"""

import logging
from typing import Callable, TypeVar

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.services.auto_strategy import AutoStrategyService
from app.services.auto_strategy.services.generated_strategy_service import (
    GeneratedStrategyService,
)
from app.services.data_collection.bybit.long_short_ratio_service import (
    BybitLongShortRatioService,
)
from app.services.data_collection.orchestration.market_data_orchestration_service import (
    MarketDataOrchestrationService,
)
from database.connection import get_db
from database.repositories.long_short_ratio_repository import LongShortRatioRepository

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _create_service(factory: Callable[[], T], service_name: str) -> T:
    """
    サービス生成の定型エラーハンドリングをまとめるヘルパー関数

    指定されたファクトリ関数を使用してサービスインスタンスを生成し、
    エラーが発生した場合は適切なHTTP例外をスローします。

    Args:
        factory: サービスインスタンスを生成する関数
        service_name: サービス名（エラーメッセージ用）

    Returns:
        T: 生成されたサービスインスタンス

    Raises:
        HTTPException: サービス初期化に失敗した場合（ステータスコード503）
    """
    try:
        return factory()
    except Exception as exc:
        logger.error(f"{service_name}初期化エラー: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{service_name}が利用できません。サーバーログを確認してください。",
        ) from exc


def get_market_data_orchestration_service(
    db: Session = Depends(get_db),
) -> MarketDataOrchestrationService:
    """
    MarketDataOrchestrationServiceのインスタンスを取得

    Args:
        db: データベースセッション

    Returns:
        MarketDataOrchestrationServiceインスタンス
    """
    return MarketDataOrchestrationService(db)


def get_auto_strategy_service() -> AutoStrategyService:
    """
    AutoStrategyServiceのインスタンスを取得（依存性注入用）

    Returns:
        AutoStrategyServiceインスタンス

    Raises:
        HTTPException: サービス初期化に失敗した場合
    """
    return _create_service(AutoStrategyService, "AutoStrategyService")


def get_generated_strategy_service_with_db(
    db: Session = Depends(get_db),
) -> GeneratedStrategyService:
    """
    GeneratedStrategyServiceのインスタンスを取得（依存性注入用）

    Args:
        db: データベースセッション（依存性注入）

    Returns:
        GeneratedStrategyServiceインスタンス
    """
    return _create_service(lambda: GeneratedStrategyService(db), "GeneratedStrategyService")


def get_bybit_open_interest_service() -> "BybitOpenInterestService":
    """BybitOpenInterestServiceのインスタンスを取得（依存性注入用）。

    Bybit取引所からのオープンインタレストデータ取得サービスを提供します。

    Returns:
        BybitOpenInterestService: オープンインタレストサービスインスタンス

    Raises:
        HTTPException: サービス初期化に失敗した場合（ステータスコード503）
    """

    def factory() -> "BybitOpenInterestService":
        """サービスのインスタンスを生成する内部ファクトリ関数。

        Returns:
            BybitOpenInterestService: 生成されたサービスインスタンス
        """
        from app.services.data_collection.bybit.open_interest_service import (
            BybitOpenInterestService,
        )

        return BybitOpenInterestService()

    return _create_service(factory, "BybitOpenInterestService")


def get_bybit_funding_rate_service() -> "BybitFundingRateService":
    """BybitFundingRateServiceのインスタンスを取得（依存性注入用）。

    Bybit取引所からのファンディングレートデータ取得サービスを提供します。

    Returns:
        BybitFundingRateService: ファンディングレートサービスインスタンス

    Raises:
        HTTPException: サービス初期化に失敗した場合（ステータスコード503）
    """

    def factory() -> "BybitFundingRateService":
        """サービスのインスタンスを生成する内部ファクトリ関数。

        Returns:
            BybitFundingRateService: 生成されたサービスインスタンス
        """
        from app.services.data_collection.bybit.funding_rate_service import (
            BybitFundingRateService,
        )

        return BybitFundingRateService()

    return _create_service(factory, "BybitFundingRateService")


# get_automl_feature_generation_service は削除されました（autofeat機能の削除に伴う）


# Dependency factories for various orchestration services (avoid direct new() in API modules)
def get_data_collection_orchestration_service() -> "DataCollectionOrchestrationService":
    """DataCollectionOrchestrationServiceのインスタンスを取得（依存性注入用）。

    OHLCVデータ、ファンディングレート、オープンインタレスト等の
    市場データ収集プロセスを統合管理するサービスを提供します。

    Returns:
        DataCollectionOrchestrationService: データ収集オーケストレーションサービスインスタンス

    Raises:
        HTTPException: サービス初期化に失敗した場合（ステータスコード503）
    """

    def factory() -> "DataCollectionOrchestrationService":
        """サービスのインスタンスを生成する内部ファクトリ関数。

        Returns:
            DataCollectionOrchestrationService: 生成されたサービスインスタンス
        """
        from app.services.data_collection.orchestration.data_collection_orchestration_service import (
            DataCollectionOrchestrationService,
        )

        return DataCollectionOrchestrationService()

    return _create_service(factory, "DataCollectionOrchestrationService")


def get_data_management_orchestration_service() -> "DataManagementOrchestrationService":
    """DataManagementOrchestrationServiceのインスタンスを取得（依存性注入用）。

    データベース内の市場データの管理・削除・リセット等の
    データ管理操作を統合管理するサービスを提供します。

    Returns:
        DataManagementOrchestrationService: データ管理オーケストレーションサービスインスタンス

    Raises:
        HTTPException: サービス初期化に失敗した場合（ステータスコード503）
    """

    def factory() -> "DataManagementOrchestrationService":
        """サービスのインスタンスを生成する内部ファクトリ関数。

        Returns:
            DataManagementOrchestrationService: 生成されたサービスインスタンス
        """
        from app.services.data_collection.orchestration.data_management_orchestration_service import (
            DataManagementOrchestrationService,
        )

        return DataManagementOrchestrationService()

    return _create_service(factory, "DataManagementOrchestrationService")


def get_open_interest_orchestration_service(
    bybit_service=Depends(get_bybit_open_interest_service),
) -> "OpenInterestOrchestrationService":
    """OpenInterestOrchestrationServiceのインスタンスを取得（依存性注入用）。

    オープンインタレストデータの収集・管理を統合管理するサービスを提供します。

    Args:
        bybit_service: BybitOpenInterestServiceインスタンス（依存性注入）

    Returns:
        OpenInterestOrchestrationService: オープンインタレストオーケストレーションサービスインスタンス

    Raises:
        HTTPException: サービス初期化に失敗した場合（ステータスコード503）
    """

    def factory() -> "OpenInterestOrchestrationService":
        """サービスのインスタンスを生成する内部ファクトリ関数。

        Returns:
            OpenInterestOrchestrationService: 生成されたサービスインスタンス
        """
        from app.services.data_collection.orchestration.open_interest_orchestration_service import (
            OpenInterestOrchestrationService,
        )

        return OpenInterestOrchestrationService(bybit_service)

    return _create_service(factory, "OpenInterestOrchestrationService")


def get_ml_training_service() -> "MLTrainingService":
    """MLTrainingServiceのインスタンスを取得（依存性注入用）。

    機械学習モデルのトレーニングフローを制御し、
    データの準備、特徴量エンジニアリング、学習、評価、保存を管理するサービスを提供します。

    Returns:
        MLTrainingService: MLトレーニングサービスインスタンス

    Raises:
        HTTPException: サービス初期化に失敗した場合（ステータスコード503）
    """

    def factory() -> "MLTrainingService":
        """サービスのインスタンスを生成する内部ファクトリ関数。

        Returns:
            MLTrainingService: 生成されたサービスインスタンス
        """
        from app.services.ml.orchestration.ml_training_orchestration_service import (
            MLTrainingService,
        )

        return MLTrainingService()

    return _create_service(factory, "MLTrainingService")


def get_backtest_orchestration_service() -> "BacktestOrchestrationService":
    """BacktestOrchestrationServiceのインスタンスを取得（依存性注入用）。

    バックテスト結果のCRUD操作、戦略管理、関連データの統合処理を担当する
    オーケストレーションサービスを提供します。

    Returns:
        BacktestOrchestrationService: バックテストオーケストレーションサービスインスタンス

    Raises:
        HTTPException: サービス初期化に失敗した場合（ステータスコード503）
    """

    def factory() -> "BacktestOrchestrationService":
        """サービスのインスタンスを生成する内部ファクトリ関数。

        Returns:
            BacktestOrchestrationService: 生成されたサービスインスタンス
        """
        from app.services.backtest.orchestration.backtest_orchestration_service import (
            BacktestOrchestrationService,
        )

        return BacktestOrchestrationService()

    return _create_service(factory, "BacktestOrchestrationService")


def get_funding_rate_orchestration_service(
    bybit_service=Depends(get_bybit_funding_rate_service),
) -> "FundingRateOrchestrationService":
    """FundingRateOrchestrationServiceのインスタンスを取得（依存性注入用）。

    ファンディングレートデータの収集・管理を統合管理するサービスを提供します。

    Args:
        bybit_service: BybitFundingRateServiceインスタンス（依存性注入）

    Returns:
        FundingRateOrchestrationService: ファンディングレートオーケストレーションサービスインスタンス

    Raises:
        HTTPException: サービス初期化に失敗した場合（ステータスコード503）
    """

    def factory() -> "FundingRateOrchestrationService":
        """サービスのインスタンスを生成する内部ファクトリ関数。

        Returns:
            FundingRateOrchestrationService: 生成されたサービスインスタンス
        """
        from app.services.data_collection.orchestration.funding_rate_orchestration_service import (
            FundingRateOrchestrationService,
        )

        return FundingRateOrchestrationService(bybit_service)

    return _create_service(factory, "FundingRateOrchestrationService")


def get_ml_management_orchestration_service() -> "MLManagementOrchestrationService":
    """MLManagementOrchestrationServiceのインスタンスを取得（依存性注入用）。

    トレーニング済みMLモデルの管理（一覧取得、削除、詳細情報取得等）を
    統合管理するサービスを提供します。

    Returns:
        MLManagementOrchestrationService: ML管理オーケストレーションサービスインスタンス

    Raises:
        HTTPException: サービス初期化に失敗した場合（ステータスコード503）
    """

    def factory() -> "MLManagementOrchestrationService":
        """サービスのインスタンスを生成する内部ファクトリ関数。

        Returns:
            MLManagementOrchestrationService: 生成されたサービスインスタンス
        """
        from app.services.ml.orchestration.ml_management_orchestration_service import (
            MLManagementOrchestrationService,
        )

        return MLManagementOrchestrationService()

    return _create_service(factory, "MLManagementOrchestrationService")


def get_long_short_ratio_repository(
    db: Session = Depends(get_db),
) -> LongShortRatioRepository:
    """
    LongShortRatioRepositoryのインスタンスを取得（依存性注入用）

    ロングショート比率データのデータベース操作を担当するリポジトリを提供します。

    Args:
        db: データベースセッション（依存性注入）

    Returns:
        LongShortRatioRepository: ロングショート比率リポジトリインスタンス
    """
    return LongShortRatioRepository(db)


def get_long_short_ratio_service() -> BybitLongShortRatioService:
    """
    BybitLongShortRatioServiceのインスタンスを取得（依存性注入用）

    Bybit取引所からのロングショート比率データ取得サービスを提供します。

    Returns:
        BybitLongShortRatioService: ロングショート比率サービスインスタンス
    """
    return BybitLongShortRatioService()
