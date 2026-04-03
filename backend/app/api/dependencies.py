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
    """サービス生成の定型エラーハンドリングをまとめる。"""
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


def get_bybit_open_interest_service():
    """BybitOpenInterestService を取得する。"""

    def factory():
        from app.services.data_collection.bybit.open_interest_service import (
            BybitOpenInterestService,
        )

        return BybitOpenInterestService()

    return _create_service(factory, "BybitOpenInterestService")


def get_bybit_funding_rate_service():
    """BybitFundingRateService を取得する。"""

    def factory():
        from app.services.data_collection.bybit.funding_rate_service import (
            BybitFundingRateService,
        )

        return BybitFundingRateService()

    return _create_service(factory, "BybitFundingRateService")


# get_automl_feature_generation_service は削除されました（autofeat機能の削除に伴う）


# Dependency factories for various orchestration services (avoid direct new() in API modules)
def get_data_collection_orchestration_service():
    """
    DataCollectionOrchestrationService のインスタンスを取得（依存性注入用）
    """

    def factory():
        from app.services.data_collection.orchestration.data_collection_orchestration_service import (
            DataCollectionOrchestrationService,
        )

        return DataCollectionOrchestrationService()

    return _create_service(factory, "DataCollectionOrchestrationService")


def get_data_management_orchestration_service():
    """
    DataManagementOrchestrationService のインスタンスを取得（依存性注入用）
    """

    def factory():
        from app.services.data_collection.orchestration.data_management_orchestration_service import (
            DataManagementOrchestrationService,
        )

        return DataManagementOrchestrationService()

    return _create_service(factory, "DataManagementOrchestrationService")


def get_open_interest_orchestration_service(
    bybit_service=Depends(get_bybit_open_interest_service),
):
    """
    OpenInterestOrchestrationService のインスタンスを取得（依存性注入用）
    """

    def factory():
        from app.services.data_collection.orchestration.open_interest_orchestration_service import (
            OpenInterestOrchestrationService,
        )

        return OpenInterestOrchestrationService(bybit_service)

    return _create_service(factory, "OpenInterestOrchestrationService")


def get_ml_training_service():
    """
    MLTrainingService のインスタンスを取得（依存性注入用）
    """

    def factory():
        from app.services.ml.orchestration.ml_training_orchestration_service import (
            MLTrainingService,
        )

        return MLTrainingService()

    return _create_service(factory, "MLTrainingService")


def get_backtest_orchestration_service():
    """
    BacktestOrchestrationService のインスタンスを取得（依存性注入用）
    """

    def factory():
        from app.services.backtest.orchestration.backtest_orchestration_service import (
            BacktestOrchestrationService,
        )

        return BacktestOrchestrationService()

    return _create_service(factory, "BacktestOrchestrationService")


def get_funding_rate_orchestration_service(
    bybit_service=Depends(get_bybit_funding_rate_service),
):
    """
    FundingRateOrchestrationService のインスタンスを取得（依存性注入用）
    """

    def factory():
        from app.services.data_collection.orchestration.funding_rate_orchestration_service import (
            FundingRateOrchestrationService,
        )

        return FundingRateOrchestrationService(bybit_service)

    return _create_service(factory, "FundingRateOrchestrationService")


def get_ml_management_orchestration_service():
    """
    MLManagementOrchestrationService のインスタンスを取得（依存性注入用）
    """

    def factory():
        from app.services.ml.orchestration.ml_management_orchestration_service import (
            MLManagementOrchestrationService,
        )

        return MLManagementOrchestrationService()

    return _create_service(factory, "MLManagementOrchestrationService")


def get_long_short_ratio_repository(
    db: Session = Depends(get_db),
) -> LongShortRatioRepository:
    """
    LongShortRatioRepository のインスタンスを取得（依存性注入用）
    """
    return LongShortRatioRepository(db)


def get_long_short_ratio_service() -> BybitLongShortRatioService:
    """
    BybitLongShortRatioService のインスタンスを取得（依存性注入用）
    """
    return BybitLongShortRatioService()
