"""
依存性注入用のファクトリ関数

FastAPIの依存性注入システムで使用するサービスファクトリ関数を提供します。
Orchestration Serviceパターンに基づいて、API層とサービス層の結合を解消します。
"""

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.services.auto_strategy import AutoStrategyService
from app.services.auto_strategy.services.generated_strategy_service import (
    GeneratedStrategyService,
)
from app.services.backtest.backtest_data_service import BacktestDataService
from app.services.backtest.backtest_service import BacktestService
from app.services.data_collection.bybit.long_short_ratio_service import (
    BybitLongShortRatioService,
)
from app.services.data_collection.orchestration.market_data_orchestration_service import (
    MarketDataOrchestrationService,
)
from database.connection import get_db
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.long_short_ratio_repository import LongShortRatioRepository
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository


def get_backtest_service(db: Session = Depends(get_db)) -> BacktestService:
    """
    BacktestServiceのインスタンスを取得

    Args:
        db: データベースセッション（依存性注入）

    Returns:
        BacktestServiceインスタンス
    """
    ohlcv_repo = OHLCVRepository(db)
    oi_repo = OpenInterestRepository(db)
    fr_repo = FundingRateRepository(db)

    data_service = BacktestDataService(
        ohlcv_repo=ohlcv_repo, oi_repo=oi_repo, fr_repo=fr_repo
    )

    return BacktestService(data_service)


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
    try:
        return AutoStrategyService()
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"AutoStrategyService初期化エラー: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AutoStrategyServiceが利用できません。サーバーログを確認してください。",
        )


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
    return GeneratedStrategyService(db)


# get_automl_feature_generation_service は削除されました（autofeat機能の削除に伴う）


# Dependency factories for various orchestration services (avoid direct new() in API modules)
def get_data_collection_orchestration_service():
    """
    DataCollectionOrchestrationService のインスタンスを取得（依存性注入用）
    """
    try:
        from app.services.data_collection.orchestration.data_collection_orchestration_service import (
            DataCollectionOrchestrationService,
        )

        return DataCollectionOrchestrationService()
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(
            f"DataCollectionOrchestrationService初期化エラー: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DataCollectionOrchestrationServiceが利用できません。サーバーログを確認してください。",
        )


def get_data_management_orchestration_service():
    """
    DataManagementOrchestrationService のインスタンスを取得（依存性注入用）
    """
    try:
        from app.services.data_collection.orchestration.data_management_orchestration_service import (
            DataManagementOrchestrationService,
        )

        return DataManagementOrchestrationService()
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(
            f"DataManagementOrchestrationService初期化エラー: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DataManagementOrchestrationServiceが利用できません。サーバーログを確認してください。",
        )


def get_open_interest_orchestration_service(
    bybit_service=Depends(
        lambda: __import__(
            "app.services.data_collection.bybit.open_interest_service",
            fromlist=["BybitOpenInterestService"],
        ).BybitOpenInterestService()
    ),
):
    """
    OpenInterestOrchestrationService のインスタンスを取得（依存性注入用）
    """
    try:
        from app.services.data_collection.orchestration.open_interest_orchestration_service import (
            OpenInterestOrchestrationService,
        )

        return OpenInterestOrchestrationService(bybit_service)
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(
            f"OpenInterestOrchestrationService初期化エラー: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenInterestOrchestrationServiceが利用できません。サーバーログを確認してください。",
        )


def get_ml_training_service():
    """
    MLTrainingService のインスタンスを取得（依存性注入用）
    """
    try:
        from app.services.ml.orchestration.ml_training_orchestration_service import (
            MLTrainingService,
        )

        return MLTrainingService()
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"MLTrainingService初期化エラー: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MLTrainingServiceが利用できません。サーバーログを確認してください。",
        )


def get_backtest_orchestration_service(
    db: Session = Depends(get_db),
):
    """
    BacktestOrchestrationService のインスタンスを取得（依存性注入用）
    """
    try:
        from app.services.backtest.orchestration.backtest_orchestration_service import (
            BacktestOrchestrationService,
        )

        return BacktestOrchestrationService()
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"BacktestOrchestrationService初期化エラー: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="BacktestOrchestrationServiceが利用できません。サーバーログを確認してください。",
        )


def get_funding_rate_orchestration_service(
    bybit_service=Depends(
        lambda: __import__(
            "app.services.data_collection.bybit.funding_rate_service",
            fromlist=["BybitFundingRateService"],
        ).BybitFundingRateService()
    ),
):
    """
    FundingRateOrchestrationService のインスタンスを取得（依存性注入用）
    """
    try:
        from app.services.data_collection.orchestration.funding_rate_orchestration_service import (
            FundingRateOrchestrationService,
        )

        return FundingRateOrchestrationService(bybit_service)
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"FundingRateOrchestrationService初期化エラー: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FundingRateOrchestrationServiceが利用できません。サーバーログを確認してください。",
        )


def get_ml_management_orchestration_service():
    """
    MLManagementOrchestrationService のインスタンスを取得（依存性注入用）
    """
    try:
        from app.services.ml.orchestration.ml_management_orchestration_service import (
            MLManagementOrchestrationService,
        )

        return MLManagementOrchestrationService()
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(
            f"MLManagementOrchestrationService初期化エラー: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MLManagementOrchestrationServiceが利用できません。サーバーログを確認してください。",
        )


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
