"""
依存性注入用のファクトリ関数

FastAPIの依存性注入システムで使用するサービスファクトリ関数を提供します。
Orchestration Serviceパターンに基づいて、API層とサービス層の結合を解消します。
"""

from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status

from database.connection import get_db
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.funding_rate_repository import FundingRateRepository

from app.services.backtest.backtest_service import BacktestService
from app.services.backtest.backtest_data_service import BacktestDataService
from app.services.strategy_integration_service import StrategyIntegrationService


from app.services.data_collection.orchestration.market_data_orchestration_service import (
    MarketDataOrchestrationService,
)
from app.services.auto_strategy import AutoStrategyService
from app.services.ml.feature_engineering.automl_feature_generation_service import (
    AutoMLFeatureGenerationService,
)


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


def get_backtest_service_with_db(db: Session) -> BacktestService:
    """
    データベースセッション付きのBacktestServiceを取得

    Args:
        db: データベースセッション

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


def get_strategy_integration_service(db: Session) -> StrategyIntegrationService:
    """
    StrategyIntegrationServiceのインスタンスを取得

    Args:
        db: データベースセッション

    Returns:
        StrategyIntegrationServiceインスタンス
    """
    return StrategyIntegrationService(db)


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


def get_strategy_integration_service_with_db(
    db: Session = Depends(get_db),
) -> StrategyIntegrationService:
    """
    StrategyIntegrationServiceのインスタンスを取得（依存性注入用）

    Args:
        db: データベースセッション（依存性注入）

    Returns:
        StrategyIntegrationServiceインスタンス
    """
    return StrategyIntegrationService(db)


def get_automl_feature_generation_service(
    db: Session = Depends(get_db),
) -> AutoMLFeatureGenerationService:
    """
    AutoMLFeatureGenerationServiceのインスタンスを取得

    Args:
        db: データベースセッション

    Returns:
        AutoMLFeatureGenerationServiceインスタンス

    Raises:
        HTTPException: サービス初期化に失敗した場合
    """
    try:
        return AutoMLFeatureGenerationService(db)
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"AutoMLFeatureGenerationService初期化エラー: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AutoMLFeatureGenerationServiceが利用できません。サーバーログを確認してください。",
        )
