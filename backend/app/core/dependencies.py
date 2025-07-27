"""
依存性注入用のファクトリ関数

FastAPIの依存性注入システムで使用するサービスファクトリ関数を提供します。
"""

from sqlalchemy.orm import Session
from fastapi import Depends

from database.connection import get_db
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.funding_rate_repository import FundingRateRepository

from app.core.services.backtest_service import BacktestService
from app.core.services.backtest_data_service import BacktestDataService
from app.core.services.strategy_integration_service import StrategyIntegrationService


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
