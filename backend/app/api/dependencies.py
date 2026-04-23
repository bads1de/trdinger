"""
依存性注入用のファクトリ関数

FastAPIの依存性注入システムで使用するサービスファクトリ関数を提供します。
Orchestration Serviceパターンに基づいて、API層とサービス層の結合を解消します。
"""

import logging
from typing import Callable, Optional, Type

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.services.auto_strategy import AutoStrategyService
from app.services.auto_strategy.services.generated_strategy_service import (
    GeneratedStrategyService,
)
from app.services.data_collection.bybit.long_short_ratio_service import (
    BybitLongShortRatioService,
)
from app.services.data_collection.orchestration.market_data_orchestration_service import (  # noqa: E501
    MarketDataOrchestrationService,
)
from database.connection import get_db
from database.repositories.long_short_ratio_repository import (
    LongShortRatioRepository,
)

logger = logging.getLogger(__name__)


def _create_service(
    factory: Callable[[], object], service_name: str
) -> object:
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


def _lazy_import_service(module_path: str, class_name: str) -> Type:
    """
    サービスクラスの遅延インポートを実行するヘルパー関数

    Args:
        module_path: モジュールのインポートパス
        class_name: クラス名

    Returns:
        Type: インポートされたサービスクラス
    """
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_service_factory(
    module_path: str,
    class_name: str,
    service_display_name: Optional[str] = None,
) -> Callable[[], object]:
    """
    ジェネリックサービスファクトリを生成する高階関数

    従来の個別ファクトリ関数を生成するための共通化ユーティリティです。
    遅延インポートとエラーハンドリングを自動的に行います。

    使用例:
        get_ml_training_service = create_service_factory(
            "app.services.ml.orchestration.ml_training_orchestration_service",
            "MLTrainingService",
            "MLTrainingService"
        )

    Args:
        module_path: モジュールのインポートパス
        class_name: クラス名
        service_display_name: エラーメッセージ用サービス名

    Returns:
        FastAPI Depends で使用可能なサービスファクトリ関数
    """
    display_name = service_display_name or class_name

    def factory() -> object:
        service_class = _lazy_import_service(module_path, class_name)
        return service_class()

    def wrapper() -> object:
        return _create_service(factory, display_name)

    return wrapper


def create_service_factory_with_deps(
    module_path: str,
    class_name: str,
    service_display_name: Optional[str] = None,
    dep_factory: Optional[Callable] = None,
) -> Callable:
    """
    依存関係を持つサービスファクトリを生成する高階関数

    別のサービスに依存するオーケストレーションサービス用のファクトリを生成します。

    Args:
        module_path: モジュールのインポートパス
        class_name: クラス名
        service_display_name: エラーメッセージ用サービス名
        dep_factory: 依存サービスファクトリ（Depends で注入）

    Returns:
        FastAPI Depends で使用可能なサービスファクトリ関数
    """
    display_name = service_display_name or class_name

    if dep_factory is not None:

        def wrapper_with_deps(dep=Depends(dep_factory)) -> object:
            def inner_factory() -> object:
                service_class = _lazy_import_service(module_path, class_name)
                return service_class(dep)

            return _create_service(inner_factory, display_name)

        return wrapper_with_deps
    else:

        def wrapper_without_deps() -> object:
            def inner_factory() -> object:
                service_class = _lazy_import_service(module_path, class_name)
                return service_class()

            return _create_service(inner_factory, display_name)

        return wrapper_without_deps


def get_market_data_orchestration_service(
    db: Session = Depends(get_db),
) -> MarketDataOrchestrationService:
    """
    MarketDataOrchestrationServiceのインスタンスを取得（依存性注入用）

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
    return _create_service(
        AutoStrategyService, "AutoStrategyService"
    )  # type: ignore[return-value]


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
    return _create_service(
        lambda: GeneratedStrategyService(db),
        "GeneratedStrategyService",
    )  # type: ignore[return-value]


# --- ジェネリックファクトリで生成したサービス ---

get_bybit_open_interest_service = create_service_factory(
    "app.services.data_collection.bybit.open_interest_service",
    "BybitOpenInterestService",
    "BybitOpenInterestService",
)

get_bybit_funding_rate_service = create_service_factory(
    "app.services.data_collection.bybit.funding_rate_service",
    "BybitFundingRateService",
    "BybitFundingRateService",
)

get_data_collection_orchestration_service = create_service_factory(
    "app.services.data_collection.orchestration."
    "data_collection_orchestration_service",
    "DataCollectionOrchestrationService",
    "DataCollectionOrchestrationService",
)

get_data_management_orchestration_service = create_service_factory(
    ("app.services.data_collection.orchestration."
     "data_management_orchestration_service"),
    "DataManagementOrchestrationService",
    "DataManagementOrchestrationService",
)

get_open_interest_orchestration_service = (
    create_service_factory_with_deps(
        "app.services.data_collection.orchestration."
        "open_interest_orchestration_service",
        "OpenInterestOrchestrationService",
        "OpenInterestOrchestrationService",
        dep_factory=get_bybit_open_interest_service,
    )
)

get_ml_training_service = create_service_factory(
    ("app.services.ml.orchestration."
     "ml_training_orchestration_service"),
    "MLTrainingService",
    "MLTrainingService",
)

get_backtest_orchestration_service = create_service_factory(
    ("app.services.backtest.orchestration."
     "backtest_orchestration_service"),
    "BacktestOrchestrationService",
    "BacktestOrchestrationService",
)

get_funding_rate_orchestration_service = (
    create_service_factory_with_deps(
        ("app.services.data_collection.orchestration."
         "funding_rate_orchestration_service"),
        "FundingRateOrchestrationService",
        "FundingRateOrchestrationService",
        dep_factory=get_bybit_funding_rate_service,
    )
)

get_ml_management_orchestration_service = create_service_factory(
    "app.services.ml.orchestration.ml_management_orchestration_service",
    "MLManagementOrchestrationService",
    "MLManagementOrchestrationService",
)


def get_long_short_ratio_repository(
    db: Session = Depends(get_db),
) -> LongShortRatioRepository:
    """
    LongShortRatioRepositoryのインスタンスを取得（依存性注入用）

    Args:
        db: データベースセッション（依存性注入）

    Returns:
        LongShortRatioRepository: ロングショート比率リポジトリインスタンス
    """
    return LongShortRatioRepository(db)


def get_long_short_ratio_service() -> BybitLongShortRatioService:
    """
    BybitLongShortRatioServiceのインスタンスを取得（依存性注入用）

    Returns:
        BybitLongShortRatioService: ロングショート比率サービスインスタンス
    """
    return BybitLongShortRatioService()
