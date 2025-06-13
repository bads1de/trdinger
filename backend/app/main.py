"""
FastAPI メインアプリケーション

Trdinger Trading API のエントリーポイント
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config.settings import settings
from app.api.market_data import router as market_data_router
from app.api.data_collection import router as data_collection_router
from app.api.funding_rates import router as funding_rates_router
from app.api.open_interest import router as open_interest_router
from app.api.data_reset import router as data_reset_router

from app.api.backtest import router as backtest_router
from app.api.auto_strategy import router as auto_strategy_router
from app.api.strategy_showcase import router as strategy_showcase_router
from app.api.indicators import router as indicators_router


def setup_logging():
    """ログ設定を初期化"""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()), format=settings.log_format
    )


def create_app() -> FastAPI:
    """FastAPIアプリケーションを作成"""

    # ログ設定
    setup_logging()

    # FastAPIアプリケーション作成
    app = FastAPI(
        title=settings.app_name,
        description="CCXT ライブラリを使用した仮想通貨取引データAPI",
        version=settings.app_version,
        debug=settings.debug,
    )

    # CORS設定
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ルーター追加
    app.include_router(market_data_router, prefix="/api")
    app.include_router(data_collection_router, prefix="/api")
    app.include_router(funding_rates_router, prefix="/api")
    app.include_router(open_interest_router, prefix="/api")
    app.include_router(data_reset_router, prefix="/api")

    app.include_router(backtest_router)
    app.include_router(auto_strategy_router)
    app.include_router(strategy_showcase_router)
    app.include_router(indicators_router, prefix="/api")

    # ヘルスチェックエンドポイント
    @app.get("/health")
    async def health_check():
        return {
            "status": "ok",
            "app_name": settings.app_name,
            "version": settings.app_version,
        }

    return app


# アプリケーションインスタンス
app = create_app()
