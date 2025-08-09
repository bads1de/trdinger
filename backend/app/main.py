"""
FastAPI メインアプリケーション

Trdinger Trading API のエントリーポイント
"""

import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.auto_strategy import router as auto_strategy_router
from app.api.automl_features import router as automl_features_router
from app.api.backtest import router as backtest_router
from app.api.data_collection import router as data_collection_router
from app.api.data_reset import router as data_reset_router
from app.api.fear_greed import router as fear_greed_router
from app.api.funding_rates import router as funding_rates_router
from app.api.market_data import router as market_data_router
from app.api.ml_management import router as ml_management_router
from app.api.ml_training import router as ml_training_router
from app.api.open_interest import router as open_interest_router
from app.api.strategies import router as strategies_router
from app.config import settings
from app.utils.duplicate_filter_handler import DuplicateFilter


def setup_logging():
    """ログ設定を初期化（重複ログフィルター付き）"""
    # ルートロガーを取得
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.logging.level.upper()))

    # 既存のハンドラーをクリア
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # コンソールハンドラーを作成
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, settings.logging.level.upper()))

    # フォーマッターを設定
    formatter = logging.Formatter(settings.logging.format)
    console_handler.setFormatter(formatter)

    # 重複フィルターを作成（1秒間隔で同じメッセージをフィルタリング）
    duplicate_filter = DuplicateFilter(capacity=200, interval=1.0)
    console_handler.addFilter(duplicate_filter)

    # ハンドラーをルートロガーに追加
    root_logger.addHandler(console_handler)

    # オートストラテジー専用ロガーの設定
    auto_strategy_logger = logging.getLogger("app.services.auto_strategy")
    auto_strategy_logger.setLevel(getattr(logging, settings.logging.level.upper()))

    # ログディレクトリが存在しない場合は作成
    log_dir = "C:/Users/buti3/trading"
    os.makedirs(log_dir, exist_ok=True)


def create_app() -> FastAPI:
    """FastAPIアプリケーションを作成"""

    # ログ設定
    setup_logging()
    logger = logging.getLogger(__name__)

    # FastAPIアプリケーション作成
    app = FastAPI(
        title=settings.app.app_name,
        description="CCXT ライブラリを使用した仮想通貨取引データAPI",
        version=settings.app.app_version,
        debug=settings.app.debug,
    )

    # CORS設定
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ルーター追加
    app.include_router(market_data_router)
    app.include_router(data_collection_router)
    app.include_router(funding_rates_router)
    app.include_router(open_interest_router)
    app.include_router(fear_greed_router)
    app.include_router(data_reset_router)
    app.include_router(backtest_router)
    app.include_router(auto_strategy_router)
    app.include_router(strategies_router)
    app.include_router(ml_training_router)
    app.include_router(ml_management_router)
    app.include_router(automl_features_router)

    # グローバル例外ハンドラ
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "サーバー内部で予期せぬエラーが発生しました。",
                "error_type": type(exc).__name__,
            },
        )

    # ヘルスチェックエンドポイント
    @app.get("/health")
    async def health_check():
        return {
            "status": "ok",
            "app_name": settings.app.app_name,
            "version": settings.app.app_version,
        }

    return app


# アプリケーションインスタンス
app = create_app()
