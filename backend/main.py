"""
アプリケーション エントリーポイント

Trdinger Trading API を起動します。
"""

import uvicorn
import logging
from app.config.settings import settings


if __name__ == "__main__":
    # ロギング設定の一元化
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
