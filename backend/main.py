"""
アプリケーション エントリーポイント

Trdinger Trading API を起動します。
"""

import uvicorn
from app.config.unified_config import settings


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
        log_level=settings.logging.level.lower(),
    )
