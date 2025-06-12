"""
アプリケーション エントリーポイント

Trdinger Trading API を起動します。
"""

import uvicorn
from app.config.settings import settings


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
