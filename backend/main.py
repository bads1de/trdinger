"""
アプリケーション エントリーポイント

Trdinger Trading API を起動します。
"""

import os
import sys
import uvicorn
from app.config.unified_config import settings

# Pythonパス調整（プロダクションモードでのインポート問題を解決）
BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
        log_level=settings.logging.level.lower(),
    )
