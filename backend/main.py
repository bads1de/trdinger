"""
アプリケーション エントリーポイント

Trdinger Trading API を起動します。
"""

import os
import sys

import uvicorn

from app.config.unified_config import unified_config

# Pythonパス調整（プロダクションモードでのインポート問題を解決）
BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=unified_config.app.host,
        port=unified_config.app.port,
        reload=unified_config.app.debug,
        log_level=unified_config.logging.level.lower(),
    )
