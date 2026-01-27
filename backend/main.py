"""
アプリケーション エントリーポイント

Trdinger Trading API を起動します。
"""

import os
import sys
import warnings

import uvicorn

# pandas-ta, pandas, numpy 関連の警告を抑制
# これらはライブラリ内部の問題であり、アプリケーションの動作に影響しない
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas_ta")

from app.config.unified_config import unified_config  # noqa: E402

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
