"""
GAワーカープロセス初期化モジュール

並列評価時にワーカープロセスで共有データを保持するためのモジュールです。
グローバル変数を使用してデータを保持し、個体評価ごとのデータロードや転送を防ぎます。
"""

from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# ワーカープロセス内で共有されるデータコンテキスト
_WORKER_DATA_CONTEXT: Dict[str, Any] = {}


def initialize_worker(data_context: Dict[str, Any]):
    """
    ワーカープロセスの初期化

    Args:
        data_context: 共有データ（OHLCVデータなど）
    """
    global _WORKER_DATA_CONTEXT
    try:
        _WORKER_DATA_CONTEXT.update(data_context)
        
        # ログ出力（デバッグ用、大量に出ないよう注意）
        # logger.info(f"Worker initialized with data keys: {list(data_context.keys())}")
    except Exception as e:
        logger.error(f"Worker initialization failed: {e}")


def get_worker_data(key: str) -> Optional[Any]:
    """
    共有データの取得

    Args:
        key: データのキー

    Returns:
        データオブジェクト、またはNone
    """
    return _WORKER_DATA_CONTEXT.get(key)
