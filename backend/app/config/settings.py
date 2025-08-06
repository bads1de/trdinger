"""
アプリケーション設定

統一設定システムのエントリーポイント
"""

from .unified_config import unified_config

settings = unified_config

__all__ = ["settings"]
