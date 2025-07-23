"""
アプリケーション設定

統一設定システムのエントリーポイント
"""

from .unified_config import unified_config

# グローバル設定インスタンス（既存のシングルトンを使用）
settings = unified_config

__all__ = ["settings"]
