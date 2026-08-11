"""
ツールレジストリ

利用可能なツールを登録・検索するためのレジストリです。
"""

import logging
from typing import Optional

from app.utils.registry import Registry

from .base import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry(Registry[str, BaseTool]):
    """
    ツールレジストリ

    すべての利用可能なツールを登録し、名前で検索できるようにします。
    """

    _instance: Optional["ToolRegistry"] = None

    def __new__(cls) -> "ToolRegistry":
        """シングルトンパターン"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """シングルトンの再初期化を防ぐ"""
        if not hasattr(self, "_items"):
            super().__init__()

    def register(self, tool: BaseTool) -> None:
        """
        ツールを登録

        Args:
            tool: 登録するツールインスタンス
        """
        if tool.name in self._items:
            logger.warning(
                f"ツール '{tool.name}' は既に登録されています。上書きします。"
            )
        self.set(tool.name, tool)


# グローバルレジストリインスタンス
tool_registry = ToolRegistry()


def register_tool(tool: BaseTool) -> BaseTool:
    """
    ツールを登録するヘルパー関数

    Args:
        tool: 登録するツール

    Returns:
        登録されたツール
    """
    tool_registry.register(tool)
    return tool
