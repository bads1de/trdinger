"""
ツールレジストリ

利用可能なツールを登録・検索するためのレジストリです。
"""

import logging
from typing import Dict, List, Optional

from .base import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    ツールレジストリ

    すべての利用可能なツールを登録し、名前で検索できるようにします。
    """

    _instance: Optional["ToolRegistry"] = None
    _tools: Dict[str, BaseTool] = {}

    def __new__(cls) -> "ToolRegistry":
        """シングルトンパターン"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    def register(self, tool: BaseTool) -> None:
        """
        ツールを登録

        Args:
            tool: 登録するツールインスタンス
        """
        if tool.name in self._tools:
            logger.warning(
                f"ツール '{tool.name}' は既に登録されています。上書きします。"
            )
        self._tools[tool.name] = tool
        logger.debug(f"ツール '{tool.name}' を登録しました")

    def get(self, name: str) -> Optional[BaseTool]:
        """
        名前でツールを取得

        Args:
            name: ツール名

        Returns:
            ツールインスタンス、見つからない場合は None
        """
        return self._tools.get(name)

    def get_all(self) -> List[BaseTool]:
        """
        すべての登録済みツールを取得

        Returns:
            ツールのリスト
        """
        return list(self._tools.values())

    def get_names(self) -> List[str]:
        """
        すべての登録済みツール名を取得

        Returns:
            ツール名のリスト
        """
        return list(self._tools.keys())

    def clear(self) -> None:
        """登録をクリア（テスト用）"""
        self._tools.clear()


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





