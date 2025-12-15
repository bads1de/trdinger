"""
ツール遺伝子モデル

ツールの設定を遺伝子として保持し、GAで進化させることができます。
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ToolGene:
    """
    ツール遺伝子

    エントリーフィルターツールの設定を保持します。
    GAによって有効/無効やパラメータが進化します。
    """

    # ツールの識別名（例: 'weekend_filter'）
    tool_name: str

    # ツールが有効かどうか
    enabled: bool = True

    # ツール固有のパラメータ
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        辞書形式に変換

        Returns:
            辞書表現
        """
        return {
            "tool_name": self.tool_name,
            "enabled": self.enabled,
            "params": self.params.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolGene":
        """
        辞書から生成

        Args:
            data: 辞書データ

        Returns:
            ToolGene インスタンス
        """
        return cls(
            tool_name=data.get("tool_name", ""),
            enabled=data.get("enabled", True),
            params=data.get("params", {}).copy(),
        )


