"""
バックテストサービス定数

バックテスト関連の静的データを定義します。
"""

from typing import Any, Dict

SUPPORTED_STRATEGIES: Dict[str, Any] = {
    "auto_strategy": {
        "name": "オートストラテジー",
        "description": "遺伝的アルゴリズムで生成された戦略",
        "parameters": {
            "strategy_gene": {
                "type": "dict",
                "required": True,
                "description": "戦略遺伝子",
            }
        },
    }
}
