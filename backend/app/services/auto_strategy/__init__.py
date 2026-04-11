"""
自動戦略生成パッケージ

遺伝的アルゴリズム（GA）を使用した取引戦略の自動生成機能を提供します。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import GAConfig
    from .genes import StrategyGene
    from .services.auto_strategy_service import AutoStrategyService
    from .positions.position_sizing_service import PositionSizingService
    from .tpsl import TPSLService

_ATTRIBUTE_EXPORTS = {
    "AutoStrategyService": ".services.auto_strategy_service",
    "StrategyGene": ".genes",
    "GAConfig": ".config",
    "TPSLService": ".tpsl",
    "PositionSizingService": ".positions.position_sizing_service",
}

__all__ = [
    "AutoStrategyService",
    "StrategyGene",
    "GAConfig",
    "TPSLService",
    "PositionSizingService",
]


def __getattr__(name: str) -> Any:
    """
    遅延インポートによる属性アクセスを提供

    モジュールの初期化時ではなく、属性が最初にアクセスされた時に
    対応するサブモジュールを動的にインポートします。
    これにより、循環インポートを回避し、初期化時のオーバーヘッドを削減します。

    Args:
        name: アクセス対象の属性名

    Returns:
        Any: 対応するサブモジュールから取得された属性値

    Raises:
        AttributeError: 指定された名前の属性が存在しない場合

    Note:
        一度インポートされた属性はglobals()にキャッシュされ、
        以降のアクセスでは再インポートされません。
    """
    module_path = _ATTRIBUTE_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """
    モジュールの公開属性一覧を返す

    モジュールレベルで既にインポートされている属性と、
    遅延インポート可能な属性の両方を含むリストを返します。
    IDEの自動補完やdir()関数での表示に使用されます。

    Returns:
        list[str]: モジュールの公開属性名のソート済みリスト
    """
    return sorted({*globals().keys(), *_ATTRIBUTE_EXPORTS})
