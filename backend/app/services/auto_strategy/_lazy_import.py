"""
遅延インポートヘルパー

モジュールレベルの__getattr__を使用した遅延インポートパターンを統一します。
循環インポートの回避と初期化時オーバーヘッドの削減を目的とします。

使用例:
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from .module import MyClass

    _EXPORTS = {
        "MyClass": ".module",
    }

    __all__ = ["MyClass"]

    # 以下の2行を追加するだけ
    from ._lazy_import import setup_lazy_import
    setup_lazy_import(globals(), _EXPORTS, __all__)
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def setup_lazy_import(
    module_globals: dict[str, Any],
    exports: dict[str, str],
    all_list: list[str],
) -> None:
    """
    モジュールに遅延インポート機能を設定

    この関数は、対象モジュールの__init__.pyで呼び出され、
    __getattr__と__dir__関数をモジュールのglobals()に注入します。

    Args:
        module_globals: 対象モジュールのglobals()辞書
        exports: {属性名: モジュールパス}のマッピング
        all_list: __all__リスト
    """
    module_name = module_globals.get("__name__", "<unknown>")

    # 既存の__getattr__を上書きしない（既に定義されている場合）
    if "__getattr__" in module_globals:
        import warnings
        warnings.warn(
            f"Module {module_name} already has __getattr__. "
            f"setup_lazy_import will not override it."
        )
        return
    
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
        module_path = exports.get(name)
        if module_path is None:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

        module = import_module(module_path, module_name)
        value = getattr(module, name)
        module_globals[name] = value
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
        return sorted({*module_globals.keys(), *exports})

    # モジュールのglobals()に関数を注入
    module_globals["__getattr__"] = __getattr__
    module_globals["__dir__"] = __dir__
