"""
_lazy_import ヘルパーのユニットテスト

遅延インポートを統一的に扱うためのヘルパー関数をテストします。
"""

from types import ModuleType
from unittest.mock import patch

import pytest

from app.services.auto_strategy import _lazy_import
from app.services.auto_strategy._lazy_import import setup_lazy_import

# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------


def _make_globals(**extra) -> dict:
    base = {"__name__": "app.services.auto_strategy._test_target_module"}
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# __getattr__ が無い素直なケース
# ---------------------------------------------------------------------------


class TestSetupLazyImportInjects:
    def test_setup_injects_getattr_and_dir(self) -> None:
        module_globals = _make_globals()
        setup_lazy_import(module_globals, {"BaseTool": ".tools.base"}, ["BaseTool"])
        assert callable(module_globals["__getattr__"])
        assert callable(module_globals["__dir__"])

    def test_setup_with_no_name_attribute_uses_unknown(self) -> None:
        """__name__ がない場合、警告で '<unknown>' が使われる"""
        module_globals = {"__getattr__": lambda name: None}
        with pytest.warns(UserWarning, match="<unknown>"):
            setup_lazy_import(module_globals, {"X": ".tools.base"}, ["X"])

    def test_getattr_resolves_known_attribute(self) -> None:
        """既存パッケージ内のモジュールを相対指定で遅延ロードできる"""
        # 実在するパッケージ名を __name__ にして、相対パスが解決できるようにする
        module_globals = {"__name__": "app.services.auto_strategy"}
        setup_lazy_import(module_globals, {"BaseTool": ".tools.base"}, ["BaseTool"])

        # アクセス前は存在しない
        assert "BaseTool" not in module_globals

        base_tool_cls = module_globals["__getattr__"]("BaseTool")
        from app.services.auto_strategy.tools.base import BaseTool

        assert base_tool_cls is BaseTool
        # 解決後にキャッシュされる
        assert module_globals["BaseTool"] is BaseTool

    def test_resolved_attribute_is_cached_in_globals(self) -> None:
        """解決済み属性は module_globals にキャッシュされ、直接アクセスで取得できる"""
        module_globals = {"__name__": "app.services.auto_strategy"}
        setup_lazy_import(module_globals, {"BaseTool": ".tools.base"}, ["BaseTool"])

        # 初回アクセスで解決
        module_globals["__getattr__"]("BaseTool")
        # module_globals にキャッシュされている
        from app.services.auto_strategy.tools.base import BaseTool

        assert module_globals["BaseTool"] is BaseTool
        # 直接アクセスでも取得できる(Python はキャッシュ済みなら __getattr__ を呼ばない)
        assert module_globals["BaseTool"] is BaseTool

    def test_getattr_raises_attribute_error_for_unknown(self) -> None:
        """未定義の属性名は AttributeError"""
        module_globals = _make_globals()
        setup_lazy_import(module_globals, {"BaseTool": ".tools.base"}, ["BaseTool"])

        with pytest.raises(AttributeError, match="has no attribute 'Missing'"):
            module_globals["__getattr__"]("Missing")

    def test_dir_includes_existing_and_lazy_attributes(self) -> None:
        """__dir__ は既存属性と遅延属性の和集合をソートして返す"""
        module_globals = _make_globals(AlreadyHere=1)
        setup_lazy_import(module_globals, {"BaseTool": ".tools.base"}, ["BaseTool"])

        names = module_globals["__dir__"]()
        assert "AlreadyHere" in names
        assert "BaseTool" in names
        # ソート済み
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# __getattr__ が既に存在する場合
# ---------------------------------------------------------------------------


class TestSetupLazyImportWhenGetattrExists:
    def test_setup_warns_and_does_nothing(self) -> None:
        """既に __getattr__ がある場合、警告を出して何もしない"""

        def existing_getattr(name: str) -> None:
            return None

        module_globals = _make_globals(__getattr__=existing_getattr)

        with pytest.warns(UserWarning, match="already has __getattr__"):
            setup_lazy_import(module_globals, {"X": ".tools.base"}, ["X"])

        # __getattr__ は上書きされず、__dir__ も追加されない
        assert module_globals["__getattr__"] is existing_getattr
        assert "__dir__" not in module_globals

    def test_warning_includes_module_name(self) -> None:
        """警告にモジュール名が含まれる"""
        module_globals = _make_globals(
            __name__="my_custom_module", __getattr__=lambda name: None
        )
        with pytest.warns(UserWarning, match="my_custom_module"):
            setup_lazy_import(module_globals, {"X": ".tools.base"}, ["X"])


# ---------------------------------------------------------------------------
# import_module の挙動
# ---------------------------------------------------------------------------


class TestSetupLazyImportCallsImportModule:
    def test_getattr_passes_relative_path_and_package(self) -> None:
        """import_module に相対パスと呼び出し元モジュール名が渡される"""
        sentinel_cls = type("Sentinel", (), {})

        with patch.object(_lazy_import, "import_module") as mock_import:
            mock_module = ModuleType("resolved")
            mock_module.DummyClass = sentinel_cls
            mock_import.return_value = mock_module

            host_name = "app.services.auto_strategy"
            module_globals = {"__name__": host_name}
            setup_lazy_import(
                module_globals, {"DummyClass": ".tools.base"}, ["DummyClass"]
            )

            value = module_globals["__getattr__"]("DummyClass")
            assert value is sentinel_cls
            # 相対パスが第1引数、ホスト名が第2引数
            mock_import.assert_called_once_with(".tools.base", host_name)


# ---------------------------------------------------------------------------
# ロギング
# ---------------------------------------------------------------------------


class TestSetupLazyImportDebugLog:
    def test_module_name_fallback_uses_unknown(self) -> None:
        """__name__ がない場合、'module {__name__!r} has no attribute' で unknown"""
        # __name__ キーを明示的に渡さず、ヘルパーも経由しない
        module_globals: dict = {}
        setup_lazy_import(module_globals, {"X": ".tools.base"}, ["X"])
        with pytest.raises(
            AttributeError, match=r"module '<unknown>' has no attribute"
        ):
            module_globals["__getattr__"]("UnknownAttr")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
