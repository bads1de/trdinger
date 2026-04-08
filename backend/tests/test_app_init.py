"""
appパッケージの__init__.pyのテスト

バージョン情報とモジュール属性を確認します。
"""

import pytest

import app as app_package


class TestAppInitAttributes:
    """app/__init__.pyの属性テスト"""

    def test_version_exists(self):
        """__version__が定義されている"""
        assert hasattr(app_package, "__version__")

    def test_version_is_string(self):
        """__version__は文字列である"""
        assert isinstance(app_package.__version__, str)

    def test_version_format(self):
        """__version__はバージョン形式である"""
        # X.Y.Z形式のバージョン番号
        version = app_package.__version__
        assert "." in version
        parts = version.split(".")
        assert len(parts) >= 1
        # 各パートは数字であることを確認（簡易チェック）
        for part in parts:
            assert part.isdigit() or part.replace(".", "").isdigit()

    def test_author_exists(self):
        """__author__が定義されている"""
        assert hasattr(app_package, "__author__")

    def test_author_is_string(self):
        """__author__は文字列である"""
        assert isinstance(app_package.__author__, str)

    def test_author_has_value(self):
        """__author__は空でない"""
        assert len(app_package.__author__) > 0

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert app_package.__doc__ is not None
        assert len(app_package.__doc__) > 0
        assert "Trdinger" in app_package.__doc__ or "Trading" in app_package.__doc__

    def test_module_name(self):
        """モジュール名が正しい"""
        assert app_package.__name__ == "app"
