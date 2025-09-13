"""unified_configのテスト"""

from backend.app.config.unified_config import unified_config
import pytest


def test_settings_removed():
    """settingsエイリアスが削除されたことを確認"""
    with pytest.raises(NameError):
        settings  # noqa: F821