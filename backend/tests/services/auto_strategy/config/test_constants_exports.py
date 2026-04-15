"""
auto_strategy.config.constants の export テスト
"""

import app.services.auto_strategy.config.constants as constants_package


class TestAutoStrategyConstantsExports:
    """config.constants の公開定数を検証する。"""

    def test_removed_common_constants_are_not_exported(self):
        """削除済みの共通定数は公開しない"""
        for name in (
            "SUPPORTED_SYMBOLS",
            "DEFAULT_TIMEFRAME",
            "ERROR_CODES",
            "CONSTRAINTS",
        ):
            assert hasattr(constants_package, name) is False
            assert name not in constants_package.__all__
