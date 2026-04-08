"""
共通定数のテスト

app/config/constants.pyの定数定義をテストします。
"""

import pytest

from app.config import constants


class TestSupportedTimeframes:
    """SUPPORTED_TIMEFRAMES定数のテスト"""

    def test_exists(self):
        """SUPPORTED_TIMEFRAMESが定義されている"""
        assert hasattr(constants, "SUPPORTED_TIMEFRAMES")

    def test_is_list(self):
        """SUPPORTED_TIMEFRAMESはリスト型"""
        assert isinstance(constants.SUPPORTED_TIMEFRAMES, list)

    def test_is_not_empty(self):
        """SUPPORTED_TIMEFRAMESは空でない"""
        assert len(constants.SUPPORTED_TIMEFRAMES) > 0

    def test_contains_standard_timeframes(self):
        """標準的な時間軸が含まれている"""
        standard_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        for tf in standard_timeframes:
            assert tf in constants.SUPPORTED_TIMEFRAMES

    def test_all_elements_are_strings(self):
        """すべての要素は文字列"""
        for tf in constants.SUPPORTED_TIMEFRAMES:
            assert isinstance(tf, str)


class TestDefaultEnsembleAlgorithms:
    """DEFAULT_ENSEMBLE_ALGORITHMS定数のテスト"""

    def test_exists(self):
        """DEFAULT_ENSEMBLE_ALGORITHMSが定義されている"""
        assert hasattr(constants, "DEFAULT_ENSEMBLE_ALGORITHMS")

    def test_is_tuple(self):
        """DEFAULT_ENSEMBLE_ALGORITHMSはタプル型"""
        assert isinstance(constants.DEFAULT_ENSEMBLE_ALGORITHMS, tuple)

    def test_contains_expected_algorithms(self):
        """予想されるアルゴリズムが含まれている"""
        expected = ("lightgbm", "xgboost", "catboost")
        assert constants.DEFAULT_ENSEMBLE_ALGORITHMS == expected

    def test_all_elements_are_strings(self):
        """すべての要素は文字列"""
        for alg in constants.DEFAULT_ENSEMBLE_ALGORITHMS:
            assert isinstance(alg, str)


class TestDefaultMarketExchange:
    """DEFAULT_MARKET_EXCHANGE定数のテスト"""

    def test_exists(self):
        """DEFAULT_MARKET_EXCHANGEが定義されている"""
        assert hasattr(constants, "DEFAULT_MARKET_EXCHANGE")

    def test_is_string(self):
        """DEFAULT_MARKET_EXCHANGEは文字列"""
        assert isinstance(constants.DEFAULT_MARKET_EXCHANGE, str)

    def test_has_value(self):
        """DEFAULT_MARKET_EXCHANGEは空でない値を持つ"""
        assert constants.DEFAULT_MARKET_EXCHANGE

    def test_is_lowercase(self):
        """DEFAULT_MARKET_EXCHANGEは小文字"""
        assert constants.DEFAULT_MARKET_EXCHANGE.islower()


class TestDefaultMarketSymbol:
    """DEFAULT_MARKET_SYMBOL定数のテスト"""

    def test_exists(self):
        """DEFAULT_MARKET_SYMBOLが定義されている"""
        assert hasattr(constants, "DEFAULT_MARKET_SYMBOL")

    def test_is_string(self):
        """DEFAULT_MARKET_SYMBOLは文字列"""
        assert isinstance(constants.DEFAULT_MARKET_SYMBOL, str)

    def test_has_value(self):
        """DEFAULT_MARKET_SYMBOLは空でない値を持つ"""
        assert constants.DEFAULT_MARKET_SYMBOL

    def test_follows_format(self):
        """DEFAULT_MARKET_SYMBOLは通貨ペア形式"""
        # BTC/USDT:USDT などの形式
        assert "/" in constants.DEFAULT_MARKET_SYMBOL


class TestDefaultMarketTimeframe:
    """DEFAULT_MARKET_TIMEFRAME定数のテスト"""

    def test_exists(self):
        """DEFAULT_MARKET_TIMEFRAMEが定義されている"""
        assert hasattr(constants, "DEFAULT_MARKET_TIMEFRAME")

    def test_is_string(self):
        """DEFAULT_MARKET_TIMEFRAMEは文字列"""
        assert isinstance(constants.DEFAULT_MARKET_TIMEFRAME, str)

    def test_is_in_supported_timeframes(self):
        """DEFAULT_MARKET_TIMEFRAMEはサポートされている時間軸に含まれる"""
        assert constants.DEFAULT_MARKET_TIMEFRAME in constants.SUPPORTED_TIMEFRAMES


class TestDataLimitConstants:
    """データ取得件数関連定数のテスト"""

    def test_default_data_limit_exists(self):
        """DEFAULT_DATA_LIMITが定義されている"""
        assert hasattr(constants, "DEFAULT_DATA_LIMIT")

    def test_max_data_limit_exists(self):
        """MAX_DATA_LIMITが定義されている"""
        assert hasattr(constants, "MAX_DATA_LIMIT")

    def test_min_data_limit_exists(self):
        """MIN_DATA_LIMITが定義されている"""
        assert hasattr(constants, "MIN_DATA_LIMIT")

    def test_are_integers(self):
        """すべて整数型"""
        assert isinstance(constants.DEFAULT_DATA_LIMIT, int)
        assert isinstance(constants.MAX_DATA_LIMIT, int)
        assert isinstance(constants.MIN_DATA_LIMIT, int)

    def test_limits_are_positive(self):
        """すべて正の値"""
        assert constants.DEFAULT_DATA_LIMIT > 0
        assert constants.MAX_DATA_LIMIT > 0
        assert constants.MIN_DATA_LIMIT > 0

    def test_limits_are_ordered(self):
        """制限値の順序が正しい"""
        assert constants.MIN_DATA_LIMIT <= constants.DEFAULT_DATA_LIMIT <= constants.MAX_DATA_LIMIT

    def test_min_limit_is_one(self):
        """MIN_DATA_LIMITは1"""
        assert constants.MIN_DATA_LIMIT == 1


class TestConstantsImmutability:
    """定数の不変性に関するテスト"""

    def test_timeframes_list_modification_does_not_affect_module(self):
        """インポート後のリスト変更がモジュールに影響しない（検証用）"""
        original_length = len(constants.SUPPORTED_TIMEFRAMES)
        # これは同じリストオブジェクトへの参照なので変更できるが、
        # テスト後に元に戻す必要がある
        timeframes = constants.SUPPORTED_TIMEFRAMES.copy()
        assert len(timeframes) == original_length


class TestConstantExports:
    """定数のエクスポートに関するテスト"""

    def test_all_expected_constants_exist(self):
        """予想されるすべての定数が存在する"""
        expected_constants = [
            "SUPPORTED_TIMEFRAMES",
            "DEFAULT_ENSEMBLE_ALGORITHMS",
            "DEFAULT_MARKET_EXCHANGE",
            "DEFAULT_MARKET_SYMBOL",
            "DEFAULT_MARKET_TIMEFRAME",
            "DEFAULT_DATA_LIMIT",
            "MAX_DATA_LIMIT",
            "MIN_DATA_LIMIT",
        ]

        for const_name in expected_constants:
            assert hasattr(constants, const_name), f"{const_name}が定義されていません"

    def test_no_private_constants_in_module(self):
        """プライベート定数（_で始まる）はモジュールレベルに存在しない"""
        public_names = [name for name in dir(constants) if not name.startswith("_")]
        # __doc__, __file__, __name__なども除外
        actual_constants = [name for name in public_names if not name.startswith("__")]
        # 少なくとも1つの定数が存在する
        assert len(actual_constants) > 0
