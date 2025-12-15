"""
バックテスト設定検証サービステスト

BacktestConfigValidatorの機能をテストします。
"""

import logging
from datetime import datetime, timedelta

import pytest

from app.services.backtest.validation.backtest_config_validator import (
    BacktestConfigValidationError,
    BacktestConfigValidator,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def validator():
    """BacktestConfigValidatorインスタンス"""
    return BacktestConfigValidator()


@pytest.fixture
def valid_config():
    """有効なバックテスト設定"""
    return {
        "strategy_name": "TestStrategy",
        "symbol": "BTC/USDT:USDT",
        "timeframe": "1h",
        "start_date": datetime(2024, 1, 1),
        "end_date": datetime(2024, 1, 31),
        "initial_capital": 10000.0,
        "commission_rate": 0.001,
        "strategy_config": {
            "strategy_type": "momentum",
            "fast_period": 10,
            "slow_period": 30,
        },
    }


class TestValidatorInitialization:
    """バリデーター初期化テスト"""

    def test_initialize_validator(self):
        """バリデーターを初期化できること"""
        validator = BacktestConfigValidator()
        assert validator is not None
        assert hasattr(validator, "REQUIRED_FIELDS")
        assert hasattr(validator, "VALID_TIMEFRAMES")

    def test_required_fields_constant(self, validator):
        """必須フィールド定数が正しいこと"""
        expected_fields = [
            "strategy_name",
            "symbol",
            "timeframe",
            "start_date",
            "end_date",
            "initial_capital",
            "commission_rate",
        ]
        assert validator.REQUIRED_FIELDS == expected_fields

    def test_valid_timeframes_constant(self, validator):
        """有効な時間軸定数が正しいこと"""
        expected_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        assert validator.VALID_TIMEFRAMES == expected_timeframes


class TestConfigValidation:
    """設定検証テスト"""

    def test_validate_valid_config(self, validator, valid_config):
        """有効な設定を正常に検証できること"""
        # エラーが発生しないことを確認
        validator.validate_config(valid_config)

    def test_validate_config_with_all_fields(self, validator):
        """全フィールドを含む設定を検証できること"""
        config = {
            "strategy_name": "FullStrategy",
            "symbol": "ETH/USDT",
            "timeframe": "4h",
            "start_date": datetime(2024, 1, 1),
            "end_date": datetime(2024, 12, 31),
            "initial_capital": 50000.0,
            "commission_rate": 0.002,
            "strategy_config": {
                "strategy_type": "trend_following",
                "parameters": {"threshold": 0.02},
            },
        }

        validator.validate_config(config)

    def test_validate_config_minimal_fields(self, validator):
        """必須フィールドのみの設定を検証できること"""
        config = {
            "strategy_name": "MinimalStrategy",
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": datetime(2024, 1, 1),
            "end_date": datetime(2024, 1, 2),
            "initial_capital": 1000.0,
            "commission_rate": 0.001,
        }

        validator.validate_config(config)


class TestRequiredFieldsValidation:
    """必須フィールド検証テスト"""

    def test_validate_missing_strategy_name(self, validator, valid_config):
        """strategy_nameが欠けている場合のエラー"""
        del valid_config["strategy_name"]

        with pytest.raises(
            BacktestConfigValidationError,
            match="必須フィールド 'strategy_name' が見つかりません",
        ):
            validator.validate_config(valid_config)

    def test_validate_missing_symbol(self, validator, valid_config):
        """symbolが欠けている場合のエラー"""
        del valid_config["symbol"]

        with pytest.raises(
            BacktestConfigValidationError,
            match="必須フィールド 'symbol' が見つかりません",
        ):
            validator.validate_config(valid_config)

    def test_validate_missing_timeframe(self, validator, valid_config):
        """timeframeが欠けている場合のエラー"""
        del valid_config["timeframe"]

        with pytest.raises(
            BacktestConfigValidationError,
            match="必須フィールド 'timeframe' が見つかりません",
        ):
            validator.validate_config(valid_config)

    def test_validate_missing_dates(self, validator, valid_config):
        """日付フィールドが欠けている場合のエラー"""
        del valid_config["start_date"]

        with pytest.raises(
            BacktestConfigValidationError,
            match="必須フィールド 'start_date' が見つかりません",
        ):
            validator.validate_config(valid_config)

    def test_validate_missing_initial_capital(self, validator, valid_config):
        """initial_capitalが欠けている場合のエラー"""
        del valid_config["initial_capital"]

        with pytest.raises(
            BacktestConfigValidationError,
            match="必須フィールド 'initial_capital' が見つかりません",
        ):
            validator.validate_config(valid_config)

    def test_validate_missing_commission_rate(self, validator, valid_config):
        """commission_rateが欠けている場合のエラー"""
        del valid_config["commission_rate"]

        with pytest.raises(
            BacktestConfigValidationError,
            match="必須フィールド 'commission_rate' が見つかりません",
        ):
            validator.validate_config(valid_config)

    def test_validate_null_field(self, validator, valid_config):
        """フィールドがNullの場合のエラー"""
        valid_config["strategy_name"] = None

        with pytest.raises(
            BacktestConfigValidationError, match="フィールド 'strategy_name' がNullです"
        ):
            validator.validate_config(valid_config)


class TestFieldValueValidation:
    """フィールド値検証テスト"""

    def test_validate_empty_strategy_name(self, validator, valid_config):
        """空の戦略名のエラー"""
        valid_config["strategy_name"] = ""

        with pytest.raises(
            BacktestConfigValidationError,
            match="strategy_nameは空でない文字列である必要があります",
        ):
            validator.validate_config(valid_config)

    def test_validate_empty_symbol(self, validator, valid_config):
        """空のシンボルのエラー"""
        valid_config["symbol"] = ""

        with pytest.raises(
            BacktestConfigValidationError,
            match="symbolは空でない文字列である必要があります",
        ):
            validator.validate_config(valid_config)

    def test_validate_whitespace_only_strategy_name(self, validator, valid_config):
        """空白のみの戦略名のエラー"""
        valid_config["strategy_name"] = "   "

        with pytest.raises(
            BacktestConfigValidationError,
            match="strategy_nameは空でない文字列である必要があります",
        ):
            validator.validate_config(valid_config)

    def test_validate_non_string_strategy_name(self, validator, valid_config):
        """文字列でない戦略名のエラー"""
        valid_config["strategy_name"] = 12345

        with pytest.raises(
            BacktestConfigValidationError,
            match="strategy_nameは空でない文字列である必要があります",
        ):
            validator.validate_config(valid_config)


class TestTimeframeValidation:
    """時間軸検証テスト"""

    def test_validate_valid_timeframes(self, validator, valid_config):
        """有効な時間軸が全て受け入れられること"""
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

        for timeframe in valid_timeframes:
            valid_config["timeframe"] = timeframe
            validator.validate_config(valid_config)

    def test_validate_invalid_timeframe(self, validator, valid_config):
        """無効な時間軸のエラー"""
        valid_config["timeframe"] = "2h"

        with pytest.raises(
            BacktestConfigValidationError,
            match="timeframeは .* のいずれかである必要があります",
        ):
            validator.validate_config(valid_config)

    def test_validate_incorrect_timeframe_format(self, validator, valid_config):
        """不正な時間軸フォーマットのエラー"""
        valid_config["timeframe"] = "1hour"

        with pytest.raises(BacktestConfigValidationError):
            validator.validate_config(valid_config)


class TestDateValidation:
    """日付検証テスト"""

    def test_validate_valid_date_range(self, validator, valid_config):
        """有効な日付範囲を検証できること"""
        valid_config["start_date"] = datetime(2024, 1, 1)
        valid_config["end_date"] = datetime(2024, 1, 31)

        validator.validate_config(valid_config)

    def test_validate_start_date_after_end_date(self, validator, valid_config):
        """開始日が終了日より後の場合のエラー"""
        valid_config["start_date"] = datetime(2024, 2, 1)
        valid_config["end_date"] = datetime(2024, 1, 1)

        with pytest.raises(
            BacktestConfigValidationError,
            match="start_dateはend_dateより前である必要があります",
        ):
            validator.validate_config(valid_config)

    def test_validate_start_date_equals_end_date(self, validator, valid_config):
        """開始日と終了日が同じ場合のエラー"""
        same_date = datetime(2024, 1, 1)
        valid_config["start_date"] = same_date
        valid_config["end_date"] = same_date

        with pytest.raises(
            BacktestConfigValidationError,
            match="start_dateはend_dateより前である必要があります",
        ):
            validator.validate_config(valid_config)

    def test_validate_future_end_date(self, validator, valid_config):
        """未来の終了日のエラー"""
        valid_config["start_date"] = datetime(2024, 1, 1)
        valid_config["end_date"] = datetime.now() + timedelta(days=30)

        with pytest.raises(
            BacktestConfigValidationError,
            match="end_dateは現在時刻より前である必要があります",
        ):
            validator.validate_config(valid_config)

    def test_validate_string_date(self, validator, valid_config):
        """文字列形式の日付を検証できること"""
        valid_config["start_date"] = "2024-01-01"
        valid_config["end_date"] = "2024-01-31"

        validator.validate_config(valid_config)

    def test_validate_iso_format_date(self, validator, valid_config):
        """ISO形式の日付を検証できること"""
        valid_config["start_date"] = "2024-01-01T00:00:00"
        valid_config["end_date"] = "2024-01-31T23:59:59"

        validator.validate_config(valid_config)

    def test_validate_invalid_date_format(self, validator, valid_config):
        """無効な日付形式のエラー"""
        valid_config["start_date"] = "invalid-date"
        valid_config["end_date"] = datetime(2024, 1, 31)

        with pytest.raises(BacktestConfigValidationError, match="日付形式が無効です"):
            validator.validate_config(valid_config)


class TestNumericFieldValidation:
    """数値フィールド検証テスト"""

    def test_validate_valid_initial_capital(self, validator, valid_config):
        """有効な初期資金を検証できること"""
        valid_capitals = [100.0, 1000.0, 10000.0, 100000.0]

        for capital in valid_capitals:
            valid_config["initial_capital"] = capital
            validator.validate_config(valid_config)

    def test_validate_zero_initial_capital(self, validator, valid_config):
        """ゼロの初期資金のエラー"""
        valid_config["initial_capital"] = 0.0

        with pytest.raises(
            BacktestConfigValidationError,
            match="initial_capitalは正の数値である必要があります",
        ):
            validator.validate_config(valid_config)

    def test_validate_negative_initial_capital(self, validator, valid_config):
        """負の初期資金のエラー"""
        valid_config["initial_capital"] = -1000.0

        with pytest.raises(
            BacktestConfigValidationError,
            match="initial_capitalは正の数値である必要があります",
        ):
            validator.validate_config(valid_config)

    def test_validate_non_numeric_initial_capital(self, validator, valid_config):
        """数値でない初期資金のエラー"""
        valid_config["initial_capital"] = "not a number"

        with pytest.raises(
            BacktestConfigValidationError,
            match="initial_capitalは数値である必要があります",
        ):
            validator.validate_config(valid_config)

    def test_validate_valid_commission_rate(self, validator, valid_config):
        """有効な手数料率を検証できること"""
        valid_rates = [0.0, 0.001, 0.01, 0.1, 1.0]

        for rate in valid_rates:
            valid_config["commission_rate"] = rate
            validator.validate_config(valid_config)

    def test_validate_negative_commission_rate(self, validator, valid_config):
        """負の手数料率のエラー"""
        valid_config["commission_rate"] = -0.001

        with pytest.raises(
            BacktestConfigValidationError,
            match="commission_rateは0から1の間である必要があります",
        ):
            validator.validate_config(valid_config)

    def test_validate_commission_rate_greater_than_one(self, validator, valid_config):
        """1を超える手数料率のエラー"""
        valid_config["commission_rate"] = 1.5

        with pytest.raises(
            BacktestConfigValidationError,
            match="commission_rateは0から1の間である必要があります",
        ):
            validator.validate_config(valid_config)

    def test_validate_non_numeric_commission_rate(self, validator, valid_config):
        """数値でない手数料率のエラー"""
        valid_config["commission_rate"] = "invalid"

        with pytest.raises(
            BacktestConfigValidationError,
            match="commission_rateは数値である必要があります",
        ):
            validator.validate_config(valid_config)


class TestStrategyConfigValidation:
    """戦略設定検証テスト"""

    def test_validate_valid_strategy_config(self, validator, valid_config):
        """有効な戦略設定を検証できること"""
        valid_config["strategy_config"] = {
            "strategy_type": "momentum",
            "parameters": {"threshold": 0.02},
        }

        validator.validate_config(valid_config)

    def test_validate_none_strategy_config(self, validator, valid_config):
        """strategy_configがNoneでも許可されること"""
        valid_config["strategy_config"] = None

        validator.validate_config(valid_config)

    def test_validate_missing_strategy_config(self, validator, valid_config):
        """strategy_configが欠けていても許可されること"""
        del valid_config["strategy_config"]

        validator.validate_config(valid_config)

    def test_validate_non_dict_strategy_config(self, validator, valid_config):
        """辞書でない戦略設定のエラー"""
        valid_config["strategy_config"] = "not a dict"

        with pytest.raises(
            BacktestConfigValidationError,
            match="strategy_configは辞書である必要があります",
        ):
            validator.validate_config(valid_config)

    def test_validate_strategy_type_in_config(self, validator, valid_config):
        """戦略タイプが文字列であることを検証"""
        valid_config["strategy_config"] = {
            "strategy_type": "trend_following",
        }

        validator.validate_config(valid_config)

    def test_validate_non_string_strategy_type(self, validator, valid_config):
        """文字列でない戦略タイプのエラー"""
        valid_config["strategy_config"] = {
            "strategy_type": 123,
        }

        with pytest.raises(
            BacktestConfigValidationError,
            match="strategy_typeは文字列である必要があります",
        ):
            validator.validate_config(valid_config)


class TestMultipleErrors:
    """複数エラーテスト"""

    def test_validate_multiple_errors_collected(self, validator):
        """複数のエラーが収集されること"""
        config = {
            "strategy_name": "",  # エラー
            "symbol": "",  # エラー
            "timeframe": "invalid",  # エラー
            "start_date": datetime(2024, 2, 1),
            "end_date": datetime(2024, 1, 1),  # エラー（start_dateより前）
            "initial_capital": -1000.0,  # エラー
            "commission_rate": 1.5,  # エラー
        }

        with pytest.raises(BacktestConfigValidationError) as exc_info:
            validator.validate_config(config)

        # 複数のエラーメッセージが含まれていることを確認
        error = exc_info.value
        assert hasattr(error, "errors")
        assert len(error.errors) > 1


class TestDateParsing:
    """日付解析テスト"""

    def test_parse_datetime_object(self, validator):
        """datetimeオブジェクトの解析"""
        date = datetime(2024, 1, 1, 12, 30, 45)
        parsed = validator._parse_date(date)

        assert parsed == date

    def test_parse_iso_format_string(self, validator):
        """ISO形式文字列の解析"""
        date_str = "2024-01-01T12:30:45+00:00"
        parsed = validator._parse_date(date_str)

        assert isinstance(parsed, datetime)
        assert parsed.year == 2024
        assert parsed.month == 1
        assert parsed.day == 1

    def test_parse_simple_date_string(self, validator):
        """シンプルな日付文字列の解析"""
        date_str = "2024-01-01"
        parsed = validator._parse_date(date_str)

        assert isinstance(parsed, datetime)
        assert parsed.year == 2024
        assert parsed.month == 1
        assert parsed.day == 1

    def test_parse_invalid_date_type(self, validator):
        """無効な日付型のエラー"""
        with pytest.raises(ValueError, match="サポートされていない日付形式"):
            validator._parse_date(12345)

    def test_parse_invalid_date_string(self, validator):
        """無効な日付文字列のエラー"""
        with pytest.raises(ValueError):
            validator._parse_date("not-a-date")


class TestEdgeCases:
    """エッジケーステスト"""

    def test_validate_very_small_initial_capital(self, validator, valid_config):
        """非常に小さい初期資金"""
        valid_config["initial_capital"] = 0.01

        validator.validate_config(valid_config)

    def test_validate_very_large_initial_capital(self, validator, valid_config):
        """非常に大きい初期資金"""
        valid_config["initial_capital"] = 1000000000.0

        validator.validate_config(valid_config)

    def test_validate_zero_commission_rate(self, validator, valid_config):
        """ゼロの手数料率"""
        valid_config["commission_rate"] = 0.0

        validator.validate_config(valid_config)

    def test_validate_one_commission_rate(self, validator, valid_config):
        """1.0の手数料率（境界値）"""
        valid_config["commission_rate"] = 1.0

        validator.validate_config(valid_config)

    def test_validate_short_date_range(self, validator, valid_config):
        """非常に短い日付範囲"""
        valid_config["start_date"] = datetime(2024, 1, 1, 0, 0, 0)
        valid_config["end_date"] = datetime(2024, 1, 1, 1, 0, 0)

        validator.validate_config(valid_config)

    def test_validate_long_date_range(self, validator, valid_config):
        """非常に長い日付範囲"""
        valid_config["start_date"] = datetime(2020, 1, 1)
        valid_config["end_date"] = datetime(2024, 1, 1)

        validator.validate_config(valid_config)

    def test_validate_empty_strategy_config(self, validator, valid_config):
        """空の戦略設定"""
        valid_config["strategy_config"] = {}

        validator.validate_config(valid_config)

    def test_validate_complex_strategy_config(self, validator, valid_config):
        """複雑な戦略設定"""
        valid_config["strategy_config"] = {
            "strategy_type": "multi_indicator",
            "indicators": [
                {"type": "SMA", "period": 20},
                {"type": "RSI", "period": 14},
                {"type": "MACD", "fast": 12, "slow": 26, "signal": 9},
            ],
            "entry_conditions": [
                {"indicator": "SMA", "condition": "cross_above"},
                {"indicator": "RSI", "condition": "below", "value": 30},
            ],
            "exit_conditions": [
                {"indicator": "RSI", "condition": "above", "value": 70},
            ],
            "risk_management": {
                "stop_loss": 0.02,
                "take_profit": 0.05,
                "position_size": 0.1,
            },
        }

        validator.validate_config(valid_config)


class TestValidationErrorDetails:
    """検証エラー詳細テスト"""

    def test_validation_error_contains_message(self, validator, valid_config):
        """検証エラーにメッセージが含まれること"""
        valid_config["strategy_name"] = ""

        with pytest.raises(BacktestConfigValidationError) as exc_info:
            validator.validate_config(valid_config)

        assert "strategy_name" in str(exc_info.value)

    def test_validation_error_contains_errors_list(self, validator):
        """検証エラーにエラーリストが含まれること"""
        config = {
            "strategy_name": "",
            "symbol": "",
            "timeframe": "1h",
            "start_date": datetime(2024, 1, 1),
            "end_date": datetime(2024, 1, 2),
            "initial_capital": -100.0,
            "commission_rate": 2.0,
        }

        with pytest.raises(BacktestConfigValidationError) as exc_info:
            validator.validate_config(config)

        error = exc_info.value
        assert hasattr(error, "errors")
        assert isinstance(error.errors, list)
        assert len(error.errors) > 0




