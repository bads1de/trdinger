"""
バックテスト設定検証サービステスト

BacktestConfig Pydanticモデルの機能をテストします。
"""

import logging
from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from app.services.backtest.backtest_config import (
    BacktestConfig,
    BacktestConfigValidationError,
)

logger = logging.getLogger(__name__)


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
            "strategy_type": "MANUAL",
            "parameters": {},
        },
    }


class TestConfigValidation:
    """設定検証テスト"""

    def test_validate_valid_config(self, valid_config):
        """有効な設定を正常に検証できること"""
        config = BacktestConfig(**valid_config)
        assert config.strategy_name == "TestStrategy"

    def test_validate_config_with_all_fields(self):
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
                "strategy_type": "MANUAL",
                "parameters": {"threshold": 0.02},
            },
        }
        result = BacktestConfig(**config)
        assert result.strategy_name == "FullStrategy"


class TestRequiredFieldsValidation:
    """必須フィールド検証テスト"""

    def test_validate_missing_strategy_name(self, valid_config):
        """strategy_nameが欠けている場合のエラー"""
        del valid_config["strategy_name"]
        with pytest.raises(ValidationError):
            BacktestConfig(**valid_config)

    def test_validate_missing_symbol(self, valid_config):
        """symbolが欠けている場合のエラー"""
        del valid_config["symbol"]
        with pytest.raises(ValidationError):
            BacktestConfig(**valid_config)

    def test_validate_missing_timeframe(self, valid_config):
        """timeframeが欠けている場合のエラー"""
        del valid_config["timeframe"]
        with pytest.raises(ValidationError):
            BacktestConfig(**valid_config)

    def test_validate_missing_dates(self, valid_config):
        """日付フィールドが欠けている場合のエラー"""
        del valid_config["start_date"]
        with pytest.raises(ValidationError):
            BacktestConfig(**valid_config)

    def test_validate_missing_initial_capital(self, valid_config):
        """initial_capitalが欠けている場合のエラー"""
        del valid_config["initial_capital"]
        with pytest.raises(ValidationError):
            BacktestConfig(**valid_config)


class TestFieldValueValidation:
    """フィールド値検証テスト"""

    def test_validate_empty_strategy_name(self, valid_config):
        """空の戦略名のエラー"""
        valid_config["strategy_name"] = ""
        with pytest.raises(ValidationError):
            BacktestConfig(**valid_config)

    def test_validate_non_string_strategy_name(self, valid_config):
        """文字列でない戦略名のエラー"""
        valid_config["strategy_name"] = 12345
        with pytest.raises(ValidationError):
            BacktestConfig(**valid_config)


class TestTimeframeValidation:
    """時間軸検証テスト"""

    def test_validate_valid_timeframes(self, valid_config):
        """有効な時間軸が全て受け入れられること"""
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        for timeframe in valid_timeframes:
            valid_config["timeframe"] = timeframe
            config = BacktestConfig(**valid_config)
            assert config.timeframe == timeframe

    def test_validate_invalid_timeframe(self, valid_config):
        """無効な時間軸のエラー"""
        valid_config["timeframe"] = "2h"
        with pytest.raises(ValidationError, match="timeframe"):
            BacktestConfig(**valid_config)


class TestDateValidation:
    """日付検証テスト"""

    def test_validate_valid_date_range(self, valid_config):
        """有効な日付範囲を検証できること"""
        valid_config["start_date"] = datetime(2024, 1, 1)
        valid_config["end_date"] = datetime(2024, 1, 31)
        config = BacktestConfig(**valid_config)
        assert config.start_date < config.end_date

    def test_validate_start_date_after_end_date(self, valid_config):
        """開始日が終了日より後の場合のエラー"""
        valid_config["start_date"] = datetime(2024, 2, 1)
        valid_config["end_date"] = datetime(2024, 1, 1)
        with pytest.raises(
            BacktestConfigValidationError,
            match="start_dateはend_dateより前である必要があります",
        ):
            BacktestConfig(**valid_config)

    def test_validate_start_date_equals_end_date(self, valid_config):
        """開始日と終了日が同じ場合のエラー"""
        same_date = datetime(2024, 1, 1)
        valid_config["start_date"] = same_date
        valid_config["end_date"] = same_date
        with pytest.raises(
            BacktestConfigValidationError,
            match="start_dateはend_dateより前である必要があります",
        ):
            BacktestConfig(**valid_config)

    def test_validate_future_end_date(self, valid_config):
        """未来の終了日のエラー"""
        valid_config["start_date"] = datetime(2024, 1, 1)
        valid_config["end_date"] = datetime.now() + timedelta(days=30)
        with pytest.raises(
            BacktestConfigValidationError,
            match="end_dateは現在時刻より前である必要があります",
        ):
            BacktestConfig(**valid_config)

    def test_validate_string_date(self, valid_config):
        """文字列形式の日付を検証できること"""
        valid_config["start_date"] = "2024-01-01"
        valid_config["end_date"] = "2024-01-31"
        config = BacktestConfig(**valid_config)
        assert isinstance(config.start_date, datetime)

    def test_validate_iso_format_date(self, valid_config):
        """ISO形式の日付を検証できること"""
        valid_config["start_date"] = "2024-01-01T00:00:00"
        valid_config["end_date"] = "2024-01-31T23:59:59"
        config = BacktestConfig(**valid_config)
        assert isinstance(config.start_date, datetime)


class TestNumericFieldValidation:
    """数値フィールド検証テスト"""

    def test_validate_valid_initial_capital(self, valid_config):
        """有効な初期資金を検証できること"""
        valid_capitals = [100.0, 1000.0, 10000.0, 100000.0]
        for capital in valid_capitals:
            valid_config["initial_capital"] = capital
            config = BacktestConfig(**valid_config)
            assert config.initial_capital == capital

    def test_validate_zero_initial_capital(self, valid_config):
        """ゼロの初期資金のエラー"""
        valid_config["initial_capital"] = 0.0
        with pytest.raises(ValidationError):
            BacktestConfig(**valid_config)

    def test_validate_negative_initial_capital(self, valid_config):
        """負の初期資金のエラー"""
        valid_config["initial_capital"] = -1000.0
        with pytest.raises(ValidationError):
            BacktestConfig(**valid_config)

    def test_validate_valid_commission_rate(self, valid_config):
        """有効な手数料率を検証できること"""
        valid_rates = [0.0, 0.001, 0.01, 0.1, 1.0]
        for rate in valid_rates:
            valid_config["commission_rate"] = rate
            config = BacktestConfig(**valid_config)
            assert config.commission_rate == rate

    def test_validate_negative_commission_rate(self, valid_config):
        """負の手数料率のエラー"""
        valid_config["commission_rate"] = -0.001
        with pytest.raises(ValidationError):
            BacktestConfig(**valid_config)

    def test_validate_commission_rate_greater_than_one(self, valid_config):
        """1を超える手数料率のエラー"""
        valid_config["commission_rate"] = 1.5
        with pytest.raises(ValidationError):
            BacktestConfig(**valid_config)


class TestEdgeCases:
    """エッジケーステスト"""

    def test_validate_very_small_initial_capital(self, valid_config):
        """非常に小さい初期資金"""
        valid_config["initial_capital"] = 0.01
        config = BacktestConfig(**valid_config)
        assert config.initial_capital == 0.01

    def test_validate_very_large_initial_capital(self, valid_config):
        """非常に大きい初期資金"""
        valid_config["initial_capital"] = 1000000000.0
        config = BacktestConfig(**valid_config)
        assert config.initial_capital == 1000000000.0

    def test_validate_zero_commission_rate(self, valid_config):
        """ゼロの手数料率"""
        valid_config["commission_rate"] = 0.0
        config = BacktestConfig(**valid_config)
        assert config.commission_rate == 0.0

    def test_validate_one_commission_rate(self, valid_config):
        """1.0の手数料率（境界値）"""
        valid_config["commission_rate"] = 1.0
        config = BacktestConfig(**valid_config)
        assert config.commission_rate == 1.0

    def test_validate_short_date_range(self, valid_config):
        """非常に短い日付範囲"""
        valid_config["start_date"] = datetime(2024, 1, 1, 0, 0, 0)
        valid_config["end_date"] = datetime(2024, 1, 1, 1, 0, 0)
        config = BacktestConfig(**valid_config)
        assert config.start_date < config.end_date

    def test_validate_long_date_range(self, valid_config):
        """非常に長い日付範囲"""
        valid_config["start_date"] = datetime(2020, 1, 1)
        valid_config["end_date"] = datetime(2024, 1, 1)
        config = BacktestConfig(**valid_config)
        assert config.start_date < config.end_date


class TestValidationErrorDetails:
    """検証エラー詳細テスト"""

    def test_validation_error_contains_message(self, valid_config):
        """検証エラーにメッセージが含まれること"""
        valid_config["strategy_name"] = ""
        with pytest.raises(ValidationError) as exc_info:
            BacktestConfig(**valid_config)
        assert "strategy_name" in str(exc_info.value)

    def test_date_validation_error_contains_errors_list(self, valid_config):
        """日付検証エラーにエラーリストが含まれること"""
        valid_config["start_date"] = datetime(2024, 2, 1)
        valid_config["end_date"] = datetime(2024, 1, 1)
        with pytest.raises(BacktestConfigValidationError) as exc_info:
            BacktestConfig(**valid_config)
        error = exc_info.value
        assert hasattr(error, "errors")
        assert isinstance(error.errors, list)
        assert len(error.errors) > 0
