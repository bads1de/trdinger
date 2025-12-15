"""
バックテスト設定スキーマのテスト

Pydanticモデルによるバリデーションとシリアライズを検証します。
"""

from datetime import datetime
from typing import Any, Dict

import pytest
from pydantic import ValidationError

# 実装予定のモジュール（まだ存在しない）
from app.services.backtest.backtest_config import (
    BacktestConfig,
    StrategyConfig,
    GeneratedGAParameters,
)


@pytest.fixture
def valid_strategy_gene_dict() -> Dict[str, Any]:
    """有効な戦略遺伝子の辞書（簡略版）"""
    return {
        "indicators": [],
        "entry_conditions": [],
        "exit_conditions": [],
        "risk_management": {"position_size": 0.1},
    }


@pytest.fixture
def valid_strategy_config_dict(valid_strategy_gene_dict) -> Dict[str, Any]:
    """有効な戦略設定の辞書"""
    return {
        "strategy_type": "GENERATED_GA",
        "parameters": {
            "strategy_gene": valid_strategy_gene_dict,
            "ml_filter_enabled": False,
        },
    }


@pytest.fixture
def valid_backtest_config_dict(valid_strategy_config_dict) -> Dict[str, Any]:
    """有効なバックテスト設定の辞書"""
    return {
        "strategy_name": "Test Strategy",
        "symbol": "BTC/USDT:USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "initial_capital": 10000.0,
        "commission_rate": 0.001,
        "strategy_config": valid_strategy_config_dict,
    }


def test_strategy_config_validation(valid_strategy_config_dict):
    """StrategyConfigのバリデーションテスト"""
    config = StrategyConfig(**valid_strategy_config_dict)
    assert config.strategy_type == "GENERATED_GA"
    assert isinstance(config.parameters, GeneratedGAParameters)
    assert config.parameters.ml_filter_enabled is False


def test_backtest_config_validation(valid_backtest_config_dict):
    """BacktestConfigのバリデーションテスト"""
    config = BacktestConfig(**valid_backtest_config_dict)
    assert config.strategy_name == "Test Strategy"
    assert config.symbol == "BTC/USDT:USDT"
    # 日付は自動的にdatetimeに変換されるべき
    assert isinstance(config.start_date, datetime)
    assert config.commission_rate == 0.001
    assert isinstance(config.strategy_config, StrategyConfig)


def test_missing_required_fields():
    """必須フィールド欠落時のエラーテスト"""
    invalid_data = {
        "symbol": "BTC/USDT:USDT",
        # strategy_nameが欠落
    }
    with pytest.raises(ValidationError) as excinfo:
        BacktestConfig(**invalid_data)

    assert "strategy_name" in str(excinfo.value)


def test_invalid_data_types(valid_backtest_config_dict):
    """不正なデータ型のエラーテスト"""
    invalid_data = valid_backtest_config_dict.copy()
    invalid_data["initial_capital"] = "invalid_number"  # 数値であるべき場所へ文字列

    with pytest.raises(ValidationError):
        BacktestConfig(**invalid_data)


def test_serialization(valid_backtest_config_dict):
    """辞書へのシリアライズテスト"""
    config = BacktestConfig(**valid_backtest_config_dict)
    data = config.model_dump()

    assert data["strategy_name"] == "Test Strategy"
    assert data["strategy_config"]["strategy_type"] == "GENERATED_GA"
    # 日付はdatetimeオブジェクトのまま（json化する際は別途対応が必要だがdumpでは維持）
    assert isinstance(data["start_date"], datetime)


