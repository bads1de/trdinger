"""
共通テストフィクスチャ

テスト全体で使用する共通のフィクスチャを定義します。
"""

import pytest
import pandas as pd
from typing import Dict, Any
from unittest.mock import Mock

from .data_generators import TestDataGenerator, PerformanceTestHelper


@pytest.fixture
def test_data_generator():
    """テストデータ生成器"""
    return TestDataGenerator()


@pytest.fixture
def performance_helper():
    """パフォーマンステストヘルパー"""
    return PerformanceTestHelper()


@pytest.fixture
def sample_ohlcv_data():
    """サンプルOHLCVデータ"""
    return TestDataGenerator.generate_ohlcv_data(length=100)


@pytest.fixture
def market_scenarios():
    """市場シナリオデータ"""
    return TestDataGenerator.generate_market_scenarios()


@pytest.fixture
def extreme_market_conditions():
    """極端な市場条件データ"""
    return TestDataGenerator.generate_extreme_market_conditions()


@pytest.fixture
def ga_config():
    """GAConfig"""
    return TestDataGenerator.generate_ga_config()


@pytest.fixture
def strategy_gene():
    """StrategyGene"""
    return TestDataGenerator.generate_strategy_gene()


@pytest.fixture
def tpsl_gene():
    """TPSLGene"""
    return TestDataGenerator.generate_tpsl_gene()


@pytest.fixture
def position_sizing_gene():
    """PositionSizingGene"""
    return TestDataGenerator.generate_position_sizing_gene()


@pytest.fixture
def mock_market_data_service():
    """モックされた市場データサービス"""
    service = Mock()
    service.fetch_ohlcv_data.return_value = [
        [1640995200000, 47000.0, 48000.0, 46500.0, 47500.0, 1000.0],
        [1641081600000, 47500.0, 48500.0, 47000.0, 48000.0, 1200.0],
    ]
    service.normalize_symbol.return_value = "BTC/USD:BTC"
    return service


@pytest.fixture
def performance_thresholds():
    """パフォーマンステストの閾値設定"""
    return {
        "small_dataset_time": 5.0,
        "medium_dataset_time": 10.0,
        "large_dataset_time": 30.0,
        "optimization_time": 60.0,
        "small_dataset_memory": 100,
        "medium_dataset_memory": 200,
        "large_dataset_memory": 500,
        "optimization_memory": 300,
        "memory_leak_threshold": 50,
    }


@pytest.fixture
def common_backtest_config():
    """共通のバックテスト設定"""
    return {
        "strategy_name": "GENERATED_TEST",
        "timeframe": "1d",
        "initial_capital": 100000,
        "commission_rate": 0.001,
        "strategy_config": {
            "strategy_type": "GENERATED_TEST",
            "parameters": {"strategy_gene": {}},
        },
    }
