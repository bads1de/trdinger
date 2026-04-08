"""
データベースモデルのテスト

database/models.pyのSQLAlchemyモデルクラスをテストします。
"""

from datetime import datetime

import pytest

from database.models import (
    BacktestResult,
    FundingRateData,
    GAExperiment,
    GeneratedStrategy,
    LongShortRatioData,
    OHLCVData,
    OpenInterestData,
)


class TestOHLCVData:
    """OHLCVDataモデルのテスト"""

    def test_table_name(self):
        """テーブル名が正しい"""
        assert OHLCVData.__tablename__ == "ohlcv_data"

    def test_repr(self):
        """__repr__メソッドが正しく動作する"""
        model = OHLCVData(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            timestamp=datetime(2024, 1, 1, 12, 0),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0,
        )
        repr_str = repr(model)

        assert "BTC/USDT:USDT" in repr_str
        assert "1h" in repr_str
        assert "50500.0" in repr_str


class TestFundingRateData:
    """FundingRateDataモデルのテスト"""

    def test_table_name(self):
        """テーブル名が正しい"""
        assert FundingRateData.__tablename__ == "funding_rate_data"

    def test_repr(self):
        """__repr__メソッドが正しく動作する"""
        model = FundingRateData(
            symbol="BTC/USDT:USDT",
            funding_rate=0.0001,
            funding_timestamp=datetime(2024, 1, 1, 8, 0),
            timestamp=datetime(2024, 1, 1, 8, 0),
        )
        repr_str = repr(model)

        assert "BTC/USDT:USDT" in repr_str
        assert "0.0001" in repr_str


class TestOpenInterestData:
    """OpenInterestDataモデルのテスト"""

    def test_table_name(self):
        """テーブル名が正しい"""
        assert OpenInterestData.__tablename__ == "open_interest_data"

    def test_repr(self):
        """__repr__メソッドが正しく動作する"""
        model = OpenInterestData(
            symbol="BTC/USDT:USDT",
            open_interest_value=1000000.0,
            data_timestamp=datetime(2024, 1, 1, 12, 0),
            timestamp=datetime(2024, 1, 1, 12, 0),
        )
        repr_str = repr(model)

        assert "BTC/USDT:USDT" in repr_str
        assert "1000000.0" in repr_str


class TestLongShortRatioData:
    """LongShortRatioDataモデルのテスト"""

    def test_table_name(self):
        """テーブル名が正しい"""
        assert LongShortRatioData.__tablename__ == "long_short_ratio_data"

    def test_repr(self):
        """__repr__メソッドが正しく動作する"""
        model = LongShortRatioData(
            symbol="BTC/USDT:USDT",
            period="1h",
            buy_ratio=0.6,
            sell_ratio=0.4,
            timestamp=datetime(2024, 1, 1, 12, 0),
        )
        repr_str = repr(model)

        assert "BTC/USDT:USDT" in repr_str
        assert "1h" in repr_str
        assert "0.6000" in repr_str
        assert "0.4000" in repr_str


class TestBacktestResult:
    """BacktestResultモデルのテスト"""

    def test_table_name(self):
        """テーブル名が正しい"""
        assert BacktestResult.__tablename__ == "backtest_results"

    def test_repr(self):
        """__repr__メソッドが正しく動作する"""
        model = BacktestResult(
            strategy_name="test_strategy",
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            initial_capital=10000.0,
            config_json={"param": "value"},
            performance_metrics={"return": 0.1},
            equity_curve=[10000, 11000],
            trade_history=[],
            status="completed",
        )
        repr_str = repr(model)

        assert "test_strategy" in repr_str
        assert "BTC/USDT:USDT" in repr_str
        assert "1h" in repr_str


class TestGAExperiment:
    """GAExperimentモデルのテスト"""

    def test_table_name(self):
        """テーブル名が正しい"""
        assert GAExperiment.__tablename__ == "ga_experiments"

    def test_repr(self):
        """__repr__メソッドが正しく動作する"""
        model = GAExperiment(
            experiment_id="exp-123",
            name="Test Experiment",
            config={"population_size": 100},
            status="running",
            progress=0.5,
        )
        repr_str = repr(model)

        assert "Test Experiment" in repr_str
        assert "running" in repr_str
        assert "0.50" in repr_str


class TestGeneratedStrategy:
    """GeneratedStrategyモデルのテスト"""

    def test_table_name(self):
        """テーブル名が正しい"""
        assert GeneratedStrategy.__tablename__ == "generated_strategies"

    def test_repr(self):
        """__repr__メソッドが正しく動作する"""
        model = GeneratedStrategy(
            experiment_id=1,
            gene_data={"conditions": []},
            generation=10,
            fitness_score=0.85,
        )
        repr_str = repr(model)

        assert "1" in repr_str
        assert "10" in repr_str
        assert "0.8500" in repr_str

    def test_repr_without_fitness(self):
        """fitness_scoreがNoneの場合の__repr__"""
        model = GeneratedStrategy(
            experiment_id=1,
            gene_data={"conditions": []},
            generation=10,
            fitness_score=None,
        )
        repr_str = repr(model)

        assert "None" in repr_str


class TestModelInheritance:
    """モデル継承のテスト"""

    def test_all_models_inherit_from_base(self):
        """すべてのモデルがBaseを継承している"""
        from database.connection import Base

        assert issubclass(OHLCVData, Base)
        assert issubclass(FundingRateData, Base)
        assert issubclass(OpenInterestData, Base)
        assert issubclass(LongShortRatioData, Base)
        assert issubclass(BacktestResult, Base)
        assert issubclass(GAExperiment, Base)
        assert issubclass(GeneratedStrategy, Base)


class TestModelAttributes:
    """モデル属性のテスト"""

    def test_ohlcv_has_required_columns(self):
        """OHLCVDataに必要なカラムがある"""
        assert hasattr(OHLCVData, "id")
        assert hasattr(OHLCVData, "symbol")
        assert hasattr(OHLCVData, "timeframe")
        assert hasattr(OHLCVData, "timestamp")
        assert hasattr(OHLCVData, "open")
        assert hasattr(OHLCVData, "high")
        assert hasattr(OHLCVData, "low")
        assert hasattr(OHLCVData, "close")
        assert hasattr(OHLCVData, "volume")

    def test_backtest_result_has_json_columns(self):
        """BacktestResultにJSONカラムがある"""
        assert hasattr(BacktestResult, "config_json")
        assert hasattr(BacktestResult, "performance_metrics")
        assert hasattr(BacktestResult, "equity_curve")
        assert hasattr(BacktestResult, "trade_history")

    def test_ga_experiment_has_relationships(self):
        """GAExperimentにリレーションがある"""
        assert hasattr(GAExperiment, "strategies")

    def test_generated_strategy_has_relationships(self):
        """GeneratedStrategyにリレーションがある"""
        assert hasattr(GeneratedStrategy, "experiment")
        assert hasattr(GeneratedStrategy, "backtest_result")
