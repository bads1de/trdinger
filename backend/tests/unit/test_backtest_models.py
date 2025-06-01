"""
バックテスト関連データベースモデルのテスト
"""

import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.connection import Base
from database.models import BacktestResult, StrategyTemplate


@pytest.fixture
def test_db():
    """テスト用のインメモリデータベース"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    yield session
    session.close()


class TestBacktestResult:
    """BacktestResultモデルのテスト"""

    def test_create_backtest_result(self, test_db):
        """バックテスト結果の作成テスト"""
        # テストデータ
        config_data = {
            "strategy": "SMA_CROSS",
            "parameters": {"n1": 20, "n2": 50},
            "initial_capital": 100000,
            "commission_rate": 0.001
        }
        
        performance_data = {
            "total_return": 25.5,
            "sharpe_ratio": 1.2,
            "max_drawdown": -15.3,
            "win_rate": 65.0,
            "total_trades": 45
        }
        
        equity_curve_data = [
            {"timestamp": "2024-01-01T00:00:00Z", "equity": 100000},
            {"timestamp": "2024-01-02T00:00:00Z", "equity": 101000}
        ]
        
        trade_history_data = [
            {
                "entry_time": "2024-01-01T10:00:00Z",
                "exit_time": "2024-01-01T15:00:00Z",
                "side": "buy",
                "size": 1.0,
                "entry_price": 50000,
                "exit_price": 51000,
                "pnl": 1000
            }
        ]

        # BacktestResultインスタンスを作成
        backtest_result = BacktestResult(
            strategy_name="SMA_CROSS",
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
            initial_capital=100000.0,
            config_json=config_data,
            performance_metrics=performance_data,
            equity_curve=equity_curve_data,
            trade_history=trade_history_data
        )

        # データベースに保存
        test_db.add(backtest_result)
        test_db.commit()

        # 検証
        assert backtest_result.id is not None
        assert backtest_result.strategy_name == "SMA_CROSS"
        assert backtest_result.symbol == "BTC/USDT"
        assert backtest_result.timeframe == "1h"
        assert backtest_result.initial_capital == 100000.0
        assert backtest_result.config_json["strategy"] == "SMA_CROSS"
        assert backtest_result.performance_metrics["total_return"] == 25.5
        assert len(backtest_result.equity_curve) == 2
        assert len(backtest_result.trade_history) == 1

    def test_backtest_result_to_dict(self, test_db):
        """BacktestResultのto_dict()メソッドテスト"""
        backtest_result = BacktestResult(
            strategy_name="TEST_STRATEGY",
            symbol="BTC/USDT",
            timeframe="1d",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 31, tzinfo=timezone.utc),
            initial_capital=50000.0,
            config_json={"test": "config"},
            performance_metrics={"return": 10.0},
            equity_curve=[],
            trade_history=[]
        )

        test_db.add(backtest_result)
        test_db.commit()

        # to_dict()メソッドをテスト
        result_dict = backtest_result.to_dict()

        assert result_dict["strategy_name"] == "TEST_STRATEGY"
        assert result_dict["symbol"] == "BTC/USDT"
        assert result_dict["timeframe"] == "1d"
        assert result_dict["initial_capital"] == 50000.0
        assert "created_at" in result_dict


class TestStrategyTemplate:
    """StrategyTemplateモデルのテスト"""

    def test_create_strategy_template(self, test_db):
        """戦略テンプレートの作成テスト"""
        config_data = {
            "strategy_type": "SMA_CROSS",
            "parameters": {
                "n1": {"type": "int", "default": 20, "min": 5, "max": 50},
                "n2": {"type": "int", "default": 50, "min": 20, "max": 200}
            },
            "entry_rules": ["sma1 > sma2"],
            "exit_rules": ["sma1 < sma2"]
        }

        strategy_template = StrategyTemplate(
            name="SMA Cross 20/50",
            description="Simple Moving Average Crossover Strategy",
            category="trend_following",
            config_json=config_data,
            is_public=True
        )

        test_db.add(strategy_template)
        test_db.commit()

        # 検証
        assert strategy_template.id is not None
        assert strategy_template.name == "SMA Cross 20/50"
        assert strategy_template.category == "trend_following"
        assert strategy_template.is_public is True
        assert strategy_template.config_json["strategy_type"] == "SMA_CROSS"

    def test_strategy_template_unique_name(self, test_db):
        """戦略テンプレート名の一意性テスト"""
        template1 = StrategyTemplate(
            name="Unique Strategy",
            description="First template",
            config_json={"test": "config1"}
        )
        
        template2 = StrategyTemplate(
            name="Unique Strategy",  # 同じ名前
            description="Second template",
            config_json={"test": "config2"}
        )

        test_db.add(template1)
        test_db.commit()

        # 同じ名前のテンプレートを追加しようとするとエラーになることを確認
        test_db.add(template2)
        with pytest.raises(Exception):  # IntegrityError等
            test_db.commit()

    def test_strategy_template_to_dict(self, test_db):
        """StrategyTemplateのto_dict()メソッドテスト"""
        template = StrategyTemplate(
            name="Test Template",
            description="Test description",
            category="test",
            config_json={"test": "config"},
            is_public=False
        )

        test_db.add(template)
        test_db.commit()

        # to_dict()メソッドをテスト
        template_dict = template.to_dict()

        assert template_dict["name"] == "Test Template"
        assert template_dict["description"] == "Test description"
        assert template_dict["category"] == "test"
        assert template_dict["is_public"] is False
        assert "created_at" in template_dict
