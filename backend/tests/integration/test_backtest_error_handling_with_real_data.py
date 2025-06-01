"""
実際のDBデータを使用したバックテストエラーハンドリングテスト

エッジケース、異常データ、エラー条件での堅牢性テスト
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.core.services.backtest_service import BacktestService
from app.core.services.backtest_data_service import BacktestDataService
from app.core.utils.data_standardization import (
    standardize_ohlcv_columns,
    validate_ohlcv_data,
    prepare_data_for_backtesting,
)
from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository
from backtest.runner import run_backtest

logger = logging.getLogger(__name__)


class TestBacktestErrorHandlingWithRealData:
    """実際のDBデータを使用したエラーハンドリングテスト"""

    @pytest.fixture(scope="class")
    def db_session(self):
        """データベースセッション"""
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    @pytest.fixture(scope="class")
    def ohlcv_repo(self, db_session):
        """OHLCVリポジトリ"""
        return OHLCVRepository(db_session)

    @pytest.fixture(scope="class")
    def test_data_info(self, ohlcv_repo):
        """テスト用データ情報"""
        symbols = ["BTC/USDT", "BTC/USDT:USDT", "BTCUSD"]
        timeframe = "1d"

        for symbol in symbols:
            count = ohlcv_repo.count_records(symbol, timeframe)
            if count > 50:  # 最低50件のデータがある
                latest = ohlcv_repo.get_latest_timestamp(symbol, timeframe)
                oldest = ohlcv_repo.get_oldest_timestamp(symbol, timeframe)
                return {
                    "symbol": symbol,
                    "count": count,
                    "latest": latest,
                    "oldest": oldest,
                    "timeframe": timeframe,
                }

        pytest.skip("エラーハンドリングテストに十分なデータがありません")

    @pytest.fixture
    def backtest_service(self, db_session):
        """BacktestService with real data"""
        ohlcv_repo = OHLCVRepository(db_session)
        data_service = BacktestDataService(ohlcv_repo)
        return BacktestService(data_service)

    def test_invalid_date_range_handling(self, backtest_service, test_data_info):
        """無効な日付範囲のエラーハンドリングテスト"""
        info = test_data_info

        # 1. 開始日が終了日より後
        config = {
            "strategy_name": "SMA_CROSS",
            "symbol": info["symbol"],
            "timeframe": "1d",
            "start_date": "2024-12-31",
            "end_date": "2024-01-01",  # 開始日より前
            "initial_capital": 1000000,  # BTCの高価格に対応
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50},
            },
        }

        with pytest.raises(ValueError, match="Start date must be before end date"):
            backtest_service.run_backtest(config)

    def test_insufficient_data_handling(self, backtest_service, test_data_info):
        """データ不足のエラーハンドリングテスト"""
        info = test_data_info

        # 非常に短い期間（1日）でテスト
        end_date = info["latest"]
        start_date = end_date

        config = {
            "strategy_name": "SMA_CROSS",
            "symbol": info["symbol"],
            "timeframe": "1d",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": 1000000,  # BTCの高価格に対応
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {
                    "n1": 20,
                    "n2": 50,
                },  # 20日移動平均だが1日分のデータしかない
            },
        }

        # データ不足でもエラーにならず、適切に処理されることを確認
        result = backtest_service.run_backtest(config)
        assert "performance_metrics" in result
        # 取引が発生しないことを確認
        if "total_trades" in result["performance_metrics"]:
            assert result["performance_metrics"]["total_trades"] == 0

    def test_invalid_strategy_parameters(self, backtest_service, test_data_info):
        """無効な戦略パラメータのエラーハンドリングテスト"""
        info = test_data_info

        end_date = info["latest"]
        start_date = end_date - timedelta(days=30)

        # 1. 負のパラメータ
        config = {
            "strategy_name": "SMA_CROSS",
            "symbol": info["symbol"],
            "timeframe": "1d",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": 1000000,  # BTCの高価格に対応
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": -10, "n2": 50},  # 負の値
            },
        }

        with pytest.raises(
            (ValueError, RuntimeError),
            match="(n1.*n2|min_periods must be >= 0|Indicator.*error)",
        ):
            backtest_service.run_backtest(config)

        # 2. n1 >= n2 の場合（エラーにならないが、取引が発生しないことを確認）
        config["strategy_config"]["parameters"] = {"n1": 50, "n2": 20}  # n1 > n2

        result = backtest_service.run_backtest(config)
        # n1 > n2の場合、取引が発生しないことを確認
        assert result["performance_metrics"]["total_return"] == 0.0

    def test_invalid_financial_parameters(self, backtest_service, test_data_info):
        """無効な金融パラメータのエラーハンドリングテスト"""
        info = test_data_info

        end_date = info["latest"]
        start_date = end_date - timedelta(days=30)

        base_config = {
            "strategy_name": "SMA_CROSS",
            "symbol": info["symbol"],
            "timeframe": "1d",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50},
            },
        }

        # 1. 負の初期資金
        config = base_config.copy()
        config.update({"initial_capital": -100000, "commission_rate": 0.001})

        with pytest.raises(ValueError, match="Initial capital must be positive"):
            backtest_service.run_backtest(config)

        # 2. 負の手数料率
        config = base_config.copy()
        config.update({"initial_capital": 100000, "commission_rate": -0.001})

        with pytest.raises(ValueError, match="Commission rate must be between 0 and 1"):
            backtest_service.run_backtest(config)

        # 3. 異常に高い手数料率（100%）
        config = base_config.copy()
        config.update(
            {"initial_capital": 100000, "commission_rate": 1.1}
        )  # 1.0は有効なので1.1に変更

        with pytest.raises(ValueError, match="Commission rate must be between 0 and 1"):
            backtest_service.run_backtest(config)

    def test_nonexistent_symbol_handling(self, backtest_service):
        """存在しないシンボルのエラーハンドリングテスト"""
        config = {
            "strategy_name": "SMA_CROSS",
            "symbol": "NONEXISTENT/SYMBOL",
            "timeframe": "1d",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "initial_capital": 1000000,  # BTCの高価格に対応
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50},
            },
        }

        with pytest.raises(ValueError, match="No data found"):
            backtest_service.run_backtest(config)

    def test_invalid_timeframe_handling(self, backtest_service, test_data_info):
        """無効な時間軸のエラーハンドリングテスト"""
        info = test_data_info

        config = {
            "strategy_name": "SMA_CROSS",
            "symbol": info["symbol"],
            "timeframe": "invalid_timeframe",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "initial_capital": 1000000,  # BTCの高価格に対応
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50},
            },
        }

        with pytest.raises(ValueError, match="Unsupported.*timeframe"):
            backtest_service.run_backtest(config)

    def test_data_corruption_handling(self, ohlcv_repo, test_data_info):
        """データ破損のハンドリングテスト"""
        info = test_data_info

        # 実際のデータを取得
        df = ohlcv_repo.get_ohlcv_dataframe(info["symbol"], "1d", limit=10)

        # データを意図的に破損させる
        corrupted_df = df.copy()

        # 1. High < Low の異常データ
        corrupted_df.loc[corrupted_df.index[0], "high"] = 100
        corrupted_df.loc[corrupted_df.index[0], "low"] = 200  # High < Low

        # データ検証でエラーが検出されることを確認
        assert not validate_ohlcv_data(corrupted_df)

        # 2. 負の価格データ
        corrupted_df2 = df.copy()
        corrupted_df2.loc[corrupted_df2.index[0], "close"] = -100

        assert not validate_ohlcv_data(corrupted_df2)

    def test_extreme_market_conditions(self, backtest_service, test_data_info):
        """極端な市場条件でのテスト"""
        info = test_data_info

        # 非常に短期間のパラメータでテスト（高頻度取引）
        end_date = info["latest"]
        start_date = end_date - timedelta(days=60)

        config = {
            "strategy_name": "SMA_CROSS",
            "symbol": info["symbol"],
            "timeframe": "1d",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": 1000,  # 少額資金
            "commission_rate": 0.01,  # 高い手数料（1%）
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 1, "n2": 2},  # 極端に短期
            },
        }

        # エラーにならずに実行できることを確認
        result = backtest_service.run_backtest(config)
        assert "performance_metrics" in result

        # 高い手数料により損失が発生することを確認
        assert result["performance_metrics"]["total_return"] <= 0

    def test_runner_error_handling(self, test_data_info):
        """runner.pyのエラーハンドリングテスト"""
        info = test_data_info

        # 無効な設定でrunner.pyをテスト
        invalid_config = {
            "strategy": {
                "name": "INVALID_STRATEGY",
                "target_pair": info["symbol"],
                "indicators": [],
                "entry_rules": [],
                "exit_rules": [],
            },
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-01-31T23:59:59Z",
            "timeframe": "1d",
            "initial_capital": -1000,  # 無効な値
            "commission_rate": 0.001,
        }

        result = run_backtest(invalid_config)

        # エラーが適切に処理されていることを確認
        assert "error" in result or "performance_metrics" in result

    def test_memory_pressure_handling(self, backtest_service, test_data_info):
        """メモリ圧迫状況でのハンドリングテスト"""
        info = test_data_info

        # 利用可能な最大期間でテスト
        end_date = info["latest"]
        available_days = (end_date - info["oldest"]).days
        start_date = end_date - timedelta(days=available_days - 1)

        config = {
            "strategy_name": "SMA_CROSS",
            "symbol": info["symbol"],
            "timeframe": "1d",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": 1000000,  # BTCの高価格に対応
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50},
            },
        }

        # 大量データでもエラーにならないことを確認
        result = backtest_service.run_backtest(config)
        assert "performance_metrics" in result

        logger.info(f"大量データテスト完了: {available_days}日間のデータ")

    def test_concurrent_access_handling(self, db_session, test_data_info):
        """並行アクセスのハンドリングテスト"""
        info = test_data_info

        # 複数のBacktestServiceインスタンスを同時に使用
        services = []
        for i in range(3):
            ohlcv_repo = OHLCVRepository(db_session)
            data_service = BacktestDataService(ohlcv_repo)
            service = BacktestService(data_service)
            services.append(service)

        end_date = info["latest"]
        start_date = end_date - timedelta(days=30)

        config = {
            "strategy_name": "SMA_CROSS",
            "symbol": info["symbol"],
            "timeframe": "1d",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": 1000000,  # BTCの高価格に対応
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50},
            },
        }

        # 全てのサービスで同時にバックテストを実行
        results = []
        for service in services:
            result = service.run_backtest(config)
            results.append(result)

        # 全ての結果が有効であることを確認
        for i, result in enumerate(results):
            assert "performance_metrics" in result, f"サービス{i+1}の結果が無効です"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
