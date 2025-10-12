"""
BacktestServiceのテスト
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from app.services.backtest.backtest_service import BacktestService
from app.services.backtest.backtest_data_service import BacktestDataService


class TestBacktestService:
    """BacktestServiceのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        self.mock_data_service = Mock(spec=BacktestDataService)
        self.service = BacktestService(self.mock_data_service)

    def test_init(self):
        """初期化のテスト"""
        assert self.service.data_service == self.mock_data_service
        assert self.service._data_service_initialized is False

    def test_ensure_data_service_initialized(self):
        """データサービス初期化のテスト"""
        with patch.object(self.service, '_initialize_data_service') as mock_init:
            self.service._ensure_data_service_initialized()
            mock_init.assert_called_once()

    def test_ensure_data_service_already_initialized(self):
        """既に初期化済みのデータサービステスト"""
        self.service._data_service_initialized = True

        with patch.object(self.service, '_initialize_data_service') as mock_init:
            self.service._ensure_data_service_initialized()
            mock_init.assert_not_called()

    def test_run_backtest_success(self):
        """バックテスト成功のテスト"""
        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-12-19",
            "strategy_config": {
                "strategy_type": "GENERATED_GA",
                "parameters": {"strategy_gene": {"id": "test_gene"}}
            }
        }

        with patch.object(self.service, '_prepare_backtest_data') as mock_prepare:
            with patch.object(self.service, '_execute_backtest') as mock_execute:
                with patch.object(self.service, '_calculate_performance_metrics') as mock_metrics:
                    mock_prepare.return_value = (Mock(), Mock())
                    mock_execute.return_value = Mock()
                    mock_metrics.return_value = {"total_return": 0.15}

                    result = self.service.run_backtest(backtest_config)

                    assert isinstance(result, dict)
                    assert "performance_metrics" in result

    def test_run_backtest_missing_config(self):
        """設定欠損バックテストのテスト"""
        backtest_config = {}  # 必須フィールドが欠けている

        with pytest.raises(ValueError, match="バックテスト設定が不完全です"):
            self.service.run_backtest(backtest_config)

    def test_prepare_backtest_data(self):
        """バックテストデータ準備のテスト"""
        symbol = "BTC/USDT:USDT"
        timeframe = "1h"
        start_date = "2024-01-01"
        end_date = "2024-12-19"

        mock_ohlcv = pd.DataFrame({
            'open': [100, 101, 99],
            'high': [102, 102, 100],
            'low': [98, 99, 98],
            'close': [101, 99, 100],
            'volume': [1000, 1100, 900]
        })

        self.mock_data_service.get_ohlcv_data.return_value = mock_ohlcv

        data, market_data = self.service._prepare_backtest_data(
            symbol, timeframe, start_date, end_date
        )

        assert data is not None
        assert market_data is not None

    def test_prepare_backtest_data_empty(self):
        """空のバックテストデータ準備のテスト"""
        symbol = "BTC/USDT:USDT"
        timeframe = "1h"
        start_date = "2024-01-01"
        end_date = "2024-12-19"

        self.mock_data_service.get_ohlcv_data.return_value = pd.DataFrame()

        data, market_data = self.service._prepare_backtest_data(
            symbol, timeframe, start_date, end_date
        )

        assert data.empty
        assert market_data is not None

    def test_execute_backtest(self):
        """バックテスト実行のテスト"""
        mock_strategy = Mock()
        mock_data = pd.DataFrame()
        mock_market_data = {}

        with patch('app.services.backtest.backtest_service.Backtest') as mock_backtest_class:
            mock_backtest = Mock()
            mock_backtest_class.return_value = mock_backtest
            mock_backtest.run.return_value = Mock()

            result = self.service._execute_backtest(mock_strategy, mock_data, mock_market_data)

            assert result is not None
            mock_backtest_class.assert_called_once()
            mock_backtest.run.assert_called_once()

    def test_calculate_performance_metrics(self):
        """パフォーメンスメトリクス計算のテスト"""
        mock_backtest_result = Mock()
        mock_backtest_result.total_return = 0.15
        mock_backtest_result.sharpe_ratio = 1.2
        mock_backtest_result.max_drawdown = 0.08
        mock_backtest_result.win_rate = 0.6
        mock_backtest_result.total_trades = 25

        metrics = self.service._calculate_performance_metrics(mock_backtest_result)

        assert isinstance(metrics, dict)
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "total_trades" in metrics

    def test_calculate_equity_curve(self):
        """エクイティカーブ計算のテスト"""
        mock_backtest_result = Mock()
        mock_backtest_result.equity_curve = [100, 110, 105, 120, 115]

        equity_curve = self.service._calculate_equity_curve(mock_backtest_result)

        assert isinstance(equity_curve, list)
        assert len(equity_curve) == 5

    def test_get_trade_history(self):
        """トレード履歴取得のテスト"""
        mock_backtest_result = Mock()
        mock_backtest_result.trades = [
            Mock(size=1, pnl=10, entry_time="2024-01-01", exit_time="2024-01-02"),
            Mock(size=-1, pnl=-5, entry_time="2024-01-03", exit_time="2024-01-04")
        ]

        trade_history = self.service._get_trade_history(mock_backtest_result)

        assert isinstance(trade_history, list)
        assert len(trade_history) == 2
        for trade in trade_history:
            assert "size" in trade
            assert "pnl" in trade
            assert "entry_time" in trade
            assert "exit_time" in trade

    def test_handle_backtest_error(self):
        """バックテストエラー処理のテスト"""
        error = Exception("Backtest error")

        with patch('app.services.backtest.backtest_service.logger') as mock_logger:
            result = self.service._handle_backtest_error(error, "test context")

            assert result is None
            mock_logger.error.assert_called_once()

    def test_validate_backtest_config_valid(self):
        """有効なバックテスト設定検証のテスト"""
        config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-12-19",
            "strategy_config": {
                "strategy_type": "GENERATED_GA",
                "parameters": {}
            }
        }

        # 例外が投げられないことを確認
        try:
            self.service._validate_backtest_config(config)
        except Exception:
            pytest.fail("例外が投げられました")

    def test_validate_backtest_config_missing_fields(self):
        """欠損フィールドのバックテスト設定検証のテスト"""
        config = {
            "symbol": "BTC/USDT:USDT",
            # timeframeが欠けている
            "start_date": "2024-01-01"
        }

        with pytest.raises(ValueError, match="timeframeが指定されていません"):
            self.service._validate_backtest_config(config)

    def test_validate_backtest_config_invalid_dates(self):
        """無効な日付のバックテスト設定検証のテスト"""
        config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-12-19",  # 終了日より後
            "end_date": "2024-01-01",   # 開始日より前
            "strategy_config": {
                "strategy_type": "GENERATED_GA",
                "parameters": {}
            }
        }

        with pytest.raises(ValueError, match="開始日は終了日より前である必要があります"):
            self.service._validate_backtest_config(config)

    def test_create_strategy_instance(self):
        """戦略インスタンス作成のテスト"""
        strategy_config = {
            "strategy_type": "GENERATED_GA",
            "parameters": {
                "strategy_gene": {"id": "test_gene", "indicators": []}
            }
        }
        data = pd.DataFrame()

        with patch('app.services.backtest.backtest_service.StrategyFactory') as mock_factory_class:
            mock_factory = Mock()
            mock_strategy_class = Mock()
            mock_strategy_instance = Mock()
            mock_factory_class.return_value.create_strategy_instance.return_value = mock_strategy_instance
            mock_factory_class.return_value.create_strategy_class.return_value = mock_strategy_class
            mock_factory_class.return_value.create_strategy_instance.return_value = mock_strategy_instance

            strategy = self.service._create_strategy_instance(strategy_config, data)

            assert strategy == mock_strategy_instance

    def test_create_strategy_instance_invalid_type(self):
        """無効な戦略タイプのインスタンス作成テスト"""
        strategy_config = {
            "strategy_type": "INVALID_TYPE",
            "parameters": {}
        }
        data = pd.DataFrame()

        with pytest.raises(ValueError, match="サポートされていない戦略タイプ"):
            self.service._create_strategy_instance(strategy_config, data)

    def test_get_market_data(self):
        """市場データ取得のテスト"""
        symbol = "BTC/USDT:USDT"
        timeframe = "1h"
        start_date = "2024-01-01"
        end_date = "2024-12-19"

        mock_ohlcv = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 102],
            'low': [98, 99],
            'close': [101, 99],
            'volume': [1000, 1100]
        })

        self.mock_data_service.get_ohlcv_data.return_value = mock_ohlcv

        market_data = self.service._get_market_data(symbol, timeframe, start_date, end_date)

        assert isinstance(market_data, dict)
        assert "ohlcv" in market_data
        assert "symbol" in market_data
        assert "timeframe" in market_data

    def test_initialize_data_service(self):
        """データサービス初期化のテスト"""
        # モックのデータリポジトリを作成
        mock_ohlcv_repo = Mock()
        mock_oi_repo = Mock()
        mock_fr_repo = Mock()

        with patch('app.services.backtest.backtest_service.OHLCVRepository') as mock_ohlcv_class:
            with patch('app.services.backtest.backtest_service.OpenInterestRepository') as mock_oi_class:
                with patch('app.services.backtest.backtest_service.FundingRateRepository') as mock_fr_class:
                    mock_ohlcv_class.return_value = mock_ohlcv_repo
                    mock_oi_class.return_value = mock_oi_repo
                    mock_fr_class.return_value = mock_fr_repo

                    self.service._initialize_data_service()

                    assert self.service._data_service_initialized is True
                    assert self.service.data_service is not None

    def test_cleanup(self):
        """クリーンアップのテスト"""
        self.service._data_service_initialized = True

        with patch.object(self.service.data_service, 'cleanup') as mock_cleanup:
            self.service.cleanup()

            mock_cleanup.assert_called_once()
            assert self.service._data_service_initialized is False

    def test_get_available_symbols(self):
        """利用可能シンボル取得のテスト"""
        expected_symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]

        with patch.object(self.service.data_service, 'get_available_symbols') as mock_get:
            mock_get.return_value = expected_symbols

            symbols = self.service.get_available_symbols()

            assert symbols == expected_symbols
            mock_get.assert_called_once()

    def test_get_timeframes(self):
        """利用可能時間足取得のテスト"""
        expected_timeframes = ["1m", "5m", "1h", "1d"]

        with patch.object(self.service.data_service, 'get_timeframes') as mock_get:
            mock_get.return_value = expected_timeframes

            timeframes = self.service.get_timeframes()

            assert timeframes == expected_timeframes
            mock_get.assert_called_once()

    def test_backtest_with_custom_parameters(self):
        """カスタムパラメータ付きバックテストのテスト"""
        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-12-19",
            "strategy_config": {
                "strategy_type": "GENERATED_GA",
                "parameters": {"strategy_gene": {"id": "test_gene"}}
            },
            "initial_capital": 100000,
            "commission_rate": 0.00055
        }

        with patch.object(self.service, '_prepare_backtest_data') as mock_prepare:
            with patch.object(self.service, '_execute_backtest') as mock_execute:
                with patch.object(self.service, '_calculate_performance_metrics') as mock_metrics:
                    mock_prepare.return_value = (Mock(), Mock())
                    mock_execute.return_value = Mock()
                    mock_metrics.return_value = {"total_return": 0.15}

                    result = self.service.run_backtest(backtest_config)

                    assert isinstance(result, dict)
                    assert result["initial_capital"] == 100000
                    assert result["commission_rate"] == 0.00055

    def test_concurrent_backtest_execution(self):
        """並列バックテスト実行のテスト"""
        configs = [
            {"config": "config1"},
            {"config": "config2"},
            {"config": "config3"}
        ]

        with patch('app.services.backtest.backtest_service.ThreadPoolExecutor') as mock_executor:
            with patch('app.services.backtest.backtest_service.as_completed') as mock_completed:
                mock_future1 = Mock()
                mock_future1.result.return_value = {"result": "test1"}
                mock_future2 = Mock()
                mock_future2.result.return_value = {"result": "test2"}
                mock_future3 = Mock()
                mock_future3.result.return_value = {"result": "test3"}

                mock_executor.return_value.__enter__.return_value.submit.side_effect = [
                    mock_future1, mock_future2, mock_future3
                ]
                mock_completed.return_value = [mock_future1, mock_future2, mock_future3]

                results = self.service.run_concurrent_backtests(configs)

                assert len(results) == 3
                assert mock_executor.return_value.__enter__.return_value.submit.call_count == 3