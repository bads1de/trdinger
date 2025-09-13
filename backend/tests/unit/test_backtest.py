"""
統合バックテスト

バックテスト実行と戦略生成の機能を統合テスト
TDD原則に基づき、各機能を包括的にテスト
"""

import warnings
import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd

from backend.app.services.backtest.execution.backtest_executor import BacktestExecutor, BacktestExecutionError
from backend.app.services.auto_strategy.generators.strategy_factory import StrategyFactory
from backend.app.services.auto_strategy.models.strategy_models import StrategyGene, IndicatorGene, Condition, ConditionGroup


class TestBacktestExecutorIntegrated:
    """BacktestExecutorの統合テスト"""

    @pytest.fixture
    def executor(self):
        """BacktestExecutorインスタンスのフィクスチャ"""
        data_service = Mock()
        return BacktestExecutor(data_service)

    def test_backtest_successful_workflow(self, executor):
        """バックテストの成功ワークフロー"""
        # Mockデータ
        data = Mock()
        data.empty = False
        executor.data_service.get_data_for_backtest.return_value = data

        # Mock戦略クラス
        strategy_class = Mock()
        strategy_parameters = {}

        symbol = "BTCUSDT"
        timeframe = "1h"
        start_date = datetime.now()
        end_date = datetime.now()
        initial_capital = 1000000.0
        commission_rate = 0.001

        # Mockバックテスト統計結果
        stats = Mock()

        # FractionalBacktestをMock
        with patch('backend.app.services.backtest.execution.backtest_executor.FractionalBacktest') as mock_fractional_bt:
            mock_bt_instance = Mock()
            mock_bt_instance.run.return_value = stats
            mock_fractional_bt.return_value = mock_bt_instance

            result = executor.execute_backtest(
                strategy_class, strategy_parameters, symbol, timeframe,
                start_date, end_date, initial_capital, commission_rate
            )

            assert result == stats
            mock_fractional_bt.assert_called_once()
            mock_bt_instance.run.assert_called_once_with(**strategy_parameters)

    def test_backtest_edge_cases(self, executor):
        """バックテストのエッジケース"""
        # 高額価格のテスト
        data = pd.DataFrame({
            'Open': [95000.0, 96000.0],
            'High': [97000.0, 98000.0],
            'Low': [94000.0, 95000.0],
            'Close': [96000.0, 97000.0],
            'Volume': [100.0, 110.0]
        })
        executor.data_service.get_data_for_backtest.return_value = data

        strategy_class = Mock()
        strategy_parameters = {}

        symbol = "BTCUSDT"
        timeframe = "1h"
        start_date = datetime.now()
        end_date = datetime.now()
        initial_capital = 10000.0
        commission_rate = 0.001

        stats = Mock()

        # 高額価格での警告なしを確認
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with patch('backend.app.services.backtest.execution.backtest_executor.FractionalBacktest') as mock_fractional_bt:
                mock_bt_instance = Mock()
                mock_bt_instance.run.return_value = stats
                mock_fractional_bt.return_value = mock_bt_instance

                executor.execute_backtest(
                    strategy_class, strategy_parameters, symbol, timeframe,
                    start_date, end_date, initial_capital, commission_rate
                )

                # UserWarningがキャプチャされているかチェック
                price_warnings = [warning for warning in w if issubclass(warning.category, UserWarning) and "initial cash value" in str(warning.message)]
                assert len(price_warnings) == 0

    def test_backtest_error_handling(self, executor):
        """バックテストのエラーハンドリング"""
        # 空データのテスト
        data = Mock()
        data.empty = True
        executor.data_service.get_data_for_backtest.return_value = data

        with pytest.raises(BacktestExecutionError, match="データが見つかりませんでした"):
            executor.execute_backtest(Mock(), {}, "BTCUSDT", "1h", datetime.now(), datetime.now(), 10000.0, 0.001)

        # データ取得失敗のテスト
        executor.data_service.get_data_for_backtest.side_effect = Exception("Data service error")

        with pytest.raises(BacktestExecutionError, match="データ取得に失敗しました"):
            executor.execute_backtest(Mock(), {}, "BTCUSDT", "1h", datetime.now(), datetime.now(), 10000.0, 0.001)

    def test_backtest_crypto_symbol_handling(self, executor, caplog):
        """暗号通貨シンボルの処理テスト"""
        data = pd.DataFrame({
            'Open': [95000.0, 96000.0],
            'High': [97000.0, 98000.0],
            'Low': [94000.0, 95000.0],
            'Close': [96000.0, 97000.0],
            'Volume': [100.0, 110.0]
        })
        executor.data_service.get_data_for_backtest.return_value = data

        strategy_class = Mock()
        strategy_parameters = {}

        symbol = "BTC/USDT:USDT"  # 暗号通貨シンボル
        timeframe = "1h"
        start_date = datetime.now()
        end_date = datetime.now()
        initial_capital = 10000.0
        commission_rate = 0.001

        stats = Mock()

        # ログレベルを設定
        caplog.set_level(logging.INFO)

        with patch('backend.app.services.backtest.execution.backtest_executor.FractionalBacktest') as mock_fractional_bt:
            mock_bt_instance = Mock()
            mock_bt_instance.run.return_value = stats
            mock_fractional_bt.return_value = mock_bt_instance

            executor.execute_backtest(
                strategy_class, strategy_parameters, symbol, timeframe,
                start_date, end_date, initial_capital, commission_rate
            )

            # "Crypto symbol"を含むログメッセージが記録されていないことを確認
            crypto_logs = [record for record in caplog.records if "Crypto symbol" in record.message]
            assert len(crypto_logs) == 0


class TestStrategyFactoryIntegrated:
    """StrategyFactoryの統合テスト"""

    @pytest.fixture
    def mock_services(self):
        """モックされた各種サービスを提供するフィクスチャ"""
        return {
            'condition_evaluator': Mock(),
            'indicator_calculator': Mock(),
            'tpsl_service': Mock(),
            'position_sizing_service': Mock()
        }

    @pytest.fixture
    def valid_gene(self):
        """有効なStrategyGeneのモック"""
        gene = Mock(spec=StrategyGene)
        gene.validate.return_value = (True, [])
        gene.id = "test-id-123"
        gene.indicators = []
        gene.entry_conditions = []
        gene.exit_conditions = []
        gene.long_conditions = []
        gene.short_conditions = []
        gene.tpsl_gene = None
        gene.position_sizing_gene = None
        gene.get_effective_long_conditions.return_value = []
        gene.get_effective_short_conditions.return_value = []
        return gene

    @pytest.fixture
    def gene_with_indicators(self, valid_gene):
        """指標を持つGeneのモック"""
        indicator_gene = Mock(spec=IndicatorGene)
        indicator_gene.enabled = True
        indicator_gene.type = "SMA"
        indicator_gene.parameters = {"period": 14}
        valid_gene.indicators = [indicator_gene]
        return valid_gene

    def test_strategy_factory_initialization(self):
        """StrategyFactoryの初期化テスト"""
        factory = StrategyFactory()
        assert hasattr(factory, 'condition_evaluator')
        assert hasattr(factory, 'indicator_calculator')
        assert hasattr(factory, 'tpsl_service')
        assert hasattr(factory, 'position_sizing_service')

    def test_strategy_generation_workflow(self, valid_gene):
        """戦略生成のワークフローテスト"""
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(valid_gene)

        assert strategy_class is not None
        assert isinstance(strategy_class, type)
        assert hasattr(strategy_class, '__name__')
        assert strategy_class.__name__.startswith("GS_")

        # クラス変数の検証
        assert hasattr(strategy_class, 'strategy_gene')
        assert strategy_class.strategy_gene == valid_gene

    def test_strategy_gene_validation(self, valid_gene):
        """戦略遺伝子の検証テスト"""
        factory = StrategyFactory()

        # 有効な遺伝子
        is_valid, errors = factory.validate_gene(valid_gene)
        assert is_valid is True
        assert errors == []

        # 無効な遺伝子
        invalid_gene = Mock(spec=StrategyGene)
        invalid_gene.validate.side_effect = Exception("Validation error")

        is_valid, errors = factory.validate_gene(invalid_gene)
        assert is_valid is False
        assert "Validation error" in str(errors[0])

    def test_generated_strategy_execution(self, valid_gene):
        """生成された戦略の実行テスト"""
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(valid_gene)

        # backtesting.py互換のデータ作成
        data = Mock()
        broker = Mock()
        strategy_instance = strategy_class(broker=broker, data=data, params=None)

        assert strategy_instance.gene == valid_gene

    def test_strategy_factory_with_custom_params(self, valid_gene):
        """カスタムパラメータでの戦略生成テスト"""
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(valid_gene)

        data = Mock()
        broker = Mock()
        custom_gene = Mock(spec=StrategyGene)
        params = {"strategy_gene": custom_gene}

        strategy_instance = strategy_class(broker=broker, data=data, params=params)

        assert strategy_instance.gene == custom_gene

    def test_strategy_class_name_generation(self, valid_gene):
        """戦略クラス名の生成テスト"""
        factory = StrategyFactory()
        valid_gene.id = "unique-strategy-id"
        strategy_class = factory.create_strategy_class(valid_gene)

        assert "GS_unique" in strategy_class.__name__

    def test_strategy_execution_with_conditions(self, mocker, valid_gene):
        """条件付き戦略実行テスト"""
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(valid_gene)

        # Mock broker with equity
        mock_broker = Mock()
        mock_broker.equity = 100000.0

        # Mock data
        mock_data = Mock()
        mock_data.Close = [50000.0]

        # Mock position
        mock_position = Mock()
        mock_position.size = 0  # no position

        strategy_instance = strategy_class(broker=mock_broker, data=mock_data, params=None)
        mocker.patch.object(type(strategy_instance), 'position', mock_position)

        # evaluator mock
        mock_evaluator = Mock()
        factory.condition_evaluator = mock_evaluator

        # Mock factory通じて条件がtrueになるように
        valid_gene.get_effective_long_conditions.return_value = [Mock()]
        mock_evaluator.evaluate_conditions.return_value = True

        # サイズ計算mock
        mock_service = Mock()
        factory.position_sizing_service = mock_service

        # buyメソッドをspy
        mocker.patch.object(strategy_instance, 'buy')

        strategy_instance.next()

        # TP/SLがない場合、buyが呼ばれているはず
        strategy_instance.buy.assert_called_once()

    def test_strategy_position_sizing(self, mocker, valid_gene):
        """ポジションサイジングテスト"""
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(valid_gene)

        # モック設定
        position_sizing_gene = Mock()
        position_sizing_gene.enabled = True
        valid_gene.position_sizing_gene = position_sizing_gene

        mock_result = Mock()
        mock_result.position_size = 0.1

        # factory.serviceをMockに置換
        mock_service = Mock()
        mock_service.calculate_position_size.return_value = mock_result
        factory.position_sizing_service = mock_service

        broker = Mock()
        broker.equity = 100000.0
        mock_data = Mock()
        mock_data.Close = [50000.0]

        strategy_instance = strategy_class(broker=broker, data=mock_data, params=None)

        size = strategy_instance._calculate_position_size()
        assert size == 0.1

    def test_strategy_exit_conditions(self, mocker, valid_gene):
        """エグジット条件テスト"""
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(valid_gene)

        mock_evaluator = Mock()
        mock_evaluator.evaluate_conditions.return_value = True
        factory.condition_evaluator = mock_evaluator

        strategy_instance = strategy_class(broker=Mock(), data=Mock(), params=None)
        strategy_instance.gene.exit_conditions = [Mock()]

        result = strategy_instance._check_exit_conditions()
        assert result is True

    def test_strategy_factory_error_handling(self):
        """StrategyFactoryのエラーハンドリング"""
        factory = StrategyFactory()

        # 無効な遺伝子でのエラー
        invalid_gene = Mock(spec=StrategyGene)
        invalid_gene.validate.return_value = (False, ["Invalid gene"])

        with pytest.raises(ValueError, match="Invalid strategy gene"):
            factory.create_strategy_class(invalid_gene)

    def test_strategy_indicator_initialization(self, mocker, valid_gene):
        """指標初期化テスト"""
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(valid_gene)

        # 失敗するindicator_gene
        indicator_gene = Mock(spec=IndicatorGene)
        indicator_gene.enabled = True
        indicator_gene.type = "UNKNOWN"
        indicator_gene.parameters = {"period": -1}
        valid_gene.indicators = [indicator_gene]

        strategy_instance = strategy_class(broker=Mock(), data=Mock(), params=None)

        # calculatorをMockに置き換え
        mock_calculator = Mock()
        mock_calculator.init_indicator.side_effect = Exception("Init failed")
        factory.indicator_calculator = mock_calculator

        # ログキャプチャ
        with pytest.warns(None):  # 警告が発生してもOK
            strategy_instance.init()

    def test_strategy_indicator_success_initialization(self, mocker, gene_with_indicators):
        """指標初期化成功テスト"""
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(gene_with_indicators)

        # calculatorをMockに置き換え
        mock_calculator = Mock()
        mock_calculator.init_indicator.return_value = None
        factory.indicator_calculator = mock_calculator

        strategy_instance = strategy_class(broker=Mock(), data=Mock(), params=None)

        # エラーがない
        strategy_instance.init()

        # 呼び出された
        mock_calculator.init_indicator.assert_called_once()


class TestFractionalBacktestIntegration:
    """FractionalBacktest統合テスト"""

    def test_fractional_backtest_crypto_compatibility(self):
        """暗号通貨でのFractionalBacktest互換性テスト"""
        from backtesting.lib import FractionalBacktest
        import pandas as pd

        # 実際のビットコインデータを想定したDataFrame
        data = pd.DataFrame({
            'Open': [95000.0, 96000.0, 97000.0],
            'High': [97000.0, 98000.0, 99000.0],
            'Low': [94000.0, 95000.0, 96000.0],
            'Close': [96000.0, 97000.0, 98000.0],
            'Volume': [100.0, 110.0, 120.0]
        })

        # ダミーのStrategyクラス
        from backtesting import Strategy

        class DummyStrategy(Strategy):
            def init(self):
                pass
            def next(self):
                pass

        # FractionalBacktestを初期化
        bt = FractionalBacktest(data, DummyStrategy, cash=10000.0)

        # 警告がないことを確認
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            stats = bt.run()  # 空のrunで統計を生成

            price_warnings = [warning for warning in w if issubclass(warning.category, UserWarning) and "initial cash value" in str(warning.message)]
            assert len(price_warnings) == 0


if __name__ == "__main__":
    pytest.main([__file__])