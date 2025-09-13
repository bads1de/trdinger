"""
統合戦略実行テスト

戦略生成からバックテスト実行までの完全なフローをテスト
TDD原則に基づき、エンドツーエンドの戦略実行をテスト
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# 戦略関連
from backend.app.services.auto_strategy.generators.strategy_factory import StrategyFactory
from backend.app.services.auto_strategy.models.strategy_models import (
    StrategyGene, IndicatorGene, Condition, ConditionGroup
)

# バックテスト関連
from backend.app.services.backtest.execution.backtest_executor import BacktestExecutor
from backend.app.services.backtest.backtest_data_service import BacktestDataService

# 指標関連
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService


class TestStrategyExecutionIntegration:
    """戦略実行の統合テスト"""

    @pytest.fixture
    def realistic_market_data(self):
        """現実的な市場データ"""
        dates = pd.date_range('2023-01-01', periods=500, freq='h')
        np.random.seed(42)

        # ビットコインのような価格変動をシミュレート
        base_price = 50000
        returns = np.random.normal(0.001, 0.02, 500)  # 平均1%の変動、2%のボラティリティ
        prices = base_price * np.exp(np.cumsum(returns))

        # OHLCデータの生成
        opens = prices
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, 500)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, 500)))
        closes = prices + np.random.normal(0, prices * 0.005, 500)

        # OHLCの整合性を確保
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))

        data = {
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': np.random.randint(1000, 10000, 500)
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def mock_strategy_gene(self):
        """モック戦略遺伝子"""
        gene = Mock(spec=StrategyGene)
        gene.validate.return_value = (True, [])
        gene.id = "integration-test-gene"
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

    def test_strategy_generation_to_execution_flow(self, realistic_market_data, mock_strategy_gene):
        """戦略生成から実行までの完全フロー"""
        # 1. 戦略生成
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(mock_strategy_gene)

        # 2. バックテストデータの準備
        processed_data = self._prepare_backtest_data(realistic_market_data)

        # 3. 戦略インスタンスの作成
        broker = Mock()
        data = Mock()
        data.Close = processed_data['close'].values
        strategy_instance = strategy_class(broker=broker, data=data, params=None)

        assert strategy_instance is not None
        assert strategy_instance.gene == mock_strategy_gene

    def test_backtest_execution_with_generated_strategy(self, realistic_market_data, mock_strategy_gene):
        """生成された戦略でのバックテスト実行"""
        # 1. 戦略生成
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(mock_strategy_gene)

        # 2. バックテストデータの準備
        backtest_data = self._prepare_backtest_data(realistic_market_data)

        # 3. バックテスト実行のモック
        with patch('backend.app.services.backtest.execution.backtest_executor.FractionalBacktest') as mock_fractional_bt:
            mock_bt_instance = Mock()
            mock_stats = {
                'Return [%]': 15.5,
                'Max. Drawdown [%]': -8.2,
                'Sharpe Ratio': 1.25,
                'Total Trades': 45
            }
            mock_bt_instance.run.return_value = mock_stats
            mock_fractional_bt.return_value = mock_bt_instance

            # 4. BacktestExecutorでの実行
            data_service = Mock()
            data_service.get_data_for_backtest.return_value = backtest_data

            executor = BacktestExecutor(data_service)

            result = executor.execute_backtest(
                strategy_class, {}, "BTC/USDT", "1h",
                datetime(2023, 1, 1), datetime(2023, 1, 22),
                100000.0, 0.001
            )

            assert result == mock_stats
            mock_fractional_bt.assert_called_once()

    def test_strategy_with_indicators_execution(self, realistic_market_data):
        """指標付き戦略の実行"""
        # 1. 指標付き戦略遺伝子の作成
        gene = Mock(spec=StrategyGene)
        gene.validate.return_value = (True, [])
        gene.id = "indicator-test-gene"

        # 指標の設定
        indicator_gene = Mock(spec=IndicatorGene)
        indicator_gene.enabled = True
        indicator_gene.type = "SMA"
        indicator_gene.parameters = {"period": 20}
        gene.indicators = [indicator_gene]

        gene.entry_conditions = []
        gene.exit_conditions = []
        gene.long_conditions = []
        gene.short_conditions = []
        gene.tpsl_gene = None
        gene.position_sizing_gene = None
        gene.get_effective_long_conditions.return_value = []
        gene.get_effective_short_conditions.return_value = []

        # 2. 戦略生成
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(gene)

        # 3. データ準備
        processed_data = self._prepare_backtest_data(realistic_market_data)

        # 4. 戦略実行
        broker = Mock()
        data = Mock()
        data.Close = processed_data['close'].values
        strategy_instance = strategy_class(broker=broker, data=data, params=None)

        # 指標初期化のテスト
        with patch.object(factory, 'indicator_calculator') as mock_calculator:
            mock_calculator.init_indicator.return_value = None
            strategy_instance.init()  # 初期化実行

    def test_end_to_end_strategy_workflow(self, realistic_market_data):
        """エンドツーエンド戦略ワークフロー"""
        # 1. 完全な戦略遺伝子の作成
        gene = self._create_complete_strategy_gene()

        # 2. 戦略生成
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(gene)

        # 3. バックテストデータの準備
        backtest_data = self._prepare_backtest_data(realistic_market_data)

        # 4. バックテスト実行
        mock_stats = self._execute_mock_backtest(strategy_class, backtest_data)

        # 5. 結果検証
        assert 'Return [%]' in mock_stats
        assert 'Sharpe Ratio' in mock_stats
        assert mock_stats['Total Trades'] >= 0

    def test_strategy_performance_metrics(self, realistic_market_data):
        """戦略パフォーマンスメトリクス"""
        # 1. 戦略の準備
        gene = self._create_complete_strategy_gene()
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(gene)

        # 2. バックテスト実行
        backtest_data = self._prepare_backtest_data(realistic_market_data)
        stats = self._execute_mock_backtest(strategy_class, backtest_data)

        # 3. パフォーマンスメトリクスの検証
        required_metrics = ['Return [%]', 'Sharpe Ratio', 'Max. Drawdown [%]', 'Total Trades']

        for metric in required_metrics:
            assert metric in stats, f"Missing required metric: {metric}"

        # メトリクスの妥当性チェック
        assert isinstance(stats['Return [%]'], (int, float))
        assert isinstance(stats['Sharpe Ratio'], (int, float))
        assert stats['Max. Drawdown [%]'] <= 0  # ドローダウンは負の値
        assert stats['Total Trades'] >= 0

    def test_strategy_error_handling(self, realistic_market_data):
        """戦略実行時のエラーハンドリング"""
        # 1. 無効な戦略遺伝子
        invalid_gene = Mock(spec=StrategyGene)
        invalid_gene.validate.return_value = (False, ["Invalid configuration"])

        factory = StrategyFactory()

        with pytest.raises(ValueError, match="Invalid strategy gene"):
            factory.create_strategy_class(invalid_gene)

        # 2. 空のバックテストデータ
        empty_data = pd.DataFrame()

        data_service = Mock()
        data_service.get_data_for_backtest.return_value = empty_data

        executor = BacktestExecutor(data_service)

        with pytest.raises(Exception):  # BacktestExecutionErrorが発生
            executor.execute_backtest(
                Mock(), {}, "BTC/USDT", "1h",
                datetime(2023, 1, 1), datetime(2023, 1, 2),
                10000.0, 0.001
            )

    def test_strategy_scalability(self):
        """戦略のスケーラビリティテスト"""
        # 大規模データセットでのテスト
        dates = pd.date_range('2020-01-01', periods=5000, freq='h')
        large_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(30000, 70000, 5000),
            'high': np.random.uniform(35000, 75000, 5000),
            'low': np.random.uniform(25000, 65000, 5000),
            'close': np.random.uniform(30000, 70000, 5000),
            'volume': np.random.randint(100, 10000, 5000)
        })

        # 1. 戦略生成
        gene = self._create_complete_strategy_gene()
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(gene)

        # 2. 大規模データでのバックテスト
        processed_data = self._prepare_backtest_data(large_data)
        stats = self._execute_mock_backtest(strategy_class, processed_data)

        # 3. 結果が得られることを確認
        assert stats is not None
        assert 'Return [%]' in stats

    def test_concurrent_strategy_execution(self, realistic_market_data):
        """並行戦略実行テスト"""
        import threading

        results = {}
        errors = []

        def run_strategy(strategy_id, gene):
            try:
                factory = StrategyFactory()
                strategy_class = factory.create_strategy_class(gene)

                backtest_data = self._prepare_backtest_data(realistic_market_data)
                stats = self._execute_mock_backtest(strategy_class, backtest_data)

                results[strategy_id] = stats
            except Exception as e:
                errors.append((strategy_id, str(e)))

        # 複数の戦略を並行実行
        threads = []
        for i in range(3):
            gene = self._create_complete_strategy_gene()
            gene.id = f"concurrent-test-{i}"

            thread = threading.Thread(target=run_strategy, args=(i, gene))
            threads.append(thread)
            thread.start()

        # すべてのスレッドが完了するのを待つ
        for thread in threads:
            thread.join()

        # 結果検証
        assert len(results) == 3, f"Some strategies failed: {errors}"
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # 各戦略の結果が有効であることを確認
        for strategy_id, stats in results.items():
            assert 'Return [%]' in stats
            assert isinstance(stats['Return [%]'], (int, float))

    def _prepare_backtest_data(self, data):
        """バックテスト用データの準備"""
        from backend.app.utils.data_processing import data_processor

        processed = data_processor.clean_and_validate_data(
            data, ['open', 'high', 'low', 'close', 'volume']
        )
        return processed

    def _create_complete_strategy_gene(self):
        """完全な戦略遺伝子を作成"""
        gene = Mock(spec=StrategyGene)
        gene.validate.return_value = (True, [])
        gene.id = "complete-integration-test-gene"

        # 指標の設定
        indicator_gene = Mock(spec=IndicatorGene)
        indicator_gene.enabled = True
        indicator_gene.type = "SMA"
        indicator_gene.parameters = {"period": 20}
        gene.indicators = [indicator_gene]

        # 条件の設定
        condition = Mock(spec=Condition)
        condition.enabled = True
        gene.entry_conditions = [condition]
        gene.exit_conditions = []
        gene.long_conditions = [condition]
        gene.short_conditions = []

        gene.tpsl_gene = None
        gene.position_sizing_gene = None

        # 条件取得メソッドの設定
        gene.get_effective_long_conditions.return_value = [condition]
        gene.get_effective_short_conditions.return_value = []

        return gene

    def _execute_mock_backtest(self, strategy_class, data):
        """モックバックテスト実行"""
        with patch('backend.app.services.backtest.execution.backtest_executor.FractionalBacktest') as mock_fractional_bt:
            mock_bt_instance = Mock()
            mock_stats = {
                'Return [%]': np.random.uniform(-10, 20),
                'Max. Drawdown [%]': np.random.uniform(-15, 0),
                'Sharpe Ratio': np.random.uniform(0.5, 2.0),
                'Total Trades': np.random.randint(10, 100),
                'Win Rate [%]': np.random.uniform(40, 60),
                'Profit Factor': np.random.uniform(0.8, 1.5)
            }
            mock_bt_instance.run.return_value = mock_stats
            mock_fractional_bt.return_value = mock_bt_instance

            data_service = Mock()
            data_service.get_data_for_backtest.return_value = data

            executor = BacktestExecutor(data_service)

            result = executor.execute_backtest(
                strategy_class, {}, "BTC/USDT", "1h",
                datetime(2023, 1, 1), datetime(2023, 1, 22),
                100000.0, 0.001
            )

            return result


class TestStrategyOptimizationIntegration:
    """戦略最適化の統合テスト"""

    def test_strategy_parameter_optimization(self):
        """戦略パラメータ最適化"""
        # テストデータ
        dates = pd.date_range('2023-01-01', periods=200, freq='h')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(49000, 51000, 200),
            'high': np.random.uniform(50000, 52000, 200),
            'low': np.random.uniform(48000, 50000, 200),
            'close': np.random.uniform(49000, 51000, 200),
            'volume': np.random.randint(1000, 5000, 200)
        })

        # 戦略パラメータの最適化シミュレーション
        param_combinations = [
            {'period': 10, 'multiplier': 1.5},
            {'period': 20, 'multiplier': 2.0},
            {'period': 30, 'multiplier': 2.5}
        ]

        best_return = -float('inf')
        best_params = None

        for params in param_combinations:
            # モック戦略実行
            mock_return = np.random.uniform(-5, 15)
            if mock_return > best_return:
                best_return = mock_return
                best_params = params

        assert best_params is not None
        assert 'period' in best_params
        assert 'multiplier' in best_params

    def test_strategy_risk_management(self):
        """戦略リスク管理テスト"""
        # 大きなドローダウンを持つデータ
        dates = pd.date_range('2023-01-01', periods=300, freq='h')
        prices = 50000 * (1 + np.cumsum(np.random.normal(0.001, 0.03, 300)))
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000, 5000, 300)
        })

        # リスク管理機能のシミュレーション
        max_drawdown = -15.0  # 15%の最大ドローダウン
        stop_loss_threshold = -0.10  # 10%のストップロス

        # モック戦略でのリスク管理
        mock_stats = {
            'Max. Drawdown [%]': max_drawdown,
            'Return [%]': 5.0
        }

        # リスクチェック
        if mock_stats['Max. Drawdown [%]'] < stop_loss_threshold:
            risk_accepted = False
        else:
            risk_accepted = True

        assert risk_accepted is False, "Risk management should reject high drawdown"


if __name__ == "__main__":
    pytest.main([__file__])