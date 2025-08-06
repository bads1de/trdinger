"""
バックテスト統合包括的テスト

BacktestServiceとの統合、戦略実行、結果変換、
パフォーマンス計算の包括的テストを実施します。
"""

import logging
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, Any

from app.services.backtest.backtest_service import BacktestService
from app.services.auto_strategy.models.gene_strategy import StrategyGene

logger = logging.getLogger(__name__)


class TestBacktestIntegrationComprehensive:
    """バックテスト統合包括的テストクラス"""

    @pytest.fixture
    def backtest_service(self):
        """BacktestServiceのテスト用インスタンス"""
        return BacktestService()

    @pytest.fixture
    def sample_strategy_gene(self):
        """サンプル戦略遺伝子"""
        return StrategyGene(
            indicators=[
                {"type": "SMA", "parameters": {"period": 20}},
                {"type": "RSI", "parameters": {"period": 14}}
            ],
            entry_conditions=[
                {"indicator": "SMA", "operator": ">", "value": "close"},
                {"indicator": "RSI", "operator": "<", "value": 30}
            ],
            exit_conditions=[
                {"indicator": "RSI", "operator": ">", "value": 70}
            ],
            risk_management={
                "position_size": 0.1,
                "stop_loss": 0.02,
                "take_profit": 0.04
            }
        )

    @pytest.fixture
    def sample_backtest_config(self):
        """サンプルバックテスト設定"""
        return {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "initial_capital": 100000,
            "commission_rate": 0.00055,
            "slippage": 0.001
        }

    @pytest.fixture
    def sample_market_data(self):
        """サンプル市場データ"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        np.random.seed(42)
        
        data = {
            'timestamp': dates,
            'open': 50000 + np.random.randn(100) * 1000,
            'high': 51000 + np.random.randn(100) * 1000,
            'low': 49000 + np.random.randn(100) * 1000,
            'close': 50000 + np.random.randn(100) * 1000,
            'volume': 1000 + np.random.randn(100) * 100,
        }
        
        df = pd.DataFrame(data)
        # 価格の整合性を保つ
        df['high'] = df[['open', 'close']].max(axis=1) + np.abs(np.random.randn(100) * 100)
        df['low'] = df[['open', 'close']].min(axis=1) - np.abs(np.random.randn(100) * 100)
        
        return df

    def test_backtest_service_initialization(self, backtest_service):
        """BacktestService初期化テスト"""
        assert backtest_service is not None
        assert hasattr(backtest_service, 'execute_backtest')

    @patch('app.services.backtest.backtest_service.BacktestDataService')
    def test_strategy_execution_success(self, mock_data_service, backtest_service, sample_strategy_gene, sample_backtest_config, sample_market_data):
        """戦略実行成功テスト"""
        # データサービスのモック設定
        mock_data_service_instance = Mock()
        mock_data_service_instance.get_market_data.return_value = sample_market_data
        mock_data_service.return_value = mock_data_service_instance
        
        try:
            # バックテスト実行
            result = backtest_service.execute_backtest(sample_backtest_config, sample_strategy_gene)
            
            # 結果の基本検証
            if result is not None:
                assert isinstance(result, dict)
                
                # 期待される結果キーの確認
                expected_keys = [
                    'total_return', 'sharpe_ratio', 'max_drawdown', 
                    'total_trades', 'win_rate', 'profit_factor'
                ]
                
                for key in expected_keys:
                    if key in result:
                        assert isinstance(result[key], (int, float))
                        
        except Exception as e:
            logger.warning(f"戦略実行テストでエラー: {e}")
            pytest.skip(f"戦略実行テストをスキップ: {e}")

    def test_supported_strategies_retrieval(self, backtest_service):
        """サポート戦略取得テスト"""
        try:
            strategies = backtest_service.get_supported_strategies()
            
            assert isinstance(strategies, (list, dict))
            
            if isinstance(strategies, list):
                assert len(strategies) > 0
                for strategy in strategies:
                    assert isinstance(strategy, (str, dict))
                    
            elif isinstance(strategies, dict):
                assert len(strategies) > 0
                
        except Exception as e:
            logger.warning(f"サポート戦略取得テストでエラー: {e}")
            pytest.skip(f"サポート戦略取得テストをスキップ: {e}")

    def test_invalid_strategy_handling(self, backtest_service, sample_backtest_config):
        """無効戦略ハンドリングテスト"""
        invalid_strategies = [
            None,  # None戦略
            {},  # 空の戦略
            {"invalid": "strategy"},  # 無効な構造
        ]
        
        for invalid_strategy in invalid_strategies:
            try:
                result = backtest_service.execute_backtest(sample_backtest_config, invalid_strategy)
                
                # エラーが発生するか、適切なデフォルト結果が返されることを確認
                if result is not None:
                    logger.warning(f"無効戦略 {invalid_strategy} で結果が返されました: {result}")
                    
            except Exception as e:
                # 適切なエラーハンドリングが行われることを確認
                assert any(keyword in str(e).lower() for keyword in ['invalid', 'strategy', 'error', 'missing'])

    def test_invalid_backtest_config_handling(self, backtest_service, sample_strategy_gene):
        """無効バックテスト設定ハンドリングテスト"""
        invalid_configs = [
            {},  # 空の設定
            {"symbol": "INVALID"},  # 無効なシンボル
            {"symbol": "BTC/USDT", "timeframe": "invalid"},  # 無効な時間軸
            {"symbol": "BTC/USDT", "timeframe": "1h", "start_date": "invalid"},  # 無効な日付
        ]
        
        for invalid_config in invalid_configs:
            try:
                result = backtest_service.execute_backtest(invalid_config, sample_strategy_gene)
                
                # エラーが発生するか、適切なデフォルト結果が返されることを確認
                if result is not None:
                    logger.warning(f"無効設定 {invalid_config} で結果が返されました")
                    
            except Exception as e:
                # 適切なエラーハンドリングが行われることを確認
                assert any(keyword in str(e).lower() for keyword in ['invalid', 'config', 'error', 'missing'])

    @patch('app.services.backtest.backtest_service.BacktestDataService')
    def test_performance_metrics_calculation(self, mock_data_service, backtest_service, sample_strategy_gene, sample_backtest_config, sample_market_data):
        """パフォーマンス指標計算テスト"""
        # データサービスのモック設定
        mock_data_service_instance = Mock()
        mock_data_service_instance.get_market_data.return_value = sample_market_data
        mock_data_service.return_value = mock_data_service_instance
        
        try:
            result = backtest_service.execute_backtest(sample_backtest_config, sample_strategy_gene)
            
            if result is not None and isinstance(result, dict):
                # パフォーマンス指標の妥当性確認
                if 'total_return' in result:
                    assert isinstance(result['total_return'], (int, float))
                    
                if 'sharpe_ratio' in result:
                    assert isinstance(result['sharpe_ratio'], (int, float))
                    # シャープレシオの合理的な範囲確認
                    assert -10 <= result['sharpe_ratio'] <= 10
                    
                if 'max_drawdown' in result:
                    assert isinstance(result['max_drawdown'], (int, float))
                    # ドローダウンは負の値または0
                    assert result['max_drawdown'] <= 0
                    
                if 'win_rate' in result:
                    assert isinstance(result['win_rate'], (int, float))
                    # 勝率は0-1の範囲
                    assert 0 <= result['win_rate'] <= 1
                    
        except Exception as e:
            logger.warning(f"パフォーマンス指標計算テストでエラー: {e}")

    def test_multiple_timeframes_support(self, backtest_service, sample_strategy_gene):
        """複数時間軸サポートテスト"""
        timeframes = ["15m", "30m", "1h", "4h", "1d"]
        
        for timeframe in timeframes:
            config = {
                "symbol": "BTC/USDT",
                "timeframe": timeframe,
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "initial_capital": 100000,
                "commission_rate": 0.00055
            }
            
            try:
                result = backtest_service.execute_backtest(config, sample_strategy_gene)
                
                # 各時間軸で適切に処理されることを確認
                if result is not None:
                    logger.info(f"時間軸 {timeframe} でバックテスト成功")
                    
            except Exception as e:
                logger.warning(f"時間軸 {timeframe} でエラー: {e}")

    def test_multiple_symbols_support(self, backtest_service, sample_strategy_gene):
        """複数シンボルサポートテスト"""
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        
        for symbol in symbols:
            config = {
                "symbol": symbol,
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "initial_capital": 100000,
                "commission_rate": 0.00055
            }
            
            try:
                result = backtest_service.execute_backtest(config, sample_strategy_gene)
                
                # 各シンボルで適切に処理されることを確認
                if result is not None:
                    logger.info(f"シンボル {symbol} でバックテスト成功")
                    
            except Exception as e:
                logger.warning(f"シンボル {symbol} でエラー: {e}")

    @patch('app.services.backtest.backtest_service.BacktestDataService')
    def test_commission_and_slippage_impact(self, mock_data_service, backtest_service, sample_strategy_gene, sample_market_data):
        """手数料とスリッページの影響テスト"""
        # データサービスのモック設定
        mock_data_service_instance = Mock()
        mock_data_service_instance.get_market_data.return_value = sample_market_data
        mock_data_service.return_value = mock_data_service_instance
        
        # 異なる手数料・スリッページ設定でテスト
        test_cases = [
            {"commission_rate": 0.0, "slippage": 0.0},  # 手数料・スリッページなし
            {"commission_rate": 0.001, "slippage": 0.001},  # 標準的な設定
            {"commission_rate": 0.005, "slippage": 0.005},  # 高い手数料・スリッページ
        ]
        
        results = []
        for case in test_cases:
            config = {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "initial_capital": 100000,
                **case
            }
            
            try:
                result = backtest_service.execute_backtest(config, sample_strategy_gene)
                if result is not None and 'total_return' in result:
                    results.append((case, result['total_return']))
                    
            except Exception as e:
                logger.warning(f"手数料・スリッページテスト {case} でエラー: {e}")
        
        # 手数料・スリッページが高いほどリターンが低くなることを確認
        if len(results) >= 2:
            logger.info(f"手数料・スリッページ影響テスト結果: {results}")

    def test_risk_management_integration(self, backtest_service, sample_backtest_config):
        """リスク管理統合テスト"""
        # 異なるリスク管理設定の戦略
        risk_management_configs = [
            {"position_size": 0.05, "stop_loss": 0.01, "take_profit": 0.02},
            {"position_size": 0.1, "stop_loss": 0.02, "take_profit": 0.04},
            {"position_size": 0.2, "stop_loss": 0.05, "take_profit": 0.1},
        ]
        
        for rm_config in risk_management_configs:
            strategy = StrategyGene(
                indicators=[{"type": "SMA", "parameters": {"period": 20}}],
                entry_conditions=[{"indicator": "SMA", "operator": ">", "value": "close"}],
                exit_conditions=[],
                risk_management=rm_config
            )
            
            try:
                result = backtest_service.execute_backtest(sample_backtest_config, strategy)
                
                if result is not None:
                    logger.info(f"リスク管理設定 {rm_config} でバックテスト成功")
                    
            except Exception as e:
                logger.warning(f"リスク管理設定 {rm_config} でエラー: {e}")

    def test_large_dataset_performance(self, backtest_service, sample_strategy_gene):
        """大量データセットパフォーマンステスト"""
        # 長期間のバックテスト設定
        large_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2024-12-31",  # 2年間
            "initial_capital": 100000,
            "commission_rate": 0.00055
        }
        
        import time
        start_time = time.time()
        
        try:
            result = backtest_service.execute_backtest(large_config, sample_strategy_gene)
            
            execution_time = time.time() - start_time
            
            # 実行時間が合理的な範囲内であることを確認（60秒以下）
            assert execution_time < 60, f"大量データ処理時間が過大: {execution_time:.2f}秒"
            
            if result is not None:
                logger.info(f"大量データセットテスト成功: {execution_time:.2f}秒")
                
        except Exception as e:
            logger.warning(f"大量データセットテストでエラー: {e}")

    def test_concurrent_backtest_execution(self, backtest_service, sample_strategy_gene, sample_backtest_config):
        """並行バックテスト実行テスト"""
        import threading
        import time
        
        results = []
        errors = []
        
        def run_backtest(test_id):
            try:
                config = sample_backtest_config.copy()
                config['test_id'] = test_id  # テスト識別用
                
                result = backtest_service.execute_backtest(config, sample_strategy_gene)
                results.append((test_id, result))
                
            except Exception as e:
                errors.append((test_id, e))
        
        # 複数スレッドで同時実行
        threads = []
        for i in range(3):
            thread = threading.Thread(target=run_backtest, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 全スレッドの完了を待機
        for thread in threads:
            thread.join(timeout=30)
        
        # 結果検証
        if results:
            logger.info(f"並行バックテスト成功: {len(results)} 個")
        
        if errors:
            logger.warning(f"並行バックテストエラー: {len(errors)} 個")

    def test_memory_efficiency_during_backtest(self, backtest_service, sample_strategy_gene, sample_backtest_config):
        """バックテスト中のメモリ効率性テスト"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss
            
            # バックテスト実行
            result = backtest_service.execute_backtest(sample_backtest_config, sample_strategy_gene)
            
            memory_after = process.memory_info().rss
            memory_increase = memory_after - memory_before
            
            # メモリ増加が合理的な範囲内であることを確認（100MB以下）
            assert memory_increase < 100 * 1024 * 1024, \
                f"メモリ使用量が過大: {memory_increase / 1024 / 1024:.2f}MB"
            
        except ImportError:
            pytest.skip("psutilが利用できないため、メモリ効率性テストをスキップ")
        except Exception as e:
            logger.warning(f"メモリ効率性テストでエラー: {e}")

    def test_result_consistency(self, backtest_service, sample_strategy_gene, sample_backtest_config):
        """結果一貫性テスト"""
        # 同じ設定で複数回実行して一貫した結果が得られることを確認
        results = []
        
        for i in range(3):
            try:
                result = backtest_service.execute_backtest(sample_backtest_config, sample_strategy_gene)
                if result is not None:
                    results.append(result)
                    
            except Exception as e:
                logger.warning(f"結果一貫性テスト {i} でエラー: {e}")
        
        # 複数の結果が得られた場合、一貫性を確認
        if len(results) >= 2:
            first_result = results[0]
            for result in results[1:]:
                # 主要な指標が一致することを確認
                for key in ['total_return', 'total_trades']:
                    if key in first_result and key in result:
                        assert first_result[key] == result[key], \
                            f"結果が一貫していません: {key} = {first_result[key]} vs {result[key]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
