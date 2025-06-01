"""
統一されたバックテストシステムの統合テスト

backtesting.pyライブラリに統一された新しいシステムのテスト
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.core.services.backtest_service import BacktestService
from app.core.utils.data_standardization import (
    standardize_ohlcv_columns,
    validate_ohlcv_data,
    prepare_data_for_backtesting,
    convert_legacy_config_to_backtest_service
)
from backtest.runner import run_backtest


class TestUnifiedBacktestSystem:
    """統一されたバックテストシステムのテストクラス"""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """標準化されたOHLCVテストデータ"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # より現実的な価格データを生成
        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)  # 2%の日次ボラティリティ
        prices = base_price * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'Open': prices + np.random.normal(0, prices * 0.001),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        return data

    @pytest.fixture
    def legacy_config(self):
        """従来の設定形式"""
        return {
            'strategy': {
                'name': 'SMA_CROSS',
                'target_pair': 'BTC/USDT',
                'indicators': [
                    {'name': 'SMA', 'params': {'period': 20}},
                    {'name': 'SMA', 'params': {'period': 50}}
                ],
                'entry_rules': [
                    {'condition': 'SMA(close, 20) > SMA(close, 50)'}
                ],
                'exit_rules': [
                    {'condition': 'SMA(close, 20) < SMA(close, 50)'}
                ]
            },
            'start_date': '2023-01-01T00:00:00Z',
            'end_date': '2023-12-31T23:59:59Z',
            'timeframe': '1d',
            'initial_capital': 100000,
            'commission_rate': 0.001
        }

    def test_data_standardization(self):
        """データ標準化機能のテスト"""
        # 小文字の列名を持つデータ
        lowercase_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })
        
        # 標準化
        standardized = standardize_ohlcv_columns(lowercase_data)
        
        # 列名が正しく変換されているか
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        assert list(standardized.columns) == expected_columns
        
        # データの妥当性チェック
        assert validate_ohlcv_data(standardized)

    def test_legacy_config_conversion(self, legacy_config):
        """従来設定の変換テスト"""
        converted = convert_legacy_config_to_backtest_service(legacy_config)
        
        # 必要なフィールドが存在するか
        required_fields = [
            'strategy_name', 'symbol', 'timeframe', 
            'start_date', 'end_date', 'initial_capital', 
            'commission_rate', 'strategy_config'
        ]
        
        for field in required_fields:
            assert field in converted
        
        # 戦略設定が正しく変換されているか
        assert converted['strategy_config']['strategy_type'] == 'SMA_CROSS'
        assert 'parameters' in converted['strategy_config']

    def test_backtest_service_with_standardized_data(self, sample_ohlcv_data):
        """標準化されたデータでのBacktestServiceテスト"""
        # データを準備
        prepared_data = prepare_data_for_backtesting(sample_ohlcv_data)
        
        # BacktestService設定
        config = {
            'strategy_name': 'SMA_CROSS',
            'symbol': 'BTC/USDT',
            'timeframe': '1d',
            'start_date': '2023-01-01',
            'end_date': '2023-04-10',
            'initial_capital': 100000,
            'commission_rate': 0.001,
            'strategy_config': {
                'strategy_type': 'SMA_CROSS',
                'parameters': {
                    'n1': 20,
                    'n2': 50
                }
            }
        }
        
        # モックデータサービスを作成
        class MockDataService:
            def get_ohlcv_for_backtest(self, **kwargs):
                return prepared_data
        
        # BacktestServiceを実行
        service = BacktestService(data_service=MockDataService())
        result = service.run_backtest(config)
        
        # 結果の検証
        assert 'performance_metrics' in result
        assert 'total_return' in result['performance_metrics']
        assert 'sharpe_ratio' in result['performance_metrics']
        assert 'max_drawdown' in result['performance_metrics']

    def test_runner_with_new_implementation(self, legacy_config):
        """新しい実装でのrunner.pyテスト"""
        # runner.pyの新しい実装をテスト
        # 注意: 実際のデータベース接続は避けて、サンプルデータを使用
        
        result = run_backtest(legacy_config)
        
        # エラーが発生していないか
        assert 'error' not in result
        
        # 基本的な結果フィールドが存在するか
        expected_fields = [
            'id', 'strategy_id', 'config', 'created_at'
        ]
        
        for field in expected_fields:
            assert field in result

    def test_performance_comparison(self, sample_ohlcv_data):
        """パフォーマンス比較テスト"""
        # 同じデータで複数の戦略パラメータをテスト
        base_config = {
            'strategy_name': 'SMA_CROSS',
            'symbol': 'BTC/USDT',
            'timeframe': '1d',
            'start_date': '2023-01-01',
            'end_date': '2023-04-10',
            'initial_capital': 100000,
            'commission_rate': 0.001,
        }
        
        # 異なるパラメータでテスト
        test_cases = [
            {'n1': 10, 'n2': 30},
            {'n1': 20, 'n2': 50},
            {'n1': 30, 'n2': 70}
        ]
        
        class MockDataService:
            def get_ohlcv_for_backtest(self, **kwargs):
                return prepare_data_for_backtesting(sample_ohlcv_data)
        
        service = BacktestService(data_service=MockDataService())
        results = []
        
        for params in test_cases:
            config = base_config.copy()
            config['strategy_config'] = {
                'strategy_type': 'SMA_CROSS',
                'parameters': params
            }
            
            result = service.run_backtest(config)
            results.append(result)
        
        # 全てのテストが成功したか
        for result in results:
            assert 'performance_metrics' in result
            assert 'total_return' in result['performance_metrics']

    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        service = BacktestService()
        
        # 無効な設定でテスト
        invalid_config = {
            'strategy_name': 'INVALID_STRATEGY',
            'symbol': 'BTC/USDT',
            'timeframe': '1d',
            'start_date': '2023-01-01',
            'end_date': '2023-01-02',  # 短すぎる期間
            'initial_capital': -1000,  # 無効な資金
            'commission_rate': 0.001,
            'strategy_config': {
                'strategy_type': 'INVALID_TYPE',
                'parameters': {}
            }
        }
        
        # エラーが適切に処理されるか
        with pytest.raises((ValueError, KeyError)):
            service.run_backtest(invalid_config)

    def test_data_validation_edge_cases(self):
        """データ検証のエッジケーステスト"""
        # 空のデータフレーム
        empty_df = pd.DataFrame()
        assert not validate_ohlcv_data(empty_df)
        
        # 無効な価格関係（High < Low）
        invalid_data = pd.DataFrame({
            'Open': [100],
            'High': [95],  # High < Low (無効)
            'Low': [105],
            'Close': [100],
            'Volume': [1000]
        })
        assert not validate_ohlcv_data(invalid_data)
        
        # 正常なデータ
        valid_data = pd.DataFrame({
            'Open': [100],
            'High': [105],
            'Low': [95],
            'Close': [102],
            'Volume': [1000]
        })
        assert validate_ohlcv_data(valid_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
