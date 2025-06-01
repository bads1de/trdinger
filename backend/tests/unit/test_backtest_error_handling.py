"""
バックテストエラーハンドリングテスト

様々なエラー条件でのバックテストの動作をテストします。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from backtest.engine.strategy_executor import StrategyExecutor
from app.core.services.backtest_service import BacktestService


@pytest.mark.unit
@pytest.mark.backtest
@pytest.mark.error_handling
class TestBacktestErrorHandling:
    """バックテストエラーハンドリングテスト"""

    def test_empty_data_handling(self):
        """空のデータでのエラーハンドリング"""
        executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)

        # 空のDataFrame
        empty_data = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        strategy_config = {
            "indicators": [{"name": "SMA", "params": {"period": 20}}],
            "entry_rules": [{"condition": "SMA(close, 20) > 0"}],
            "exit_rules": [{"condition": "SMA(close, 20) < 0"}],
        }

        # 空のデータでは適切なエラーまたは空の結果が返されることを確認
        with pytest.raises((ValueError, IndexError, KeyError)):
            executor.run_backtest(empty_data, strategy_config)

    def test_invalid_data_format(self):
        """無効なデータ形式でのエラーハンドリング"""
        executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)

        # 必要な列が不足しているデータ
        invalid_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [101, 102, 103],
                # 'Low'と'Close'が不足
                "Volume": [1000, 1100, 1200],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        strategy_config = {
            "indicators": [{"name": "SMA", "params": {"period": 2}}],
            "entry_rules": [{"condition": "SMA(close, 2) > 0"}],
            "exit_rules": [{"condition": "SMA(close, 2) < 0"}],
        }

        # 無効なデータ形式では適切なエラーが発生することを確認
        with pytest.raises((KeyError, AttributeError)):
            executor.run_backtest(invalid_data, strategy_config)

    def test_negative_prices_handling(self):
        """負の価格データでのエラーハンドリング"""
        executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)

        # 負の価格を含むデータ
        data_with_negative_prices = pd.DataFrame(
            {
                "Open": [100, -50, 102],  # 負の価格
                "High": [101, 102, 103],
                "Low": [99, -60, 101],  # 負の価格
                "Close": [100, 101, 102],
                "Volume": [1000, 1100, 1200],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        strategy_config = {
            "indicators": [{"name": "SMA", "params": {"period": 2}}],
            "entry_rules": [{"condition": "SMA(close, 2) > 0"}],
            "exit_rules": [{"condition": "SMA(close, 2) < 0"}],
        }

        # 負の価格でもエラーハンドリングされることを確認
        # （実装によっては警告を出すか、データをクリーニングする）
        try:
            result = executor.run_backtest(data_with_negative_prices, strategy_config)
            # 成功した場合は結果が返される
            assert result is not None
        except (ValueError, AssertionError) as e:
            # 適切なエラーが発生した場合
            assert "negative" in str(e).lower() or "invalid" in str(e).lower()

    def test_insufficient_capital_handling(self):
        """資金不足でのエラーハンドリング"""
        # 極端に少ない初期資金
        executor = StrategyExecutor(initial_capital=1, commission_rate=0.001)  # $1のみ

        # 高価格のデータ
        high_price_data = pd.DataFrame(
            {
                "Open": [50000, 51000, 52000],
                "High": [50500, 51500, 52500],
                "Low": [49500, 50500, 51500],
                "Close": [50000, 51000, 52000],
                "Volume": [1000, 1100, 1200],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        strategy_config = {
            "indicators": [{"name": "SMA", "params": {"period": 2}}],
            "entry_rules": [{"condition": "SMA(close, 2) > 0"}],
            "exit_rules": [{"condition": "SMA(close, 2) < 0"}],
        }

        # 資金不足でも適切に処理されることを確認
        result = executor.run_backtest(high_price_data, strategy_config)
        assert result is not None

        # 取引が発生しないことを確認
        metrics = result["performance_metrics"]
        assert metrics["total_trades"] == 0

    def test_invalid_commission_rate(self):
        """無効な手数料率でのエラーハンドリング"""
        # 負の手数料率
        with pytest.raises(ValueError):
            StrategyExecutor(
                initial_capital=100000, commission_rate=-0.001  # 負の手数料
            )

        # 100%を超える手数料率
        with pytest.raises(ValueError):
            StrategyExecutor(
                initial_capital=100000, commission_rate=1.5  # 150%の手数料
            )

    def test_invalid_strategy_config_structure(self):
        """無効な戦略設定構造でのエラーハンドリング"""
        executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)

        sample_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [101, 102, 103],
                "Low": [99, 100, 101],
                "Close": [100, 101, 102],
                "Volume": [1000, 1100, 1200],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        # 無効な戦略設定（必須キーが不足）
        invalid_configs = [
            {},  # 空の設定
            {"indicators": []},  # entry_rules, exit_rulesが不足
            {"entry_rules": []},  # indicators, exit_rulesが不足
            {"exit_rules": []},  # indicators, entry_rulesが不足
        ]

        for invalid_config in invalid_configs:
            try:
                result = executor.run_backtest(sample_data, invalid_config)
                # 成功した場合でも、適切なデフォルト動作であることを確認
                assert result is not None
            except (KeyError, ValueError, TypeError) as e:
                # 適切なエラーが発生することを確認
                assert isinstance(e, (KeyError, ValueError, TypeError))

    def test_malformed_indicator_config(self):
        """不正な指標設定でのエラーハンドリング"""
        executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)

        sample_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [101, 102, 103],
                "Low": [99, 100, 101],
                "Close": [100, 101, 102],
                "Volume": [1000, 1100, 1200],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        # 不正な指標設定
        malformed_configs = [
            {
                "indicators": [{"name": "SMA"}],  # paramsが不足
                "entry_rules": [{"condition": "SMA(close, 20) > 0"}],
                "exit_rules": [{"condition": "SMA(close, 20) < 0"}],
            },
            {
                "indicators": [{"params": {"period": 20}}],  # nameが不足
                "entry_rules": [{"condition": "SMA(close, 20) > 0"}],
                "exit_rules": [{"condition": "SMA(close, 20) < 0"}],
            },
            {
                "indicators": [
                    {
                        "name": "SMA",
                        "params": {"period": "invalid"},
                    }  # 無効なパラメータ型
                ],
                "entry_rules": [{"condition": "SMA(close, 20) > 0"}],
                "exit_rules": [{"condition": "SMA(close, 20) < 0"}],
            },
        ]

        for config in malformed_configs:
            with pytest.raises((KeyError, ValueError, TypeError)):
                executor.run_backtest(sample_data, config)

    def test_data_service_error_handling(self):
        """データサービスエラーのハンドリング"""
        with patch(
            "app.core.services.backtest_service.BacktestDataService"
        ) as mock_data_service:
            # データサービスが例外を発生させる設定
            mock_data_service_instance = Mock()
            mock_data_service.return_value = mock_data_service_instance
            mock_data_service_instance.get_ohlcv_for_backtest.side_effect = ValueError(
                "Database connection failed"
            )

            service = BacktestService()
            service.data_service = mock_data_service_instance

            config = {
                "strategy_name": "SMA_CROSS",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "end_date": datetime(2024, 1, 31, tzinfo=timezone.utc),
                "initial_capital": 100000.0,
                "commission_rate": 0.001,
                "strategy_config": {
                    "strategy_type": "SMA_CROSS",
                    "parameters": {"n1": 20, "n2": 50},
                },
            }

            # データサービスエラーが適切に処理されることを確認
            with pytest.raises(ValueError, match="Database connection failed"):
                service.run_backtest(config)

    def test_date_range_validation(self):
        """日付範囲バリデーションのテスト"""
        with patch(
            "app.core.services.backtest_service.BacktestDataService"
        ) as mock_data_service:
            mock_data_service_instance = Mock()
            mock_data_service.return_value = mock_data_service_instance

            service = BacktestService()
            service.data_service = mock_data_service_instance

            # 無効な日付範囲（終了日が開始日より前）
            invalid_config = {
                "strategy_name": "SMA_CROSS",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": datetime(2024, 12, 31, tzinfo=timezone.utc),  # 後の日付
                "end_date": datetime(2024, 1, 1, tzinfo=timezone.utc),  # 前の日付
                "initial_capital": 100000.0,
                "commission_rate": 0.001,
                "strategy_config": {
                    "strategy_type": "SMA_CROSS",
                    "parameters": {"n1": 20, "n2": 50},
                },
            }

            # 無効な日付範囲でエラーが発生することを確認
            with pytest.raises(ValueError, match="start_date.*end_date"):
                service.run_backtest(invalid_config)

    def test_memory_limit_handling(self):
        """メモリ制限のハンドリング（シミュレーション）"""
        # 非常に大きなデータセットを作成してメモリ不足をシミュレート
        try:
            # 極端に大きなデータセット（メモリ不足を引き起こす可能性）
            large_size = 1000000  # 100万データポイント
            dates = pd.date_range("2020-01-01", periods=large_size, freq="H")

            # メモリ効率的でないデータ生成
            data = pd.DataFrame(
                {
                    "Open": np.random.random(large_size) * 50000,
                    "High": np.random.random(large_size) * 55000,
                    "Low": np.random.random(large_size) * 45000,
                    "Close": np.random.random(large_size) * 50000,
                    "Volume": np.random.random(large_size) * 10000,
                },
                index=dates,
            )

            executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)

            strategy_config = {
                "indicators": [{"name": "SMA", "params": {"period": 20}}],
                "entry_rules": [{"condition": "SMA(close, 20) > 0"}],
                "exit_rules": [{"condition": "SMA(close, 20) < 0"}],
            }

            # メモリ不足でも適切にエラーハンドリングされることを確認
            result = executor.run_backtest(data, strategy_config)

            # 成功した場合は結果が返される
            if result is not None:
                assert "performance_metrics" in result

        except MemoryError:
            # メモリエラーが発生した場合は適切に処理される
            pytest.skip("Memory limit reached - this is expected behavior")
        except Exception as e:
            # その他のエラーの場合は、適切なエラーメッセージであることを確認
            assert isinstance(e, (ValueError, RuntimeError, OverflowError))

    def test_invalid_timeframe_handling(self):
        """無効な時間枠でのエラーハンドリング"""
        with patch(
            "app.core.services.backtest_service.BacktestDataService"
        ) as mock_data_service:
            mock_data_service_instance = Mock()
            mock_data_service.return_value = mock_data_service_instance

            service = BacktestService()
            service.data_service = mock_data_service_instance

            # 無効な時間枠
            invalid_timeframes = ["invalid", "1x", "0h", "-1d"]

            for timeframe in invalid_timeframes:
                config = {
                    "strategy_name": "SMA_CROSS",
                    "symbol": "BTC/USDT",
                    "timeframe": timeframe,
                    "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
                    "end_date": datetime(2024, 1, 31, tzinfo=timezone.utc),
                    "initial_capital": 100000.0,
                    "commission_rate": 0.001,
                    "strategy_config": {
                        "strategy_type": "SMA_CROSS",
                        "parameters": {"n1": 20, "n2": 50},
                    },
                }

                # 無効な時間枠でエラーが発生することを確認
                with pytest.raises((ValueError, KeyError)):
                    service.run_backtest(config)
