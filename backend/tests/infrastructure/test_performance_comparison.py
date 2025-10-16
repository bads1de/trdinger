"""レジーム別バックテスト比較のパフォーマンステスト"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from app.services.backtest.backtest_service import BacktestService
from app.services.auto_strategy.services.regime_detector import RegimeDetector


class TestPerformanceComparison:
    """パフォーマンス比較テストクラス"""

    @pytest.fixture
    def sample_backtest_result(self):
        """サンプルバックテスト結果"""
        return {
            "performance_metrics": {
                "sharpe_ratio": 1.5,
                "max_drawdown": -0.15,
                "win_rate": 0.65,
                "total_return": 0.25,
                "total_trades": 100,
                "profit_factor": 1.8,
            },
            "equity_curve": [1.0, 1.02, 0.98, 1.05],
            "trade_history": [
                {"profit": 100, "entry_price": 100, "exit_price": 105},
                {"profit": -50, "entry_price": 105, "exit_price": 100},
            ],
        }

    @pytest.fixture
    def regime_detector_mock(self):
        """レジーム検知モック"""
        mock = Mock(spec=RegimeDetector)
        mock.detect_regimes.return_value = np.array(
            [0, 0, 1, 1, 2, 2]
        )  # trend, range, high_vol
        return mock

    @pytest.fixture
    def backtest_service_mock(self):
        """バックテストサービスモック"""
        mock = Mock(spec=BacktestService)
        mock.run_backtest.return_value = {
            "performance_metrics": {
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.12,
                "win_rate": 0.58,
                "total_return": 0.18,
                "total_trades": 80,
                "profit_factor": 1.5,
            }
        }
        return mock

    def test_calculate_sharpe_ratio(self, sample_backtest_result):
        """Sharpe比率計算テスト"""
        from backend.scripts.performance_comparison import calculate_sharpe_ratio

        equity_curve = pd.Series(sample_backtest_result["equity_curve"])
        returns = equity_curve.pct_change().dropna()

        sharpe = calculate_sharpe_ratio(returns)

        # 正の値であることを確認
        assert isinstance(sharpe, float)
        assert sharpe > 0

    def test_calculate_max_drawdown(self, sample_backtest_result):
        """最大ドローダウン計算テスト"""
        from backend.scripts.performance_comparison import calculate_max_drawdown

        equity_curve = pd.Series(sample_backtest_result["equity_curve"])

        max_dd = calculate_max_drawdown(equity_curve)

        # 負の値であることを確認（ドローダウンは負）
        assert isinstance(max_dd, float)
        assert max_dd <= 0

    def test_calculate_win_rate(self, sample_backtest_result):
        """勝率計算テスト"""
        from backend.scripts.performance_comparison import calculate_win_rate

        trade_history = sample_backtest_result["trade_history"]

        win_rate = calculate_win_rate(trade_history)

        # 0-1の範囲であることを確認
        assert isinstance(win_rate, float)
        assert 0 <= win_rate <= 1

        # サンプルデータでは勝率0.5になるはず
        assert win_rate == 0.5

    def test_regime_based_comparison_basic(
        self, regime_detector_mock, backtest_service_mock
    ):
        """レジーム別比較基本テスト"""
        # モックデータを設定
        ohlcv_data = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104, 105],
                "high": [105, 106, 107, 108, 109, 110],
                "low": [95, 96, 97, 98, 99, 100],
                "close": [102, 103, 104, 105, 106, 107],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500],
            }
        )

        with (
            patch(
                "backend.scripts.performance_comparison.RegimeDetector",
                return_value=regime_detector_mock,
            ),
            patch(
                "backend.scripts.performance_comparison.BacktestService",
                return_value=backtest_service_mock,
            ),
            patch(
                "backend.scripts.performance_comparison.OHLCVRepository"
            ) as mock_repo_class,
        ):

            mock_repo = Mock()
            mock_repo.get_ohlcv_dataframe.return_value = ohlcv_data
            mock_repo_class.return_value = mock_repo

            from backend.scripts.performance_comparison import (
                regime_based_backtest_comparison,
            )

            results = regime_based_backtest_comparison(
                symbol="BTC/USDT",
                timeframe="1h",
                start_date="2023-01-01",
                end_date="2023-01-02",
                regime_adaptation_enabled=True,
            )

            # 結果が辞書であることを確認
            assert isinstance(results, dict)
            assert "regime_results" in results
            assert "summary" in results

    def test_output_to_console(self, sample_backtest_result):
        """コンソール出力テスト"""
        from backend.scripts.performance_comparison import output_to_console

        regime_results = {
            "trend": sample_backtest_result["performance_metrics"],
            "range": sample_backtest_result["performance_metrics"],
            "high_volatility": sample_backtest_result["performance_metrics"],
        }

        # 出力がエラーなく実行されることを確認
        try:
            output_to_console(regime_results, regime_adaptation_enabled=True)
        except Exception as e:
            pytest.fail(f"コンソール出力でエラー発生: {e}")

    def test_save_to_csv(self, sample_backtest_result):
        """CSV保存テスト"""
        from backend.scripts.performance_comparison import save_to_csv

        regime_results = {
            "trend": sample_backtest_result["performance_metrics"],
            "range": sample_backtest_result["performance_metrics"],
            "high_volatility": sample_backtest_result["performance_metrics"],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_results.csv")

            # CSV保存がエラーなく実行されることを確認
            try:
                save_to_csv(regime_results, file_path, regime_adaptation_enabled=True)
                # ファイルが作成されたことを確認
                assert os.path.exists(file_path)
            except Exception as e:
                pytest.fail(f"CSV保存でエラー発生: {e}")

    def test_plot_results(self, sample_backtest_result):
        """グラフ生成テスト"""
        from backend.scripts.performance_comparison import plot_results

        regime_results = {
            "trend": sample_backtest_result["performance_metrics"],
            "range": sample_backtest_result["performance_metrics"],
            "high_volatility": sample_backtest_result["performance_metrics"],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_plot.png")

            # matplotlibが利用可能でない場合を考慮
            try:
                plot_results(regime_results, file_path, regime_adaptation_enabled=True)
                # ファイルが作成されたことを確認（matplotlibが利用可能な場合）
                if hasattr(pd, "plot"):  # matplotlibが利用可能かの簡易チェック
                    assert os.path.exists(file_path)
            except ImportError:
                # matplotlibが利用できない場合はスキップ
                pytest.skip("matplotlibが利用できません")
            except Exception as e:
                pytest.fail(f"グラフ生成でエラー発生: {e}")

    def test_metrics_calculation_accuracy(self):
        """メトリクス計算精度テスト"""
        from backend.scripts.performance_comparison import (
            calculate_sharpe_ratio,
            calculate_max_drawdown,
            calculate_win_rate,
        )

        # テストデータ作成
        # Sharpe比率テスト
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005])
        sharpe = calculate_sharpe_ratio(returns)
        assert abs(sharpe - 7.4) < 0.1  # 概算値

        # 最大ドローダウンテスト
        equity = pd.Series([1.0, 1.05, 1.02, 0.98, 1.03])
        max_dd = calculate_max_drawdown(equity)
        assert max_dd < 0  # 負の値
        assert abs(max_dd - -0.067) < 0.01  # 概算値

        # 勝率テスト
        trades = [{"profit": 100}, {"profit": -50}, {"profit": 200}, {"profit": -25}]
        win_rate = calculate_win_rate(trades)
        assert win_rate == 0.5  # 2勝2敗
