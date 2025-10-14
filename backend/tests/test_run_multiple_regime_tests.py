import unittest
import os
import sys
import pandas as pd
import numpy as np
from unittest.mock import patch

# パス追加
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_multiple_regime_tests import run_multiple_regime_tests


class TestRunMultipleRegimeTests(unittest.TestCase):
    """複数回レジームテストのユニットテスト"""

    @patch("scripts.run_multiple_regime_tests.regime_based_backtest_comparison")
    def test_run_multiple_tests_structure(self, mock_comparison):
        """テスト実行の構造確認"""
        # モック設定
        mock_comparison.return_value = {
            "regime_results": {
                "trend": {
                    "performance_metrics": {"sharpe_ratio": 1.5, "max_drawdown": -0.1}
                },
                "range": {
                    "performance_metrics": {"sharpe_ratio": 0.8, "max_drawdown": -0.05}
                },
            },
            "summary": {},
        }

        # 実行
        result = run_multiple_regime_tests(num_runs=2)

        # 検証
        self.assertIn("results", result)
        self.assertIn("summary", result)
        self.assertIn("csv_path", result)

        # 結果数は 2 runs * 2 enabled/disabled = 4
        self.assertEqual(len(result["results"]), 4)

        # 各結果に必要なキーが含まれる
        for r in result["results"]:
            self.assertIn("run", r)
            self.assertIn("regime_adaptation_enabled", r)
            self.assertIn("avg_sharpe_ratio", r)
            self.assertIn("avg_max_drawdown", r)

    @patch("scripts.run_multiple_regime_tests.regime_based_backtest_comparison")
    def test_statistics_calculation(self, mock_comparison):
        """統計計算の確認"""
        # 適応有効: Sharpe 1.0, 1.2
        # 適応無効: Sharpe 0.5, 0.7
        mock_comparison.side_effect = [
            # enabled run1
            {
                "regime_results": {
                    "trend": {
                        "performance_metrics": {
                            "sharpe_ratio": 1.0,
                            "max_drawdown": -0.1,
                        }
                    }
                },
                "summary": {},
            },
            # enabled run2
            {
                "regime_results": {
                    "trend": {
                        "performance_metrics": {
                            "sharpe_ratio": 1.2,
                            "max_drawdown": -0.2,
                        }
                    }
                },
                "summary": {},
            },
            # disabled run1
            {
                "regime_results": {
                    "trend": {
                        "performance_metrics": {
                            "sharpe_ratio": 0.5,
                            "max_drawdown": -0.3,
                        }
                    }
                },
                "summary": {},
            },
            # disabled run2
            {
                "regime_results": {
                    "trend": {
                        "performance_metrics": {
                            "sharpe_ratio": 0.7,
                            "max_drawdown": -0.4,
                        }
                    }
                },
                "summary": {},
            },
        ]

        result = run_multiple_regime_tests(num_runs=2)

        summary = result["summary"]

        # 適応有効: mean 1.1, std 0.1
        self.assertAlmostEqual(summary["enabled"]["sharpe_mean"], 1.1, places=1)
        self.assertAlmostEqual(summary["enabled"]["sharpe_std"], 0.1, places=1)

        # 適応無効: mean 0.6, std 0.1
        self.assertAlmostEqual(summary["disabled"]["sharpe_mean"], 0.6, places=1)
        self.assertAlmostEqual(summary["disabled"]["sharpe_std"], 0.1, places=1)

    def test_csv_output(self):
        """CSV出力の確認"""
        # 一時的に実行してCSVが作成されるか確認
        with patch(
            "scripts.run_multiple_regime_tests.regime_based_backtest_comparison"
        ) as mock_comp:
            mock_comp.return_value = {
                "regime_results": {
                    "trend": {
                        "performance_metrics": {
                            "sharpe_ratio": 1.0,
                            "max_drawdown": -0.1,
                        }
                    }
                },
                "summary": {},
            }

            result = run_multiple_regime_tests(num_runs=1)

            csv_path = result["csv_path"]
            self.assertTrue(os.path.exists(csv_path))

            # CSV内容確認
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 2)  # enabled + disabled

            # クリーンアップ
            if os.path.exists(csv_path):
                os.remove(csv_path)


if __name__ == "__main__":
    unittest.main()
