#!/usr/bin/env python3
"""
複数回レジームテスト実行スクリプト
performance_comparison.pyを複数回実行し、パフォーマンス向上を確認
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# パス追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.performance_comparison import regime_based_backtest_comparison


def run_multiple_regime_tests(num_runs: int = 5) -> Dict[str, Any]:
    """
    複数回レジームテストを実行

    Args:
        num_runs: 実行回数

    Returns:
        集計結果
    """
    results = []

    for regime_enabled in [True, False]:
        for run in range(num_runs):
            # 異なるシード設定
            seed = 42 + run * 10
            np.random.seed(seed)

            # データサブセット: 開始日をランダム化
            base_start = pd.to_datetime('2023-01-01')
            start_offset = np.random.randint(0, 30)
            start_date = (base_start + pd.Timedelta(days=start_offset)).strftime('%Y-%m-%d')
            end_date = (pd.to_datetime(start_date) + pd.Timedelta(days=365)).strftime('%Y-%m-%d')

            # バックテスト実行
            result = regime_based_backtest_comparison(
                symbol="BTC/USDT",
                timeframe="1h",
                start_date=start_date,
                end_date=end_date,
                regime_adaptation_enabled=regime_enabled,
                output_csv=None,  # 個別出力なし
                output_plot=None
            )

            # レジーム別結果から平均メトリクス計算
            sharpe_values = []
            drawdown_values = []

            for regime_result in result['regime_results'].values():
                metrics = regime_result['performance_metrics']
                sharpe_values.append(metrics.get('sharpe_ratio', 0))
                drawdown_values.append(metrics.get('max_drawdown', 0))

            avg_sharpe = np.mean(sharpe_values) if sharpe_values else 0.0
            avg_drawdown = np.mean(drawdown_values) if drawdown_values else 0.0

            results.append({
                'run': run + 1,
                'regime_adaptation_enabled': regime_enabled,
                'seed': seed,
                'start_date': start_date,
                'end_date': end_date,
                'avg_sharpe_ratio': avg_sharpe,
                'avg_max_drawdown': avg_drawdown
            })

    # CSV保存
    df = pd.DataFrame(results)
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'multiple_regime_tests_results.csv')
    df.to_csv(csv_path, index=False)

    # 統計集計
    enabled_results = [r for r in results if r['regime_adaptation_enabled']]
    disabled_results = [r for r in results if not r['regime_adaptation_enabled']]

    enabled_sharpe = [r['avg_sharpe_ratio'] for r in enabled_results]
    disabled_sharpe = [r['avg_sharpe_ratio'] for r in disabled_results]
    enabled_drawdown = [r['avg_max_drawdown'] for r in enabled_results]
    disabled_drawdown = [r['avg_max_drawdown'] for r in disabled_results]

    summary = {
        'enabled': {
            'sharpe_mean': np.mean(enabled_sharpe),
            'sharpe_std': np.std(enabled_sharpe),
            'drawdown_mean': np.mean(enabled_drawdown),
            'drawdown_std': np.std(enabled_drawdown)
        },
        'disabled': {
            'sharpe_mean': np.mean(disabled_sharpe),
            'sharpe_std': np.std(disabled_sharpe),
            'drawdown_mean': np.mean(disabled_drawdown),
            'drawdown_std': np.std(disabled_drawdown)
        }
    }

    # コンソール出力
    print("\n=== 複数回レジームテスト結果 ===")
    print(f"実行回数: {num_runs}")
    print()
    print("レジーム適応有効:")
    print(".3f")
    print(".3f")
    print()
    print("レジーム適応無効:")
    print(".3f")
    print(".3f")

    return {
        'results': results,
        'summary': summary,
        'csv_path': csv_path
    }


if __name__ == "__main__":
    run_multiple_regime_tests()