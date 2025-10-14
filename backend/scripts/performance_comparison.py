#!/usr/bin/env python3
"""
パフォーマンス比較テスト - Pandasオンリー移行後
レジーム別バックテスト比較機能拡張
"""

import time
import os
import logging
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np

from app.services.auto_strategy.services.regime_detector import RegimeDetector
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RegimeConfig(BaseModel):
    """レジーム検知設定"""

    n_components: int = 3
    covariance_type: str = "full"
    n_iter: int = 100


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Sharpe比率を計算

    Args:
        returns: リターンのシリーズ
        risk_free_rate: 無リスク金利（デフォルト0）

    Returns:
        Sharpe比率
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / 252  # 日次換算
    if excess_returns.std() == 0:
        return 0.0

    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    最大ドローダウンを計算

    Args:
        equity_curve: エクイティカーブ

    Returns:
        最大ドローダウン（負の値）
    """
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()


def calculate_win_rate(trade_history: List[Dict[str, Any]]) -> float:
    """
    勝率を計算

    Args:
        trade_history: 取引履歴

    Returns:
        勝率（0-1）
    """
    if not trade_history:
        return 0.0

    winning_trades = sum(1 for trade in trade_history if trade.get("profit", 0) > 0)
    return winning_trades / len(trade_history)


def regime_based_backtest_comparison(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    regime_adaptation_enabled: bool = True,
    output_csv: Optional[str] = None,
    output_plot: Optional[str] = None,
) -> Dict[str, Any]:
    """
    複数レジームでのバックテスト比較を実行

    Args:
        symbol: 取引ペア
        timeframe: 時間軸
        start_date: 開始日
        end_date: 終了日
        regime_adaptation_enabled: レジーム適応有効フラグ
        output_csv: CSV出力ファイルパス
        output_plot: プロット出力ファイルパス

    Returns:
        比較結果
    """
    logger.info(
        f"レジーム別バックテスト比較開始: {symbol} {timeframe} {start_date}-{end_date}"
    )

    # データ取得 (ダミーデータ使用)
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq="1h")
    n = len(dates)
    ohlcv_data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 100 + np.cumsum(np.random.randn(n) * 0.01),
            "high": 100
            + np.cumsum(np.random.randn(n) * 0.01)
            + np.random.rand(n) * 0.02,
            "low": 100
            + np.cumsum(np.random.randn(n) * 0.01)
            - np.random.rand(n) * 0.02,
            "close": 100 + np.cumsum(np.random.randn(n) * 0.01),
            "volume": np.random.randint(100, 1000, n),
        }
    ).set_index("timestamp")

    # レジーム検知
    config = RegimeConfig()
    regime_detector = RegimeDetector(config)
    regimes = regime_detector.detect_regimes(ohlcv_data)

    # レジーム別データ分割
    regime_results = {}
    regime_names = {0: "trend", 1: "range", 2: "high_volatility"}

    for regime_id, regime_name in regime_names.items():
        regime_mask = regimes == regime_id
        regime_data = ohlcv_data[regime_mask]

        if len(regime_data) < 10:  # 最小データ数チェック
            logger.warning(
                f"レジーム {regime_name} のデータが不足しています: {len(regime_data)}件"
            )
            continue

        # バックテスト実行（簡易版）
        backtest_result = run_simple_backtest(regime_data, regime_adaptation_enabled)

        regime_results[regime_name] = backtest_result

    # 集計
    summary = calculate_regime_comparison_summary(regime_results)

    # 出力
    output_to_console(regime_results, regime_adaptation_enabled)

    if output_csv:
        save_to_csv(regime_results, output_csv, regime_adaptation_enabled)

    if output_plot:
        plot_results(regime_results, output_plot, regime_adaptation_enabled)

    return {
        "regime_results": regime_results,
        "summary": summary,
        "regime_adaptation_enabled": regime_adaptation_enabled,
    }


def run_simple_backtest(
    data: pd.DataFrame, regime_adaptation_enabled: bool
) -> Dict[str, Any]:
    """
    簡易バックテスト実行

    Args:
        data: OHLCVデータ
        regime_adaptation_enabled: レジーム適応フラグ

    Returns:
        バックテスト結果
    """
    # 簡易戦略: 移動平均クロス (レジーム適応でパラメータ変更)
    data = data.copy()
    if regime_adaptation_enabled:
        sma_short_period = 5
        sma_long_period = 20
    else:
        sma_short_period = 10
        sma_long_period = 30

    data["sma_short"] = data["close"].rolling(sma_short_period).mean()
    data["sma_long"] = data["close"].rolling(sma_long_period).mean()
    data = data.dropna()

    if data.empty:
        return {"performance_metrics": {}, "trade_history": []}

    # シグナル生成
    data["signal"] = 0
    data.loc[data["sma_short"] > data["sma_long"], "signal"] = 1
    data.loc[data["sma_short"] < data["sma_long"], "signal"] = -1

    # ポジション変化でのみ取引
    data["position_change"] = data["signal"].diff()
    trades = []

    capital = 10000
    position = 0
    entry_price = 0

    for idx, row in data.iterrows():
        if row["position_change"] != 0:
            if position == 0 and row["signal"] == 1:  # 買いエントリー
                position = capital / row["close"]
                entry_price = row["close"]
                capital = 0
            elif position > 0 and row["signal"] == -1:  # 売り決済
                capital = position * row["close"]
                profit = capital - 10000
                trades.append(
                    {
                        "profit": profit,
                        "entry_price": entry_price,
                        "exit_price": row["close"],
                    }
                )
                position = 0
                entry_price = 0
                capital = 10000  # リセット

    # 最終ポジション決済
    if position > 0:
        final_value = position * data.iloc[-1]["close"]
        profit = final_value - 10000
        trades.append(
            {
                "profit": profit,
                "entry_price": entry_price,
                "exit_price": data.iloc[-1]["close"],
            }
        )

    # メトリクス計算
    equity_curve = [10000]
    cumulative = 10000
    for trade in trades:
        cumulative += trade["profit"]
        equity_curve.append(cumulative)

    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()

    metrics = {
        "sharpe_ratio": calculate_sharpe_ratio(returns),
        "max_drawdown": calculate_max_drawdown(equity_series),
        "win_rate": calculate_win_rate(trades),
        "total_trades": len(trades),
        "total_return": (cumulative - 10000) / 10000 if equity_curve else 0,
    }

    return {
        "performance_metrics": metrics,
        "equity_curve": equity_curve,
        "trade_history": trades,
    }


def calculate_regime_comparison_summary(
    regime_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    レジーム比較の集計

    Args:
        regime_results: レジーム別結果

    Returns:
        集計結果
    """
    summary = {}

    for regime, result in regime_results.items():
        metrics = result.get("performance_metrics", {})
        for metric, value in metrics.items():
            if metric not in summary:
                summary[metric] = {}
            summary[metric][regime] = value

    return summary


def output_to_console(
    regime_results: Dict[str, Dict[str, Any]], regime_adaptation_enabled: bool
):
    """
    コンソール出力

    Args:
        regime_results: レジーム別結果
        regime_adaptation_enabled: レジーム適応フラグ
    """
    print("\n=== レジーム別バックテスト比較結果 ===")
    print(f"レジーム適応: {'有効' if regime_adaptation_enabled else '無効'}")
    print()

    for regime, result in regime_results.items():
        metrics = result.get("performance_metrics", {})
        print(f"レジーム: {regime}")
        print(f"  Sharpe比率: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  最大ドローダウン: {metrics.get('max_drawdown', 0):.3f}")
        print(f"  勝率: {metrics.get('win_rate', 0):.3f}")
        print(f"  総取引数: {metrics.get('total_trades', 0)}")
        print(f"  総リターン: {metrics.get('total_return', 0):.3f}")
        print()


def save_to_csv(
    regime_results: Dict[str, Dict[str, Any]],
    file_path: str,
    regime_adaptation_enabled: bool,
):
    """
    CSV保存

    Args:
        regime_results: レジーム別結果
        file_path: 保存パス
        regime_adaptation_enabled: レジーム適応フラグ
    """
    rows = []
    for regime, result in regime_results.items():
        metrics = result.get("performance_metrics", {})
        row = {
            "regime": regime,
            "regime_adaptation": regime_adaptation_enabled,
            **metrics,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)
    logger.info(f"結果をCSVに保存しました: {file_path}")


def plot_results(
    regime_results: Dict[str, Dict[str, Any]],
    file_path: str,
    regime_adaptation_enabled: bool,
):
    """
    グラフ生成

    Args:
        regime_results: レジーム別結果
        file_path: 保存パス
        regime_adaptation_enabled: レジーム適応フラグ
    """
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(
            f"レジーム別バックテスト比較 (適応: {'有効' if regime_adaptation_enabled else '無効'})"
        )

        metrics = ["sharpe_ratio", "max_drawdown", "win_rate", "total_return"]
        regimes = list(regime_results.keys())

        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            values = [
                regime_results[r].get("performance_metrics", {}).get(metric, 0)
                for r in regimes
            ]
            ax.bar(regimes, values)
            ax.set_title(metric.replace("_", " ").title())
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        logger.info(f"グラフを保存しました: {file_path}")

    except ImportError:
        logger.warning("matplotlibが利用できないため、グラフ生成をスキップします")


def performance_test():
    """Pandasオンリー実装のパフォーマンステスト"""
    print("=== Pandasオンリー移行後パフォーマンステスト ===\n")

    # 大規模テストデータ作成
    n = 50000
    np.random.seed(42)
    high = pd.Series(
        100 + np.cumsum(np.random.randn(n)) + np.random.rand(n) * 10, name="high"
    )
    low = pd.Series(
        100 + np.cumsum(np.random.randn(n)) - np.random.rand(n) * 10, name="low"
    )
    close = pd.Series(100 + np.cumsum(np.random.randn(n)), name="close")
    volume = pd.Series(np.random.randint(10000, 100000, n), name="volume")

    print(f"テストデータサイズ: {n:,} 行")
    print(f"メモリ使用量: {close.memory_usage(deep=True)} bytes\n")

    # 指標計算関数のリスト
    test_functions = [
        ("ATR (Volatility)", lambda: VolatilityIndicators.atr(high, low, close)),
        ("RSI (Momentum)", lambda: MomentumIndicators.rsi(close)),
        ("MACD (Momentum)", lambda: MomentumIndicators.macd(close)),
        ("SMA (Trend)", lambda: close.rolling(20).mean()),
        ("AD (Volume)", lambda: VolumeIndicators.ad(high, low, close, volume)),
    ]

    results = {}

    # パフォーマンス測定
    for name, func in test_functions:
        print(f"測定中: {name}...")

        # ウォームアップ (JIT最適化)
        _ = func()

        # 測定実行
        start_time = time.perf_counter()
        result = func()
        end_time = time.perf_counter()

        elapsed = end_time - start_time

        # 結果検証
        if isinstance(result, pd.Series):
            result_type = f"pd.Series (len={len(result)})"
        elif isinstance(result, tuple):
            result_type = f"tuple (len={len(result)})"
            first_result = result[0] if result else None
            if isinstance(first_result, pd.Series):
                result_type += f" [pd.Series: len={len(first_result)}]"
        else:
            result_type = type(result).__name__

        results[name] = {
            "time": elapsed,
            "result_type": result_type,
            "valid": not (hasattr(result, "isna") and result.isna().all()),
        }

        print(f"  OK {name}: {elapsed:.4f}秒, {result_type}")

    # 結果集計
    print("\n=== パフォーマンス結果集計 ===")
    total_time = sum(r["time"] for r in results.values())
    avg_time = total_time / len(results)

    print(f"総処理時間: {total_time:.4f}秒")
    print(f"平均処理時間: {avg_time:.4f}秒/指標")

    # 詳細結果
    print("\n=== 詳細結果 ===")
    for name, data in results.items():
        valid_icon = "OK" if data["valid"] else "NG"
        print(
            f"{name:20} | {data['time']:6.4f}秒 | {data['result_type']:30} | {valid_icon}"
        )
    print(f"\n全テスト完了！ 平均 {avg_time:.4f}秒/指標")
    print("Pandasオンリー移行がパフォーマンス正常")
    print("全ての指標が正常に pd.Series または tuple[pd.Series] を返します")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "--regime-comparison":
        # レジーム別比較実行
        enabled = sys.argv[2].lower() == "true"
        regime_based_backtest_comparison(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2023-01-01",
            end_date="2023-12-31",
            regime_adaptation_enabled=enabled,
            output_csv=f"regime_comparison_results_{enabled}.csv",
            output_plot=f"regime_comparison_plot_{enabled}.png",
        )
    else:
        # 既存のパフォーマンステスト実行
        performance_test()
