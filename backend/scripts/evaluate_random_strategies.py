"""
ランダム生成戦略を1年の合成データで評価し、Markdownレポートを作成

- 時間軸: 1h（1時間足）
- 期間: 1年（例: 2023-01-01 〜 2023-12-31）
- モード: テクニカル指標のみ（auto-strategy仕様に準拠）
- 出力: backend/reports/auto_strategy_year_synthetic_report.md

合格基準:
- 最低100トレード以上
- シャープレシオ1.0以上
- 勝率50%以上

注意:
- 実運用DBは触らず、合成データで BacktestExecutor を直接使います
- サーバー起動は不要
"""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# このスクリプトのフォルダ（backend/scripts）から backend を特定し、レポート出力先を安定化
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_REPORTS_DIR = os.path.join(_BACKEND_DIR, "reports")

# 'backend' を import path に追加
import sys

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
from app.services.backtest.execution.backtest_executor import BacktestExecutor
from app.services.backtest.conversion.backtest_result_converter import (
    BacktestResultConverter,
)
from app.services.indicators.config import indicator_registry
from tests.common.test_stubs import _SyntheticDataService


def _make_one_year_hourly_data(start: datetime, bars: int = 24 * 365) -> pd.DataFrame:
    """1時間足の合成価格データを作成（トレンド + 周期 + ノイズ）"""
    idx = pd.date_range(start=start, periods=bars, freq="1h")
    # 緩やかな上昇トレンド + 波動 + ランダムノイズ
    t = np.linspace(0, 10 * np.pi, bars)
    trend = np.linspace(100, 200, bars)
    wavy = 5 * np.sin(t) + 3 * np.sin(2.5 * t)
    noise = np.random.normal(0, 0.4, size=bars)
    close = trend + wavy + noise
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.random.normal(0.0005, 0.0003, size=bars))
    low = np.minimum(open_, close) * (1 - np.random.normal(0.0005, 0.0003, size=bars))
    vol = np.full(bars, 1000.0)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def check_passing_criteria(metrics: Dict[str, float]) -> bool:
    """
    合格基準をチェック

    Args:
        metrics: 戦略のメトリクス

    Returns:
        合格基準を満たすかどうか
    """
    return (
        metrics["num_trades"] >= 100
        and metrics["sharpe_ratio"] >= 1.0
        and metrics["win_rate"] >= 50.0
    )


def evaluate_strategies(n: int = 250, seed: int = 42) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)

    # 合成データサービスとして BacktestExecutor に渡すための簡易クラス
    data_service = _SyntheticDataService(data_generator_func=_make_one_year_hourly_data)
    executor = BacktestExecutor(data_service)
    factory = StrategyFactory()
    converter = BacktestResultConverter()

    # ログ抑制（Auto-Strategyまわりの冗長ログをWARNING以上に）
    import logging

    logging.getLogger("app.services.auto_strategy").setLevel(logging.WARNING)

    start = datetime(2023, 1, 1)
    end = datetime(2024, 1, 1) - timedelta(hours=1)

    ga_cfg = GAConfig(
        indicator_mode="technical_only",
        max_indicators=5,
        min_indicators=3,
        max_conditions=7,
        min_conditions=3,
        # VALID_INDICATOR_TYPESに含まれる安全な指標のみを使用
        allowed_indicators=[
            "RSI",
            "SMA",
            "EMA",
            "MACD",
            "BBANDS",
            "ATR",
            "STOCH",
            "CCI",
            "WILLR",
            "MFI",
            "ADX",
            "AROON",
            "ROC",
            "MOM",
            "TRIX",
            "UO",
            "CMO",
            "DX",
            "MINUS_DI",
            "PLUS_DI",
            "WMA",
            "MACDFIX",
            "MACDEXT",
            "STOCHRSI",
            "KAMA",
            "T3",
            "TRIMA",
            "PPO",
            "APO",
            "AROONOSC",
            "BOP",
        ],
    )

    results: List[Dict[str, Any]] = []

    for i in range(n):
        try:
            # ランダム戦略生成（aggressive profile と 1h を注入）
            gene = RandomGeneGenerator(
                config=ga_cfg,
                enable_smart_generation=True,
                smart_context={"timeframe": "1h", "threshold_profile": "aggressive"},
            ).generate_random_gene()
            strategy_class = factory.create_strategy_class(gene)
        except Exception as e:
            print(f"戦略 #{i+1} 生成エラー: {e}")
            continue

        try:
            # バックテスト実行
            stats = executor.execute_backtest(
                strategy_class=strategy_class,
                strategy_parameters={"strategy_gene": gene},
                symbol="BTC:USDT",
                timeframe="1h",
                start_date=start,
                end_date=end,
                initial_capital=10000.0,
                commission_rate=0.001,
            )
        except Exception as e:
            print(f"戦略 #{i+1} バックテストエラー: {e}")
            continue

        stats_dict = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
        # 重要メトリクスを抽出
        metrics = {
            "total_return": float(stats_dict.get("Return [%]", 0.0)),
            "sharpe_ratio": float(stats_dict.get("Sharpe Ratio", 0.0)),
            "max_drawdown": float(stats_dict.get("Max. Drawdown [%]", 0.0)),
            "win_rate": float(stats_dict.get("Win Rate [%]", 0.0)),
            "num_trades": int(stats_dict.get("# Trades", 0)),
            "profit_factor": float(stats_dict.get("Profit Factor", 0.0)),
        }

        # カテゴリだけざっくり
        categories = []
        for ind in gene.indicators:
            cfg = indicator_registry.get_indicator_config(ind.type)
            if cfg and getattr(cfg, "category", None):
                categories.append(cfg.category)

        # 合格基準チェック
        is_passing = check_passing_criteria(metrics)

        results.append(
            {
                "index": i + 1,
                "metrics": metrics,
                "categories": sorted(set(categories)),
                "gene": gene,
                "is_passing": is_passing,
            }
        )

    # 合格基準分析
    passing_strategies = [r for r in results if r["is_passing"]]
    trading_strategies = [r for r in results if r["metrics"]["num_trades"] > 0]

    # レポート生成
    os.makedirs("backend/reports", exist_ok=True)
    report_path = os.path.join(
        "backend/reports", "auto_strategy_year_synthetic_report.md"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Auto-Strategy Year-long Synthetic Backtest Report\n\n")
        f.write("- Timeframe: 1h\n")
        f.write("- Period: 2023-01-01 .. 2023-12-31\n")
        f.write(f"- Strategies evaluated: {n}\n")
        f.write("- Mode: Technical indicators only\n\n")

        f.write("## 合格基準\n")
        f.write("- 最低100トレード以上\n")
        f.write("- シャープレシオ1.0以上\n")
        f.write("- 勝率50%以上\n\n")

        f.write("## 結果サマリー\n")
        f.write(
            f"- 取引実行戦略: {len(trading_strategies)}/{n} ({len(trading_strategies)/n:.1%})\n"
        )
        f.write(
            f"- **合格戦略: {len(passing_strategies)}/{n} ({len(passing_strategies)/n:.1%})**\n\n"
        )

        # 合格戦略の詳細
        if passing_strategies:
            f.write("## 🎉 合格戦略詳細\n\n")
            for r in passing_strategies:
                m = r["metrics"]
                f.write(f"### Strategy #{r['index']} ✅\n")
                f.write(
                    f"- **Trades: {m['num_trades']}, Sharpe: {m['sharpe_ratio']:.3f}, Return[%]: {m['total_return']:.2f}**\n"
                )
                f.write(
                    f"- MaxDD[%]: {m['max_drawdown']:.2f}, WinRate[%]: {m['win_rate']:.2f}, PF: {m['profit_factor']:.2f}\n"
                )
                f.write(
                    f"- Categories: {', '.join(r['categories']) if r['categories'] else 'n/a'}\n"
                )
                # インジケータ簡易ダンプ
                inds = ", ".join([ind.type for ind in r["gene"].indicators])
                f.write(f"- Indicators: {inds}\n\n")
        else:
            f.write("## ❌ 合格戦略\n\n")
            f.write("合格基準を満たす戦略は見つかりませんでした。\n\n")

        # 上位の指標（全体から）
        sorted_res = sorted(
            results,
            key=lambda r: (r["metrics"]["sharpe_ratio"], r["metrics"]["total_return"]),
            reverse=True,
        )
        top = sorted_res[: min(5, len(sorted_res))]

        f.write("## Top 5 strategies (by Sharpe, Return)\n\n")
        for r in top:
            m = r["metrics"]
            status = "✅" if r["is_passing"] else "❌"
            f.write(f"### Strategy #{r['index']} {status}\n")
            f.write(
                f"- Trades: {m['num_trades']}, Sharpe: {m['sharpe_ratio']:.3f}, Return[%]: {m['total_return']:.2f}, MaxDD[%]: {m['max_drawdown']:.2f}, WinRate[%]: {m['win_rate']:.2f}, PF: {m['profit_factor']:.2f}\n"
            )
            f.write(
                f"- Categories: {', '.join(r['categories']) if r['categories'] else 'n/a'}\n"
            )
            # インジケータ簡易ダンプ
            inds = ", ".join([ind.type for ind in r["gene"].indicators])
            f.write(f"- Indicators: {inds}\n\n")

        # 取引回数での上位
        top_by_trades = sorted(
            results, key=lambda r: r["metrics"]["num_trades"], reverse=True
        )[: min(5, len(results))]
        f.write("## Top strategies (by Trades)\n\n")
        for r in top_by_trades:
            m = r["metrics"]
            f.write(f"### Strategy #{r['index']}\n")
            f.write(
                f"- Trades: {m['num_trades']}, Sharpe: {m['sharpe_ratio']:.3f}, Return[%]: {m['total_return']:.2f}, MaxDD[%]: {m['max_drawdown']:.2f}, WinRate[%]: {m['win_rate']:.2f}, PF: {m['profit_factor']:.2f}\n\n"
            )

        # 多様性分析
        f.write("## 戦略多様性分析\n\n")
        all_categories = []
        all_indicators = []
        for r in results:
            all_categories.extend(r["categories"])
            all_indicators.extend([ind.type for ind in r["gene"].indicators])

        unique_categories = sorted(set(all_categories))
        unique_indicators = sorted(set(all_indicators))

        f.write(f"- 使用された指標カテゴリ: {len(unique_categories)}種類\n")
        f.write(f"  - {', '.join(unique_categories)}\n")
        f.write(f"- 使用された指標: {len(unique_indicators)}種類\n")
        f.write(
            f"  - {', '.join(unique_indicators[:10])}{'...' if len(unique_indicators) > 10 else ''}\n\n"
        )

        f.write("## All strategies summary\n\n")
        for r in results:
            m = r["metrics"]
            status = "✅" if r["is_passing"] else "❌"
            line = (
                f"#{r['index']:02d} {status} | Trades {m['num_trades']:3d} | Sharpe {m['sharpe_ratio']:.3f} | "
                f"Ret[%] {m['total_return']:.2f} | MaxDD[%] {m['max_drawdown']:.2f} | WinRate[%] {m['win_rate']:.2f} | PF {m['profit_factor']:.2f} | "
                f"{', '.join(r['categories']) if r['categories'] else 'n/a'}\n"
            )
            f.write(line)

    return {
        "report_path": report_path,
        "results": results,
        "passing_strategies": passing_strategies,
        "total_evaluated": n,
        "passing_count": len(passing_strategies),
    }


if __name__ == "__main__":
    print("🚀 オートストラテジー評価開始...")
    print("- テクニカル指標のみのモード")
    print("- 1年分の合成データ")
    print("- 合格基準: 100トレード以上、シャープレシオ1.0以上、勝率50%以上")
    print()

    out = evaluate_strategies(n=250, seed=42)

    print(f"📊 評価完了!")
    print(f"- 評価戦略数: {out['total_evaluated']}")
    print(f"- 合格戦略数: {out['passing_count']}")
    print(f"- 合格率: {out['passing_count']/out['total_evaluated']:.1%}")
    print(f"- レポート: {out['report_path']}")

    if out["passing_count"] > 0:
        print("✅ 合格基準を満たす戦略が見つかりました！")
    else:
        print("❌ 合格基準を満たす戦略は見つかりませんでした。")
        print("   パラメータ調整や評価数増加を検討してください。")
