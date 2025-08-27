#!/usr/bin/env python3
"""
実際に取引が発生する戦略テスト
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ログレベルを設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_strategy(strategy_name, strategy_gene, description):
    """個別戦略のテスト"""
    print(f"\n{'='*80}")
    print(f"テスト戦略: {strategy_name}")
    print(f"説明: {description}")
    print(f"{'='*80}")

    try:
        from app.services.auto_strategy.generators.strategy_factory import (
            StrategyFactory,
        )
        from app.services.backtest.backtest_service import BacktestService
        from app.services.backtest.backtest_data_service import BacktestDataService
        from database.connection import SessionLocal
        from database.repositories.ohlcv_repository import OHLCVRepository
        from database.repositories.open_interest_repository import (
            OpenInterestRepository,
        )
        from database.repositories.funding_rate_repository import FundingRateRepository

        # データベースセッションとリポジトリを初期化
        with SessionLocal() as db:
            ohlcv_repo = OHLCVRepository(db)
            oi_repo = OpenInterestRepository(db)
            fr_repo = FundingRateRepository(db)

            data_service = BacktestDataService(
                ohlcv_repo=ohlcv_repo, oi_repo=oi_repo, fr_repo=fr_repo
            )
            backtest_service = BacktestService(data_service)

            # StrategyFactoryで戦略クラスを生成
            strategy_factory = StrategyFactory()
            strategy_class = strategy_factory.create_strategy_class(strategy_gene)

            # バックテスト設定
            config = {
                "strategy_name": strategy_name,
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "start_date": datetime.now() - timedelta(days=7),
                "end_date": datetime.now() - timedelta(days=1),
                "initial_capital": 10000000.0,  # 1000万円
                "commission_rate": 0.001,
                "strategy_class": strategy_class,
                "strategy_config": {
                    "strategy_gene": {
                        "id": strategy_gene.id,
                        "indicators": [
                            {
                                "type": ind.type,
                                "parameters": ind.parameters,
                                "enabled": ind.enabled,
                            }
                            for ind in strategy_gene.indicators
                        ],
                        "entry_conditions": [
                            {
                                "left_operand": cond.left_operand,
                                "operator": cond.operator,
                                "right_operand": cond.right_operand,
                            }
                            for cond in strategy_gene.entry_conditions
                        ],
                        "exit_conditions": [
                            {
                                "left_operand": cond.left_operand,
                                "operator": cond.operator,
                                "right_operand": cond.right_operand,
                            }
                            for cond in strategy_gene.exit_conditions
                        ],
                    }
                },
            }

            print("バックテスト実行中...")
            result = backtest_service.run_backtest(config)

            # 結果の表示
            metrics = result.get("performance_metrics", {})
            trade_history = result.get("trade_history", [])

            print(f"✅ {strategy_name} - 完了")
            print(f"   総取引数: {metrics.get('total_trades', 0)}")
            print(f"   最終資産: {metrics.get('final_equity', 0):,.0f}円")
            print(f"   利益率: {metrics.get('profit_factor', 0):.4f}")
            print(f"   取引履歴: {len(trade_history)}件")

            return True, metrics, trade_history

    except Exception as e:
        print(f"❌ {strategy_name} - エラー: {e}")
        import traceback

        traceback.print_exc()
        return False, {}, []


def create_working_strategies():
    """実際に取引が発生する戦略パターンを作成"""
    from app.services.auto_strategy.models.gene_strategy import (
        StrategyGene,
        IndicatorGene,
        Condition,
    )

    strategies = []

    # 戦略1: 緩いSTOCH戦略（取引が発生しやすい条件）
    strategies.append(
        {
            "name": "STOCH_Loose",
            "description": "緩いSTOCH戦略 - 取引が発生しやすい条件",
            "gene": StrategyGene(
                id="stoch_loose_001",
                indicators=[
                    IndicatorGene(
                        type="STOCH",
                        parameters={
                            "fastk_period": 5,
                            "slowk_period": 3,
                            "slowd_period": 3,
                        },
                        enabled=True,
                    )
                ],
                entry_conditions=[
                    Condition(
                        left_operand="STOCH_0", operator="<", right_operand=80.0
                    )  # 80未満（緩い条件）
                ],
                exit_conditions=[
                    Condition(
                        left_operand="STOCH_0", operator=">", right_operand=90.0
                    )  # 90超（緩い条件）
                ],
            ),
        }
    )

    # 戦略2: 常に取引するAO戦略
    strategies.append(
        {
            "name": "AO_Always_Trade",
            "description": "AO戦略 - 常に取引する条件",
            "gene": StrategyGene(
                id="ao_always_001",
                indicators=[IndicatorGene(type="AO", parameters={}, enabled=True)],
                entry_conditions=[
                    Condition(
                        left_operand="AO", operator=">", right_operand=-1000.0
                    )  # 非常に緩い条件
                ],
                exit_conditions=[
                    Condition(
                        left_operand="AO", operator="<", right_operand=1000.0
                    )  # 非常に緩い条件
                ],
            ),
        }
    )

    # 戦略3: 緩いKDJ戦略
    strategies.append(
        {
            "name": "KDJ_Loose",
            "description": "緩いKDJ戦略 - 取引が発生しやすい条件",
            "gene": StrategyGene(
                id="kdj_loose_001",
                indicators=[IndicatorGene(type="KDJ", parameters={}, enabled=True)],
                entry_conditions=[
                    Condition(
                        left_operand="KDJ_0", operator="<", right_operand=80.0
                    )  # 80未満（緩い条件）
                ],
                exit_conditions=[
                    Condition(
                        left_operand="KDJ_0", operator=">", right_operand=90.0
                    )  # 90超（緩い条件）
                ],
            ),
        }
    )

    # 戦略4: 複合指標戦略（STOCH + AO）
    strategies.append(
        {
            "name": "STOCH_AO_Combo",
            "description": "STOCH + AO複合戦略",
            "gene": StrategyGene(
                id="stoch_ao_combo_001",
                indicators=[
                    IndicatorGene(
                        type="STOCH",
                        parameters={
                            "fastk_period": 5,
                            "slowk_period": 3,
                            "slowd_period": 3,
                        },
                        enabled=True,
                    ),
                    IndicatorGene(type="AO", parameters={}, enabled=True),
                ],
                entry_conditions=[
                    Condition(
                        left_operand="STOCH_0", operator="<", right_operand=90.0
                    )  # 緩い条件
                ],
                exit_conditions=[
                    Condition(
                        left_operand="AO", operator=">", right_operand=-500.0
                    )  # 緩い条件
                ],
            ),
        }
    )

    # 戦略5: 極端に緩い条件の戦略
    strategies.append(
        {
            "name": "Always_Trade",
            "description": "常に取引する戦略 - テスト用",
            "gene": StrategyGene(
                id="always_trade_001",
                indicators=[
                    IndicatorGene(
                        type="STOCH",
                        parameters={
                            "fastk_period": 5,
                            "slowk_period": 3,
                            "slowd_period": 3,
                        },
                        enabled=True,
                    )
                ],
                entry_conditions=[
                    Condition(
                        left_operand="STOCH_0", operator=">", right_operand=0.0
                    )  # ほぼ常にTrue
                ],
                exit_conditions=[
                    Condition(
                        left_operand="STOCH_0", operator="<", right_operand=200.0
                    )  # ほぼ常にTrue
                ],
            ),
        }
    )

    return strategies


def main():
    """メイン関数"""
    print("🚀 実際に取引が発生する戦略テスト開始")
    print("=" * 80)

    strategies = create_working_strategies()
    results = []

    for strategy in strategies:
        success, metrics, trades = test_strategy(
            strategy["name"], strategy["gene"], strategy["description"]
        )

        results.append(
            {
                "name": strategy["name"],
                "success": success,
                "metrics": metrics,
                "trades": len(trades),
            }
        )

    # 総合結果の表示
    print(f"\n{'='*80}")
    print("📊 総合結果")
    print(f"{'='*80}")

    successful_strategies = [r for r in results if r["success"]]
    failed_strategies = [r for r in results if not r["success"]]
    trading_strategies = [
        r for r in successful_strategies if r["metrics"].get("total_trades", 0) > 0
    ]

    print(f"✅ 成功した戦略: {len(successful_strategies)}/{len(results)}")
    print(
        f"📈 取引が発生した戦略: {len(trading_strategies)}/{len(successful_strategies)}"
    )
    print(f"❌ 失敗した戦略: {len(failed_strategies)}/{len(results)}")

    if trading_strategies:
        print("\n🎯 取引が発生した戦略の詳細:")
        for result in trading_strategies:
            metrics = result["metrics"]
            print(f"  {result['name']}:")
            print(f"    取引数: {metrics.get('total_trades', 0)}")
            print(f"    利益率: {metrics.get('profit_factor', 0):.4f}")
            print(f"    最終資産: {metrics.get('final_equity', 0):,.0f}円")
            print(
                f"    リターン: {((metrics.get('final_equity', 10000000) / 10000000) - 1) * 100:.2f}%"
            )

    if successful_strategies and not trading_strategies:
        print("\n⚠️  すべての戦略で取引が発生しませんでした")
        print("成功した戦略（取引なし）:")
        for result in successful_strategies:
            metrics = result["metrics"]
            print(f"  {result['name']}: 取引数 {metrics.get('total_trades', 0)}")

    if failed_strategies:
        print("\n失敗した戦略:")
        for result in failed_strategies:
            print(f"  ❌ {result['name']}")

    print(f"\n🎯 テスト完了: {len(trading_strategies)} 戦略で取引が発生")


if __name__ == "__main__":
    main()
