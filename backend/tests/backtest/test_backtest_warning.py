#!/usr/bin/env python3
"""
バックテスト実行の警告メッセージテスト
修正されたメッセージを確認
"""

import sys
import os

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_backtest_warning():
    """バックテスト警告メッセージをテスト"""
    print("=" * 60)
    print("Backtest Warning Message Test")
    print("=" * 60)

    try:
        # 最小限のオートストラテジー戦略遺伝子を作成
        from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
        from app.services.auto_strategy.serializers.gene_serialization import GeneSerializer
        from app.services.auto_strategy.models.strategy_models import StrategyGene

        print("Creating minimal strategy gene...")

        # 空の取引条件だけを持つ最小限の戦略遺伝子
        minimal_gene_data = {
            "gene_id": "test_no_trade_001",
            "factors": [],
            "conditions": [],  # 条件なしで取引が発生しない戦略
            "tpsl_rules": {
                "take_profit": {"type": "percentage", "value": 2.0},
                "stop_loss": {"type": "percentage", "value": 1.0}
            },
            "indicators": [],
            "trajectory_type": "trend_following"
        }

        # 戦略遺伝子を構築
        serializer = GeneSerializer()
        minimal_gene = serializer.dict_to_strategy_gene(minimal_gene_data, StrategyGene)

        # 戦略クラスを生成
        strategy_factory = StrategyFactory()
        strategy_class = strategy_factory.create_strategy_class(minimal_gene)

        print("Strategy class created successfully!")
        print("Running backtest with no-trade strategy...")

        # オートストラテジーの設定でバックテストを実行
        from app.services.backtest.backtest_service import BacktestService
        backtest_service = BacktestService()

        config = {
            "strategy_name": "Test_No_Trade_Auto",
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01T00:00:00",
            "end_date": "2023-01-07T00:00:00",  # 短期間でテスト
            "initial_capital": 100000.0,
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "GENERATED_AUTO",
                "parameters": {
                    "strategy_gene": serializer.strategy_gene_to_dict(minimal_gene)
                }
            }
        }

        print("Running backtest...")
        result = backtest_service.run_backtest(config)

        print("\nTest Results:")
        print("=" * 30)
        metrics = result.get("performance_metrics", {})
        print(f"Total Trades: {metrics.get('total_trades', 'N/A')}")
        print(f"Total Return: {metrics.get('total_return', 'N/A')}")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
        print(f"Win Rate: {metrics.get('win_rate', 'N/A')}")

        trade_history = result.get("trade_history", [])
        print(f"Trade History Length: {len(trade_history)}")

        print("\n[SUCCESS] Backtest completed! Check logs for warning messages.")

    except Exception as e:
        print(f"Error in backtest test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_backtest_warning()