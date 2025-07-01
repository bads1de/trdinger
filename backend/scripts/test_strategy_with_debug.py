#!/usr/bin/env python3
"""
戦略テスト機能を使用してデバッグログを確認するスクリプト

auto-strategy機能の指標初期化問題を調査します。
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from app.core.services.auto_strategy.services.auto_strategy_service import (
    AutoStrategyService,
)
from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene,
    IndicatorGene,
    Condition,
)

# ログレベルを詳細に設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_strategy_with_debug():
    """戦略テスト機能を使用してデバッグログを確認"""

    print("🔍 戦略テスト機能デバッグ開始")
    print("=" * 60)

    try:
        # 1. AutoStrategyServiceの初期化
        print("📦 AutoStrategyService初期化中...")
        service = AutoStrategyService()
        print(f"  ✅ AutoStrategyService初期化完了")

        # 2. 簡単な戦略遺伝子を作成
        print("\n🧬 戦略遺伝子作成中...")

        # RSI指標のみを使用した簡単な戦略
        indicators = [
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)
        ]

        entry_conditions = [
            Condition(left_operand="RSI", operator="<", right_operand=30)
        ]

        exit_conditions = [
            Condition(left_operand="RSI", operator=">", right_operand=70)
        ]

        gene = StrategyGene(
            id="DEBUG_TEST_001",
            indicators=indicators,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            risk_management={"stop_loss": 0.05, "take_profit": 0.1},
        )

        print(f"  ✅ 戦略遺伝子作成完了: ID {gene.id}")
        print(f"  📊 指標数: {len(gene.indicators)}")
        print(f"  📊 エントリー条件数: {len(gene.entry_conditions)}")
        print(f"  📊 イグジット条件数: {len(gene.exit_conditions)}")

        # 3. 戦略遺伝子の妥当性検証
        print("\n🔍 戦略遺伝子妥当性検証中...")
        is_valid, errors = gene.validate()

        if is_valid:
            print("  ✅ 戦略遺伝子は有効です")
        else:
            print(f"  ❌ 戦略遺伝子が無効: {errors}")
            return False

        # 4. バックテスト設定
        print("\n⚙️ バックテスト設定作成中...")
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",  # 短期間でテスト
            "initial_capital": 100000,
            "commission_rate": 0.001,
        }
        print(
            f"  ✅ バックテスト設定: {backtest_config['symbol']} {backtest_config['timeframe']}"
        )
        print(
            f"  📅 期間: {backtest_config['start_date']} - {backtest_config['end_date']}"
        )

        # 5. 戦略テスト実行
        print("\n🚀 戦略テスト実行中...")
        print("  → test_strategy_generation()呼び出し...")

        result = service.test_strategy_generation(gene, backtest_config)

        print(f"\n📊 戦略テスト結果:")
        print(f"  成功: {result.get('success', False)}")

        if result.get("success"):
            print("  ✅ 戦略テスト成功")
            backtest_result = result.get("backtest_result", {})
            if backtest_result:
                print(f"    取引回数: {backtest_result.get('trades_count', 'N/A')}")
                print(f"    最終資産: {backtest_result.get('final_value', 'N/A')}")
                print(f"    リターン: {backtest_result.get('return_pct', 'N/A')}%")
        else:
            print("  ❌ 戦略テスト失敗")
            errors = result.get("errors", [])
            if errors:
                print(f"    エラー: {errors}")

        return result.get("success", False)

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_strategy_with_debug()
