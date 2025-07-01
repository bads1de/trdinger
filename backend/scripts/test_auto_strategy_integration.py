#!/usr/bin/env python3
"""
auto-strategy機能の統合テストスクリプト

指標初期化修正後、取引回数0問題が解決されたことを確認します。
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from app.core.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, IndicatorGene, Condition

# ログレベルを詳細に設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_auto_strategy_integration():
    """auto-strategy機能の統合テスト"""
    
    print("🔍 auto-strategy機能統合テスト開始")
    print("=" * 60)
    
    try:
        # 1. AutoStrategyServiceの初期化
        print("📦 AutoStrategyService初期化中...")
        service = AutoStrategyService()
        print(f"  ✅ AutoStrategyService初期化完了")
        
        # 2. より現実的な戦略遺伝子を作成
        print("\n🧬 現実的な戦略遺伝子作成中...")
        
        # RSI + SMA を使用した戦略
        indicators = [
            IndicatorGene(
                type="RSI",
                parameters={"period": 14},
                enabled=True
            ),
            IndicatorGene(
                type="SMA",
                parameters={"period": 20},
                enabled=True
            )
        ]
        
        # RSI < 30 AND close > SMA でエントリー
        entry_conditions = [
            Condition(
                left_operand="RSI",
                operator="<",
                right_operand=30
            ),
            Condition(
                left_operand="close",
                operator=">",
                right_operand="SMA"
            )
        ]
        
        # RSI > 70 OR close < SMA でイグジット
        exit_conditions = [
            Condition(
                left_operand="RSI",
                operator=">",
                right_operand=70
            ),
            Condition(
                left_operand="close",
                operator="<",
                right_operand="SMA"
            )
        ]
        
        gene = StrategyGene(
            id="INTEGRATION_TEST_001",
            indicators=indicators,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            risk_management={"stop_loss": 0.03, "take_profit": 0.06}
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
        
        # 4. バックテスト設定（より長期間でテスト）
        print("\n⚙️ バックテスト設定作成中...")
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",  # 2ヶ月間でテスト
            "initial_capital": 100000,
            "commission_rate": 0.001
        }
        print(f"  ✅ バックテスト設定: {backtest_config['symbol']} {backtest_config['timeframe']}")
        print(f"  📅 期間: {backtest_config['start_date']} - {backtest_config['end_date']}")
        
        # 5. 戦略テスト実行
        print("\n🚀 戦略テスト実行中...")
        print("  → test_strategy_generation()呼び出し...")
        
        result = service.test_strategy_generation(gene, backtest_config)
        
        print(f"\n📊 戦略テスト結果:")
        print(f"  成功: {result.get('success', False)}")
        
        if result.get('success'):
            print("  ✅ 戦略テスト成功")
            backtest_result = result.get('backtest_result', {})
            if backtest_result:
                trades_count = backtest_result.get('trades_count', 0)
                final_value = backtest_result.get('final_value', 0)
                return_pct = backtest_result.get('return_pct', 0)
                
                print(f"    📈 取引回数: {trades_count}")
                print(f"    💰 最終資産: {final_value:,.2f}")
                print(f"    📊 リターン: {return_pct:.2f}%")
                
                # 取引回数0問題の確認
                if trades_count > 0:
                    print("  🎉 取引回数0問題が解決されました！")
                    return True
                else:
                    print("  ⚠️ 取引回数が0です。条件が厳しすぎる可能性があります。")
                    return False
        else:
            print("  ❌ 戦略テスト失敗")
            errors = result.get('errors', [])
            if errors:
                print(f"    エラー: {errors}")
            return False
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_strategy():
    """より簡単な戦略でテスト"""
    
    print("\n🔍 簡単な戦略でのテスト開始")
    print("=" * 60)
    
    try:
        service = AutoStrategyService()
        
        # 非常に簡単な戦略（SMAクロス）
        indicators = [
            IndicatorGene(
                type="SMA",
                parameters={"period": 10},
                enabled=True
            ),
            IndicatorGene(
                type="SMA",
                parameters={"period": 20},
                enabled=True
            )
        ]
        
        # close > SMA_10 でエントリー
        entry_conditions = [
            Condition(
                left_operand="close",
                operator=">",
                right_operand="SMA"
            )
        ]
        
        # close < SMA_10 でイグジット
        exit_conditions = [
            Condition(
                left_operand="close",
                operator="<",
                right_operand="SMA"
            )
        ]
        
        gene = StrategyGene(
            id="SIMPLE_TEST_001",
            indicators=indicators,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            risk_management={"stop_loss": 0.05, "take_profit": 0.1}
        )
        
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": "2024-01-01",
            "end_date": "2024-04-01",  # 3ヶ月間
            "initial_capital": 100000,
            "commission_rate": 0.001
        }
        
        print(f"📊 簡単な戦略テスト: {len(gene.indicators)}個の指標")
        
        result = service.test_strategy_generation(gene, backtest_config)
        
        if result.get('success'):
            backtest_result = result.get('backtest_result', {})
            trades_count = backtest_result.get('trades_count', 0)
            print(f"  📈 取引回数: {trades_count}")
            
            if trades_count > 0:
                print("  🎉 簡単な戦略でも取引が発生しました！")
                return True
            else:
                print("  ⚠️ 簡単な戦略でも取引回数が0です。")
                return False
        else:
            print("  ❌ 簡単な戦略テスト失敗")
            return False
            
    except Exception as e:
        print(f"❌ 簡単な戦略テストエラー: {e}")
        return False


if __name__ == "__main__":
    print("🚀 auto-strategy統合テスト開始")
    
    # 1. 現実的な戦略テスト
    success1 = test_auto_strategy_integration()
    
    # 2. 簡単な戦略テスト
    success2 = test_simple_strategy()
    
    print(f"\n📊 統合テスト結果:")
    print(f"  現実的な戦略: {'✅ 成功' if success1 else '❌ 失敗'}")
    print(f"  簡単な戦略: {'✅ 成功' if success2 else '❌ 失敗'}")
    
    if success1 or success2:
        print("\n🎉 auto-strategy機能の取引回数0問題が解決されました！")
    else:
        print("\n⚠️ まだ問題が残っている可能性があります。")
