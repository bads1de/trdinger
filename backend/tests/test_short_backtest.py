#!/usr/bin/env python3
"""
ショートバックテストテスト

確実にショートポジションが作成される戦略でバックテストを実行し、
取引履歴にショートポジションが含まれることを確認します。
"""

import sys
import os
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.services.backtest_service import BacktestService
from app.core.services.auto_strategy.models.gene_strategy import StrategyGene, IndicatorGene, Condition
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_simple_short_strategy():
    """確実にショートポジションが作成される簡単な戦略"""
    return StrategyGene(
        indicators=[
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
        ],
        long_entry_conditions=[
            Condition(left_operand="close", operator=">", right_operand=999999)  # 絶対に満たされない条件
        ],
        short_entry_conditions=[
            Condition(left_operand="close", operator=">", right_operand=1)  # ほぼ常に満たされる条件
        ],
        exit_conditions=[
            Condition(left_operand="close", operator="!=", right_operand=0)  # 常に決済
        ],
        risk_management={"position_size": 0.1},
    )

def create_alternating_strategy():
    """ロングとショートを交互に行う戦略"""
    return StrategyGene(
        indicators=[
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
        ],
        long_entry_conditions=[
            Condition(left_operand="close", operator=">", right_operand="open")  # 陽線でロング
        ],
        short_entry_conditions=[
            Condition(left_operand="close", operator="<", right_operand="open")  # 陰線でショート
        ],
        exit_conditions=[
            Condition(left_operand="close", operator="!=", right_operand="open")  # 常に決済
        ],
        risk_management={"position_size": 0.1},
    )

def run_short_backtest():
    """ショートバックテストを実行"""
    print("=== ショートバックテスト実行 ===")
    
    try:
        # バックテストサービスを初期化
        backtest_service = BacktestService()
        
        # テスト戦略を作成
        strategies = [
            ("Simple Short Strategy", create_simple_short_strategy()),
            ("Alternating Strategy", create_alternating_strategy()),
        ]
        
        for strategy_name, gene in strategies:
            print(f"\n--- {strategy_name} ---")
            
            # 戦略クラスを作成
            factory = StrategyFactory()
            strategy_class = factory.create_strategy_class(gene)
            
            # バックテスト設定
            config = {
                "strategy_name": strategy_name,
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-10",
                "initial_capital": 100000,
                "commission_rate": 0.001,
                "strategy_config": {
                    "strategy_type": "AUTO_STRATEGY",
                    "strategy_gene": gene,
                }
            }
            
            # バックテストを実行
            result = backtest_service.run_backtest(config)
            
            # 結果を分析
            trade_history = result.get("trade_history", [])
            print(f"総取引数: {len(trade_history)}")
            
            if trade_history:
                long_trades = [t for t in trade_history if t.get("size", 0) > 0]
                short_trades = [t for t in trade_history if t.get("size", 0) < 0]
                
                print(f"ロング取引: {len(long_trades)}")
                print(f"ショート取引: {len(short_trades)}")
                
                # 最初の数取引を表示
                print("最初の5取引:")
                for i, trade in enumerate(trade_history[:5]):
                    size = trade.get("size", 0)
                    direction = "LONG" if size > 0 else "SHORT"
                    print(f"  {i+1}. {direction} - サイズ: {size:.4f}")
                
                if short_trades:
                    print("✅ ショート取引が確認されました！")
                else:
                    print("❌ ショート取引が見つかりませんでした")
            else:
                print("❌ 取引が発生しませんでした")
                
    except Exception as e:
        print(f"❌ バックテストエラー: {e}")
        import traceback
        traceback.print_exc()

def main():
    """メイン実行"""
    print("🔍 ショートバックテストテスト開始\n")
    
    try:
        run_short_backtest()
        
        print("\n✅ テスト完了")
        print("\n📋 確認ポイント:")
        print("1. ショート取引が実際に発生するか")
        print("2. 取引履歴でショートポジションが記録されるか")
        print("3. サイズが負の値になっているか")
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
