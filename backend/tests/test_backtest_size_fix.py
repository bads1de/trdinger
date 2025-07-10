#!/usr/bin/env python3
"""
バックテストでのポジションサイズエラー修正テスト
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.services.backtest_service import BacktestService
from app.core.services.auto_strategy.models.gene_strategy import StrategyGene, IndicatorGene, Condition
from app.core.services.auto_strategy.models.gene_position_sizing import (
    PositionSizingGene,
    PositionSizingMethod,
)
from app.core.services.auto_strategy.models.gene_tpsl import TPSLGene, TPSLMethod


def test_backtest_size_fix():
    """バックテストでのポジションサイズエラー修正テスト"""
    print("=== バックテストポジションサイズエラー修正テスト ===\n")
    
    # 様々なポジションサイズ設定でテスト
    test_cases = [
        {
            "name": "小数ポジションサイズ（割合）",
            "position_sizing_gene": PositionSizingGene(
                method=PositionSizingMethod.FIXED_RATIO,
                fixed_ratio=0.1,  # 10%（0.1未満になる可能性）
                enabled=True,
            ),
        },
        {
            "name": "大きなポジションサイズ（整数）",
            "position_sizing_gene": PositionSizingGene(
                method=PositionSizingMethod.FIXED_QUANTITY,
                fixed_quantity=5.7,  # 6に丸められる
                enabled=True,
            ),
        },
        {
            "name": "ボラティリティベース",
            "position_sizing_gene": PositionSizingGene(
                method=PositionSizingMethod.VOLATILITY_BASED,
                risk_per_trade=0.02,  # 2%
                atr_multiplier=2.0,
                enabled=True,
            ),
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case['name']}")
        
        # TP/SL遺伝子を作成
        tpsl_gene = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            stop_loss_pct=0.02,  # 2%
            take_profit_pct=0.04,  # 4%
            enabled=True,
        )
        
        # シンプルな戦略遺伝子を作成
        strategy_gene = StrategyGene(
            id=f"size_fix_test_{i}",
            indicators=[
                IndicatorGene(
                    type="SMA",
                    parameters={"period": 10},
                    enabled=True,
                    json_config={"indicator_name": "SMA", "parameters": {"period": 10}},
                )
            ],
            entry_conditions=[
                Condition(
                    left_operand="close",
                    operator=">",
                    right_operand="SMA_10",
                )
            ],
            long_entry_conditions=[
                Condition(
                    left_operand="close",
                    operator=">",
                    right_operand="SMA_10",
                )
            ],
            short_entry_conditions=[
                Condition(
                    left_operand="close",
                    operator="<",
                    right_operand="SMA_10",
                )
            ],
            exit_conditions=[],  # TP/SLを使用するため空
            risk_management={"position_size": 0.1},  # フォールバック用
            position_sizing_gene=test_case["position_sizing_gene"],
            tpsl_gene=tpsl_gene,
        )
        
        # バックテスト設定
        config = {
            "strategy_name": f"SizeFixTest_{i}",
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-07",  # 短期間でテスト
            "initial_capital": 10000.0,
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_gene": strategy_gene.to_dict(),
            },
        }
        
        try:
            # バックテストサービスを初期化
            backtest_service = BacktestService()
            
            print(f"  バックテスト実行中...")
            result = backtest_service.run_backtest(config)
            
            print(f"  ✅ 成功: {result['status']}")
            print(f"  総取引数: {len(result.get('trade_history', []))}")
            
            # 最初の数取引のサイズを確認
            trade_history = result.get('trade_history', [])
            if trade_history:
                print(f"  最初の3取引のサイズ:")
                for j, trade in enumerate(trade_history[:3]):
                    size = trade.get('size', 0)
                    print(f"    取引 {j+1}: {size:.6f}")
            else:
                print(f"  取引履歴なし")
                
        except Exception as e:
            print(f"  ❌ エラー: {e}")
            if "size must be a positive fraction" in str(e):
                print(f"    → ポジションサイズ制約エラー（修正が必要）")
            else:
                print(f"    → その他のエラー")
        
        print()


if __name__ == "__main__":
    test_backtest_size_fix()
