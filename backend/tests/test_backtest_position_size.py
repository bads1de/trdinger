#!/usr/bin/env python3
"""
実際のバックテストでポジションサイズの精度をテストするスクリプト
"""

import sys
import os
import json
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.services.backtest_service import BacktestService
from app.core.services.auto_strategy.models.gene_strategy import StrategyGene, IndicatorGene, Condition
from app.core.services.auto_strategy.models.gene_position_sizing import (
    PositionSizingGene,
    PositionSizingMethod,
)
from app.core.services.auto_strategy.models.gene_tpsl import TPSLGene, TPSLMethod


def test_backtest_position_size_precision():
    """バックテストでのポジションサイズ精度をテスト"""
    print("=== バックテストポジションサイズ精度テスト ===\n")
    
    # ボラティリティベースのポジションサイジング遺伝子を作成
    position_sizing_gene = PositionSizingGene(
        method=PositionSizingMethod.VOLATILITY_BASED,
        risk_per_trade=0.025,  # 2.5%
        atr_multiplier=1.8,
        min_position_size=0.001,  # より小さい最小値
        max_position_size=50.0,
        enabled=True,
    )
    
    # TP/SL遺伝子を作成
    tpsl_gene = TPSLGene(
        method=TPSLMethod.FIXED_PERCENTAGE,
        stop_loss_pct=0.03,  # 3%
        take_profit_pct=0.06,  # 6%
        enabled=True,
    )
    
    # シンプルな戦略遺伝子を作成
    strategy_gene = StrategyGene(
        id="precision_test_strategy",
        indicators=[
            IndicatorGene(
                type="SMA",
                parameters={"period": 20},
                enabled=True,
                json_config={"indicator_name": "SMA", "parameters": {"period": 20}},
            )
        ],
        long_entry_conditions=[
            Condition(
                left_operand="Close",
                operator=">",
                right_operand="SMA_20",
            )
        ],
        short_entry_conditions=[
            Condition(
                left_operand="Close",
                operator="<",
                right_operand="SMA_20",
            )
        ],
        exit_conditions=[],  # TP/SLを使用するため空
        risk_management={"position_size": 0.1},  # フォールバック用
        position_sizing_gene=position_sizing_gene,
        tpsl_gene=tpsl_gene,
    )
    
    # バックテスト設定
    config = {
        "strategy_name": "PositionSizePrecisionTest",
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "initial_capital": 10000.0,
        "commission_rate": 0.001,
        "strategy_config": {
            "strategy_gene": strategy_gene.to_dict(),
        },
    }
    
    try:
        # バックテストサービスを初期化
        backtest_service = BacktestService()
        
        print("バックテストを実行中...")
        result = backtest_service.run_backtest(config)
        
        print(f"バックテスト完了: {result['status']}")
        print(f"総取引数: {len(result.get('trade_history', []))}")
        
        # トレードヒストリーのポジションサイズを確認
        trade_history = result.get('trade_history', [])
        if trade_history:
            print("\n=== トレードヒストリーのポジションサイズ ===")
            for i, trade in enumerate(trade_history[:10]):  # 最初の10取引を表示
                size = trade.get('size', 0)
                print(f"取引 {i+1}: サイズ = {size:.8f}")
                
            # 統計情報
            sizes = [abs(trade.get('size', 0)) for trade in trade_history]
            if sizes:
                print(f"\n=== ポジションサイズ統計 ===")
                print(f"最小サイズ: {min(sizes):.8f}")
                print(f"最大サイズ: {max(sizes):.8f}")
                print(f"平均サイズ: {sum(sizes)/len(sizes):.8f}")
                
                # 小数点精度の確認
                decimal_places = []
                for size in sizes:
                    size_str = f"{size:.8f}".rstrip('0')
                    if '.' in size_str:
                        decimal_places.append(len(size_str.split('.')[1]))
                    else:
                        decimal_places.append(0)
                
                print(f"小数点桁数の範囲: {min(decimal_places)} - {max(decimal_places)}")
                
                # 整数かどうかの確認
                integer_count = sum(1 for size in sizes if size == int(size))
                print(f"整数のサイズ: {integer_count}/{len(sizes)} ({integer_count/len(sizes)*100:.1f}%)")
        else:
            print("取引履歴がありません")
            
    except Exception as e:
        print(f"バックテストエラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_backtest_position_size_precision()
