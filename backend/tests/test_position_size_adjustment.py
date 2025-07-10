#!/usr/bin/env python3
"""
ポジションサイズ調整のテスト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.auto_strategy.models.gene_strategy import StrategyGene
from app.core.services.auto_strategy.models.gene_position_sizing import (
    PositionSizingGene,
    PositionSizingMethod,
)


def test_position_size_adjustment():
    """ポジションサイズ調整のテスト"""
    print("=== ポジションサイズ調整テスト ===\n")
    
    # ダミーの戦略遺伝子を作成
    strategy_gene = StrategyGene(
        id="test_adjustment",
        indicators=[],
        long_entry_conditions=[],
        short_entry_conditions=[],
        exit_conditions=[],
        risk_management={"position_size": 0.1},
        position_sizing_gene=PositionSizingGene(),
    )
    
    # StrategyFactoryを作成
    factory = StrategyFactory()
    
    # ダミーのStrategyクラスを作成してテスト
    class TestStrategy:
        def __init__(self):
            self.gene = strategy_gene
        
        def _adjust_position_size_for_backtesting(self, size: float) -> float:
            """
            backtesting.pyの制約に合わせてポジションサイズを調整
            """
            if size == 0:
                return 0
            
            abs_size = abs(size)
            sign = 1 if size > 0 else -1
            
            # 1未満の場合は割合として扱う（そのまま使用）
            if abs_size < 1:
                # 最小値チェック（backtesting.pyは0より大きい必要がある）
                if abs_size <= 0:
                    return 0
                return size
            
            # 1以上の場合は整数に丸める（単位数として扱う）
            else:
                rounded_size = round(abs_size)
                # 丸めた結果が0になった場合は1にする
                if rounded_size == 0:
                    rounded_size = 1
                return sign * rounded_size
    
    strategy = TestStrategy()
    
    # テストケース
    test_cases = [
        # (入力サイズ, 期待される出力, 説明)
        (0.0, 0.0, "ゼロサイズ"),
        (0.1, 0.1, "小数（割合として有効）"),
        (0.5, 0.5, "小数（割合として有効）"),
        (0.99, 0.99, "1未満の小数"),
        (1.0, 1.0, "ちょうど1.0"),
        (1.1, 1.0, "1.1 → 1に丸める"),
        (1.5, 2.0, "1.5 → 2に丸める"),
        (2.3, 2.0, "2.3 → 2に丸める"),
        (2.7, 3.0, "2.7 → 3に丸める"),
        (10.0, 10.0, "整数"),
        (100.5, 101.0, "大きな数の丸め"),
        (-0.1, -0.1, "負の小数"),
        (-1.5, -2.0, "負の数の丸め"),
        (-2.3, -2.0, "負の数の丸め"),
        (0.001, 0.001, "非常に小さな正の数"),
        (20000.0, 20000.0, "大きな整数"),
        (20000.7, 20001.0, "大きな数の丸め"),
    ]
    
    print("テストケース実行:")
    for i, (input_size, expected, description) in enumerate(test_cases, 1):
        result = strategy._adjust_position_size_for_backtesting(input_size)
        status = "✅" if result == expected else "❌"
        print(f"  {i:2d}. {description}: {input_size} → {result} (期待: {expected}) {status}")
        
        if result != expected:
            print(f"      エラー: 期待値と異なります")
    
    # backtesting.pyの制約チェック
    print("\nbacktesting.py制約チェック:")
    for i, (input_size, expected, description) in enumerate(test_cases, 1):
        result = strategy._adjust_position_size_for_backtesting(input_size)
        
        if result == 0:
            constraint_ok = True
            constraint_msg = "ゼロサイズ（取引なし）"
        elif 0 < abs(result) < 1:
            constraint_ok = True
            constraint_msg = "割合として有効"
        elif abs(result) >= 1 and round(abs(result)) == abs(result):
            constraint_ok = True
            constraint_msg = "整数として有効"
        else:
            constraint_ok = False
            constraint_msg = "制約違反"
        
        status = "✅" if constraint_ok else "❌"
        print(f"  {i:2d}. {description}: {result} → {constraint_msg} {status}")


if __name__ == "__main__":
    test_position_size_adjustment()
