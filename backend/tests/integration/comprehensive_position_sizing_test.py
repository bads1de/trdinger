"""
資金管理機能の包括的テスト

ポジションサイズ計算、リスク管理パラメータ、複数ポジション時の資金配分などを検証します。
"""

import sys
import os
from typing import Dict, Any, Optional, List

# テスト対象のモジュールをインポートするためのパス設定
backend_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, backend_path)

from app.services.auto_strategy.models.gene_position_sizing import (
    PositionSizingGene,
    PositionSizingMethod,
)
from app.services.auto_strategy.calculators.position_sizing_calculator import (
    PositionSizingCalculatorService,
)


class PositionSizingTestSuite:
    """資金管理機能の包括的テストスイート"""
    
    def __init__(self):
        self.calculator = PositionSizingCalculatorService()
        self.test_results = []
        
    def run_all_tests(self):
        """全テストを実行"""
        print("🚀 資金管理機能包括的テスト開始")
        print("=" * 60)
        
        tests = [
            self.test_fixed_ratio_calculation,
            self.test_fixed_quantity_calculation,
            self.test_volatility_based_calculation,
            self.test_half_optimal_f_calculation,
            self.test_risk_management_parameters,
            self.test_account_balance_scaling,
            self.test_position_size_limits,
            self.test_multiple_position_allocation,
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
                    print("✅ PASS")
                else:
                    print("❌ FAIL")
            except Exception as e:
                print(f"❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 60)
        print(f"📊 テスト結果: {passed}/{total} 成功")
        
        if passed == total:
            print("🎉 全テスト成功！資金管理機能は正常に動作しています。")
        else:
            print(f"⚠️  {total - passed}個のテストが失敗しました。")
            
        return passed == total
    
    def test_fixed_ratio_calculation(self) -> bool:
        """固定比率計算テスト"""
        print("\n=== 固定比率計算テスト ===")
        
        gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=0.1,  # 10%
            min_position_size=0.01,
            max_position_size=1.0,
            enabled=True
        )
        
        test_cases = [
            (10000, 50000, 0.1),   # 残高10000, 価格50000, 期待10%
            (50000, 25000, 0.1),   # 残高50000, 価格25000, 期待10%
            (100000, 100000, 0.1), # 残高100000, 価格100000, 期待10%
        ]
        
        for account_balance, current_price, expected_ratio in test_cases:
            position_size = gene.calculate_position_size(
                account_balance=account_balance,
                current_price=current_price
            )
            
            ratio_match = abs(position_size - expected_ratio) < 1e-6
            
            print(f"   残高: {account_balance}, 価格: {current_price}")
            print(f"   計算結果: {position_size:.6f}, 期待値: {expected_ratio:.6f}")
            print(f"   精度チェック: {'✅' if ratio_match else '❌'}")
            
            if not ratio_match:
                return False
                
        return True
    
    def test_fixed_quantity_calculation(self) -> bool:
        """固定数量計算テスト"""
        print("\n=== 固定数量計算テスト ===")
        
        gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_QUANTITY,
            fixed_quantity=0.5,
            min_position_size=0.01,
            max_position_size=2.0,
            enabled=True
        )
        
        test_cases = [
            (10000, 50000, 0.5),   # 異なる残高・価格でも固定数量
            (50000, 25000, 0.5),
            (100000, 100000, 0.5),
        ]
        
        for account_balance, current_price, expected_quantity in test_cases:
            position_size = gene.calculate_position_size(
                account_balance=account_balance,
                current_price=current_price
            )
            
            quantity_match = abs(position_size - expected_quantity) < 1e-6
            
            print(f"   残高: {account_balance}, 価格: {current_price}")
            print(f"   計算結果: {position_size:.6f}, 期待値: {expected_quantity:.6f}")
            print(f"   精度チェック: {'✅' if quantity_match else '❌'}")
            
            if not quantity_match:
                return False
                
        return True
    
    def test_volatility_based_calculation(self) -> bool:
        """ボラティリティベース計算テスト"""
        print("\n=== ボラティリティベース計算テスト ===")

        gene = PositionSizingGene(
            method=PositionSizingMethod.VOLATILITY_BASED,
            atr_multiplier=2.0,  # ATR倍率
            risk_per_trade=0.02,  # 2%のリスク
            min_position_size=0.01,
            max_position_size=1.0,
            enabled=True
        )
        
        # 市場データのシミュレーション
        market_data = {
            "atr": 1000.0,  # ATR値
        }

        account_balance = 100000
        current_price = 50000

        position_size = gene.calculate_position_size(
            account_balance=account_balance,
            current_price=current_price,
            market_data=market_data
        )

        # ボラティリティベースの期待値計算
        # atr_pct = 1000 / 50000 = 0.02 (2%)
        # volatility_factor = 0.02 * 2.0 = 0.04
        # position_ratio = 0.02 / 0.04 = 0.5
        atr_pct = market_data["atr"] / current_price
        volatility_factor = atr_pct * gene.atr_multiplier
        expected_size = gene.risk_per_trade / volatility_factor
        expected_size = max(gene.min_position_size, min(expected_size, gene.max_position_size))

        volatility_match = abs(position_size - expected_size) < 1e-6

        print(f"   ATR: {market_data['atr']:.0f} ({atr_pct:.1%})")
        print(f"   ATR倍率: {gene.atr_multiplier}")
        print(f"   リスク率: {gene.risk_per_trade:.1%}")
        print(f"   計算結果: {position_size:.6f}, 期待値: {expected_size:.6f}")
        print(f"   精度チェック: {'✅' if volatility_match else '❌'}")

        return volatility_match
    
    def test_half_optimal_f_calculation(self) -> bool:
        """ハーフオプティマルF計算テスト"""
        print("\n=== ハーフオプティマルF計算テスト ===")
        
        gene = PositionSizingGene(
            method=PositionSizingMethod.HALF_OPTIMAL_F,
            optimal_f_multiplier=0.5,  # ハーフオプティマルF
            min_position_size=0.01,
            max_position_size=1.0,
            enabled=True
        )
        
        # 取引履歴のシミュレーション（10件以上必要）
        trade_history = [
            {"pnl": 1000},
            {"pnl": -500},
            {"pnl": 1500},
            {"pnl": -300},
            {"pnl": 800},
            {"pnl": 1200},
            {"pnl": -400},
            {"pnl": 900},
            {"pnl": -600},
            {"pnl": 1100},
            {"pnl": -350},
            {"pnl": 750},
        ]
        
        account_balance = 100000
        current_price = 50000
        
        position_size = gene.calculate_position_size(
            account_balance=account_balance,
            current_price=current_price,
            trade_history=trade_history
        )
        
        # 手動でオプティマルF計算
        wins = [t for t in trade_history if t["pnl"] > 0]
        losses = [t for t in trade_history if t["pnl"] < 0]

        if wins and losses:
            win_rate = len(wins) / len(trade_history)
            avg_win = sum(t["pnl"] for t in wins) / len(wins)
            avg_loss = abs(sum(t["pnl"] for t in losses) / len(losses))
            
            optimal_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            half_optimal_f = max(0, optimal_f * gene.optimal_f_multiplier)
            
            # 範囲制限を適用
            expected_size = max(gene.min_position_size, min(half_optimal_f, gene.max_position_size))
            
            optimal_f_match = abs(position_size - expected_size) < 1e-6
            
            print(f"   勝率: {win_rate:.1%}")
            print(f"   平均利益: {avg_win:.2f}, 平均損失: {avg_loss:.2f}")
            print(f"   オプティマルF: {optimal_f:.6f}")
            print(f"   ハーフオプティマルF: {half_optimal_f:.6f}")
            print(f"   計算結果: {position_size:.6f}, 期待値: {expected_size:.6f}")
            print(f"   精度チェック: {'✅' if optimal_f_match else '❌'}")
            
            return optimal_f_match
        else:
            print("   取引履歴が不十分です")
            return position_size == gene.min_position_size
    
    def test_risk_management_parameters(self) -> bool:
        """リスク管理パラメータテスト"""
        print("\n=== リスク管理パラメータテスト ===")

        # 現在の仕様では最大制限は無効（資金管理ロジックで制御）
        gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=0.5,  # 50%
            min_position_size=0.01,
            max_position_size=float('inf'),  # 無制限（現在の仕様）
            enabled=True
        )

        account_balance = 100000
        current_price = 50000

        position_size = gene.calculate_position_size(
            account_balance=account_balance,
            current_price=current_price
        )

        # 現在の仕様では最小制限のみ適用、計算値がそのまま使用される
        above_minimum = position_size >= gene.min_position_size
        expected_size = gene.fixed_ratio  # 50%が期待値
        size_correct = abs(position_size - expected_size) < 0.01

        print(f"   設定比率: {gene.fixed_ratio:.1%}")
        print(f"   最大制限: 無制限（現在の仕様）")
        print(f"   計算結果: {position_size:.1%}")
        print(f"   期待値: {expected_size:.1%}")
        print(f"   計算正確性: {'✅' if size_correct else '❌'}")
        print(f"   最小値チェック: {'✅' if above_minimum else '❌'}")

        return size_correct and above_minimum

    def test_account_balance_scaling(self) -> bool:
        """残高スケーリングテスト"""
        print("\n=== 残高スケーリングテスト ===")

        gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=0.1,  # 10%
            min_position_size=0.01,
            max_position_size=1.0,
            enabled=True
        )

        current_price = 50000
        balance_levels = [1000, 10000, 100000, 1000000]

        for balance in balance_levels:
            position_size = gene.calculate_position_size(
                account_balance=balance,
                current_price=current_price
            )

            # 固定比率なので残高に関係なく同じ比率になるはず
            expected_ratio = gene.fixed_ratio
            ratio_match = abs(position_size - expected_ratio) < 1e-6

            print(f"   残高: {balance:,}, ポジションサイズ: {position_size:.6f}")
            print(f"   比率一貫性: {'✅' if ratio_match else '❌'}")

            if not ratio_match:
                return False

        return True

    def test_position_size_limits(self) -> bool:
        """ポジションサイズ制限テスト"""
        print("\n=== ポジションサイズ制限テスト ===")

        # 最小制限テスト
        gene_min = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=0.001,  # 0.1%（最小値以下）
            min_position_size=0.01,  # 1%最小
            max_position_size=1.0,
            enabled=True
        )

        position_size = gene_min.calculate_position_size(
            account_balance=100000,
            current_price=50000
        )

        min_enforced = position_size >= gene_min.min_position_size
        print(f"   最小制限テスト: 設定0.1%, 最小1%, 結果{position_size:.1%}")
        print(f"   最小制限適用: {'✅' if min_enforced else '❌'}")

        # 最大制限テスト（現在の仕様では無制限）
        gene_max = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=2.0,  # 200%（従来なら制限される値）
            min_position_size=0.01,
            max_position_size=float('inf'),  # 無制限（現在の仕様）
            enabled=True
        )

        position_size = gene_max.calculate_position_size(
            account_balance=100000,
            current_price=50000
        )

        # 現在の仕様では最大制限は適用されない
        expected_size = gene_max.fixed_ratio  # 200%が期待値
        size_correct = abs(position_size - expected_size) < 0.01
        print(f"   最大制限テスト: 設定200%, 最大無制限, 結果{position_size:.1%}")
        print(f"   最大制限適用: ❌（現在の仕様では無制限）")

        return min_enforced and size_correct

    def test_multiple_position_allocation(self) -> bool:
        """複数ポジション資金配分テスト"""
        print("\n=== 複数ポジション資金配分テスト ===")

        # 複数の戦略のシミュレーション
        strategies = [
            PositionSizingGene(
                method=PositionSizingMethod.FIXED_RATIO,
                fixed_ratio=0.1,
                min_position_size=0.01,
                max_position_size=0.3,
                enabled=True
            ),
            PositionSizingGene(
                method=PositionSizingMethod.FIXED_RATIO,
                fixed_ratio=0.15,
                min_position_size=0.01,
                max_position_size=0.3,
                enabled=True
            ),
            PositionSizingGene(
                method=PositionSizingMethod.FIXED_RATIO,
                fixed_ratio=0.2,
                min_position_size=0.01,
                max_position_size=0.3,
                enabled=True
            ),
        ]

        account_balance = 100000
        current_price = 50000

        total_allocation = 0
        individual_allocations = []

        for i, strategy in enumerate(strategies):
            position_size = strategy.calculate_position_size(
                account_balance=account_balance,
                current_price=current_price
            )
            individual_allocations.append(position_size)
            total_allocation += position_size

            print(f"   戦略{i+1}: {position_size:.1%}")

        print(f"   合計配分: {total_allocation:.1%}")

        # 合計配分が100%を超えていないかチェック（リスク管理の観点）
        reasonable_total = total_allocation <= 1.0  # 100%以下

        # 各戦略が適切な範囲内かチェック
        all_within_limits = all(
            strategy.min_position_size <= allocation <= strategy.max_position_size
            for strategy, allocation in zip(strategies, individual_allocations)
        )

        print(f"   合計配分妥当性: {'✅' if reasonable_total else '❌'}")
        print(f"   個別制限遵守: {'✅' if all_within_limits else '❌'}")

        return reasonable_total and all_within_limits


def main():
    """メインテスト実行"""
    test_suite = PositionSizingTestSuite()
    return test_suite.run_all_tests()


if __name__ == "__main__":
    main()
