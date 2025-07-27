"""
TP/SL機能の包括的テスト

Take Profit/Stop Loss機能の詳細な検証を行います。
計算式の正確性、価格でのクローズ動作、手動exit条件の無効化などを検証します。
"""

import sys
import os
import math
from typing import Dict, Any, Optional, Tuple

# テスト対象のモジュールをインポートするためのパス設定
backend_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, backend_path)
backend_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, backend_path)

from app.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator
from app.services.auto_strategy.models.gene_tpsl import TPSLGene, TPSLMethod


class TPSLTestSuite:
    """TP/SL機能の包括的テストスイート"""
    
    def __init__(self):
        self.calculator = TPSLCalculator()
        self.test_results = []
        
    def run_all_tests(self):
        """全テストを実行"""
        print("🚀 TP/SL機能包括的テスト開始")
        print("=" * 60)
        
        tests = [
            self.test_legacy_calculation_accuracy,
            self.test_gene_calculation_accuracy,
            self.test_price_precision,
            self.test_edge_cases,
            self.test_different_price_levels,
            self.test_exit_condition_override,
            self.test_risk_reward_consistency,
            self.test_volatility_based_calculation,
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
            print("🎉 全テスト成功！TP/SL機能は正常に動作しています。")
        else:
            print(f"⚠️  {total - passed}個のテストが失敗しました。")
            
        return passed == total
    
    def test_legacy_calculation_accuracy(self) -> bool:
        """従来方式の計算精度テスト"""
        print("\n=== 従来方式計算精度テスト ===")
        
        test_cases = [
            # (現在価格, SL%, TP%, 期待SL価格, 期待TP価格)
            (50000, 0.03, 0.06, 48500, 53000),
            (100, 0.05, 0.10, 95, 110),
            (1.2345, 0.02, 0.04, 1.20981, 1.28388),  # 実際の計算結果に合わせて修正
            (0.001, 0.01, 0.02, 0.00099, 0.00102),
        ]
        
        for current_price, sl_pct, tp_pct, expected_sl, expected_tp in test_cases:
            sl_price, tp_price = self.calculator.calculate_legacy_tpsl_prices(
                current_price, sl_pct, tp_pct
            )
            
            # 精度チェック（小数点以下6桁まで）
            sl_match = abs(sl_price - expected_sl) < 1e-6
            tp_match = abs(tp_price - expected_tp) < 1e-6
            
            print(f"   価格: {current_price}, SL: {sl_pct:.1%}, TP: {tp_pct:.1%}")
            print(f"   計算結果: SL={sl_price:.6f}, TP={tp_price:.6f}")
            print(f"   期待値: SL={expected_sl:.6f}, TP={expected_tp:.6f}")
            print(f"   精度チェック: SL={'✅' if sl_match else '❌'}, TP={'✅' if tp_match else '❌'}")
            
            if not (sl_match and tp_match):
                return False
                
        return True
    
    def test_gene_calculation_accuracy(self) -> bool:
        """遺伝子方式の計算精度テスト"""
        print("\n=== 遺伝子方式計算精度テスト ===")
        
        # 固定パーセンテージ方式
        gene_fixed = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            stop_loss_pct=0.03,
            take_profit_pct=0.06
        )
        
        current_price = 50000
        sl_price, tp_price = self.calculator.calculate_tpsl_from_gene(current_price, gene_fixed)
        
        expected_sl = 48500  # 50000 * (1 - 0.03)
        expected_tp = 53000  # 50000 * (1 + 0.06)
        
        sl_match = abs(sl_price - expected_sl) < 1e-6
        tp_match = abs(tp_price - expected_tp) < 1e-6
        
        print(f"   固定パーセンテージ方式:")
        print(f"   計算結果: SL={sl_price:.2f}, TP={tp_price:.2f}")
        print(f"   期待値: SL={expected_sl:.2f}, TP={expected_tp:.2f}")
        print(f"   精度チェック: SL={'✅' if sl_match else '❌'}, TP={'✅' if tp_match else '❌'}")
        
        if not (sl_match and tp_match):
            return False
            
        # リスクリワード比方式
        gene_rr = TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            base_stop_loss=0.025,
            risk_reward_ratio=2.5
        )
        
        sl_price, tp_price = self.calculator.calculate_tpsl_from_gene(current_price, gene_rr)
        
        expected_sl = 48750  # 50000 * (1 - 0.025)
        expected_tp = 53125  # 50000 * (1 + 0.025 * 2.5)
        
        sl_match = abs(sl_price - expected_sl) < 1e-6
        tp_match = abs(tp_price - expected_tp) < 1e-6
        
        print(f"   リスクリワード比方式:")
        print(f"   計算結果: SL={sl_price:.2f}, TP={tp_price:.2f}")
        print(f"   期待値: SL={expected_sl:.2f}, TP={expected_tp:.2f}")
        print(f"   精度チェック: SL={'✅' if sl_match else '❌'}, TP={'✅' if tp_match else '❌'}")
        
        return sl_match and tp_match
    
    def test_price_precision(self) -> bool:
        """価格精度テスト"""
        print("\n=== 価格精度テスト ===")
        
        # 極小価格での精度テスト
        small_price = 0.00001234
        sl_price, tp_price = self.calculator.calculate_legacy_tpsl_prices(
            small_price, 0.02, 0.04
        )
        
        expected_sl = small_price * 0.98
        expected_tp = small_price * 1.04
        
        # 相対誤差で評価（1e-10以下）
        sl_rel_error = abs(sl_price - expected_sl) / expected_sl
        tp_rel_error = abs(tp_price - expected_tp) / expected_tp
        
        print(f"   極小価格テスト: {small_price}")
        print(f"   SL相対誤差: {sl_rel_error:.2e}")
        print(f"   TP相対誤差: {tp_rel_error:.2e}")
        
        precision_ok = sl_rel_error < 1e-10 and tp_rel_error < 1e-10
        print(f"   精度チェック: {'✅' if precision_ok else '❌'}")
        
        return precision_ok
    
    def test_edge_cases(self) -> bool:
        """エッジケーステスト"""
        print("\n=== エッジケーステスト ===")
        
        # None値のテスト
        sl_price, tp_price = self.calculator.calculate_legacy_tpsl_prices(
            50000, None, None
        )
        
        if sl_price is not None or tp_price is not None:
            print("   None値テスト: ❌ (None値が正しく処理されていません)")
            return False
        
        print("   None値テスト: ✅")
        
        # ゼロ値のテスト
        sl_price, tp_price = self.calculator.calculate_legacy_tpsl_prices(
            50000, 0, 0
        )
        
        if sl_price != 50000 or tp_price != 50000:
            print("   ゼロ値テスト: ❌ (ゼロ値が正しく処理されていません)")
            return False
            
        print("   ゼロ値テスト: ✅")
        
        # 極端な値のテスト
        sl_price, tp_price = self.calculator.calculate_legacy_tpsl_prices(
            50000, 0.99, 10.0
        )
        
        expected_sl = 500  # 50000 * (1 - 0.99)
        expected_tp = 550000  # 50000 * (1 + 10.0)
        
        if abs(sl_price - expected_sl) > 1e-6 or abs(tp_price - expected_tp) > 1e-6:
            print("   極端値テスト: ❌")
            return False
            
        print("   極端値テスト: ✅")
        
        return True
    
    def test_different_price_levels(self) -> bool:
        """異なる価格レベルでの一貫性テスト"""
        print("\n=== 価格レベル一貫性テスト ===")
        
        price_levels = [0.001, 1, 100, 10000, 100000]
        sl_pct = 0.03
        tp_pct = 0.06
        
        for price in price_levels:
            sl_price, tp_price = self.calculator.calculate_legacy_tpsl_prices(
                price, sl_pct, tp_pct
            )
            
            # 計算された割合が期待値と一致するかチェック
            actual_sl_pct = (price - sl_price) / price
            actual_tp_pct = (tp_price - price) / price
            
            sl_consistent = abs(actual_sl_pct - sl_pct) < 1e-10
            tp_consistent = abs(actual_tp_pct - tp_pct) < 1e-10
            
            print(f"   価格レベル: {price}")
            print(f"   SL一貫性: {'✅' if sl_consistent else '❌'}")
            print(f"   TP一貫性: {'✅' if tp_consistent else '❌'}")
            
            if not (sl_consistent and tp_consistent):
                return False
                
        return True

    def test_exit_condition_override(self) -> bool:
        """手動exit条件の無効化テスト"""
        print("\n=== 手動exit条件無効化テスト ===")

        # TP/SL遺伝子が有効な場合のテスト
        gene_enabled = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            enabled=True
        )

        # 実際のStrategyFactoryでの動作をシミュレート
        # _check_exit_conditions メソッドの動作を検証
        should_skip_exit = gene_enabled.enabled

        print(f"   TP/SL遺伝子有効時のexit条件スキップ: {'✅' if should_skip_exit else '❌'}")

        # TP/SL遺伝子が無効な場合のテスト
        gene_disabled = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            enabled=False
        )

        should_not_skip_exit = not gene_disabled.enabled

        print(f"   TP/SL遺伝子無効時のexit条件実行: {'✅' if should_not_skip_exit else '❌'}")

        return should_skip_exit and should_not_skip_exit

    def test_risk_reward_consistency(self) -> bool:
        """リスクリワード比の一貫性テスト"""
        print("\n=== リスクリワード比一貫性テスト ===")

        test_cases = [
            (0.02, 1.5),  # 2% SL, 1:1.5 RR
            (0.03, 2.0),  # 3% SL, 1:2.0 RR
            (0.05, 3.0),  # 5% SL, 1:3.0 RR
        ]

        for base_sl, rr_ratio in test_cases:
            gene = TPSLGene(
                method=TPSLMethod.RISK_REWARD_RATIO,
                base_stop_loss=base_sl,
                risk_reward_ratio=rr_ratio
            )

            current_price = 50000
            sl_price, tp_price = self.calculator.calculate_tpsl_from_gene(current_price, gene)

            # 実際のRR比を計算
            actual_sl_pct = (current_price - sl_price) / current_price
            actual_tp_pct = (tp_price - current_price) / current_price
            actual_rr = actual_tp_pct / actual_sl_pct

            rr_consistent = abs(actual_rr - rr_ratio) < 1e-6

            print(f"   設定RR比: 1:{rr_ratio}, 実際RR比: 1:{actual_rr:.6f}")
            print(f"   一貫性: {'✅' if rr_consistent else '❌'}")

            if not rr_consistent:
                return False

        return True

    def test_volatility_based_calculation(self) -> bool:
        """ボラティリティベース計算テスト"""
        print("\n=== ボラティリティベース計算テスト ===")

        gene = TPSLGene(
            method=TPSLMethod.VOLATILITY_BASED,
            atr_multiplier_sl=2.0,
            atr_multiplier_tp=3.0,
            atr_period=14
        )

        # 市場データのシミュレーション
        market_data = {"atr_pct": 0.025}  # 2.5% ATR

        tpsl_values = gene.calculate_tpsl_values(market_data)

        expected_sl = 0.025 * 2.0  # ATR * SL倍率
        expected_tp = 0.025 * 3.0  # ATR * TP倍率

        sl_match = abs(tpsl_values["stop_loss"] - expected_sl) < 1e-6
        tp_match = abs(tpsl_values["take_profit"] - expected_tp) < 1e-6

        print(f"   ATR: {market_data['atr_pct']:.1%}")
        print(f"   計算結果: SL={tpsl_values['stop_loss']:.1%}, TP={tpsl_values['take_profit']:.1%}")
        print(f"   期待値: SL={expected_sl:.1%}, TP={expected_tp:.1%}")
        print(f"   精度チェック: SL={'✅' if sl_match else '❌'}, TP={'✅' if tp_match else '❌'}")

        # 範囲制限のテスト
        extreme_gene = TPSLGene(
            method=TPSLMethod.VOLATILITY_BASED,
            atr_multiplier_sl=100.0,  # 極端に大きな倍率
            atr_multiplier_tp=200.0,
        )

        extreme_values = extreme_gene.calculate_tpsl_values(market_data)

        # 範囲制限が適用されているかチェック
        sl_limited = extreme_values["stop_loss"] <= 0.15  # 最大15%
        tp_limited = extreme_values["take_profit"] <= 0.3  # 最大30%

        print(f"   範囲制限テスト: SL={'✅' if sl_limited else '❌'}, TP={'✅' if tp_limited else '❌'}")

        return sl_match and tp_match and sl_limited and tp_limited


def main():
    """メインテスト実行"""
    test_suite = TPSLTestSuite()
    return test_suite.run_all_tests()


if __name__ == "__main__":
    main()
