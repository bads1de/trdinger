"""
統合テストとエッジケース検証

TP/SL/資金管理の相互作用、エッジケース動作、計算精度と丸め処理を検証します。
"""

import sys
import os
from typing import Dict, Any, Optional, Tuple

# テスト対象のモジュールをインポートするためのパス設定
backend_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, backend_path)

from app.core.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator
from app.core.services.auto_strategy.models.gene_tpsl import TPSLGene, TPSLMethod
from app.core.services.auto_strategy.models.gene_position_sizing import (
    PositionSizingGene,
    PositionSizingMethod,
)


class IntegrationTestSuite:
    """統合テストとエッジケース検証スイート"""
    
    def __init__(self):
        self.tpsl_calculator = TPSLCalculator()
        self.test_results = []
        
    def run_all_tests(self):
        """全テストを実行"""
        print("🚀 統合テストとエッジケース検証開始")
        print("=" * 60)
        
        tests = [
            self.test_tpsl_position_sizing_interaction,
            self.test_extreme_market_conditions,
            self.test_calculation_precision,
            self.test_rounding_behavior,
            self.test_concurrent_operations,
            self.test_memory_efficiency,
            self.test_error_handling,
            self.test_performance_under_load,
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
            print("🎉 全テスト成功！統合機能は正常に動作しています。")
        else:
            print(f"⚠️  {total - passed}個のテストが失敗しました。")
            
        return passed == total
    
    def test_tpsl_position_sizing_interaction(self) -> bool:
        """TP/SLとポジションサイジングの相互作用テスト"""
        print("\n=== TP/SL・ポジションサイジング相互作用テスト ===")
        
        # TP/SL遺伝子
        tpsl_gene = TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            base_stop_loss=0.03,
            risk_reward_ratio=2.0,
            enabled=True
        )
        
        # ポジションサイジング遺伝子
        position_gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=0.1,
            min_position_size=0.01,
            max_position_size=0.5,
            enabled=True
        )
        
        # 市場条件
        current_price = 50000
        account_balance = 100000
        
        # TP/SL価格計算
        sl_price, tp_price = self.tpsl_calculator.calculate_tpsl_from_gene(
            current_price, tpsl_gene
        )
        
        # ポジションサイズ計算
        position_size = position_gene.calculate_position_size(
            account_balance=account_balance,
            current_price=current_price
        )
        
        # 実際のリスク計算
        sl_risk_pct = (current_price - sl_price) / current_price
        actual_risk_amount = account_balance * position_size * sl_risk_pct
        
        print(f"   現在価格: {current_price:,}")
        print(f"   SL価格: {sl_price:,.2f} ({sl_risk_pct:.1%}リスク)")
        print(f"   TP価格: {tp_price:,.2f}")
        print(f"   ポジションサイズ: {position_size:.1%}")
        print(f"   実際のリスク金額: {actual_risk_amount:,.2f}")
        
        # 相互作用の妥当性チェック
        reasonable_risk = actual_risk_amount <= account_balance * 0.1  # 10%以下
        valid_tpsl = sl_price < current_price < tp_price
        valid_position = 0 < position_size <= 1.0
        
        print(f"   リスク妥当性: {'✅' if reasonable_risk else '❌'}")
        print(f"   TP/SL妥当性: {'✅' if valid_tpsl else '❌'}")
        print(f"   ポジション妥当性: {'✅' if valid_position else '❌'}")
        
        return reasonable_risk and valid_tpsl and valid_position
    
    def test_extreme_market_conditions(self) -> bool:
        """極端な市場条件でのテスト"""
        print("\n=== 極端な市場条件テスト ===")
        
        extreme_conditions = [
            {"price": 0.00001, "name": "極小価格"},
            {"price": 1000000, "name": "極大価格"},
            {"price": 1.0, "name": "単位価格"},
        ]
        
        tpsl_gene = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            enabled=True
        )
        
        all_valid = True
        
        for condition in extreme_conditions:
            price = condition["price"]
            name = condition["name"]
            
            try:
                sl_price, tp_price = self.tpsl_calculator.calculate_tpsl_from_gene(
                    price, tpsl_gene
                )
                
                # 基本的な妥当性チェック
                valid_sl = sl_price > 0 and sl_price < price
                valid_tp = tp_price > price
                
                print(f"   {name} (価格: {price})")
                print(f"     SL: {sl_price:.8f}, TP: {tp_price:.8f}")
                print(f"     妥当性: SL={'✅' if valid_sl else '❌'}, TP={'✅' if valid_tp else '❌'}")
                
                if not (valid_sl and valid_tp):
                    all_valid = False
                    
            except Exception as e:
                print(f"   {name}: ❌ エラー - {e}")
                all_valid = False
        
        return all_valid
    
    def test_calculation_precision(self) -> bool:
        """計算精度テスト"""
        print("\n=== 計算精度テスト ===")
        
        # 高精度が要求される計算のテスト
        test_cases = [
            (50000.123456789, 0.030000001, 0.060000002),
            (0.000012345678, 0.019999999, 0.040000001),
            (999999.999999, 0.050000001, 0.100000002),
        ]
        
        precision_ok = True
        
        for price, sl_pct, tp_pct in test_cases:
            sl_price, tp_price = self.tpsl_calculator.calculate_legacy_tpsl_prices(
                price, sl_pct, tp_pct
            )
            
            # 期待値計算
            expected_sl = price * (1 - sl_pct)
            expected_tp = price * (1 + tp_pct)
            
            # 相対誤差計算
            sl_rel_error = abs(sl_price - expected_sl) / expected_sl if expected_sl != 0 else 0
            tp_rel_error = abs(tp_price - expected_tp) / expected_tp if expected_tp != 0 else 0
            
            # 許容誤差（浮動小数点精度の限界を考慮）
            tolerance = 1e-14
            
            sl_precise = sl_rel_error < tolerance
            tp_precise = tp_rel_error < tolerance
            
            print(f"   価格: {price:.9f}")
            print(f"   SL相対誤差: {sl_rel_error:.2e}, TP相対誤差: {tp_rel_error:.2e}")
            print(f"   精度: SL={'✅' if sl_precise else '❌'}, TP={'✅' if tp_precise else '❌'}")
            
            if not (sl_precise and tp_precise):
                precision_ok = False
        
        return precision_ok
    
    def test_rounding_behavior(self) -> bool:
        """丸め処理動作テスト"""
        print("\n=== 丸め処理動作テスト ===")
        
        # 丸め誤差が累積する可能性のあるケース
        base_price = 1.0 / 3.0  # 0.333...
        
        tpsl_gene = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            stop_loss_pct=1.0 / 3.0,  # 33.333...%
            take_profit_pct=2.0 / 3.0,  # 66.666...%
            enabled=True
        )
        
        sl_price, tp_price = self.tpsl_calculator.calculate_tpsl_from_gene(
            base_price, tpsl_gene
        )
        
        # 丸め誤差の確認
        expected_sl = base_price * (1 - 1.0/3.0)
        expected_tp = base_price * (1 + 2.0/3.0)
        
        sl_diff = abs(sl_price - expected_sl)
        tp_diff = abs(tp_price - expected_tp)
        
        print(f"   基準価格: {base_price:.15f}")
        print(f"   SL差分: {sl_diff:.2e}, TP差分: {tp_diff:.2e}")
        
        # 丸め誤差が許容範囲内かチェック
        rounding_ok = sl_diff < 1e-15 and tp_diff < 1e-15
        
        print(f"   丸め処理妥当性: {'✅' if rounding_ok else '❌'}")
        
        return rounding_ok
    
    def test_concurrent_operations(self) -> bool:
        """並行操作テスト"""
        print("\n=== 並行操作テスト ===")
        
        # 複数の計算を同時実行（スレッドセーフティの確認）
        import threading
        import time
        
        results = []
        errors = []
        
        def calculate_tpsl(thread_id):
            try:
                for i in range(100):
                    price = 50000 + thread_id * 1000 + i
                    sl_price, tp_price = self.tpsl_calculator.calculate_legacy_tpsl_prices(
                        price, 0.03, 0.06
                    )
                    results.append((thread_id, i, sl_price, tp_price))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # 5つのスレッドで並行実行
        threads = []
        for i in range(5):
            thread = threading.Thread(target=calculate_tpsl, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 全スレッドの完了を待機
        for thread in threads:
            thread.join()
        
        concurrent_ok = len(errors) == 0 and len(results) == 500
        
        print(f"   実行結果数: {len(results)}")
        print(f"   エラー数: {len(errors)}")
        print(f"   並行処理妥当性: {'✅' if concurrent_ok else '❌'}")
        
        return concurrent_ok

    def test_memory_efficiency(self) -> bool:
        """メモリ効率テスト"""
        print("\n=== メモリ効率テスト ===")

        import gc
        import psutil
        import os

        # 初期メモリ使用量
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 大量の計算を実行
        for i in range(10000):
            price = 50000 + i
            tpsl_gene = TPSLGene(
                method=TPSLMethod.FIXED_PERCENTAGE,
                stop_loss_pct=0.03,
                take_profit_pct=0.06,
                enabled=True
            )
            sl_price, tp_price = self.tpsl_calculator.calculate_tpsl_from_gene(
                price, tpsl_gene
            )

        # ガベージコレクション実行
        gc.collect()

        # 最終メモリ使用量
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"   初期メモリ: {initial_memory:.2f} MB")
        print(f"   最終メモリ: {final_memory:.2f} MB")
        print(f"   増加量: {memory_increase:.2f} MB")

        # メモリリークがないかチェック（50MB以下の増加は許容）
        memory_ok = memory_increase < 50

        print(f"   メモリ効率: {'✅' if memory_ok else '❌'}")

        return memory_ok

    def test_error_handling(self) -> bool:
        """エラーハンドリングテスト"""
        print("\n=== エラーハンドリングテスト ===")

        error_cases = [
            {"price": -1000, "name": "負の価格"},
            {"price": 0, "name": "ゼロ価格"},
            {"price": float('inf'), "name": "無限大価格"},
            {"price": float('nan'), "name": "NaN価格"},
        ]

        error_handling_ok = True

        for case in error_cases:
            price = case["price"]
            name = case["name"]

            try:
                sl_price, tp_price = self.tpsl_calculator.calculate_legacy_tpsl_prices(
                    price, 0.03, 0.06
                )

                # 結果の妥当性チェック
                if price <= 0:
                    # 負やゼロの価格では適切なエラーまたはフォールバック処理が期待される
                    valid_result = sl_price is None or tp_price is None
                else:
                    # 無限大やNaNでは適切な処理が期待される
                    valid_result = not (
                        (sl_price is not None and (sl_price == float('inf') or sl_price != sl_price)) or
                        (tp_price is not None and (tp_price == float('inf') or tp_price != tp_price))
                    )

                print(f"   {name}: {'✅' if valid_result else '❌'}")

                if not valid_result:
                    error_handling_ok = False

            except Exception as e:
                # 例外が発生した場合は適切なエラーハンドリング
                print(f"   {name}: ✅ (例外処理: {type(e).__name__})")

        return error_handling_ok

    def test_performance_under_load(self) -> bool:
        """負荷下でのパフォーマンステスト"""
        print("\n=== 負荷下パフォーマンステスト ===")

        import time

        # 大量計算のパフォーマンス測定
        start_time = time.time()

        calculation_count = 50000
        for i in range(calculation_count):
            price = 50000 + (i % 10000)
            sl_price, tp_price = self.tpsl_calculator.calculate_legacy_tpsl_prices(
                price, 0.03, 0.06
            )

        end_time = time.time()
        execution_time = end_time - start_time
        calculations_per_second = calculation_count / execution_time

        print(f"   計算回数: {calculation_count:,}")
        print(f"   実行時間: {execution_time:.3f}秒")
        print(f"   秒間計算数: {calculations_per_second:,.0f}")

        # パフォーマンス基準（秒間10,000計算以上）
        performance_ok = calculations_per_second >= 10000

        print(f"   パフォーマンス: {'✅' if performance_ok else '❌'}")

        return performance_ok


def main():
    """メインテスト実行"""
    test_suite = IntegrationTestSuite()
    return test_suite.run_all_tests()


if __name__ == "__main__":
    main()
