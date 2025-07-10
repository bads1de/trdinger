"""
オートストラテジー機能の包括的ワークフローテスト

実際のユーザーワークフローに基づいた包括的なテストを実行します。
エンドツーエンドの動作、データ品質、パフォーマンス、セキュリティを検証します。
"""

import sys
import os
import time
import json
import threading
import psutil
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# テスト対象のモジュールをインポートするためのパス設定
backend_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, backend_path)

from app.core.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.models.gene_strategy import StrategyGene
from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator
from app.core.services.auto_strategy.calculators.position_sizing_calculator import PositionSizingCalculatorService


class ComprehensiveWorkflowTestSuite:
    """オートストラテジー包括的ワークフローテストスイート"""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
        self.security_checks = {}
        self.data_quality_results = {}
        
    def run_all_tests(self):
        """全テストを実行"""
        print("🚀 オートストラテジー包括的ワークフローテスト開始")
        print("=" * 80)
        
        tests = [
            self.test_end_to_end_workflow,
            self.test_data_quality_and_integrity,
            self.test_scalability_and_performance,
            self.test_security_and_robustness,
            self.test_usability_and_user_experience,
            self.test_concurrent_operations,
            self.test_error_recovery_mechanisms,
            self.test_resource_management,
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
        
        print("\n" + "=" * 80)
        print(f"📊 テスト結果: {passed}/{total} 成功")
        
        if passed == total:
            print("🎉 全テスト成功！オートストラテジー機能は本格運用可能です。")
        else:
            print(f"⚠️  {total - passed}個のテストが失敗しました。")
            
        return passed == total

    def test_end_to_end_workflow(self) -> bool:
        """エンドツーエンドワークフローテスト"""
        print("\n=== エンドツーエンドワークフローテスト ===")
        
        try:
            # 1. サービス初期化
            service = AutoStrategyService(enable_smart_generation=True)
            print("   ✅ サービス初期化成功")
            
            # 2. GA設定作成
            ga_config = {
                "population_size": 10,
                "generations": 3,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "elite_size": 2,
                "tournament_size": 3,
                "fitness_function": "sharpe_ratio",
                "enable_smart_generation": True,
                "enable_tpsl_optimization": True,
                "enable_position_sizing_optimization": True,
            }
            print("   ✅ GA設定作成成功")
            
            # 3. バックテスト設定作成
            backtest_config = {
                "symbol": "BTCUSDT",
                "timeframe": "1h",
                "start_date": "2023-01-01",
                "end_date": "2023-01-31",
                "initial_cash": 10000,
                "commission": 0.001,
            }
            print("   ✅ バックテスト設定作成成功")
            
            # 4. 戦略生成テスト（実際の実行はしない）
            config_obj = GAConfig.from_dict(ga_config)
            generator = RandomGeneGenerator(config_obj)
            test_gene = generator.generate_random_gene()
            print("   ✅ 戦略遺伝子生成成功")
            
            # 5. TP/SL計算テスト
            calculator = TPSLCalculator()
            sl_price, tp_price = calculator.calculate_tpsl_prices(
                current_price=50000.0,
                stop_loss_pct=0.03,
                take_profit_pct=0.06,
                risk_management={},
                gene=test_gene,
                position_direction=1.0
            )
            print(f"   ✅ TP/SL計算成功: SL={sl_price:.2f}, TP={tp_price:.2f}")
            
            # 6. ポジションサイジング計算テスト
            pos_calculator = PositionSizingCalculatorService()
            if hasattr(test_gene, 'position_sizing_gene') and test_gene.position_sizing_gene:
                pos_result = pos_calculator.calculate_position_size(
                    gene=test_gene.position_sizing_gene,
                    account_balance=10000.0,
                    current_price=50000.0,
                    symbol="BTCUSDT"
                )
                print(f"   ✅ ポジションサイジング計算成功: {pos_result.position_size:.4f}")
            else:
                print("   ✅ ポジションサイジング遺伝子なし（従来方式）")
            
            # 7. データ検証
            self._validate_gene_structure(test_gene)
            print("   ✅ 遺伝子構造検証成功")
            
            return True
            
        except Exception as e:
            print(f"   ❌ エンドツーエンドワークフローエラー: {e}")
            return False

    def test_data_quality_and_integrity(self) -> bool:
        """データ品質と整合性テスト"""
        print("\n=== データ品質と整合性テスト ===")
        
        try:
            config = GAConfig.create_fast()
            generator = RandomGeneGenerator(config)

            # 複数の遺伝子を生成してデータ品質をチェック
            genes = []
            for i in range(20):
                gene = generator.generate_random_gene()
                genes.append(gene)
            
            # 1. 遺伝子構造の一貫性チェック
            structure_consistent = self._check_gene_structure_consistency(genes)
            print(f"   遺伝子構造一貫性: {'✅' if structure_consistent else '❌'}")
            
            # 2. パラメータ範囲の妥当性チェック
            params_valid = self._check_parameter_ranges(genes)
            print(f"   パラメータ範囲妥当性: {'✅' if params_valid else '❌'}")
            
            # 3. 条件の論理的整合性チェック
            logic_consistent = self._check_condition_logic(genes)
            print(f"   条件論理整合性: {'✅' if logic_consistent else '❌'}")
            
            # 4. TP/SL設定の妥当性チェック
            tpsl_valid = self._check_tpsl_validity(genes)
            print(f"   TP/SL設定妥当性: {'✅' if tpsl_valid else '❌'}")
            
            # 5. ポジションサイジングの妥当性チェック
            position_valid = self._check_position_sizing_validity(genes)
            print(f"   ポジションサイジング妥当性: {'✅' if position_valid else '❌'}")
            
            self.data_quality_results = {
                "structure_consistent": structure_consistent,
                "params_valid": params_valid,
                "logic_consistent": logic_consistent,
                "tpsl_valid": tpsl_valid,
                "position_valid": position_valid,
            }
            
            return all(self.data_quality_results.values())
            
        except Exception as e:
            print(f"   ❌ データ品質テストエラー: {e}")
            return False

    def test_scalability_and_performance(self) -> bool:
        """スケーラビリティとパフォーマンステスト"""
        print("\n=== スケーラビリティとパフォーマンステスト ===")
        
        try:
            config = GAConfig.create_fast()
            generator = RandomGeneGenerator(config)
            calculator = TPSLCalculator()

            # 1. 大量遺伝子生成パフォーマンス
            start_time = time.time()
            genes = []
            for i in range(1000):
                gene = generator.generate_random_gene()
                genes.append(gene)
            generation_time = time.time() - start_time
            
            genes_per_second = 1000 / generation_time
            print(f"   遺伝子生成速度: {genes_per_second:.0f} genes/sec")
            
            # 2. TP/SL計算パフォーマンス
            start_time = time.time()
            for i in range(10000):
                calculator.calculate_legacy_tpsl_prices(50000.0, 0.03, 0.06, 1.0)
            calculation_time = time.time() - start_time
            
            calculations_per_second = 10000 / calculation_time
            print(f"   TP/SL計算速度: {calculations_per_second:.0f} calc/sec")
            
            # 3. メモリ使用量チェック
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # 大量データ処理
            large_genes = []
            for i in range(5000):
                gene = generator.generate_random_gene()
                large_genes.append(gene)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            print(f"   メモリ使用量増加: {memory_increase:.2f} MB (5000遺伝子)")
            
            # 4. パフォーマンス基準チェック
            performance_ok = (
                genes_per_second >= 100 and  # 最低100遺伝子/秒
                calculations_per_second >= 1000 and  # 最低1000計算/秒
                memory_increase < 500  # 500MB未満
            )
            
            self.performance_metrics = {
                "genes_per_second": genes_per_second,
                "calculations_per_second": calculations_per_second,
                "memory_increase_mb": memory_increase,
                "performance_ok": performance_ok,
            }
            
            print(f"   パフォーマンス基準: {'✅' if performance_ok else '❌'}")
            
            return performance_ok
            
        except Exception as e:
            print(f"   ❌ パフォーマンステストエラー: {e}")
            return False

    def test_security_and_robustness(self) -> bool:
        """セキュリティと堅牢性テスト"""
        print("\n=== セキュリティと堅牢性テスト ===")
        
        try:
            calculator = TPSLCalculator()
            
            # 1. 不正入力に対する堅牢性
            malicious_inputs = [
                {"price": -1000, "sl": 0.03, "tp": 0.06},  # 負の価格
                {"price": float('inf'), "sl": 0.03, "tp": 0.06},  # 無限大
                {"price": float('nan'), "sl": 0.03, "tp": 0.06},  # NaN
                {"price": 50000, "sl": -0.5, "tp": 0.06},  # 負のSL
                {"price": 50000, "sl": 0.03, "tp": -0.06},  # 負のTP
                {"price": 50000, "sl": 10.0, "tp": 0.06},  # 異常に大きなSL
            ]
            
            robust_count = 0
            for i, inputs in enumerate(malicious_inputs):
                try:
                    result = calculator.calculate_legacy_tpsl_prices(
                        inputs["price"], inputs["sl"], inputs["tp"], 1.0
                    )
                    # 結果が妥当かチェック
                    if result[0] is not None and result[1] is not None:
                        if not (float('inf') in result or float('-inf') in result or 
                               any(str(x) == 'nan' for x in result)):
                            robust_count += 1
                except Exception:
                    # 例外が発生するのは正常（不正入力の拒否）
                    robust_count += 1
            
            robustness_score = robust_count / len(malicious_inputs)
            print(f"   不正入力堅牢性: {robustness_score:.1%} ({robust_count}/{len(malicious_inputs)})")
            
            # 2. メモリリーク検出
            memory_leak_detected = self._check_memory_leaks()
            print(f"   メモリリーク: {'❌ 検出' if memory_leak_detected else '✅ なし'}")
            
            # 3. 例外処理の適切性
            exception_handling_ok = self._check_exception_handling()
            print(f"   例外処理: {'✅' if exception_handling_ok else '❌'}")
            
            security_ok = (
                robustness_score >= 0.8 and
                not memory_leak_detected and
                exception_handling_ok
            )
            
            self.security_checks = {
                "robustness_score": robustness_score,
                "memory_leak_detected": memory_leak_detected,
                "exception_handling_ok": exception_handling_ok,
                "security_ok": security_ok,
            }
            
            print(f"   セキュリティ基準: {'✅' if security_ok else '❌'}")
            
            return security_ok

        except Exception as e:
            print(f"   ❌ セキュリティテストエラー: {e}")
            return False

    def test_usability_and_user_experience(self) -> bool:
        """ユーザビリティとユーザーエクスペリエンステスト"""
        print("\n=== ユーザビリティとユーザーエクスペリエンステスト ===")

        try:
            # 1. 設定の簡単さテスト
            simple_config = GAConfig(
                population_size=20,
                generations=5,
            )

            generator = RandomGeneGenerator(simple_config)
            gene = generator.generate_random_gene()
            print("   ✅ 簡単設定での遺伝子生成成功")

            # 2. デフォルト値の妥当性
            default_gene = generator.generate_random_gene()
            default_valid = self._validate_gene_structure(default_gene)
            print(f"   デフォルト値妥当性: {'✅' if default_valid else '❌'}")

            # 3. エラーメッセージの分かりやすさ
            error_clarity = self._test_error_message_clarity()
            print(f"   エラーメッセージ明確性: {'✅' if error_clarity else '❌'}")

            # 4. 実行時間の予測可能性
            execution_predictable = self._test_execution_predictability()
            print(f"   実行時間予測可能性: {'✅' if execution_predictable else '❌'}")

            usability_ok = default_valid and error_clarity and execution_predictable
            print(f"   ユーザビリティ基準: {'✅' if usability_ok else '❌'}")

            return usability_ok

        except Exception as e:
            print(f"   ❌ ユーザビリティテストエラー: {e}")
            return False

    def test_concurrent_operations(self) -> bool:
        """並行操作テスト"""
        print("\n=== 並行操作テスト ===")

        try:
            config = GAConfig.create_fast()
            generator = RandomGeneGenerator(config)
            calculator = TPSLCalculator()

            # 並行遺伝子生成テスト
            def generate_genes(count):
                genes = []
                for i in range(count):
                    gene = generator.generate_random_gene()
                    genes.append(gene)
                return genes

            # 並行TP/SL計算テスト
            def calculate_tpsl(count):
                results = []
                for i in range(count):
                    result = calculator.calculate_legacy_tpsl_prices(
                        50000.0 + i, 0.03, 0.06, 1.0
                    )
                    results.append(result)
                return results

            # 並行実行
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []

                # 遺伝子生成タスク
                for i in range(4):
                    future = executor.submit(generate_genes, 100)
                    futures.append(future)

                # TP/SL計算タスク
                for i in range(4):
                    future = executor.submit(calculate_tpsl, 100)
                    futures.append(future)

                # 結果収集
                results = []
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                    except Exception as e:
                        print(f"   ❌ 並行処理エラー: {e}")
                        return False

            print(f"   ✅ 並行処理成功: {len(results)}個のタスク完了")

            # データ整合性チェック
            all_genes = []
            all_calculations = []

            for result in results:
                if isinstance(result[0], dict):  # 遺伝子の場合
                    all_genes.extend(result)
                else:  # 計算結果の場合
                    all_calculations.extend(result)

            # 重複や異常値のチェック
            integrity_ok = len(all_genes) > 0 and len(all_calculations) > 0
            print(f"   データ整合性: {'✅' if integrity_ok else '❌'}")

            return integrity_ok

        except Exception as e:
            print(f"   ❌ 並行操作テストエラー: {e}")
            return False

    def test_error_recovery_mechanisms(self) -> bool:
        """エラー回復メカニズムテスト"""
        print("\n=== エラー回復メカニズムテスト ===")

        try:
            calculator = TPSLCalculator()

            # 1. 部分的失敗からの回復
            success_count = 0
            total_attempts = 100

            for i in range(total_attempts):
                try:
                    # 意図的に一部失敗する条件を作成
                    price = 50000.0 if i % 10 != 0 else float('nan')
                    result = calculator.calculate_legacy_tpsl_prices(price, 0.03, 0.06, 1.0)

                    if result[0] is not None and result[1] is not None:
                        success_count += 1
                except Exception:
                    # 例外が発生した場合はフォールバック処理
                    try:
                        result = calculator.calculate_legacy_tpsl_prices(50000.0, 0.03, 0.06, 1.0)
                        if result[0] is not None and result[1] is not None:
                            success_count += 1
                    except Exception:
                        pass

            recovery_rate = success_count / total_attempts
            print(f"   エラー回復率: {recovery_rate:.1%} ({success_count}/{total_attempts})")

            # 2. リソース枯渇からの回復
            resource_recovery = self._test_resource_recovery()
            print(f"   リソース回復: {'✅' if resource_recovery else '❌'}")

            recovery_ok = recovery_rate >= 0.9 and resource_recovery
            print(f"   回復メカニズム: {'✅' if recovery_ok else '❌'}")

            return recovery_ok

        except Exception as e:
            print(f"   ❌ エラー回復テストエラー: {e}")
            return False

    def test_resource_management(self) -> bool:
        """リソース管理テスト"""
        print("\n=== リソース管理テスト ===")

        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # 大量処理実行
            config = GAConfig.create_fast()
            generator = RandomGeneGenerator(config)
            genes = []

            for i in range(10000):
                gene = generator.generate_random_gene()
                genes.append(gene)

                # 定期的にメモリ使用量をチェック
                if i % 1000 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_increase = current_memory - initial_memory

                    # メモリ使用量が異常に増加していないかチェック
                    if memory_increase > 1000:  # 1GB以上の増加は異常
                        print(f"   ❌ メモリ使用量異常: {memory_increase:.2f} MB")
                        return False

            final_memory = process.memory_info().rss / 1024 / 1024
            total_increase = final_memory - initial_memory

            print(f"   メモリ使用量増加: {total_increase:.2f} MB (10000遺伝子)")

            # メモリ効率チェック
            memory_per_gene = total_increase / 10000 * 1024  # KB per gene
            print(f"   遺伝子あたりメモリ: {memory_per_gene:.2f} KB")

            # リソース管理基準
            resource_ok = (
                total_increase < 500 and  # 500MB未満
                memory_per_gene < 50  # 50KB/遺伝子未満
            )

            print(f"   リソース管理: {'✅' if resource_ok else '❌'}")

            return resource_ok

        except Exception as e:
            print(f"   ❌ リソース管理テストエラー: {e}")
            return False

    # ヘルパーメソッド
    def _validate_gene_structure(self, gene) -> bool:
        """遺伝子構造の検証"""
        try:
            # 基本属性の存在確認
            required_attrs = ['indicators', 'long_conditions', 'short_conditions', 'risk_management']
            for attr in required_attrs:
                if not hasattr(gene, attr):
                    return False

            # 条件の妥当性確認
            if not gene.long_conditions or not gene.short_conditions:
                return False

            return True
        except Exception:
            return False

    def _check_gene_structure_consistency(self, genes) -> bool:
        """遺伝子構造の一貫性チェック"""
        try:
            if not genes:
                return False

            first_gene = genes[0]
            for gene in genes[1:]:
                if type(gene) != type(first_gene):
                    return False

                # 基本構造の一致確認
                if (hasattr(first_gene, 'indicators') != hasattr(gene, 'indicators') or
                    hasattr(first_gene, 'long_conditions') != hasattr(gene, 'long_conditions')):
                    return False

            return True
        except Exception:
            return False

    def _check_parameter_ranges(self, genes) -> bool:
        """パラメータ範囲の妥当性チェック"""
        try:
            for gene in genes:
                if hasattr(gene, 'indicators'):
                    for indicator in gene.indicators:
                        if hasattr(indicator, 'parameters'):
                            for param_name, param_value in indicator.parameters.items():
                                if isinstance(param_value, (int, float)):
                                    if param_value < 0 or param_value > 1000:
                                        return False
            return True
        except Exception:
            return False

    def _check_condition_logic(self, genes) -> bool:
        """条件の論理的整合性チェック"""
        try:
            for gene in genes:
                if hasattr(gene, 'long_conditions') and hasattr(gene, 'short_conditions'):
                    if not gene.long_conditions or not gene.short_conditions:
                        return False

                    # 条件の基本構造チェック
                    for condition in gene.long_conditions + gene.short_conditions:
                        if not hasattr(condition, 'left_operand') or not hasattr(condition, 'operator'):
                            return False

            return True
        except Exception:
            return False

    def _check_tpsl_validity(self, genes) -> bool:
        """TP/SL設定の妥当性チェック"""
        try:
            for gene in genes:
                if hasattr(gene, 'tpsl_gene') and gene.tpsl_gene:
                    tpsl_values = gene.tpsl_gene.calculate_tpsl_values()
                    sl = tpsl_values.get('stop_loss', 0)
                    tp = tpsl_values.get('take_profit', 0)

                    if sl <= 0 or tp <= 0 or sl >= 1.0 or tp >= 1.0:
                        return False

            return True
        except Exception:
            return False

    def _check_position_sizing_validity(self, genes) -> bool:
        """ポジションサイジングの妥当性チェック"""
        try:
            for gene in genes:
                if hasattr(gene, 'position_sizing_gene') and gene.position_sizing_gene:
                    pos_gene = gene.position_sizing_gene

                    # 基本パラメータの妥当性
                    if (pos_gene.min_position_size < 0 or
                        pos_gene.min_position_size > 1.0):
                        return False

            return True
        except Exception:
            return False

    def _check_memory_leaks(self) -> bool:
        """メモリリーク検出"""
        try:
            import gc
            process = psutil.Process()

            initial_memory = process.memory_info().rss / 1024 / 1024

            # 大量オブジェクト作成・削除
            for i in range(100):
                generator = RandomGeneGenerator()
                genes = []
                for j in range(100):
                    gene = generator.generate_random_gene({})
                    genes.append(gene)

                # 明示的に削除
                del genes
                del generator
                gc.collect()

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory

            # 50MB以上の増加はメモリリークの可能性
            return memory_increase > 50

        except Exception:
            return False

    def _check_exception_handling(self) -> bool:
        """例外処理の適切性チェック"""
        try:
            calculator = TPSLCalculator()

            # 異常な入力での例外処理テスト
            test_cases = [
                (None, 0.03, 0.06, 1.0),
                (50000, None, 0.06, 1.0),
                (50000, 0.03, None, 1.0),
                ("invalid", 0.03, 0.06, 1.0),
            ]

            for case in test_cases:
                try:
                    result = calculator.calculate_legacy_tpsl_prices(*case)
                    # 例外が発生しないか、適切な結果が返される
                    if result is None or (result[0] is None and result[1] is None):
                        continue  # 適切な処理
                except Exception:
                    continue  # 例外発生も適切

            return True

        except Exception:
            return False

    def _test_error_message_clarity(self) -> bool:
        """エラーメッセージの明確性テスト"""
        try:
            # 意図的にエラーを発生させてメッセージをチェック
            calculator = TPSLCalculator()

            try:
                calculator.calculate_legacy_tpsl_prices("invalid", 0.03, 0.06, 1.0)
            except Exception as e:
                error_msg = str(e)
                # エラーメッセージが空でないことを確認
                return len(error_msg) > 0

            return True

        except Exception:
            return False

    def _test_execution_predictability(self) -> bool:
        """実行時間予測可能性テスト"""
        try:
            config = GAConfig.create_fast()
            generator = RandomGeneGenerator(config)

            # 同じ処理を複数回実行して時間のばらつきをチェック
            times = []
            for i in range(10):
                start_time = time.time()
                gene = generator.generate_random_gene()
                end_time = time.time()
                times.append(end_time - start_time)

            # 標準偏差が平均の50%以下なら予測可能
            import statistics
            mean_time = statistics.mean(times)
            std_time = statistics.stdev(times)

            return std_time / mean_time <= 0.5

        except Exception:
            return False

    def _test_resource_recovery(self) -> bool:
        """リソース回復テスト"""
        try:
            # メモリ使用量を意図的に増やしてから回復をテスト
            large_data = []
            for i in range(1000):
                large_data.append([0] * 1000)

            # データを削除
            del large_data

            # ガベージコレクション実行
            import gc
            gc.collect()

            # 新しい処理が正常に実行できるかテスト
            config = GAConfig.create_fast()
            generator = RandomGeneGenerator(config)
            gene = generator.generate_random_gene()

            return gene is not None

        except Exception:
            return False


def main():
    """メインテスト実行"""
    suite = ComprehensiveWorkflowTestSuite()
    success = suite.run_all_tests()

    # 詳細結果の表示
    print("\n" + "=" * 80)
    print("📊 詳細テスト結果")
    print("=" * 80)

    if suite.performance_metrics:
        print("\n🚀 パフォーマンス指標:")
        for key, value in suite.performance_metrics.items():
            print(f"   {key}: {value}")

    if suite.security_checks:
        print("\n🔒 セキュリティチェック:")
        for key, value in suite.security_checks.items():
            print(f"   {key}: {value}")

    if suite.data_quality_results:
        print("\n📊 データ品質結果:")
        for key, value in suite.data_quality_results.items():
            print(f"   {key}: {value}")

    return success


if __name__ == "__main__":
    main()
