"""
OI/FR対応機能の包括的テスト

多角的な観点からOI/FR機能の問題点を洗い出します。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
import traceback
import logging

from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene, IndicatorGene, Condition, encode_gene_to_list, decode_list_to_gene
)
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator

# ログ設定
logging.basicConfig(level=logging.WARNING)  # ノイズを減らす

class ComprehensiveOIFRTest:
    """包括的OI/FRテストクラス"""
    
    def __init__(self):
        self.factory = StrategyFactory()
        self.generator = RandomGeneGenerator()
        self.test_results = []
        
    def log_test_result(self, test_name, success, details=""):
        """テスト結果をログ"""
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details
        })
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {details}")

    def test_edge_cases(self):
        """エッジケーステスト"""
        print("\n=== エッジケーステスト ===")
        
        # 1. 空のOI/FRデータ
        try:
            mock_data = Mock()
            mock_data.Close = pd.Series([100])
            mock_data.OpenInterest = pd.Series([])  # 空
            mock_data.FundingRate = pd.Series([])   # 空
            
            gene = StrategyGene(
                indicators=[],
                entry_conditions=[Condition("FundingRate", ">", 0.001)],
                exit_conditions=[Condition("close", "<", 95)]
            )
            
            strategy_class = self.factory.create_strategy_class(gene)
            strategy = strategy_class(data=mock_data, params={})
            
            # 空データでのアクセス
            fr_value = strategy._get_oi_fr_value("FundingRate")
            oi_value = strategy._get_oi_fr_value("OpenInterest")
            
            success = fr_value == 0.0 and oi_value == 0.0
            self.log_test_result("空OI/FRデータ処理", success, f"FR={fr_value}, OI={oi_value}")
            
        except Exception as e:
            self.log_test_result("空OI/FRデータ処理", False, f"例外: {e}")

        # 2. NaN値を含むデータ
        try:
            mock_data = Mock()
            mock_data.Close = pd.Series([100, 101, 102])
            mock_data.OpenInterest = pd.Series([1000000, np.nan, 2000000])
            mock_data.FundingRate = pd.Series([0.001, np.nan, 0.002])
            
            gene = StrategyGene(
                indicators=[],
                entry_conditions=[Condition("FundingRate", ">", 0.001)],
                exit_conditions=[Condition("close", "<", 95)]
            )
            
            strategy_class = self.factory.create_strategy_class(gene)
            strategy = strategy_class(data=mock_data, params={})
            
            # NaN値でのアクセス（最後の値は有効）
            fr_value = strategy._get_oi_fr_value("FundingRate")
            oi_value = strategy._get_oi_fr_value("OpenInterest")
            
            success = fr_value == 0.002 and oi_value == 2000000
            self.log_test_result("NaN値含有データ処理", success, f"FR={fr_value}, OI={oi_value}")
            
        except Exception as e:
            self.log_test_result("NaN値含有データ処理", False, f"例外: {e}")

        # 3. 極端な値
        try:
            mock_data = Mock()
            mock_data.Close = pd.Series([100])
            mock_data.OpenInterest = pd.Series([1e15])  # 極大値
            mock_data.FundingRate = pd.Series([-1.0])   # 極端な負値
            
            gene = StrategyGene(
                indicators=[],
                entry_conditions=[Condition("FundingRate", "<", 0)],
                exit_conditions=[Condition("OpenInterest", ">", 1e14)]
            )
            
            strategy_class = self.factory.create_strategy_class(gene)
            strategy = strategy_class(data=mock_data, params={})
            
            # 条件評価
            entry_result = strategy._check_entry_conditions()
            exit_result = strategy._check_exit_conditions()
            
            success = entry_result and exit_result
            self.log_test_result("極端値処理", success, f"Entry={entry_result}, Exit={exit_result}")
            
        except Exception as e:
            self.log_test_result("極端値処理", False, f"例外: {e}")

    def test_data_type_compatibility(self):
        """データ型互換性テスト"""
        print("\n=== データ型互換性テスト ===")
        
        # 1. 異なるデータ型でのOI/FRアクセス
        data_types = [
            ("list", [100, 101, 102]),
            ("numpy", np.array([100, 101, 102])),
            ("pandas", pd.Series([100, 101, 102])),
        ]
        
        for dtype_name, data_values in data_types:
            try:
                mock_data = Mock()
                mock_data.Close = pd.Series([100, 101, 102])
                setattr(mock_data, "OpenInterest", data_values)
                setattr(mock_data, "FundingRate", data_values)
                
                gene = StrategyGene(
                    indicators=[],
                    entry_conditions=[Condition("FundingRate", ">", 50)],
                    exit_conditions=[Condition("close", "<", 95)]
                )
                
                strategy_class = self.factory.create_strategy_class(gene)
                strategy = strategy_class(data=mock_data, params={})
                
                # データアクセステスト
                fr_value = strategy._get_oi_fr_value("FundingRate")
                oi_value = strategy._get_oi_fr_value("OpenInterest")
                
                success = fr_value is not None and oi_value is not None
                self.log_test_result(f"{dtype_name}型データ互換性", success, f"FR={fr_value}, OI={oi_value}")
                
            except Exception as e:
                self.log_test_result(f"{dtype_name}型データ互換性", False, f"例外: {e}")

    def test_condition_operator_coverage(self):
        """条件演算子網羅テスト"""
        print("\n=== 条件演算子網羅テスト ===")
        
        operators = [">", "<", ">=", "<=", "==", "cross_above", "cross_below"]
        
        for op in operators:
            try:
                mock_data = Mock()
                mock_data.Close = pd.Series([100, 101, 102])
                mock_data.OpenInterest = pd.Series([1000000, 1100000, 1200000])
                mock_data.FundingRate = pd.Series([0.001, 0.002, 0.003])
                
                gene = StrategyGene(
                    indicators=[],
                    entry_conditions=[Condition("FundingRate", op, 0.002)],
                    exit_conditions=[Condition("close", "<", 95)]
                )
                
                strategy_class = self.factory.create_strategy_class(gene)
                strategy = strategy_class(data=mock_data, params={})
                
                # 条件評価テスト
                result = strategy._evaluate_condition(gene.entry_conditions[0])
                
                success = isinstance(result, bool)
                self.log_test_result(f"演算子'{op}'処理", success, f"結果={result}")
                
            except Exception as e:
                self.log_test_result(f"演算子'{op}'処理", False, f"例外: {e}")

    def test_concurrent_access(self):
        """並行アクセステスト"""
        print("\n=== 並行アクセステスト ===")
        
        import threading
        import time
        
        results = []
        
        def worker(worker_id):
            try:
                for i in range(5):
                    gene = StrategyGene(
                        indicators=[IndicatorGene("SMA", {"period": 20})],
                        entry_conditions=[Condition("FundingRate", ">", 0.001)],
                        exit_conditions=[Condition("OpenInterest", "<", 5000000)]
                    )
                    
                    is_valid, errors = self.factory.validate_gene(gene)
                    if is_valid:
                        strategy_class = self.factory.create_strategy_class(gene)
                        results.append(f"Worker{worker_id}-{i}: Success")
                    else:
                        results.append(f"Worker{worker_id}-{i}: Invalid - {errors}")
                        
                    time.sleep(0.01)  # 短い待機
                    
            except Exception as e:
                results.append(f"Worker{worker_id}: Error - {e}")
        
        # 複数スレッドで並行実行
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        success_count = sum(1 for r in results if "Success" in r)
        total_count = len(results)
        
        success = success_count > total_count * 0.8  # 80%以上成功
        self.log_test_result("並行アクセス", success, f"成功率={success_count}/{total_count}")

    def test_memory_usage(self):
        """メモリ使用量テスト"""
        print("\n=== メモリ使用量テスト ===")
        
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # 大量の戦略生成
            strategies = []
            for i in range(100):
                gene = StrategyGene(
                    indicators=[IndicatorGene("SMA", {"period": 20})],
                    entry_conditions=[Condition("FundingRate", ">", 0.001 * i)],
                    exit_conditions=[Condition("OpenInterest", "<", 5000000 * i)]
                )
                
                strategy_class = self.factory.create_strategy_class(gene)
                strategies.append(strategy_class)
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # クリーンアップ
            del strategies
            gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_increase = peak_memory - initial_memory
            memory_recovered = peak_memory - final_memory
            
            success = memory_increase < 100  # 100MB未満の増加
            self.log_test_result("メモリ使用量", success, 
                               f"増加={memory_increase:.1f}MB, 回収={memory_recovered:.1f}MB")
            
        except Exception as e:
            self.log_test_result("メモリ使用量", False, f"例外: {e}")

    def test_encoding_decoding_integrity(self):
        """エンコード・デコード整合性テスト"""
        print("\n=== エンコード・デコード整合性テスト ===")
        
        try:
            # OI/FR条件を含む複雑な戦略
            original_gene = StrategyGene(
                indicators=[
                    IndicatorGene("SMA", {"period": 20}),
                    IndicatorGene("RSI", {"period": 14}),
                ],
                entry_conditions=[
                    Condition("close", ">", "SMA_20"),
                    Condition("FundingRate", ">", 0.001),
                ],
                exit_conditions=[
                    Condition("RSI_14", ">", 70),
                    Condition("OpenInterest", "<", 5000000),
                ]
            )
            
            # エンコード
            encoded = encode_gene_to_list(original_gene)
            
            # デコード
            decoded_gene = decode_list_to_gene(encoded)
            
            # 整合性チェック
            original_valid, _ = self.factory.validate_gene(original_gene)
            decoded_valid, _ = self.factory.validate_gene(decoded_gene)
            
            success = original_valid and decoded_valid
            self.log_test_result("エンコード・デコード整合性", success, 
                               f"元={original_valid}, 復元={decoded_valid}")
            
        except Exception as e:
            self.log_test_result("エンコード・デコード整合性", False, f"例外: {e}")

    def test_performance_benchmarks(self):
        """性能ベンチマークテスト"""
        print("\n=== 性能ベンチマークテスト ===")
        
        import time
        
        # 1. 戦略生成速度
        start_time = time.time()
        for i in range(50):
            gene = self.generator.generate_random_gene()
            self.factory.validate_gene(gene)
        generation_time = time.time() - start_time
        
        # 2. 条件評価速度
        mock_data = Mock()
        mock_data.Close = pd.Series([100] * 1000)
        mock_data.OpenInterest = pd.Series([1000000] * 1000)
        mock_data.FundingRate = pd.Series([0.001] * 1000)
        
        gene = StrategyGene(
            indicators=[],
            entry_conditions=[Condition("FundingRate", ">", 0.0005)],
            exit_conditions=[Condition("OpenInterest", ">", 500000)]
        )
        
        strategy_class = self.factory.create_strategy_class(gene)
        strategy = strategy_class(data=mock_data, params={})
        
        start_time = time.time()
        for i in range(100):
            strategy._check_entry_conditions()
            strategy._check_exit_conditions()
        evaluation_time = time.time() - start_time
        
        generation_success = generation_time < 5.0  # 5秒未満
        evaluation_success = evaluation_time < 1.0  # 1秒未満
        
        self.log_test_result("戦略生成速度", generation_success, f"{generation_time:.2f}秒/50戦略")
        self.log_test_result("条件評価速度", evaluation_success, f"{evaluation_time:.3f}秒/100評価")

    def run_all_tests(self):
        """全テスト実行"""
        print("🔍 OI/FR機能包括テスト開始\n")
        
        # 各テストカテゴリを実行
        self.test_edge_cases()
        self.test_data_type_compatibility()
        self.test_condition_operator_coverage()
        self.test_concurrent_access()
        self.test_memory_usage()
        self.test_encoding_decoding_integrity()
        self.test_performance_benchmarks()
        
        # 結果サマリー
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - successful_tests
        
        print(f"\n📊 包括テスト結果サマリー:")
        print(f"  総テスト数: {total_tests}")
        print(f"  成功: {successful_tests}")
        print(f"  失敗: {failed_tests}")
        print(f"  成功率: {successful_tests/total_tests*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ 失敗したテスト:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result['details']}")
        
        overall_success = failed_tests == 0
        if overall_success:
            print("\n🎉 全ての包括テストが成功しました！")
            print("✅ OI/FR対応機能は本番環境で使用可能です。")
        else:
            print(f"\n⚠️ {failed_tests}個のテストが失敗しました。修正が必要です。")
        
        return overall_success

def main():
    """メイン実行"""
    tester = ComprehensiveOIFRTest()
    return tester.run_all_tests()

if __name__ == "__main__":
    main()
