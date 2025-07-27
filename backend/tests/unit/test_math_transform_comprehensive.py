#!/usr/bin/env python3
"""
数学変換系指標の包括的テストスクリプト

修正後の全ての数学変換関数が正しく動作することを確認します。
"""

import sys
import os
import numpy as np
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app.services.indicators.technical_indicators.math_transform import MathTransformIndicators

# ログ設定（WARNING以上のみ表示）
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_all_math_functions():
    """全ての数学変換関数をテスト"""
    print("=== 全数学変換関数テスト ===")
    
    # テストデータセット
    test_cases = {
        "normal_data": np.array([0.1, 0.5, 1.0, 2.0, 5.0]),
        "negative_data": np.array([-5.0, -2.0, -1.0, -0.5, -0.1]),
        "mixed_data": np.array([-2.0, -0.5, 0.0, 0.5, 2.0]),
        "extreme_data": np.array([-100.0, -10.0, 0.0, 10.0, 100.0]),
        "small_values": np.array([1e-10, 1e-5, 1e-3, 1e-1, 1.0])
    }
    
    # テスト対象の関数リスト
    functions_to_test = [
        ("acos", "逆余弦"),
        ("asin", "逆正弦"),
        ("atan", "逆正接"),
        ("ceil", "天井関数"),
        ("cos", "余弦"),
        ("cosh", "双曲線余弦"),
        ("exp", "指数関数"),
        ("floor", "床関数"),
        ("ln", "自然対数"),
        ("log10", "常用対数"),
        ("sin", "正弦"),
        ("sinh", "双曲線正弦"),
        ("sqrt", "平方根"),
        ("tan", "正接"),
        ("tanh", "双曲線正接")
    ]
    
    results = {}
    
    for func_name, description in functions_to_test:
        print(f"\n--- {func_name.upper()} ({description}) ---")
        results[func_name] = {}
        
        func = getattr(MathTransformIndicators, func_name)
        
        for data_name, data in test_cases.items():
            try:
                result = func(data)
                nan_count = np.sum(np.isnan(result))
                inf_count = np.sum(np.isinf(result))
                
                results[func_name][data_name] = {
                    "success": True,
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                    "result_range": f"[{np.nanmin(result):.3f}, {np.nanmax(result):.3f}]"
                }
                
                status = "✓" if nan_count == 0 and inf_count == 0 else "⚠"
                print(f"  {data_name}: {status} NaN:{nan_count}, Inf:{inf_count}, Range:{results[func_name][data_name]['result_range']}")
                
            except Exception as e:
                results[func_name][data_name] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"  {data_name}: ✗ Error: {e}")
    
    return results

def test_edge_cases():
    """エッジケースのテスト"""
    print("\n=== エッジケーステスト ===")
    
    edge_cases = {
        "zeros": np.zeros(5),
        "ones": np.ones(5),
        "negative_ones": -np.ones(5),
        "very_large": np.array([1e10, 1e20, 1e50, 1e100, 1e200]),
        "very_small": np.array([1e-10, 1e-20, 1e-50, 1e-100, 1e-200]),
        "nan_input": np.array([np.nan, 1.0, 2.0, np.nan, 3.0]),
        "inf_input": np.array([np.inf, -np.inf, 1.0, 2.0, 3.0])
    }
    
    # 特に問題が起きやすい関数をテスト
    critical_functions = ["acos", "asin", "ln", "log10", "sqrt"]
    
    for func_name in critical_functions:
        print(f"\n--- {func_name.upper()} エッジケース ---")
        func = getattr(MathTransformIndicators, func_name)
        
        for case_name, data in edge_cases.items():
            try:
                result = func(data)
                nan_count = np.sum(np.isnan(result))
                inf_count = np.sum(np.isinf(result))
                
                status = "✓" if nan_count == 0 and inf_count == 0 else "⚠"
                print(f"  {case_name}: {status} NaN:{nan_count}, Inf:{inf_count}")
                
            except Exception as e:
                print(f"  {case_name}: ✗ Error: {e}")

def test_performance():
    """パフォーマンステスト"""
    print("\n=== パフォーマンステスト ===")
    
    import time
    
    # 大きなデータセット
    large_data = np.random.uniform(-2.0, 2.0, 10000)
    
    functions_to_test = ["acos", "asin", "ln", "log10", "sqrt"]
    
    for func_name in functions_to_test:
        func = getattr(MathTransformIndicators, func_name)
        
        start_time = time.time()
        result = func(large_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        nan_count = np.sum(np.isnan(result))
        
        print(f"{func_name.upper()}: {execution_time:.4f}秒, NaN数: {nan_count}")

def generate_summary(results):
    """テスト結果のサマリーを生成"""
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    
    total_tests = 0
    successful_tests = 0
    functions_with_issues = []
    
    for func_name, func_results in results.items():
        func_total = len(func_results)
        func_success = sum(1 for r in func_results.values() if r.get("success", False) and r.get("nan_count", 0) == 0 and r.get("inf_count", 0) == 0)
        
        total_tests += func_total
        successful_tests += func_success
        
        if func_success < func_total:
            functions_with_issues.append(func_name)
        
        print(f"{func_name.upper()}: {func_success}/{func_total} テスト成功")
    
    print(f"\n全体: {successful_tests}/{total_tests} テスト成功 ({successful_tests/total_tests*100:.1f}%)")
    
    if functions_with_issues:
        print(f"\n問題のある関数: {', '.join(functions_with_issues)}")
    else:
        print("\n✓ 全ての関数が正常に動作しています！")

def main():
    """メイン実行関数"""
    print("数学変換系指標の包括的テスト開始")
    print("=" * 60)
    
    # 全関数テスト
    results = test_all_math_functions()
    
    # エッジケーステスト
    test_edge_cases()
    
    # パフォーマンステスト
    test_performance()
    
    # サマリー生成
    generate_summary(results)
    
    print("\n" + "=" * 60)
    print("テスト完了")

if __name__ == "__main__":
    main()
