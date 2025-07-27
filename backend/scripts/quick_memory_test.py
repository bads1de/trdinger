#!/usr/bin/env python3
"""
軽量なAutoMLメモリテスト

最適化効果を素早く確認するためのテストスクリプト
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.services.ml.feature_engineering.automl_features.autofeat_calculator import AutoFeatCalculator
from app.core.services.ml.feature_engineering.automl_features.automl_config import AutoFeatConfig

def generate_simple_test_data(rows: int = 1000, features: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """
    シンプルなテスト用データを生成
    """
    print(f"テストデータ生成: {rows}行 x {features}特徴量")
    
    # 数値特徴量のみ生成（処理を軽量化）
    data = {}
    for i in range(features):
        data[f'feature_{i}'] = np.random.normal(0, 1, rows)
    
    df = pd.DataFrame(data)
    
    # シンプルなターゲット変数
    target = df[f'feature_0'] * 0.5 + df[f'feature_1'] * 0.3 + np.random.normal(0, 0.1, rows)
    
    return df, pd.Series(target, name='target')

def test_memory_optimization():
    """メモリ最適化効果をテスト"""
    print("=== 軽量メモリ最適化テスト ===")
    
    # テストデータ生成
    df, target = generate_simple_test_data(1000, 5)
    data_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    print(f"テストデータサイズ: {data_size_mb:.2f}MB")
    
    # 最適化前の設定（重い設定）
    config_heavy = AutoFeatConfig(
        max_features=50,
        feateng_steps=2,
        max_gb=2.0,
        featsel_runs=2,
        verbose=1,
        n_jobs=2,
    )
    
    # 最適化後の設定（軽い設定）
    config_light = AutoFeatConfig(
        max_features=10,
        feateng_steps=1,
        max_gb=0.5,
        featsel_runs=1,
        verbose=0,
        n_jobs=1,
    )
    
    # 動的最適化設定
    config_dynamic = AutoFeatConfig().get_memory_optimized_config(data_size_mb)
    
    print(f"\n動的最適化設定:")
    print(f"  max_features: {config_dynamic.max_features}")
    print(f"  feateng_steps: {config_dynamic.feateng_steps}")
    print(f"  max_gb: {config_dynamic.max_gb}")
    print(f"  featsel_runs: {config_dynamic.featsel_runs}")
    print(f"  n_jobs: {config_dynamic.n_jobs}")
    
    # テスト実行
    results = {}
    
    for config_name, config in [
        ("軽量設定", config_light),
        ("動的最適化", config_dynamic),
    ]:
        print(f"\n--- {config_name}テスト ---")
        
        calculator = AutoFeatCalculator(config)
        
        start_time = time.time()
        try:
            with calculator as calc:
                result_df, info = calc.generate_features(df, target, max_features=10)
            
            execution_time = time.time() - start_time
            
            results[config_name] = {
                "success": True,
                "execution_time": execution_time,
                "generated_features": info.get("generated_features", 0),
                "memory_before": calc._memory_usage_before,
                "memory_after": calc._memory_usage_after,
                "memory_freed": calc._memory_usage_before - calc._memory_usage_after,
                "peak_memory": max(calc._memory_usage_before, calc._memory_usage_after),
            }
            
            print(f"✅ 成功: {execution_time:.2f}秒")
            print(f"   生成特徴量: {info.get('generated_features', 0)}個")
            print(f"   ピークメモリ: {results[config_name]['peak_memory']:.2f}MB")
            print(f"   メモリ解放: {results[config_name]['memory_freed']:.2f}MB")
            
        except Exception as e:
            execution_time = time.time() - start_time
            results[config_name] = {
                "success": False,
                "execution_time": execution_time,
                "error": str(e),
            }
            print(f"❌ エラー: {e}")
    
    return results

def test_data_size_scaling():
    """データサイズによるスケーリングテスト"""
    print("\n=== データサイズスケーリングテスト ===")
    
    test_sizes = [
        (500, 3),    # 小量
        (1000, 5),   # 中小量
        (2000, 7),   # 中量
    ]
    
    results = []
    
    for rows, features in test_sizes:
        df, target = generate_simple_test_data(rows, features)
        data_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        print(f"\n--- {rows}行 x {features}特徴量 ({data_size_mb:.2f}MB) ---")
        
        # 動的最適化設定を取得
        config = AutoFeatConfig().get_memory_optimized_config(data_size_mb)
        
        print(f"最適化設定: max_features={config.max_features}, "
              f"feateng_steps={config.feateng_steps}, "
              f"max_gb={config.max_gb}")
        
        calculator = AutoFeatCalculator(config)
        
        start_time = time.time()
        try:
            with calculator as calc:
                result_df, info = calc.generate_features(df, target, max_features=config.max_features)
            
            execution_time = time.time() - start_time
            
            result = {
                "data_size_mb": data_size_mb,
                "rows": rows,
                "features": features,
                "execution_time": execution_time,
                "generated_features": info.get("generated_features", 0),
                "memory_peak": max(calc._memory_usage_before, calc._memory_usage_after),
                "success": True,
            }
            
            results.append(result)
            
            print(f"✅ 成功: {execution_time:.2f}秒, "
                  f"生成特徴量: {result['generated_features']}個, "
                  f"ピークメモリ: {result['memory_peak']:.2f}MB")
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = {
                "data_size_mb": data_size_mb,
                "rows": rows,
                "features": features,
                "execution_time": execution_time,
                "success": False,
                "error": str(e),
            }
            results.append(result)
            print(f"❌ エラー: {e}")
    
    return results

def test_pandas_optimization():
    """pandasメモリ最適化テスト"""
    print("\n=== pandasメモリ最適化テスト ===")
    
    # テストデータ生成（様々なデータ型を含む）
    df = pd.DataFrame({
        'int_col': np.random.randint(0, 100, 1000),
        'float_col': np.random.normal(0, 1, 1000),
        'category_col': np.random.choice(['A', 'B', 'C'], 1000),
        'large_int': np.random.randint(0, 1000000, 1000),
    })
    
    original_memory = df.memory_usage(deep=True).sum()
    print(f"最適化前メモリ: {original_memory / 1024 / 1024:.2f}MB")
    
    # メモリ最適化を適用
    from app.core.services.ml.feature_engineering.automl_features.memory_utils import optimize_dataframe_dtypes
    
    optimized_df = optimize_dataframe_dtypes(df, aggressive=True)
    optimized_memory = optimized_df.memory_usage(deep=True).sum()
    
    reduction = (original_memory - optimized_memory) / original_memory * 100
    
    print(f"最適化後メモリ: {optimized_memory / 1024 / 1024:.2f}MB")
    print(f"削減率: {reduction:.1f}%")
    
    return {
        "original_memory_mb": original_memory / 1024 / 1024,
        "optimized_memory_mb": optimized_memory / 1024 / 1024,
        "reduction_percent": reduction,
    }

def main():
    """メイン実行関数"""
    print("軽量AutoMLメモリテスト開始")
    print("=" * 50)
    
    try:
        # メモリ最適化効果テスト
        optimization_results = test_memory_optimization()
        
        # データサイズスケーリングテスト
        scaling_results = test_data_size_scaling()
        
        # pandasメモリ最適化テスト
        pandas_results = test_pandas_optimization()
        
        print("\n" + "=" * 50)
        print("テスト完了")
        
        # 結果サマリー
        print("\n=== 結果サマリー ===")
        
        print("\n1. メモリ最適化効果:")
        for config_name, result in optimization_results.items():
            if result.get("success"):
                print(f"   {config_name}: {result['execution_time']:.2f}秒, "
                      f"ピークメモリ: {result['peak_memory']:.2f}MB")
            else:
                print(f"   {config_name}: エラー - {result.get('error', 'Unknown')}")
        
        print("\n2. データサイズスケーリング:")
        for result in scaling_results:
            if result.get("success"):
                print(f"   {result['data_size_mb']:.2f}MB: {result['execution_time']:.2f}秒, "
                      f"ピークメモリ: {result['memory_peak']:.2f}MB")
            else:
                print(f"   {result['data_size_mb']:.2f}MB: エラー - {result.get('error', 'Unknown')}")
        
        print(f"\n3. pandasメモリ最適化: {pandas_results['reduction_percent']:.1f}%削減")
        
    except Exception as e:
        print(f"テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
