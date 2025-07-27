#!/usr/bin/env python3
"""
NaN修正のテストスクリプト
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

from app.services.ml.feature_engineering.automl_features.autofeat_calculator import AutoFeatCalculator
from app.services.ml.feature_engineering.automl_features.automl_config import AutoFeatConfig

def generate_data_with_nan(rows: int = 500) -> tuple[pd.DataFrame, pd.Series]:
    """
    NaN値を含むテストデータを生成
    """
    print(f"NaN値を含むテストデータ生成: {rows}行")
    
    # 基本データ生成
    data = {
        'feature_1': np.random.normal(0, 1, rows),
        'feature_2': np.random.normal(0, 1, rows),
        'feature_3': np.random.normal(0, 1, rows),
        'feature_4': np.random.normal(0, 1, rows),
        'feature_5': np.random.normal(0, 1, rows),
    }
    
    df = pd.DataFrame(data)
    
    # 意図的にNaN値を挿入
    nan_indices = np.random.choice(rows, size=int(rows * 0.1), replace=False)
    df.loc[nan_indices, 'feature_1'] = np.nan
    
    # 無限値も挿入
    inf_indices = np.random.choice(rows, size=int(rows * 0.05), replace=False)
    df.loc[inf_indices, 'feature_2'] = np.inf
    
    # ターゲット変数（一部にNaN）
    target = df['feature_1'] * 0.5 + df['feature_2'] * 0.3 + np.random.normal(0, 0.1, rows)
    target_nan_indices = np.random.choice(rows, size=int(rows * 0.05), replace=False)
    target.iloc[target_nan_indices] = np.nan
    
    print(f"NaN値: {df.isnull().sum().sum()}個")
    print(f"無限値: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}個")
    print(f"ターゲットNaN: {target.isnull().sum()}個")
    
    return df, pd.Series(target, name='target')

def test_nan_handling():
    """NaN処理のテスト"""
    print("=== NaN処理テスト ===")
    
    # NaN値を含むデータでテスト
    df, target = generate_data_with_nan(500)
    data_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    print(f"データサイズ: {data_size_mb:.2f}MB")
    
    # 最適化設定を取得
    config = AutoFeatConfig().get_memory_optimized_config(data_size_mb)
    
    print(f"最適化設定: max_features={config.max_features}, "
          f"max_gb={config.max_gb}")
    
    calculator = AutoFeatCalculator(config)
    
    start_time = time.time()
    try:
        with calculator as calc:
            result_df, info = calc.generate_features(df, target, max_features=5)
        
        execution_time = time.time() - start_time
        
        print(f"✅ 成功: {execution_time:.2f}秒")
        print(f"   入力データ形状: {df.shape}")
        print(f"   出力データ形状: {result_df.shape}")
        print(f"   生成特徴量: {info.get('generated_features', 0)}個")
        print(f"   ピークメモリ: {max(calc._memory_usage_before, calc._memory_usage_after):.2f}MB")
        
        # 出力データのNaNチェック
        if result_df.isnull().any().any():
            print(f"⚠️  出力データにNaN値が残っています: {result_df.isnull().sum().sum()}個")
        else:
            print("✅ 出力データにNaN値はありません")
        
        return True
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ エラー: {e}")
        return False

def test_clean_data():
    """クリーンなデータでのテスト"""
    print("\n=== クリーンデータテスト ===")
    
    # クリーンなデータ生成
    rows = 500
    data = {
        'feature_1': np.random.normal(0, 1, rows),
        'feature_2': np.random.normal(0, 1, rows),
        'feature_3': np.random.normal(0, 1, rows),
    }
    
    df = pd.DataFrame(data)
    target = df['feature_1'] * 0.5 + df['feature_2'] * 0.3 + np.random.normal(0, 0.1, rows)
    target = pd.Series(target, name='target')
    
    data_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"データサイズ: {data_size_mb:.2f}MB")
    
    config = AutoFeatConfig().get_memory_optimized_config(data_size_mb)
    calculator = AutoFeatCalculator(config)
    
    start_time = time.time()
    try:
        with calculator as calc:
            result_df, info = calc.generate_features(df, target, max_features=5)
        
        execution_time = time.time() - start_time
        
        print(f"✅ 成功: {execution_time:.2f}秒")
        print(f"   入力データ形状: {df.shape}")
        print(f"   出力データ形状: {result_df.shape}")
        print(f"   生成特徴量: {info.get('generated_features', 0)}個")
        print(f"   ピークメモリ: {max(calc._memory_usage_before, calc._memory_usage_after):.2f}MB")
        
        return True
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ エラー: {e}")
        return False

def main():
    """メイン実行関数"""
    print("NaN修正テスト開始")
    print("=" * 40)
    
    try:
        # NaN処理テスト
        nan_success = test_nan_handling()
        
        # クリーンデータテスト
        clean_success = test_clean_data()
        
        print("\n" + "=" * 40)
        print("テスト結果:")
        print(f"  NaN処理テスト: {'✅ 成功' if nan_success else '❌ 失敗'}")
        print(f"  クリーンデータテスト: {'✅ 成功' if clean_success else '❌ 失敗'}")
        
        if nan_success and clean_success:
            print("\n🎉 全てのテストが成功しました！")
        else:
            print("\n⚠️  一部のテストが失敗しました。")
        
    except Exception as e:
        print(f"テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
