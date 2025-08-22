#!/usr/bin/env python3
"""
パラメータ関連問題の検出テストスクリプト
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

def create_test_data():
    """テスト用のOHLCVデータを生成"""
    np.random.seed(42)
    
    n = 100
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    
    price = 50000
    prices = [price]
    
    for _ in range(n-1):
        change = np.random.normal(0, 0.02)
        price = price * (1 + change)
        prices.append(price)
    
    data = []
    for i, close in enumerate(prices):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.randint(1000000, 10000000)
        
        data.append({
            'Open': open_price,
            'High': max(open_price, high, close),
            'Low': min(open_price, low, close),
            'Close': close,
            'Volume': volume
        })
    
    return pd.DataFrame(data, index=dates)

def test_problematic_indicators():
    """問題が発生しやすい指標をテスト"""
    print("🧪 問題発生しやすい指標テスト")
    print("=" * 50)
    
    df = create_test_data()
    service = TechnicalIndicatorService()
    
    # 同様エラーが発生しやすい指標
    test_cases = [
        # 修正済みを確認
        ("STC", {"period": 10, "fast": 23, "slow": 50}, "STC専用tclength"),
        ("KDJ", {"k": 14, "d": 3}, "KDJ専用k,d"),
        
        # 新規に追加した指標
        ("STOCHRSI", {"period": 14, "k_period": 5, "d_period": 3}, "STOCHRSI特殊パラメータ"),
        ("KST", {"r1": 10, "r2": 15, "r3": 20, "r4": 30}, "KST特殊パラメータ"),
        ("SMI", {"fast": 13, "slow": 25, "signal": 2}, "SMI特殊パラメータ"),
        ("PVO", {"fast": 12, "slow": 26, "signal": 9}, "PVO特殊パラメータ"),
        
        # lengthパラメータを持つが誤処理されやすい指標
        ("RMI", {"length": 20, "mom": 20}, "RMI length+mom"),
        ("DPO", {"length": 20}, "DPO length"),
        ("CHOP", {"length": 14}, "CHOP length"),
        ("VORTEX", {"length": 14}, "VORTEX length"),
        ("CFO", {"period": 9}, "CFO period→length"),
        ("CTI", {"period": 12}, "CTI period→length"),
    ]
    
    success_count = 0
    total_count = len(test_cases)
    
    for indicator_type, params, description in test_cases:
        try:
            print(f"📊 {indicator_type} テスト: {description}")
            result = service.calculate_indicator(df, indicator_type, params)
            
            if isinstance(result, (np.ndarray, tuple)):
                print(f"  ✅ {indicator_type} 正常動作")
                success_count += 1
            else:
                print(f"  ❌ {indicator_type} 結果形式エラー: {type(result)}")
                
        except Exception as e:
            print(f"  ❌ {indicator_type} エラー: {e}")
    
    print(f"\n問題発生しやすい指標テスト結果: {success_count}/{total_count} 成功")
    return success_count == total_count

def test_edge_cases():
    """エッジケースをテスト"""
    print("\n🧪 エッジケース指標テスト")
    print("=" * 50)
    
    df = create_test_data()
    service = TechnicalIndicatorService()
    
    # エッジケース
    edge_cases = [
        # periodパラメータが誤って渡される可能性のある指標
        ("OBV", {"period": 14}, "OBV period無視"),
        ("VWAP", {"period": 14}, "VWAP period無視"),
        ("AD", {"period": 14}, "AD period無視"),
        ("AO", {"period": 14}, "AO period無視"),
        ("BOP", {"period": 14}, "BOP period無視"),
        ("PPO", {"period": 14}, "PPO period無視"),
        ("APO", {"period": 14}, "APO period無視"),
        ("ULTOSC", {"period": 14}, "ULTOSC period無視"),
        
        # 価格変換系
        ("TYPPRICE", {"period": 14}, "TYPPRICE period無視"),
        ("AVGPRICE", {"period": 14}, "AVGPRICE period無視"),
        ("MEDPRICE", {"period": 14}, "MEDPRICE period無視"),
        ("WCLPRICE", {"period": 14}, "WCLPRICE period無視"),
    ]
    
    success_count = 0
    total_count = len(edge_cases)
    
    for indicator_type, params, description in edge_cases:
        try:
            print(f"📊 {indicator_type} テスト: {description}")
            result = service.calculate_indicator(df, indicator_type, params)
            
            if isinstance(result, (np.ndarray, tuple)):
                print(f"  ✅ {indicator_type} 正常動作 - periodパラメータ無視")
                success_count += 1
            else:
                print(f"  ❌ {indicator_type} 結果形式エラー: {type(result)}")
                
        except Exception as e:
            print(f"  ❌ {indicator_type} エラー: {e}")
    
    print(f"\nエッジケース指標テスト結果: {success_count}/{total_count} 成功")
    return success_count == total_count

if __name__ == "__main__":
    print("パラメータ関連問題検出テストスクリプト")
    print("=" * 60)
    
    all_passed = True
    
    # テスト実行
    all_passed &= test_problematic_indicators()
    all_passed &= test_edge_cases()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎊 すべてのテストが成功しました！")
        print("✅ パラメータマッピング問題は解決されています")
    else:
        print("⚠️  まだ問題が残っています")
        print("上記のエラーを確認して修正してください")
    print("=" * 60)