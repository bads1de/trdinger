#!/usr/bin/env python3
"""
WMA実装のテストスクリプト

新しく実装したWMAIndicatorクラスの動作確認を行います。
"""

import sys
import os
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_wma_indicator():
    """WMAIndicatorクラスのテスト"""
    try:
        from app.core.services.indicators import WMAIndicator
        
        print("✅ WMAIndicatorのインポート成功")
        
        # テストデータの作成
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        test_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 115, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        # WMAIndicatorのインスタンス化
        wma_indicator = WMAIndicator()
        print("✅ WMAIndicatorのインスタンス化成功")
        
        # WMA計算のテスト
        period = 20
        result = wma_indicator.calculate(test_data, period)
        
        print(f"✅ WMA計算成功 (期間: {period})")
        print(f"   結果の型: {type(result)}")
        print(f"   結果の長さ: {len(result)}")
        print(f"   最初の5つの値: {result.head()}")
        
        # 説明の取得テスト
        description = wma_indicator.get_description()
        print(f"✅ 説明取得成功: {description}")
        
        return True
        
    except Exception as e:
        print(f"❌ WMAIndicatorテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_indicator_factory():
    """指標ファクトリーのテスト"""
    try:
        from app.core.services.indicators import get_indicator_by_type
        
        # WMAの取得テスト
        wma_indicator = get_indicator_by_type("WMA")
        print("✅ ファクトリーからのWMA取得成功")
        print(f"   指標タイプ: {wma_indicator.indicator_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ ファクトリーテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_random_gene_generator():
    """RandomGeneGeneratorのテスト"""
    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        
        # ジェネレーターのインスタンス化
        generator = RandomGeneGenerator()
        print("✅ RandomGeneGeneratorのインスタンス化成功")
        
        # 利用可能な指標の確認
        print(f"   利用可能な指標数: {len(generator.available_indicators)}")
        print(f"   WMAが含まれているか: {'WMA' in generator.available_indicators}")
        
        # ランダム遺伝子の生成テスト
        gene = generator.generate_random_gene()
        print("✅ ランダム遺伝子生成成功")
        print(f"   生成された指標数: {len(gene.indicators)}")
        
        # WMAが含まれる遺伝子を探す
        for i in range(10):
            gene = generator.generate_random_gene()
            wma_indicators = [ind for ind in gene.indicators if ind.type == "WMA"]
            if wma_indicators:
                print(f"✅ WMAを含む遺伝子生成成功 (試行{i+1}回目)")
                print(f"   WMA指標: {wma_indicators[0].type}, パラメータ: {wma_indicators[0].parameters}")
                break
        
        return True
        
    except Exception as e:
        print(f"❌ RandomGeneGeneratorテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("🧪 WMA実装テスト開始\n")
    
    tests = [
        ("WMAIndicatorクラス", test_wma_indicator),
        ("指標ファクトリー", test_indicator_factory),
        ("RandomGeneGenerator", test_random_gene_generator),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}のテスト:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "="*50)
    print("📊 テスト結果サマリー:")
    print("="*50)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 全てのテストが成功しました！")
        print("WMAの実装とオートストラテジー統合が完了しています。")
    else:
        print("⚠️  一部のテストが失敗しました。")
        print("エラーを確認して修正してください。")
    print("="*50)

if __name__ == "__main__":
    main()
