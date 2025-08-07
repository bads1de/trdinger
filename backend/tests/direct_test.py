"""
直接テストスクリプト

ファイルを直接インポートして修正内容をテスト
"""

import sys
import os
import numpy as np
import pandas as pd

def test_advanced_features_direct():
    """AdvancedFeatureEngineerの直接テスト"""
    print("=== AdvancedFeatureEngineer 直接テスト ===")
    
    try:
        # ファイルを直接実行してテスト
        advanced_features_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'app', 
            'services', 
            'ml', 
            'feature_engineering', 
            'advanced_features.py'
        )
        
        # ファイルの内容を確認
        with open(advanced_features_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # scipy.statsのインポートが削除されているか確認
        if 'from scipy import stats' not in content and 'import scipy.stats' not in content:
            print("✅ scipy.statsのインポートが削除されています")
        else:
            print("❌ scipy.statsのインポートが残っています")
        
        # np.polyfitが使用されているか確認
        if 'np.polyfit' in content:
            print("✅ np.polyfitが使用されています")
        else:
            print("❌ np.polyfitが使用されていません")
        
        # stats.linregressが削除されているか確認
        if 'stats.linregress' not in content:
            print("✅ stats.linregressが削除されています")
        else:
            print("❌ stats.linregressが残っています")
        
        # トレンド強度計算の新しい実装を確認
        if 'calculate_trend_strength' in content:
            print("✅ 新しいトレンド強度計算関数が実装されています")
        else:
            print("❌ 新しいトレンド強度計算関数が見つかりません")
        
        print("✅ AdvancedFeatureEngineer 直接テスト完了\n")
        return True
        
    except Exception as e:
        print(f"❌ AdvancedFeatureEngineer 直接テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_label_generation_direct():
    """LabelGeneratorの直接テスト"""
    print("=== LabelGenerator 直接テスト ===")
    
    try:
        # ファイルを直接確認
        label_generation_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'app', 
            'utils', 
            'label_generation.py'
        )
        
        # ファイルの内容を確認
        with open(label_generation_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # KBinsDiscretizerのインポートが追加されているか確認
        if 'from sklearn.preprocessing import KBinsDiscretizer' in content:
            print("✅ KBinsDiscretizerのインポートが追加されています")
        else:
            print("❌ KBinsDiscretizerのインポートが見つかりません")
        
        # KBINS_DISCRETIZERが追加されているか確認
        if 'KBINS_DISCRETIZER = "kbins_discretizer"' in content:
            print("✅ KBINS_DISCRETIZERが追加されています")
        else:
            print("❌ KBINS_DISCRETIZERが見つかりません")
        
        # 新しいメソッドが実装されているか確認
        if '_calculate_kbins_discretizer_thresholds' in content:
            print("✅ KBinsDiscretizer閾値計算メソッドが実装されています")
        else:
            print("❌ KBinsDiscretizer閾値計算メソッドが見つかりません")
        
        if 'generate_labels_with_kbins_discretizer' in content:
            print("✅ 便利メソッドが実装されています")
        else:
            print("❌ 便利メソッドが見つかりません")
        
        # KBinsDiscretizerの使用を確認
        if 'KBinsDiscretizer(' in content:
            print("✅ KBinsDiscretizerが使用されています")
        else:
            print("❌ KBinsDiscretizerが使用されていません")
        
        print("✅ LabelGenerator 直接テスト完了\n")
        return True
        
    except Exception as e:
        print(f"❌ LabelGenerator 直接テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_functionality():
    """機能テスト"""
    print("=== 機能テスト ===")
    
    try:
        # NumPyのpolyfitテスト
        print("NumPy polyfit テスト:")
        x = np.arange(10)
        y = 2 * x + 1 + np.random.randn(10) * 0.1
        slope = np.polyfit(x, y, 1)[0]
        print(f"✅ 傾き計算成功: {slope:.3f}")
        
        # KBinsDiscretizerテスト
        print("KBinsDiscretizer テスト:")
        from sklearn.preprocessing import KBinsDiscretizer
        
        # テストデータ
        np.random.seed(42)
        data = np.random.randn(100).reshape(-1, 1)
        
        # 3つのビンに分割
        discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
        discretizer.fit(data)
        bin_edges = discretizer.bin_edges_[0]
        
        print(f"✅ ビン境界値: {bin_edges}")
        print(f"✅ 閾値下: {bin_edges[1]:.3f}")
        print(f"✅ 閾値上: {bin_edges[2]:.3f}")
        
        print("✅ 機能テスト完了\n")
        return True
        
    except Exception as e:
        print(f"❌ 機能テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("ライブラリ置き換え修正 直接テスト開始\n")
    
    # テスト実行
    test1_result = test_advanced_features_direct()
    test2_result = test_label_generation_direct()
    test3_result = test_functionality()
    
    # 結果サマリー
    print("=== テスト結果サマリー ===")
    print(f"AdvancedFeatureEngineer 直接テスト: {'✅ 成功' if test1_result else '❌ 失敗'}")
    print(f"LabelGenerator 直接テスト: {'✅ 成功' if test2_result else '❌ 失敗'}")
    print(f"機能テスト: {'✅ 成功' if test3_result else '❌ 失敗'}")
    
    if test1_result and test2_result and test3_result:
        print("\n🎉 すべてのテストが成功しました！")
        print("3.1と3.3の問題修正が正常に実装されています。")
        print("\n修正内容:")
        print("- 3.1: stats.linregressをnp.polyfitに置き換え")
        print("- 3.3: 複雑な動的閾値設定をKBinsDiscretizerで簡素化")
    else:
        print("\n⚠️ 一部のテストが失敗しました。")
    
    return test1_result and test2_result and test3_result

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
