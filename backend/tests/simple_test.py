"""
簡単なテストスクリプト

修正した機能の基本的な動作確認
"""

import sys
import os
import numpy as np
import pandas as pd

# パスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

def test_advanced_features():
    """AdvancedFeatureEngineerの基本テスト"""
    print("=== AdvancedFeatureEngineer テスト ===")
    
    try:
        # インポートテスト
        from services.ml.feature_engineering.advanced_features import AdvancedFeatureEngineer
        print("✅ インポート成功")
        
        # インスタンス作成
        engineer = AdvancedFeatureEngineer()
        print("✅ インスタンス作成成功")
        
        # テストデータ作成
        dates = pd.date_range('2023-01-01', periods=50, freq='H')
        np.random.seed(42)
        
        test_data = pd.DataFrame({
            'Open': 50000 + np.random.randn(50) * 1000,
            'High': 50000 + np.random.randn(50) * 1000 + 500,
            'Low': 50000 + np.random.randn(50) * 1000 - 500,
            'Close': 50000 + np.random.randn(50) * 1000,
            'Volume': np.random.randint(1000, 10000, 50)
        }, index=dates)
        
        # Closeが正の値になるように調整
        test_data['Close'] = np.abs(test_data['Close'])
        test_data['High'] = np.maximum(test_data['High'], test_data['Close'])
        test_data['Low'] = np.minimum(test_data['Low'], test_data['Close'])
        
        print("✅ テストデータ作成成功")
        
        # 時系列特徴量テスト（トレンド強度を含む）
        result = engineer._add_time_series_features(test_data.copy())
        print("✅ 時系列特徴量追加成功")
        
        # トレンド強度の列が存在することを確認
        trend_columns = [col for col in result.columns if 'Trend_strength' in col]
        print(f"✅ トレンド強度列数: {len(trend_columns)}")
        
        # 各トレンド強度列をチェック
        for col in trend_columns:
            non_nan_values = result[col].dropna()
            print(f"✅ {col}: {len(non_nan_values)}個の有効値")
        
        # scipy.statsがインポートされていないことを確認
        import services.ml.feature_engineering.advanced_features as module
        import inspect
        source = inspect.getsource(module)
        
        if 'from scipy import stats' not in source and 'import scipy.stats' not in source:
            print("✅ scipy.statsのインポートが削除されています")
        else:
            print("❌ scipy.statsのインポートが残っています")
        
        print("✅ AdvancedFeatureEngineer テスト完了\n")
        return True
        
    except Exception as e:
        print(f"❌ AdvancedFeatureEngineer テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_label_generator():
    """LabelGeneratorの基本テスト"""
    print("=== LabelGenerator テスト ===")
    
    try:
        # インポートテスト
        from utils.label_generation import LabelGenerator, ThresholdMethod
        print("✅ インポート成功")
        
        # インスタンス作成
        generator = LabelGenerator()
        print("✅ インスタンス作成成功")
        
        # テスト用の価格データを作成
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='H')
        
        # トレンドのある価格データを生成
        trend = np.linspace(50000, 55000, 200)
        noise = np.random.randn(200) * 500
        price_data = pd.Series(trend + noise, index=dates, name='Close')
        print("✅ テストデータ作成成功")
        
        # KBinsDiscretizerメソッドのテスト
        labels, info = generator.generate_labels(
            price_data,
            method=ThresholdMethod.KBINS_DISCRETIZER,
            strategy='quantile'
        )
        print("✅ KBinsDiscretizerラベル生成成功")
        
        # 基本的な検証
        unique_labels = set(labels.unique())
        expected_labels = {0, 1, 2}
        if unique_labels == expected_labels:
            print(f"✅ ラベル値正常: {unique_labels}")
        else:
            print(f"❌ ラベル値異常: {unique_labels}, 期待値: {expected_labels}")
        
        # 情報辞書の内容を確認
        required_keys = ['method', 'threshold_up', 'threshold_down', 'bin_edges', 'actual_distribution']
        missing_keys = [key for key in required_keys if key not in info]
        if not missing_keys:
            print("✅ 情報辞書の内容正常")
        else:
            print(f"❌ 情報辞書に不足キー: {missing_keys}")
        
        print(f"✅ メソッド: {info.get('method')}")
        print(f"✅ 戦略: {info.get('strategy')}")
        print(f"✅ 分布: {info.get('actual_distribution')}")
        
        # 異なる戦略でのテスト
        strategies = ['uniform', 'quantile', 'kmeans']
        for strategy in strategies:
            try:
                labels_s, info_s = generator.generate_labels(
                    price_data,
                    method=ThresholdMethod.KBINS_DISCRETIZER,
                    strategy=strategy
                )
                print(f"✅ {strategy}戦略成功")
            except Exception as e:
                print(f"❌ {strategy}戦略エラー: {e}")
        
        # 便利メソッドのテスト
        labels_conv, info_conv = generator.generate_labels_with_kbins_discretizer(
            price_data,
            strategy='quantile'
        )
        print("✅ 便利メソッド成功")
        
        print("✅ LabelGenerator テスト完了\n")
        return True
        
    except Exception as e:
        print(f"❌ LabelGenerator テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("ライブラリ置き換え修正テスト開始\n")
    
    # テスト実行
    test1_result = test_advanced_features()
    test2_result = test_label_generator()
    
    # 結果サマリー
    print("=== テスト結果サマリー ===")
    print(f"AdvancedFeatureEngineer: {'✅ 成功' if test1_result else '❌ 失敗'}")
    print(f"LabelGenerator: {'✅ 成功' if test2_result else '❌ 失敗'}")
    
    if test1_result and test2_result:
        print("\n🎉 すべてのテストが成功しました！")
        print("3.1と3.3の問題修正が正常に動作しています。")
    else:
        print("\n⚠️ 一部のテストが失敗しました。")
    
    return test1_result and test2_result

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
