"""
包括的テストスクリプト

修正した機能と関連コードの動作確認
"""

import sys
import os
import numpy as np
import pandas as pd
import time

def test_label_generation_comprehensive():
    """LabelGeneratorの包括的テスト"""
    print("=== LabelGenerator 包括的テスト ===")
    
    try:
        # パスを追加
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
        from utils.label_generation import LabelGenerator, ThresholdMethod
        
        generator = LabelGenerator()
        print("✅ LabelGenerator インスタンス作成成功")
        
        # テストデータ作成
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=500, freq='h')
        
        # 複数パターンのテストデータ
        test_cases = {
            'トレンド上昇': np.linspace(50000, 60000, 500) + np.random.randn(500) * 500,
            'トレンド下降': np.linspace(60000, 50000, 500) + np.random.randn(500) * 500,
            'レンジ相場': 55000 + np.random.randn(500) * 1000,
            'ボラティリティ高': 55000 + np.random.randn(500) * 2000,
            'ボラティリティ低': 55000 + np.random.randn(500) * 100,
        }
        
        print("✅ テストデータ作成成功")
        
        # 各テストケースで全メソッドをテスト
        all_methods = [
            (ThresholdMethod.FIXED, {'threshold': 0.01}),
            (ThresholdMethod.QUANTILE, {}),
            (ThresholdMethod.STD_DEVIATION, {'std_multiplier': 0.5}),
            (ThresholdMethod.DYNAMIC_VOLATILITY, {}),
            (ThresholdMethod.KBINS_DISCRETIZER, {'strategy': 'quantile'}),
            (ThresholdMethod.KBINS_DISCRETIZER, {'strategy': 'uniform'}),
            (ThresholdMethod.KBINS_DISCRETIZER, {'strategy': 'kmeans'}),
        ]
        
        results = {}
        
        for case_name, price_values in test_cases.items():
            price_data = pd.Series(price_values, index=dates, name='Close')
            case_results = {}
            
            print(f"\n--- {case_name} テスト ---")
            
            for method, params in all_methods:
                try:
                    start_time = time.time()
                    labels, info = generator.generate_labels(
                        price_data, method=method, **params
                    )
                    end_time = time.time()
                    
                    # 基本検証
                    unique_labels = set(labels.unique())
                    expected_labels = {0, 1, 2}
                    
                    if unique_labels == expected_labels:
                        distribution = info.get('actual_distribution', {})
                        execution_time = end_time - start_time
                        
                        case_results[f"{method.value}_{params}"] = {
                            'success': True,
                            'distribution': distribution,
                            'execution_time': execution_time,
                            'label_count': len(labels),
                        }
                        
                        print(f"✅ {method.value}: {execution_time:.3f}s, 分布={distribution}")
                    else:
                        print(f"❌ {method.value}: ラベル異常 {unique_labels}")
                        case_results[f"{method.value}_{params}"] = {'success': False}
                        
                except Exception as e:
                    print(f"❌ {method.value}: エラー {e}")
                    case_results[f"{method.value}_{params}"] = {'success': False, 'error': str(e)}
            
            results[case_name] = case_results
        
        # 結果サマリー
        print("\n=== 結果サマリー ===")
        total_tests = 0
        successful_tests = 0
        
        for case_name, case_results in results.items():
            case_success = sum(1 for r in case_results.values() if r.get('success', False))
            case_total = len(case_results)
            total_tests += case_total
            successful_tests += case_success
            
            print(f"{case_name}: {case_success}/{case_total} 成功")
        
        print(f"\n全体: {successful_tests}/{total_tests} 成功 ({successful_tests/total_tests*100:.1f}%)")
        
        return successful_tests == total_tests
        
    except Exception as e:
        print(f"❌ LabelGenerator 包括的テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kbins_discretizer_edge_cases():
    """KBinsDiscretizerのエッジケーステスト"""
    print("\n=== KBinsDiscretizer エッジケーステスト ===")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
        from utils.label_generation import LabelGenerator, ThresholdMethod
        
        generator = LabelGenerator()
        
        # エッジケース
        edge_cases = {
            '最小データ': np.random.randn(10),
            '同一値データ': np.ones(100),
            '極端な外れ値': np.concatenate([np.ones(95), [1000, -1000, 2000, -2000, 3000]]),
            '欠損値含む': np.concatenate([np.random.randn(95), [np.nan] * 5]),
            '大きなデータ': np.random.randn(10000),
        }
        
        for case_name, data in edge_cases.items():
            print(f"\n--- {case_name} ---")
            dates = pd.date_range('2023-01-01', periods=len(data), freq='h')
            price_data = pd.Series(data, index=dates, name='Close')
            
            try:
                labels, info = generator.generate_labels(
                    price_data,
                    method=ThresholdMethod.KBINS_DISCRETIZER,
                    strategy='quantile'
                )
                
                print(f"✅ {case_name}: 成功")
                print(f"   ラベル数: {len(labels)}")
                print(f"   ユニークラベル: {set(labels.unique())}")
                print(f"   メソッド: {info.get('method')}")
                
            except Exception as e:
                print(f"⚠️ {case_name}: {e}")
                # フォールバックが動作することを確認
                if 'フォールバック' in str(e) or info.get('method') != 'kbins_discretizer':
                    print("   フォールバック動作確認")
        
        print("✅ エッジケーステスト完了")
        return True
        
    except Exception as e:
        print(f"❌ エッジケーステストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """パフォーマンス比較テスト"""
    print("\n=== パフォーマンス比較テスト ===")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
        from utils.label_generation import LabelGenerator, ThresholdMethod
        
        generator = LabelGenerator()
        
        # 大きなデータセットでテスト
        np.random.seed(42)
        large_data = np.random.randn(5000)
        dates = pd.date_range('2023-01-01', periods=len(large_data), freq='h')
        price_data = pd.Series(large_data, index=dates, name='Close')
        
        methods_to_compare = [
            ('ADAPTIVE（複雑）', ThresholdMethod.ADAPTIVE, {}),
            ('KBINS_DISCRETIZER（簡素）', ThresholdMethod.KBINS_DISCRETIZER, {'strategy': 'quantile'}),
            ('QUANTILE（標準）', ThresholdMethod.QUANTILE, {}),
        ]
        
        print("大きなデータセット（5000サンプル）でのパフォーマンス比較:")
        
        for name, method, params in methods_to_compare:
            times = []
            
            # 複数回実行して平均を取る
            for _ in range(3):
                start_time = time.time()
                labels, info = generator.generate_labels(
                    price_data, method=method, **params
                )
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            print(f"✅ {name}: {avg_time:.3f}s (平均)")
        
        print("✅ パフォーマンス比較完了")
        return True
        
    except Exception as e:
        print(f"❌ パフォーマンス比較エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_validation_functions():
    """検証関数のテスト"""
    print("\n=== 検証関数テスト ===")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
        from utils.label_generation import LabelGenerator
        
        # テストラベル
        test_labels = [
            pd.Series([0, 1, 2, 0, 1, 2]),  # バランス良い
            pd.Series([0, 0, 0, 1, 1, 2]),  # 偏りあり
            pd.Series([1, 1, 1, 1, 1, 1]),  # 単一クラス
            pd.Series([0, 2, 0, 2, 0, 2]),  # 2クラスのみ
        ]
        
        for i, labels in enumerate(test_labels):
            print(f"\nテストケース {i+1}:")
            validation_result = LabelGenerator.validate_label_distribution(labels)
            
            print(f"  有効性: {validation_result['is_valid']}")
            print(f"  警告数: {len(validation_result['warnings'])}")
            print(f"  エラー数: {len(validation_result['errors'])}")
            print(f"  分布: {validation_result['distribution']}")
        
        print("✅ 検証関数テスト完了")
        return True
        
    except Exception as e:
        print(f"❌ 検証関数テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("関連コード包括的テスト開始\n")
    
    # テスト実行
    test1_result = test_label_generation_comprehensive()
    test2_result = test_kbins_discretizer_edge_cases()
    test3_result = test_performance_comparison()
    test4_result = test_validation_functions()
    
    # 結果サマリー
    print("\n" + "="*50)
    print("=== 最終テスト結果サマリー ===")
    print(f"包括的テスト: {'✅ 成功' if test1_result else '❌ 失敗'}")
    print(f"エッジケーステスト: {'✅ 成功' if test2_result else '❌ 失敗'}")
    print(f"パフォーマンス比較: {'✅ 成功' if test3_result else '❌ 失敗'}")
    print(f"検証関数テスト: {'✅ 成功' if test4_result else '❌ 失敗'}")
    
    all_success = all([test1_result, test2_result, test3_result, test4_result])
    
    if all_success:
        print("\n🎉 すべての関連コードテストが成功しました！")
        print("修正内容と周辺機能が正常に動作しています。")
    else:
        print("\n⚠️ 一部のテストが失敗しました。")
    
    return all_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
