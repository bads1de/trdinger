#!/usr/bin/env python3
"""
最終テクニカル指標統合テスト
ADXR + Price Transform指標（AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE）の動作確認
"""

import sys
import os
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def create_test_data(periods=100):
    """テスト用のOHLCVデータを作成"""
    dates = pd.date_range('2024-01-01', periods=periods, freq='D')
    np.random.seed(42)
    
    base_price = 50000
    returns = np.random.normal(0, 0.02, periods)
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    return pd.DataFrame({
        'open': close_prices * (1 + np.random.normal(0, 0.001, periods)),
        'high': close_prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
        'low': close_prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, periods),
    }, index=dates)

def test_new_indicators():
    """新規指標のテスト"""
    print("\n🧪 最終追加テクニカル指標テスト")
    print("=" * 60)
    
    test_data = create_test_data(100)
    print(f"📊 テストデータ作成: {len(test_data)}件")
    
    # 新規指標のテスト
    new_indicators_tests = [
        ("ADXR", "app.core.services.indicators.momentum_indicators", "ADXRIndicator", 14),
        ("AVGPRICE", "app.core.services.indicators.price_transform_indicators", "AVGPRICEIndicator", 1),
        ("MEDPRICE", "app.core.services.indicators.price_transform_indicators", "MEDPRICEIndicator", 1),
        ("TYPPRICE", "app.core.services.indicators.price_transform_indicators", "TYPPRICEIndicator", 1),
        ("WCLPRICE", "app.core.services.indicators.price_transform_indicators", "WCLPRICEIndicator", 1),
    ]
    
    success_count = 0
    for indicator_type, module_name, class_name, period in new_indicators_tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            indicator_class = getattr(module, class_name)
            indicator = indicator_class()
            
            # 計算テスト
            result = indicator.calculate(test_data, period)
            
            # 結果検証
            assert isinstance(result, pd.Series)
            assert len(result) == len(test_data)
            
            # 値の範囲チェック
            valid_values = result.dropna()
            if len(valid_values) > 0:
                if indicator_type == "ADXR":
                    assert all(valid_values >= 0) and all(valid_values <= 100)
                elif indicator_type in ["AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"]:
                    # 価格変換指標は価格レベルの値を持つ
                    assert all(valid_values > 0)  # 正の価格値
            
            print(f"✅ {indicator_type}: 計算成功 (期間: {period}, 有効値: {len(valid_values)})")
            success_count += 1
            
        except Exception as e:
            print(f"❌ {indicator_type}: 計算失敗 - {e}")
    
    print(f"\n📊 計算テスト結果: {success_count}/{len(new_indicators_tests)} 成功")
    return success_count == len(new_indicators_tests)

def test_factory_functions():
    """ファクトリー関数のテスト"""
    print("\n🧪 ファクトリー関数テスト")
    print("=" * 60)
    
    try:
        # モメンタム指標のファクトリー関数テスト
        from app.core.services.indicators.momentum_indicators import get_momentum_indicator
        
        adxr_indicator = get_momentum_indicator("ADXR")
        assert adxr_indicator is not None
        assert adxr_indicator.indicator_type == "ADXR"
        print("✅ ADXR: モメンタム指標ファクトリー関数成功")
        
        # 価格変換指標のファクトリー関数テスト
        from app.core.services.indicators.price_transform_indicators import get_price_transform_indicator
        
        price_indicators = ["AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"]
        success_count = 1  # ADXR already tested
        
        for indicator_type in price_indicators:
            try:
                indicator = get_price_transform_indicator(indicator_type)
                assert indicator is not None
                assert indicator.indicator_type == indicator_type
                print(f"✅ {indicator_type}: 価格変換指標ファクトリー関数成功")
                success_count += 1
            except Exception as e:
                print(f"❌ {indicator_type}: ファクトリー関数失敗 - {e}")
        
        print(f"\n📊 ファクトリー関数テスト結果: {success_count}/5 成功")
        return success_count == 5
        
    except Exception as e:
        print(f"❌ ファクトリー関数テスト失敗: {e}")
        return False

def test_unified_factory():
    """統合ファクトリー関数のテスト"""
    print("\n🧪 統合ファクトリー関数テスト")
    print("=" * 60)
    
    try:
        from app.core.services.indicators import get_indicator_by_type
        
        all_new_indicators = ["ADXR", "AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"]
        success_count = 0
        
        for indicator_type in all_new_indicators:
            try:
                indicator = get_indicator_by_type(indicator_type)
                assert indicator is not None
                assert indicator.indicator_type == indicator_type
                print(f"✅ {indicator_type}: 統合ファクトリー関数成功")
                success_count += 1
            except Exception as e:
                print(f"❌ {indicator_type}: 統合ファクトリー関数失敗 - {e}")
        
        print(f"\n📊 統合ファクトリー関数テスト結果: {success_count}/{len(all_new_indicators)} 成功")
        return success_count == len(all_new_indicators)
        
    except Exception as e:
        print(f"❌ 統合ファクトリー関数テスト失敗: {e}")
        return False

def test_auto_strategy_integration():
    """オートストラテジー統合テスト"""
    print("\n🧪 オートストラテジー統合テスト")
    print("=" * 60)
    
    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        
        generator = RandomGeneGenerator()
        new_indicators = ["ADXR", "AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"]
        
        print(f"📊 利用可能な指標数: {len(generator.available_indicators)}")
        
        success_count = 0
        for indicator in new_indicators:
            if indicator in generator.available_indicators:
                print(f"✅ {indicator}: オートストラテジー統合済み")
                success_count += 1
            else:
                print(f"❌ {indicator}: オートストラテジー未統合")
        
        # パラメータ生成テスト
        param_success = 0
        for indicator in new_indicators:
            try:
                params = generator._generate_indicator_parameters(indicator)
                assert isinstance(params, dict)
                assert "period" in params
                print(f"✅ {indicator}: パラメータ生成成功 - {params}")
                param_success += 1
            except Exception as e:
                print(f"❌ {indicator}: パラメータ生成失敗 - {e}")
        
        print(f"\n📊 統合テスト結果: {success_count}/{len(new_indicators)} 統合済み")
        print(f"📊 パラメータ生成結果: {param_success}/{len(new_indicators)} 成功")
        
        return success_count == len(new_indicators) and param_success == len(new_indicators)
        
    except Exception as e:
        print(f"❌ オートストラテジー統合テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_total_indicator_count():
    """総指標数の確認"""
    print("\n🧪 総指標数確認テスト")
    print("=" * 60)
    
    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        
        generator = RandomGeneGenerator()
        total_indicators = len(generator.available_indicators)
        
        print(f"📊 現在の利用可能指標数: {total_indicators}")
        
        # 期待される指標数（前回39 + 今回5 = 44）
        expected_count = 44
        
        if total_indicators >= expected_count:
            print(f"✅ 指標数確認成功: {total_indicators}種類（期待値: {expected_count}以上）")
            return True
        else:
            print(f"❌ 指標数不足: {total_indicators}種類（期待値: {expected_count}以上）")
            return False
        
    except Exception as e:
        print(f"❌ 総指標数確認テスト失敗: {e}")
        return False

def main():
    """メインテスト実行"""
    print("🚀 最終テクニカル指標統合テスト開始")
    print("=" * 80)
    
    tests = [
        ("新規指標計算テスト", test_new_indicators),
        ("ファクトリー関数テスト", test_factory_functions),
        ("統合ファクトリー関数テスト", test_unified_factory),
        ("オートストラテジー統合テスト", test_auto_strategy_integration),
        ("総指標数確認テスト", test_total_indicator_count),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "="*80)
    print("📊 テスト結果サマリー:")
    print("="*80)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 全てのテストが成功しました！")
        print("新規実装された5個の指標（ADXR + Price Transform 4種類）が正常に動作しています。")
        print("オートストラテジー生成での使用も可能です。")
        print("総指標数が44種類以上に拡張されました。")
    else:
        print("⚠️  一部のテストが失敗しました。")
        print("エラーを確認して修正してください。")
    print("="*80)

if __name__ == "__main__":
    main()
