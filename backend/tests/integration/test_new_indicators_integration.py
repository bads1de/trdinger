#!/usr/bin/env python3
"""
新規追加テクニカル指標の統合テスト
BOP, APO, PPO, AROONOSC, DX指標の動作確認
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
    print("\n🧪 新規追加テクニカル指標テスト")
    print("=" * 60)
    
    test_data = create_test_data(100)
    print(f"📊 テストデータ作成: {len(test_data)}件")
    
    # 新規指標のテスト
    new_indicators_tests = [
        ("BOP", "app.core.services.indicators.momentum_indicators", "BOPIndicator", 1),
        ("APO", "app.core.services.indicators.momentum_indicators", "APOIndicator", 12),
        ("PPO", "app.core.services.indicators.momentum_indicators", "PPOIndicator", 12),
        ("AROONOSC", "app.core.services.indicators.momentum_indicators", "AROONOSCIndicator", 14),
        ("DX", "app.core.services.indicators.momentum_indicators", "DXIndicator", 14),
    ]
    
    success_count = 0
    for indicator_type, module_name, class_name, period in new_indicators_tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            indicator_class = getattr(module, class_name)
            indicator = indicator_class()
            
            # 計算テスト
            if indicator_type in ["APO", "PPO"]:
                result = indicator.calculate(test_data, period, slow_period=26)
            else:
                result = indicator.calculate(test_data, period)
            
            # 結果検証
            assert isinstance(result, pd.Series)
            assert len(result) == len(test_data)
            
            # 値の範囲チェック
            valid_values = result.dropna()
            if len(valid_values) > 0:
                if indicator_type == "BOP":
                    assert all(valid_values >= -1) and all(valid_values <= 1)
                elif indicator_type == "AROONOSC":
                    assert all(valid_values >= -100) and all(valid_values <= 100)
                elif indicator_type == "DX":
                    assert all(valid_values >= 0) and all(valid_values <= 100)
            
            print(f"✅ {indicator_type}: 計算成功 (期間: {period}, 有効値: {len(valid_values)})")
            success_count += 1
            
        except Exception as e:
            print(f"❌ {indicator_type}: 計算失敗 - {e}")
    
    print(f"\n📊 計算テスト結果: {success_count}/{len(new_indicators_tests)} 成功")
    return success_count == len(new_indicators_tests)

def test_factory_function():
    """ファクトリー関数のテスト"""
    print("\n🧪 ファクトリー関数テスト")
    print("=" * 60)
    
    try:
        from app.core.services.indicators.momentum_indicators import get_momentum_indicator
        
        new_indicators = ["BOP", "APO", "PPO", "AROONOSC", "DX"]
        success_count = 0
        
        for indicator_type in new_indicators:
            try:
                indicator = get_momentum_indicator(indicator_type)
                assert indicator is not None
                assert indicator.indicator_type == indicator_type
                print(f"✅ {indicator_type}: ファクトリー関数成功")
                success_count += 1
            except Exception as e:
                print(f"❌ {indicator_type}: ファクトリー関数失敗 - {e}")
        
        print(f"\n📊 ファクトリー関数テスト結果: {success_count}/{len(new_indicators)} 成功")
        return success_count == len(new_indicators)
        
    except Exception as e:
        print(f"❌ ファクトリー関数テスト失敗: {e}")
        return False

def test_auto_strategy_integration():
    """オートストラテジー統合テスト"""
    print("\n🧪 オートストラテジー統合テスト")
    print("=" * 60)
    
    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        
        generator = RandomGeneGenerator()
        new_indicators = ["BOP", "APO", "PPO", "AROONOSC", "DX"]
        
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

def test_ta_lib_functions():
    """TA-Lib関数の直接テスト"""
    print("\n🧪 TA-Lib関数直接テスト")
    print("=" * 60)
    
    try:
        import talib
        test_data = create_test_data(100)
        
        # TA-Lib関数の直接テスト
        ta_lib_tests = [
            ("BOP", lambda: talib.BOP(test_data['open'].values, test_data['high'].values, 
                                     test_data['low'].values, test_data['close'].values)),
            ("APO", lambda: talib.APO(test_data['close'].values, fastperiod=12, slowperiod=26)),
            ("PPO", lambda: talib.PPO(test_data['close'].values, fastperiod=12, slowperiod=26)),
            ("AROONOSC", lambda: talib.AROONOSC(test_data['high'].values, test_data['low'].values, timeperiod=14)),
            ("DX", lambda: talib.DX(test_data['high'].values, test_data['low'].values, 
                                   test_data['close'].values, timeperiod=14)),
        ]
        
        success_count = 0
        for name, func in ta_lib_tests:
            try:
                result = func()
                assert result is not None
                assert len(result) == len(test_data)
                print(f"✅ {name}: TA-Lib関数成功")
                success_count += 1
            except Exception as e:
                print(f"❌ {name}: TA-Lib関数失敗 - {e}")
        
        print(f"\n📊 TA-Lib関数テスト結果: {success_count}/{len(ta_lib_tests)} 成功")
        return success_count == len(ta_lib_tests)
        
    except Exception as e:
        print(f"❌ TA-Lib関数テスト失敗: {e}")
        return False

def main():
    """メインテスト実行"""
    print("🚀 新規追加テクニカル指標統合テスト開始")
    print("=" * 80)
    
    tests = [
        ("TA-Lib関数直接テスト", test_ta_lib_functions),
        ("新規指標計算テスト", test_new_indicators),
        ("ファクトリー関数テスト", test_factory_function),
        ("オートストラテジー統合テスト", test_auto_strategy_integration),
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
        print("新規実装された5個の指標（BOP, APO, PPO, AROONOSC, DX）が正常に動作しています。")
        print("オートストラテジー生成での使用も可能です。")
    else:
        print("⚠️  一部のテストが失敗しました。")
        print("エラーを確認して修正してください。")
    print("="*80)

if __name__ == "__main__":
    main()
