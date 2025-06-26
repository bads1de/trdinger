#!/usr/bin/env python3
"""
STOCH指標エラーの調査と修正テスト
"""

import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import Mock

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(__file__))

def test_stoch_availability():
    """STOCH指標の利用可能性テスト"""
    print("🔍 STOCH指標利用可能性テスト")
    print("=" * 60)
    
    try:
        from app.core.services.auto_strategy.factories.indicator_calculator import IndicatorCalculator
        
        calculator = IndicatorCalculator()
        
        print("利用可能な指標:")
        for indicator_name in calculator.indicator_adapters.keys():
            print(f"  - {indicator_name}")
        
        # STOCHが含まれているかチェック
        if "STOCH" in calculator.indicator_adapters:
            print("\n✅ STOCH指標は利用可能です")
            
            # STOCH設定の詳細を確認
            stoch_config = calculator.indicator_adapters["STOCH"]
            print(f"  設定: {stoch_config}")
            
        else:
            print("\n❌ STOCH指標が利用可能な指標リストに含まれていません")
            
        return "STOCH" in calculator.indicator_adapters
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stoch_calculation():
    """STOCH指標の計算テスト"""
    print("\n🧮 STOCH指標計算テスト")
    print("=" * 60)
    
    try:
        from app.core.services.auto_strategy.factories.indicator_calculator import IndicatorCalculator
        
        calculator = IndicatorCalculator()
        
        # テストデータ作成
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        np.random.seed(42)
        
        price = 45000
        prices = []
        for _ in range(100):
            change = np.random.normal(0, 0.02)
            price *= (1 + change)
            prices.append(price)
        
        close_data = pd.Series(prices, index=dates)
        high_data = pd.Series([p * 1.01 for p in prices], index=dates)
        low_data = pd.Series([p * 0.99 for p in prices], index=dates)
        volume_data = pd.Series([1000] * 100, index=dates)
        
        # STOCH計算テスト
        print("STOCH計算を実行中...")
        
        result, indicator_name = calculator.calculate_indicator(
            "STOCH",
            {"period": 14},
            close_data,
            high_data,
            low_data,
            volume_data
        )
        
        if result is not None:
            print(f"✅ STOCH計算成功")
            print(f"  指標名: {indicator_name}")
            print(f"  結果タイプ: {type(result)}")
            if hasattr(result, 'columns'):
                print(f"  カラム: {list(result.columns)}")
            print(f"  データ数: {len(result)}")
        else:
            print("❌ STOCH計算失敗")
            
        return result is not None
        
    except Exception as e:
        print(f"❌ STOCH計算エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stoch_initialization():
    """STOCH指標の初期化テスト"""
    print("\n🔧 STOCH指標初期化テスト")
    print("=" * 60)
    
    try:
        from app.core.services.auto_strategy.factories.indicator_initializer import IndicatorInitializer
        from app.core.services.auto_strategy.models.strategy_gene import IndicatorGene
        from unittest.mock import Mock
        
        initializer = IndicatorInitializer()
        
        # STOCH指標遺伝子を作成
        stoch_gene = IndicatorGene(
            type="STOCH",
            parameters={"period": 14},
            enabled=True
        )
        
        # テストデータ作成
        mock_data = Mock()
        mock_data.Close = pd.Series([45000 + i for i in range(100)])
        mock_data.High = pd.Series([45100 + i for i in range(100)])
        mock_data.Low = pd.Series([44900 + i for i in range(100)])
        mock_data.Volume = pd.Series([1000] * 100)
        
        # モック戦略インスタンス
        mock_strategy = Mock()
        mock_strategy.indicators = {}
        mock_strategy.I = Mock(return_value=Mock())
        
        # STOCH初期化テスト
        print("STOCH初期化を実行中...")
        
        result = initializer.initialize_indicator(stoch_gene, mock_data, mock_strategy)
        
        if result:
            print(f"✅ STOCH初期化成功")
            print(f"  返された指標名: {result}")
            print(f"  登録された指標: {list(mock_strategy.indicators.keys())}")
        else:
            print("❌ STOCH初期化失敗")
            
        return result is not None
        
    except Exception as e:
        print(f"❌ STOCH初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stoch_condition_evaluation():
    """STOCH条件評価テスト"""
    print("\n📊 STOCH条件評価テスト")
    print("=" * 60)
    
    try:
        from app.core.services.auto_strategy.factories.condition_evaluator import ConditionEvaluator
        from app.core.services.auto_strategy.models.strategy_gene import Condition
        
        evaluator = ConditionEvaluator()
        
        # モック戦略インスタンス（STOCHが登録されている状態）
        mock_strategy = Mock()
        mock_strategy.indicators = {
            "STOCH": Mock(),
        }
        
        # STOCH指標の値を設定
        mock_strategy.indicators["STOCH"].__getitem__ = Mock(return_value=25.0)
        mock_strategy.indicators["STOCH"].__len__ = Mock(return_value=100)
        
        # STOCH条件のテスト
        test_conditions = [
            ("STOCH < 30", Condition("STOCH", "<", 30)),
            ("STOCH > 20", Condition("STOCH", ">", 20)),
        ]
        
        print("STOCH条件評価結果:")
        all_success = True
        
        for description, condition in test_conditions:
            try:
                result = evaluator.evaluate_condition(condition, mock_strategy)
                print(f"  {description}: {result}")
            except Exception as e:
                print(f"  {description}: ❌ エラー - {e}")
                all_success = False
        
        return all_success
        
    except Exception as e:
        print(f"❌ STOCH条件評価エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_indicator_constants():
    """指標定数の確認テスト"""
    print("\n📋 指標定数確認テスト")
    print("=" * 60)
    
    try:
        from app.core.services.indicators.constants import ALL_INDICATORS, MOMENTUM_INDICATORS
        
        print("全指標リスト:")
        for indicator in ALL_INDICATORS:
            print(f"  - {indicator}")
        
        print(f"\nモメンタム系指標:")
        for indicator in MOMENTUM_INDICATORS:
            print(f"  - {indicator}")
        
        stoch_in_all = "STOCH" in ALL_INDICATORS
        stoch_in_momentum = "STOCH" in MOMENTUM_INDICATORS
        
        print(f"\nSTOCH in ALL_INDICATORS: {stoch_in_all}")
        print(f"STOCH in MOMENTUM_INDICATORS: {stoch_in_momentum}")
        
        return stoch_in_all and stoch_in_momentum
        
    except Exception as e:
        print(f"❌ 指標定数確認エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("🎯 STOCH指標エラー調査テスト")
    print("=" * 80)
    print("目的: 'STOCH' が見つかりませんエラーの原因調査")
    print("=" * 80)
    
    tests = [
        ("指標定数確認", test_indicator_constants),
        ("STOCH利用可能性", test_stoch_availability),
        ("STOCH計算", test_stoch_calculation),
        ("STOCH初期化", test_stoch_initialization),
        ("STOCH条件評価", test_stoch_condition_evaluation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name}テスト実行エラー: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 80)
    print("📊 テスト結果サマリー")
    print("=" * 80)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 全てのテストが成功しました！")
        print("✅ STOCH指標は正常に動作しています")
    else:
        print("⚠️ 一部のテストが失敗しました")
        print("STOCH指標に問題があります")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
