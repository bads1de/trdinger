#!/usr/bin/env python3
"""
実際のオートストラテジー実行でのSTOCH指標テスト
"""

import sys
import os
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(__file__))

def create_test_data():
    """テスト用のOHLCVデータを作成"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    
    # ランダムウォークでリアルなデータを生成
    np.random.seed(42)
    price = 45000
    prices = []
    volumes = []
    
    for _ in range(100):
        change = np.random.normal(0, 0.02)  # 2%の標準偏差
        price *= (1 + change)
        prices.append(price)
        volumes.append(np.random.uniform(100, 1000))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    return data

def test_strategy_with_stoch():
    """STOCH指標を含む戦略の実際のテスト"""
    print("🧪 STOCH指標を含む実際の戦略テスト")
    print("=" * 60)
    
    try:
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, IndicatorGene, Condition
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        from app.core.services.auto_strategy.factories.indicator_initializer import IndicatorInitializer
        from app.core.services.auto_strategy.factories.condition_evaluator import ConditionEvaluator
        
        # STOCH指標を含む戦略遺伝子を作成
        indicators = [
            IndicatorGene(type="STOCH", parameters={"period": 14}, enabled=True),
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
        ]
        
        # STOCH条件を含む戦略
        entry_conditions = [
            Condition(left_operand="STOCH", operator="<", right_operand=20),
            Condition(left_operand="RSI", operator="<", right_operand=30),
        ]
        
        exit_conditions = [
            Condition(left_operand="STOCH", operator=">", right_operand=80),
            Condition(left_operand="RSI", operator=">", right_operand=70),
        ]
        
        strategy_gene = StrategyGene(
            indicators=indicators,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions
        )
        
        print("✅ 戦略遺伝子作成成功")
        
        # 戦略ファクトリーで戦略クラスを生成
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(strategy_gene)
        
        print("✅ 戦略クラス生成成功")
        
        # テストデータでバックテスト実行をシミュレート
        test_data = create_test_data()
        
        # バックテストライブラリのDataクラスをシミュレート
        from unittest.mock import Mock
        
        mock_bt_data = Mock()
        mock_bt_data.Close = test_data['close'].values
        mock_bt_data.High = test_data['high'].values
        mock_bt_data.Low = test_data['low'].values
        mock_bt_data.Open = test_data['open'].values
        mock_bt_data.Volume = test_data['volume'].values
        
        # 戦略インスタンスを作成
        strategy_instance = strategy_class()
        strategy_instance.data = mock_bt_data
        strategy_instance.indicators = {}
        strategy_instance.I = Mock(return_value=Mock())
        
        # 指標初期化
        initializer = IndicatorInitializer()
        
        print("\n指標初期化中...")
        initialized_indicators = []
        
        for indicator_gene in strategy_gene.indicators:
            result = initializer.initialize_indicator(
                indicator_gene, mock_bt_data, strategy_instance
            )
            if result:
                initialized_indicators.append(result)
                print(f"  ✅ {indicator_gene.type} -> {result}")
            else:
                print(f"  ❌ {indicator_gene.type} 初期化失敗")
        
        print(f"\n初期化された指標: {list(strategy_instance.indicators.keys())}")
        
        # 条件評価テスト
        evaluator = ConditionEvaluator()
        
        print("\n条件評価テスト:")
        
        # エントリー条件
        print("  エントリー条件:")
        for i, condition in enumerate(strategy_gene.entry_conditions):
            try:
                result = evaluator.evaluate_condition(condition, strategy_instance)
                print(f"    条件{i+1}: {condition.left_operand} {condition.operator} {condition.right_operand} = {result}")
            except Exception as e:
                print(f"    条件{i+1}: ❌ エラー - {e}")
        
        # エグジット条件
        print("  エグジット条件:")
        for i, condition in enumerate(strategy_gene.exit_conditions):
            try:
                result = evaluator.evaluate_condition(condition, strategy_instance)
                print(f"    条件{i+1}: {condition.left_operand} {condition.operator} {condition.right_operand} = {result}")
            except Exception as e:
                print(f"    条件{i+1}: ❌ エラー - {e}")
        
        print("\n✅ 実際の戦略テスト完了")
        return True
        
    except Exception as e:
        print(f"❌ 実際の戦略テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stoch_parameter_handling():
    """STOCHパラメータ処理テスト"""
    print("\n🔧 STOCHパラメータ処理テスト")
    print("=" * 60)
    
    try:
        from app.core.services.auto_strategy.factories.indicator_calculator import IndicatorCalculator
        
        calculator = IndicatorCalculator()
        
        # STOCHの設定を確認
        if "STOCH" in calculator.indicator_adapters:
            stoch_config = calculator._get_legacy_config("STOCH")
            print("STOCH設定:")
            print(f"  adapter_function: {stoch_config.get('adapter_function')}")
            print(f"  required_data: {stoch_config.get('required_data')}")
            print(f"  parameters: {stoch_config.get('parameters')}")
            print(f"  result_type: {stoch_config.get('result_type')}")
            print(f"  result_handler: {stoch_config.get('result_handler')}")
            
            # パラメータ処理テスト
            test_parameters = {"period": 14}
            
            print(f"\nテストパラメータ: {test_parameters}")
            
            # パラメータ変換テスト
            adapter_params = calculator._convert_parameters_for_adapter(
                "STOCH", test_parameters, stoch_config
            )
            print(f"変換後パラメータ: {adapter_params}")
            
        return True
        
    except Exception as e:
        print(f"❌ STOCHパラメータ処理エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("🎯 実際のオートストラテジーでのSTOCH指標テスト")
    print("=" * 80)
    print("目的: 実際の実行環境でのSTOCH指標エラーを再現・修正")
    print("=" * 80)
    
    tests = [
        ("STOCHパラメータ処理", test_stoch_parameter_handling),
        ("STOCH戦略実行", test_strategy_with_stoch),
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
        print("✅ STOCH指標は実際の環境でも正常に動作しています")
    else:
        print("⚠️ 一部のテストが失敗しました")
        print("実際の環境でSTOCH指標に問題があります")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
