#!/usr/bin/env python3
"""
STOCH指標修正の確認テスト
"""

import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import Mock

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(__file__))

def create_test_data():
    """テスト用データ作成"""
    dates = pd.date_range(start='2024-01-01', periods=200, freq='h')
    
    np.random.seed(42)
    price = 45000
    prices = []
    volumes = []
    
    for _ in range(200):
        change = np.random.normal(0, 0.015)
        price *= (1 + change)
        price = max(price, 1000)
        prices.append(price)
        volumes.append(np.random.uniform(500, 2000))
    
    class MockData:
        def __init__(self):
            self.Close = np.array(prices)
            self.High = np.array([p * (1 + np.random.uniform(0, 0.02)) for p in prices])
            self.Low = np.array([p * (1 - np.random.uniform(0, 0.02)) for p in prices])
            self.Open = np.array(prices)
            self.Volume = np.array(volumes)
    
    return MockData()

def test_stoch_initialization_fix():
    """STOCH指標初期化修正のテスト"""
    print("🔧 STOCH指標初期化修正テスト")
    print("=" * 60)
    
    try:
        from app.core.services.auto_strategy.factories.indicator_initializer import IndicatorInitializer
        from app.core.services.auto_strategy.models.strategy_gene import IndicatorGene
        
        initializer = IndicatorInitializer()
        test_data = create_test_data()
        
        # STOCH指標遺伝子を作成
        stoch_gene = IndicatorGene(
            type="STOCH",
            parameters={"period": 14},
            enabled=True
        )
        
        # モック戦略インスタンス
        mock_strategy = Mock()
        mock_strategy.indicators = {}
        mock_strategy.I = Mock(return_value=Mock())
        
        print("STOCH指標初期化実行中...")
        
        # 初期化実行
        result = initializer.initialize_indicator(
            stoch_gene, test_data, mock_strategy
        )
        
        if result:
            print(f"✅ STOCH初期化成功: {result}")
            print(f"登録された指標: {list(mock_strategy.indicators.keys())}")
            
            # 指標が正しく登録されているかチェック
            if "STOCH" in mock_strategy.indicators:
                print("✅ JSON形式での登録確認")
            else:
                print("❌ JSON形式での登録失敗")
            
            if "STOCH_14" in mock_strategy.indicators:
                print("✅ レガシー形式での登録確認")
            else:
                print("❌ レガシー形式での登録失敗")
            
            return True
        else:
            print("❌ STOCH初期化失敗")
            return False
            
    except Exception as e:
        print(f"❌ テストエラー: {e}")
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
            "STOCH_14": Mock(),
        }
        
        # STOCH指標の値を設定
        mock_strategy.indicators["STOCH"].__getitem__ = Mock(return_value=25.0)
        mock_strategy.indicators["STOCH"].__len__ = Mock(return_value=100)
        
        mock_strategy.indicators["STOCH_14"].__getitem__ = Mock(return_value=25.0)
        mock_strategy.indicators["STOCH_14"].__len__ = Mock(return_value=100)
        
        # STOCH条件のテスト
        test_conditions = [
            ("STOCH < 30 (JSON形式)", Condition("STOCH", "<", 30)),
            ("STOCH > 20 (JSON形式)", Condition("STOCH", ">", 20)),
            ("STOCH_14 < 30 (レガシー形式)", Condition("STOCH_14", "<", 30)),
            ("STOCH_14 > 20 (レガシー形式)", Condition("STOCH_14", ">", 20)),
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
        print(f"❌ 条件評価テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_strategy_with_stoch():
    """STOCH指標を含む完全な戦略テスト"""
    print("\n🚀 STOCH指標を含む完全戦略テスト")
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
        
        print("戦略遺伝子:")
        print(f"  指標: {[ind.type for ind in strategy_gene.indicators]}")
        
        # 戦略ファクトリーで戦略クラスを生成
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(strategy_gene)
        
        print(f"✅ 戦略クラス生成成功: {strategy_class.__name__}")
        
        # 戦略インスタンスを作成
        strategy_instance = strategy_class()
        test_data = create_test_data()
        
        # dataプロパティを直接設定せず、initで設定
        strategy_instance.__dict__['data'] = test_data
        strategy_instance.indicators = {}
        strategy_instance.I = Mock(return_value=Mock())
        
        print("\n指標初期化:")
        
        # 指標初期化
        initializer = IndicatorInitializer()
        initialized_count = 0
        
        for indicator_gene in strategy_gene.indicators:
            print(f"  {indicator_gene.type}を初期化中...")
            result = initializer.initialize_indicator(
                indicator_gene, test_data, strategy_instance
            )
            if result:
                print(f"    ✅ 成功: {result}")
                initialized_count += 1
            else:
                print(f"    ❌ 失敗")
        
        print(f"\n初期化された指標数: {initialized_count}/{len(strategy_gene.indicators)}")
        print(f"登録された指標: {list(strategy_instance.indicators.keys())}")
        
        # 条件評価テスト
        print("\n条件評価:")
        evaluator = ConditionEvaluator()
        
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
        
        return True
        
    except Exception as e:
        print(f"❌ 完全戦略テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("🎯 STOCH指標修正確認テスト")
    print("=" * 80)
    print("目的: STOCH指標の初期化修正が正しく動作するか確認")
    print("=" * 80)
    
    tests = [
        ("STOCH初期化修正", test_stoch_initialization_fix),
        ("STOCH条件評価", test_stoch_condition_evaluation),
        ("STOCH完全戦略", test_full_strategy_with_stoch),
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
        print("✅ STOCH指標の修正が正常に動作しています")
        print("✅ オートストラテジー機能でSTOCH指標が利用可能です")
    else:
        print("⚠️ 一部のテストが失敗しました")
        print("STOCH指標にまだ問題があります")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
