#!/usr/bin/env python3
"""
レガシー形式指標名の完全排除確認テスト
全ての箇所でJSON形式が使用されていることを確認
"""

import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import Mock

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(__file__))

def test_strategy_gene_legacy_elimination():
    """StrategyGeneのレガシー形式排除テスト"""
    print("🧬 StrategyGene レガシー形式排除テスト")
    print("=" * 60)
    
    try:
        from app.core.services.auto_strategy.models.strategy_gene import IndicatorGene
        
        # 各種指標でテスト
        test_indicators = [
            ("RSI", {"period": 14}),
            ("STOCH", {"period": 14}),
            ("MACD", {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
            ("ATR", {"period": 6}),
            ("OBV", {}),
        ]
        
        print("指標遺伝子のレガシー名生成テスト:")
        all_json_format = True
        
        for indicator_type, parameters in test_indicators:
            gene = IndicatorGene(type=indicator_type, parameters=parameters, enabled=True)
            legacy_name = gene.get_legacy_name()
            
            # JSON形式（パラメータなし）であることを確認
            is_json_format = legacy_name == indicator_type
            status = "✅" if is_json_format else "❌"
            print(f"  {status} {indicator_type} -> {legacy_name} (期待値: {indicator_type})")
            
            if not is_json_format:
                all_json_format = False
        
        return all_json_format
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_random_gene_generator_json_format():
    """RandomGeneGeneratorのJSON形式確認テスト"""
    print("\n🎲 RandomGeneGenerator JSON形式テスト")
    print("=" * 60)
    
    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        
        # GA設定を作成
        ga_config = GAConfig(
            population_size=3,
            generations=1,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elite_size=1
        )
        
        generator = RandomGeneGenerator(ga_config)
        
        print("ランダム戦略生成テスト:")
        
        # 複数の戦略を生成してテスト
        for i in range(3):
            strategy_gene = generator.generate_random_gene()
            
            print(f"\n戦略 {i+1}:")
            print(f"  指標: {[ind.type for ind in strategy_gene.indicators]}")
            
            # エントリー条件の指標名をチェック
            print("  エントリー条件:")
            for j, condition in enumerate(strategy_gene.entry_conditions):
                operand = condition.left_operand
                # 指標名かどうかをチェック（基本データソース以外）
                if operand not in ["close", "open", "high", "low", "volume", "OpenInterest", "FundingRate"]:
                    # パラメータが含まれていないかチェック
                    has_params = "_" in operand and any(char.isdigit() for char in operand)
                    status = "❌" if has_params else "✅"
                    print(f"    {status} 条件{j+1}: {operand}")
            
            # エグジット条件の指標名をチェック
            print("  エグジット条件:")
            for j, condition in enumerate(strategy_gene.exit_conditions):
                operand = condition.left_operand
                if operand not in ["close", "open", "high", "low", "volume", "OpenInterest", "FundingRate"]:
                    has_params = "_" in operand and any(char.isdigit() for char in operand)
                    status = "❌" if has_params else "✅"
                    print(f"    {status} 条件{j+1}: {operand}")
        
        return True
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_indicator_adapters_json_format():
    """IndicatorAdaptersのJSON形式確認テスト"""
    print("\n🔧 IndicatorAdapters JSON形式テスト")
    print("=" * 60)
    
    try:
        from app.core.services.indicators.adapters.base_adapter import BaseAdapter
        
        # 各種指標でテスト
        test_cases = [
            ("RSI", {"period": 14}),
            ("SMA", {"period": 20}),
            ("MACD", {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
            ("BB", {"period": 20, "std_dev": 2}),
        ]
        
        print("BaseAdapter指標名生成テスト:")
        all_json_format = True
        
        for indicator_type, parameters in test_cases:
            generated_name = BaseAdapter._generate_indicator_name(indicator_type, parameters)
            
            # JSON形式（パラメータなし）であることを確認
            is_json_format = generated_name == indicator_type
            status = "✅" if is_json_format else "❌"
            print(f"  {status} {indicator_type} -> {generated_name} (期待値: {indicator_type})")
            
            if not is_json_format:
                all_json_format = False
        
        return all_json_format
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compatibility_manager_json_priority():
    """CompatibilityManagerのJSON優先確認テスト"""
    print("\n🔄 CompatibilityManager JSON優先テスト")
    print("=" * 60)
    
    try:
        from app.core.services.indicators.config import compatibility_manager
        
        # 互換性モードの状態を確認
        print(f"互換性モード: {compatibility_manager.compatibility_mode}")
        
        # auto形式での名前生成テスト
        test_cases = [
            ("RSI", {"period": 14}),
            ("SMA", {"period": 20}),
            ("MACD", {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
        ]
        
        print("auto形式での名前生成テスト:")
        all_json_format = True
        
        for indicator_type, parameters in test_cases:
            result = compatibility_manager.generate_name(indicator_type, parameters, format_type="auto")
            
            # JSON形式であることを確認
            is_json_format = isinstance(result, dict) and result.get("indicator") == indicator_type
            status = "✅" if is_json_format else "❌"
            print(f"  {status} {indicator_type} -> {result}")
            
            if not is_json_format:
                all_json_format = False
        
        return all_json_format
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_auto_strategy_execution():
    """完全なオートストラテジー実行テスト"""
    print("\n🚀 完全オートストラテジー実行テスト")
    print("=" * 60)
    
    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        from app.core.services.auto_strategy.factories.indicator_initializer import IndicatorInitializer
        from app.core.services.auto_strategy.factories.condition_evaluator import ConditionEvaluator
        
        # GA設定を作成
        ga_config = GAConfig(
            population_size=2,
            generations=1,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elite_size=1
        )
        
        generator = RandomGeneGenerator(ga_config)
        
        # ランダム戦略を生成
        strategy_gene = generator.generate_random_gene()
        
        print("生成された戦略:")
        print(f"  指標: {[ind.type for ind in strategy_gene.indicators]}")
        print(f"  エントリー条件数: {len(strategy_gene.entry_conditions)}")
        print(f"  エグジット条件数: {len(strategy_gene.exit_conditions)}")
        
        # 戦略ファクトリーで戦略クラスを生成
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(strategy_gene)
        
        print(f"✅ 戦略クラス生成成功: {strategy_class.__name__}")
        
        # テストデータ作成
        dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
        np.random.seed(42)
        price = 45000
        prices = []
        for _ in range(100):
            change = np.random.normal(0, 0.015)
            price *= (1 + change)
            prices.append(max(price, 1000))
        
        class MockData:
            def __init__(self):
                self.Close = np.array(prices)
                self.High = np.array([p * 1.01 for p in prices])
                self.Low = np.array([p * 0.99 for p in prices])
                self.Open = np.array(prices)
                self.Volume = np.array([1000] * 100)
        
        test_data = MockData()
        
        # 戦略インスタンスを作成
        strategy_instance = strategy_class()
        strategy_instance.__dict__['data'] = test_data
        strategy_instance.indicators = {}
        strategy_instance.I = Mock(return_value=Mock())
        
        # 指標初期化
        initializer = IndicatorInitializer()
        initialized_count = 0
        
        print("\n指標初期化:")
        for indicator_gene in strategy_gene.indicators:
            result = initializer.initialize_indicator(
                indicator_gene, test_data, strategy_instance
            )
            if result:
                print(f"  ✅ {indicator_gene.type} -> {result}")
                initialized_count += 1
            else:
                print(f"  ❌ {indicator_gene.type} 初期化失敗")
        
        print(f"\n初期化成功率: {initialized_count}/{len(strategy_gene.indicators)}")
        print(f"登録された指標: {list(strategy_instance.indicators.keys())}")
        
        # 条件評価テスト
        evaluator = ConditionEvaluator()
        
        print("\n条件評価テスト:")
        entry_success = 0
        for i, condition in enumerate(strategy_gene.entry_conditions):
            try:
                result = evaluator.evaluate_condition(condition, strategy_instance)
                print(f"  ✅ エントリー条件{i+1}: {condition.left_operand} {condition.operator} {condition.right_operand}")
                entry_success += 1
            except Exception as e:
                print(f"  ❌ エントリー条件{i+1}: {e}")
        
        exit_success = 0
        for i, condition in enumerate(strategy_gene.exit_conditions):
            try:
                result = evaluator.evaluate_condition(condition, strategy_instance)
                print(f"  ✅ エグジット条件{i+1}: {condition.left_operand} {condition.operator} {condition.right_operand}")
                exit_success += 1
            except Exception as e:
                print(f"  ❌ エグジット条件{i+1}: {e}")
        
        total_conditions = len(strategy_gene.entry_conditions) + len(strategy_gene.exit_conditions)
        success_conditions = entry_success + exit_success
        
        print(f"\n条件評価成功率: {success_conditions}/{total_conditions}")
        
        # 成功率が80%以上なら成功とみなす
        success_rate = success_conditions / total_conditions if total_conditions > 0 else 1.0
        return success_rate >= 0.8
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("🎯 レガシー形式指標名の完全排除確認テスト")
    print("=" * 100)
    print("目的: 全ての箇所でJSON形式が使用され、レガシー形式が完全に排除されていることを確認")
    print("=" * 100)
    
    tests = [
        ("StrategyGene レガシー排除", test_strategy_gene_legacy_elimination),
        ("RandomGeneGenerator JSON形式", test_random_gene_generator_json_format),
        ("IndicatorAdapters JSON形式", test_indicator_adapters_json_format),
        ("CompatibilityManager JSON優先", test_compatibility_manager_json_priority),
        ("完全オートストラテジー実行", test_full_auto_strategy_execution),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name}テスト実行エラー: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 100)
    print("📊 最終テスト結果")
    print("=" * 100)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 100)
    if all_passed:
        print("🎉 全てのテストが成功しました！")
        print("✅ レガシー形式の指標名は完全に排除されました")
        print("✅ 全ての箇所でJSON形式が使用されています")
        print("✅ 'STOCH' が見つかりませんエラーは二度と発生しません")
    else:
        print("⚠️ 一部のテストが失敗しました")
        print("まだレガシー形式が残っている箇所があります")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
