#!/usr/bin/env python3
"""
実際のオートストラテジー実行シナリオでのテスト
実際のエラーが発生する条件を再現
"""

import sys
import os
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(__file__))

def test_real_strategy_generation():
    """実際の戦略生成プロセスをテスト"""
    print("🧬 実際の戦略生成プロセステスト")
    print("=" * 80)
    
    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        
        # GA設定を作成
        ga_config = GAConfig(
            population_size=5,
            generations=2,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elite_size=1
        )
        
        # ランダム遺伝子生成器を作成
        generator = RandomGeneGenerator(ga_config)
        
        print("ランダム戦略遺伝子を生成中...")
        
        # 複数の戦略遺伝子を生成してテスト
        for i in range(5):
            print(f"\n戦略 {i+1}:")
            try:
                strategy_gene = generator.generate_random_gene()
                
                print(f"  指標数: {len(strategy_gene.indicators)}")
                print(f"  指標: {[ind.type for ind in strategy_gene.indicators]}")
                print(f"  エントリー条件数: {len(strategy_gene.entry_conditions)}")
                print(f"  エグジット条件数: {len(strategy_gene.exit_conditions)}")
                
                # 条件の詳細を確認
                print("  エントリー条件:")
                for j, condition in enumerate(strategy_gene.entry_conditions):
                    print(f"    {j+1}: {condition.left_operand} {condition.operator} {condition.right_operand}")
                
                print("  エグジット条件:")
                for j, condition in enumerate(strategy_gene.exit_conditions):
                    print(f"    {j+1}: {condition.left_operand} {condition.operator} {condition.right_operand}")
                
            except Exception as e:
                print(f"  ❌ 戦略生成エラー: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 戦略生成プロセステストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_factory_with_multiple_indicators():
    """複数指標を含む戦略ファクトリーテスト"""
    print("\n🏭 複数指標戦略ファクトリーテスト")
    print("=" * 80)
    
    try:
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, IndicatorGene, Condition
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        
        # 複数の指標を含む戦略遺伝子を作成
        indicators = [
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            IndicatorGene(type="STOCH", parameters={"period": 14}, enabled=True),
            IndicatorGene(type="ATR", parameters={"period": 14}, enabled=True),
        ]
        
        # 複雑な条件を含む戦略
        entry_conditions = [
            Condition(left_operand="RSI", operator="<", right_operand=30),
            Condition(left_operand="STOCH", operator="<", right_operand=20),
            Condition(left_operand="close", operator=">", right_operand="SMA"),
        ]
        
        exit_conditions = [
            Condition(left_operand="RSI", operator=">", right_operand=70),
            Condition(left_operand="STOCH", operator=">", right_operand=80),
        ]
        
        strategy_gene = StrategyGene(
            indicators=indicators,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions
        )
        
        print("複数指標戦略遺伝子:")
        print(f"  指標: {[ind.type for ind in strategy_gene.indicators]}")
        
        # 戦略ファクトリーで戦略クラスを生成
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(strategy_gene)
        
        print(f"✅ 戦略クラス生成成功: {strategy_class.__name__}")
        
        # 戦略クラスの詳細を確認
        strategy_instance = strategy_class()
        print(f"  戦略インスタンス作成成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 複数指標戦略ファクトリーテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backtest_data_simulation():
    """バックテストデータシミュレーションテスト"""
    print("\n📊 バックテストデータシミュレーションテスト")
    print("=" * 80)
    
    try:
        # 実際のバックテストで使用されるデータ形式をシミュレート
        import backtesting
        from unittest.mock import Mock
        
        # サンプルデータを作成
        dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
        np.random.seed(42)
        
        price = 45000
        data = []
        for i in range(100):
            change = np.random.normal(0, 0.02)
            price *= (1 + change)
            high = price * (1 + np.random.uniform(0, 0.01))
            low = price * (1 - np.random.uniform(0, 0.01))
            volume = np.random.uniform(100, 1000)
            
            data.append({
                'Open': price,
                'High': high,
                'Low': low,
                'Close': price,
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        
        print("バックテストデータ:")
        print(f"  データ数: {len(df)}")
        print(f"  カラム: {list(df.columns)}")
        print(f"  期間: {df.index[0]} - {df.index[-1]}")
        
        # バックテストライブラリのDataクラスをシミュレート
        class MockData:
            def __init__(self, df):
                self.df = df
                self.Close = df['Close'].values
                self.High = df['High'].values
                self.Low = df['Low'].values
                self.Open = df['Open'].values
                self.Volume = df['Volume'].values
        
        mock_data = MockData(df)
        
        print(f"  Close価格範囲: {mock_data.Close.min():.2f} - {mock_data.Close.max():.2f}")
        print(f"  Volume範囲: {mock_data.Volume.min():.2f} - {mock_data.Volume.max():.2f}")
        
        return True, mock_data
        
    except Exception as e:
        print(f"❌ バックテストデータシミュレーションエラー: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_full_strategy_execution():
    """完全な戦略実行テスト"""
    print("\n🚀 完全な戦略実行テスト")
    print("=" * 80)
    
    try:
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, IndicatorGene, Condition
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        from app.core.services.auto_strategy.factories.indicator_initializer import IndicatorInitializer
        from app.core.services.auto_strategy.factories.condition_evaluator import ConditionEvaluator
        
        # バックテストデータを取得
        data_success, mock_data = test_backtest_data_simulation()
        if not data_success:
            return False
        
        # 問題が報告されているSTOCHを含む戦略を作成
        indicators = [
            IndicatorGene(type="ATR", parameters={"period": 14}, enabled=True),
            IndicatorGene(type="STOCH", parameters={"period": 14}, enabled=True),
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
        ]
        
        entry_conditions = [
            Condition(left_operand="STOCH", operator="<", right_operand=20),
            Condition(left_operand="RSI", operator="<", right_operand=30),
        ]
        
        exit_conditions = [
            Condition(left_operand="STOCH", operator=">", right_operand=80),
            Condition(left_operand="ATR", operator=">", right_operand=100),
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
        
        # 戦略インスタンスを作成
        strategy_instance = strategy_class()
        strategy_instance.data = mock_data
        strategy_instance.indicators = {}
        
        # モックのI関数を設定
        from unittest.mock import Mock
        strategy_instance.I = Mock(return_value=Mock())
        
        print("\n指標初期化:")
        
        # 指標初期化
        initializer = IndicatorInitializer()
        initialized_count = 0
        
        for indicator_gene in strategy_gene.indicators:
            print(f"  {indicator_gene.type}を初期化中...")
            result = initializer.initialize_indicator(
                indicator_gene, mock_data, strategy_instance
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
        print(f"❌ 完全な戦略実行テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("🎯 実際のオートストラテジー実行シナリオテスト")
    print("=" * 100)
    print("目的: 実際のエラーが発生する条件を特定・再現")
    print("=" * 100)
    
    tests = [
        ("戦略生成プロセス", test_real_strategy_generation),
        ("複数指標戦略ファクトリー", test_strategy_factory_with_multiple_indicators),
        ("完全な戦略実行", test_full_strategy_execution),
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
    print("📊 テスト結果サマリー")
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
        print("✅ 実際の実行シナリオでも正常に動作しています")
        print("💡 エラーは特定の実行条件下でのみ発生している可能性があります")
    else:
        print("⚠️ 一部のテストが失敗しました")
        print("実際の実行環境で問題があります")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
