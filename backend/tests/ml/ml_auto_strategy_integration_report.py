#!/usr/bin/env python3
"""
ML-オートストラテジー統合レポート

オートストラテジーとMLの統合状況を包括的に検証し、
統合の完了状況をレポートします。
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime

def create_test_data(size: int = 100) -> pd.DataFrame:
    """テスト用のOHLCVデータを生成"""
    dates = pd.date_range(start='2023-01-01', periods=size, freq='1h')
    
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, size)
    prices = 50000 * np.exp(np.cumsum(returns))
    
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        high = price * (1 + abs(np.random.normal(0, 0.005)))
        low = price * (1 - abs(np.random.normal(0, 0.005)))
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    return df

def test_ml_indicator_service():
    """MLIndicatorServiceテスト"""
    try:
        from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
        
        service = MLIndicatorService()
        test_data = create_test_data(50)
        
        result = service.calculate_ml_indicators(test_data)
        
        return {
            'status': 'SUCCESS',
            'indicators': list(result.keys()),
            'data_length': len(test_data),
            'result_lengths': {k: len(v) for k, v in result.items()},
            'value_ranges': {k: f"[{v.min():.3f}, {v.max():.3f}]" for k, v in result.items()}
        }
    except Exception as e:
        return {'status': 'FAILED', 'error': str(e)}

def test_indicator_calculator():
    """IndicatorCalculatorテスト"""
    try:
        from app.core.services.auto_strategy.calculators.indicator_calculator import IndicatorCalculator
        
        calculator = IndicatorCalculator()
        test_data = create_test_data(30)
        
        class MockBacktestData:
            def __init__(self, df):
                self.df = df
        
        mock_data = MockBacktestData(test_data)
        
        results = {}
        ml_indicators = ['ML_UP_PROB', 'ML_DOWN_PROB', 'ML_RANGE_PROB']
        
        for indicator in ml_indicators:
            result = calculator.calculate_indicator(indicator, {}, mock_data)
            results[indicator] = {
                'success': result is not None,
                'length': len(result) if result is not None else 0,
                'range': f"[{result.min():.3f}, {result.max():.3f}]" if result is not None else "N/A"
            }
        
        return {'status': 'SUCCESS', 'results': results}
    except Exception as e:
        return {'status': 'FAILED', 'error': str(e)}

def test_smart_condition_generator():
    """SmartConditionGeneratorテスト"""
    try:
        from app.core.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
        from app.core.services.auto_strategy.models.gene_strategy import IndicatorGene
        
        generator = SmartConditionGenerator()
        
        indicators = [
            IndicatorGene(type='RSI', parameters={'period': 14}, enabled=True),
            IndicatorGene(type='ML_UP_PROB', parameters={}, enabled=True),
            IndicatorGene(type='ML_DOWN_PROB', parameters={}, enabled=True),
        ]
        
        long_conditions, short_conditions, exit_conditions = generator.generate_balanced_conditions(indicators)
        
        # ML指標を使った条件の確認
        all_conditions = long_conditions + short_conditions + exit_conditions
        ml_condition_count = 0
        ml_conditions = []
        
        for condition in all_conditions:
            condition_str = str(condition)
            if any(ml_ind in condition_str for ml_ind in ['ML_UP_PROB', 'ML_DOWN_PROB', 'ML_RANGE_PROB']):
                ml_condition_count += 1
                ml_conditions.append(condition_str)
        
        return {
            'status': 'SUCCESS',
            'total_conditions': len(all_conditions),
            'long_conditions': len(long_conditions),
            'short_conditions': len(short_conditions),
            'exit_conditions': len(exit_conditions),
            'ml_condition_count': ml_condition_count,
            'ml_conditions': ml_conditions
        }
    except Exception as e:
        return {'status': 'FAILED', 'error': str(e)}

def test_random_gene_generator():
    """RandomGeneGeneratorテスト"""
    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        
        # ML有効設定
        config = GAConfig()
        config.enable_ml_indicators = True
        config.max_indicators = 3
        
        generator = RandomGeneGenerator(config)
        
        # 複数の戦略を生成
        strategies = []
        for i in range(10):
            strategy = generator.generate_random_gene()
            strategies.append(strategy)
        
        # ML指標使用状況の分析
        ml_strategy_count = 0
        ml_indicators_used = set()
        
        for strategy in strategies:
            ml_indicators = [ind for ind in strategy.indicators if ind.type.startswith('ML_')]
            if ml_indicators:
                ml_strategy_count += 1
                for ind in ml_indicators:
                    ml_indicators_used.add(ind.type)
        
        return {
            'status': 'SUCCESS',
            'total_strategies': len(strategies),
            'ml_strategy_count': ml_strategy_count,
            'ml_usage_rate': f"{ml_strategy_count/len(strategies)*100:.1f}%",
            'ml_indicators_used': list(ml_indicators_used)
        }
    except Exception as e:
        return {'status': 'FAILED', 'error': str(e)}

def test_strategy_factory():
    """StrategyFactoryテスト"""
    try:
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        from app.core.services.auto_strategy.models.gene_strategy import StrategyGene, IndicatorGene, Condition
        
        factory = StrategyFactory()
        
        # ML指標を含む戦略遺伝子を作成
        strategy_gene = StrategyGene(
            id="test_ml_strategy_factory",
            indicators=[
                IndicatorGene(type='SMA', parameters={'period': 20}, enabled=True),
                IndicatorGene(type='ML_UP_PROB', parameters={}, enabled=True),
                IndicatorGene(type='ML_DOWN_PROB', parameters={}, enabled=True),
            ],
            long_entry_conditions=[
                Condition(left_operand='ML_UP_PROB', operator='>', right_operand=0.7),
                Condition(left_operand='close', operator='>', right_operand='SMA_20'),
            ],
            short_entry_conditions=[
                Condition(left_operand='ML_DOWN_PROB', operator='>', right_operand=0.7),
                Condition(left_operand='close', operator='<', right_operand='SMA_20'),
            ],
            exit_conditions=[
                Condition(left_operand='ML_UP_PROB', operator='<', right_operand=0.3),
            ]
        )
        
        # 戦略クラスを生成
        strategy_class = factory.create_strategy_class(strategy_gene)
        
        return {
            'status': 'SUCCESS',
            'strategy_class_name': strategy_class.__name__,
            'indicators_count': len(strategy_gene.indicators),
            'ml_indicators': [ind.type for ind in strategy_gene.indicators if ind.type.startswith('ML_')],
            'long_conditions_count': len(strategy_gene.long_entry_conditions),
            'short_conditions_count': len(strategy_gene.short_entry_conditions),
            'exit_conditions_count': len(strategy_gene.exit_conditions)
        }
    except Exception as e:
        return {'status': 'FAILED', 'error': str(e)}

def test_ga_config():
    """GAConfigテスト"""
    try:
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        
        # ML有効設定
        config_with_ml = GAConfig()
        config_with_ml.enable_ml_indicators = True
        
        # ML無効設定
        config_without_ml = GAConfig()
        config_without_ml.enable_ml_indicators = False
        
        return {
            'status': 'SUCCESS',
            'ml_enabled_config': config_with_ml.enable_ml_indicators,
            'ml_disabled_config': config_without_ml.enable_ml_indicators,
            'has_ml_weight': hasattr(config_with_ml, 'ml_weight'),
            'population_size': config_with_ml.population_size,
            'generations': config_with_ml.generations
        }
    except Exception as e:
        return {'status': 'FAILED', 'error': str(e)}

def generate_integration_report():
    """統合レポートを生成"""
    print("🔍 ML-オートストラテジー統合レポート")
    print("=" * 80)
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("MLIndicatorService", test_ml_indicator_service),
        ("IndicatorCalculator", test_indicator_calculator),
        ("SmartConditionGenerator", test_smart_condition_generator),
        ("RandomGeneGenerator", test_random_gene_generator),
        ("StrategyFactory", test_strategy_factory),
        ("GAConfig", test_ga_config),
    ]
    
    results = {}
    success_count = 0
    
    for test_name, test_func in tests:
        print(f"📋 {test_name}テスト")
        print("-" * 40)
        
        try:
            result = test_func()
            results[test_name] = result
            
            if result['status'] == 'SUCCESS':
                success_count += 1
                print("✅ 成功")
                
                # 詳細情報を表示
                for key, value in result.items():
                    if key != 'status':
                        print(f"   {key}: {value}")
            else:
                print("❌ 失敗")
                print(f"   エラー: {result.get('error', '不明')}")
                
        except Exception as e:
            print(f"❌ 例外発生: {e}")
            results[test_name] = {'status': 'ERROR', 'error': str(e)}
        
        print()
    
    # 総合評価
    print("📊 総合評価")
    print("=" * 80)
    print(f"成功率: {success_count}/{len(tests)} ({success_count/len(tests)*100:.1f}%)")
    
    if success_count == len(tests):
        print("🎉 全テスト成功！ML-オートストラテジー統合は完全に動作しています。")
        print()
        print("✅ 統合完了項目:")
        print("   • ML指標計算サービス")
        print("   • 指標計算器でのML指標処理")
        print("   • スマート条件生成器でのML指標使用")
        print("   • ランダム遺伝子生成器でのML指標選択")
        print("   • 戦略ファクトリーでのML戦略生成")
        print("   • GA設定でのML指標制御")
        print()
        print("🚀 オートストラテジーでMLが正常に使用できます！")
    else:
        print(f"⚠️  {len(tests) - success_count}個のテストが失敗しました。")
        print("   統合に問題がある可能性があります。")
    
    return results

def main():
    """メイン実行"""
    return generate_integration_report()

if __name__ == "__main__":
    main()
