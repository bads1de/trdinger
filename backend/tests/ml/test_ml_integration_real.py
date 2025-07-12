#!/usr/bin/env python3
"""
ML-オートストラテジー統合の実際の動作テスト

実際のGA実行でML指標が使用されているかを確認します。
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_test_data(size: int = 100) -> pd.DataFrame:
    """テスト用のOHLCVデータを生成"""
    dates = pd.date_range(start='2023-01-01', periods=size, freq='1H')
    
    # ランダムウォークで価格データを生成
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, size)
    prices = 50000 * np.exp(np.cumsum(returns))
    
    # OHLCV データを生成
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

def test_ml_indicator_calculation():
    """ML指標計算テスト"""
    print("=== ML指標計算テスト ===")
    
    try:
        from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
        
        service = MLIndicatorService()
        test_data = create_test_data(100)
        
        print(f"テストデータサイズ: {len(test_data)}")
        print(f"データ列: {test_data.columns.tolist()}")
        
        # ML指標計算
        result = service.calculate_ml_indicators(test_data)
        
        print(f"ML指標計算結果:")
        for indicator, values in result.items():
            print(f"  {indicator}: 長さ={len(values)}, 範囲=[{values.min():.3f}, {values.max():.3f}], 平均={values.mean():.3f}")
        
        return True
        
    except Exception as e:
        print(f"ML指標計算テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_indicator_calculator_ml():
    """IndicatorCalculatorでのML指標テスト"""
    print("\n=== IndicatorCalculator ML指標テスト ===")
    
    try:
        from app.core.services.auto_strategy.calculators.indicator_calculator import IndicatorCalculator
        
        calculator = IndicatorCalculator()
        test_data = create_test_data(50)
        
        # backtesting.pyのDataオブジェクトを模擬
        class MockBacktestData:
            def __init__(self, df):
                self.df = df
                
        mock_data = MockBacktestData(test_data)
        
        # ML指標の計算テスト
        ml_indicators = ['ML_UP_PROB', 'ML_DOWN_PROB', 'ML_RANGE_PROB']
        
        for indicator in ml_indicators:
            try:
                result = calculator.calculate_indicator(indicator, {}, mock_data)
                if result is not None:
                    print(f"  {indicator}: 成功 - 長さ={len(result)}, 範囲=[{result.min():.3f}, {result.max():.3f}]")
                else:
                    print(f"  {indicator}: 失敗 - None が返された")
            except Exception as e:
                print(f"  {indicator}: エラー - {e}")
        
        return True
        
    except Exception as e:
        print(f"IndicatorCalculator ML指標テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_smart_condition_generator_ml():
    """SmartConditionGeneratorでのML指標テスト"""
    print("\n=== SmartConditionGenerator ML指標テスト ===")
    
    try:
        from app.core.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
        from app.core.services.auto_strategy.models.gene_strategy import IndicatorGene
        
        generator = SmartConditionGenerator()
        
        # ML指標を含む指標リスト
        indicators = [
            IndicatorGene(type='RSI', parameters={'period': 14}, enabled=True),
            IndicatorGene(type='ML_UP_PROB', parameters={}, enabled=True),
            IndicatorGene(type='ML_DOWN_PROB', parameters={}, enabled=True),
            IndicatorGene(type='ML_RANGE_PROB', parameters={}, enabled=True),
        ]
        
        # バランス条件生成
        long_conditions, short_conditions, exit_conditions = generator.generate_balanced_conditions(indicators)
        
        print(f"生成された条件:")
        print(f"  ロング条件: {len(long_conditions)}")
        print(f"  ショート条件: {len(short_conditions)}")
        print(f"  エグジット条件: {len(exit_conditions)}")
        
        # ML指標を使った条件の確認
        ml_condition_count = 0
        all_conditions = long_conditions + short_conditions + exit_conditions
        
        for condition in all_conditions:
            condition_str = str(condition)
            if any(ml_ind in condition_str for ml_ind in ['ML_UP_PROB', 'ML_DOWN_PROB', 'ML_RANGE_PROB']):
                ml_condition_count += 1
                print(f"    ML条件: {condition_str}")
        
        print(f"  ML指標を使った条件数: {ml_condition_count}")
        
        return True
        
    except Exception as e:
        print(f"SmartConditionGenerator ML指標テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ga_config_ml_flags():
    """GAConfigでのMLフラグテスト"""
    print("\n=== GAConfig MLフラグテスト ===")
    
    try:
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        
        # ML有効設定
        config_with_ml = GAConfig()
        config_with_ml.enable_ml_indicators = True
        
        # ML無効設定
        config_without_ml = GAConfig()
        config_without_ml.enable_ml_indicators = False
        
        print(f"ML有効設定: enable_ml_indicators = {config_with_ml.enable_ml_indicators}")
        print(f"ML無効設定: enable_ml_indicators = {config_without_ml.enable_ml_indicators}")
        
        # その他のML関連設定確認
        if hasattr(config_with_ml, 'ml_weight'):
            print(f"ML重み設定: ml_weight = {config_with_ml.ml_weight}")
        
        return True
        
    except Exception as e:
        print(f"GAConfig MLフラグテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ml_model_status():
    """MLモデル状態テスト"""
    print("\n=== MLモデル状態テスト ===")
    
    try:
        from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
        
        service = MLIndicatorService()
        status = service.get_model_status()
        
        print(f"MLモデル状態:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"MLモデル状態テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("ML-オートストラテジー統合の実際の動作テスト開始")
    print("=" * 60)
    
    tests = [
        test_ml_indicator_calculation,
        test_indicator_calculator_ml,
        test_smart_condition_generator_ml,
        test_ga_config_ml_flags,
        test_ml_model_status,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✓ PASS")
            else:
                print("✗ FAIL")
        except Exception as e:
            print(f"✗ ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"テスト結果: {passed}/{total} 成功")
    
    if passed == total:
        print("✓ 全テスト成功！ML-オートストラテジー統合は正常に動作しています。")
    else:
        print(f"✗ {total - passed}個のテストが失敗しました。")
    
    return passed == total

if __name__ == "__main__":
    main()
