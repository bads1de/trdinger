#!/usr/bin/env python3
"""
全テクニカル指標の初期化状況を確認するテスト
どの指標が初期化に失敗しているかを包括的にチェック
"""

import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import Mock

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(__file__))

def create_comprehensive_test_data():
    """包括的なテスト用OHLCVデータを作成"""
    dates = pd.date_range(start='2024-01-01', periods=200, freq='h')
    
    np.random.seed(42)
    price = 45000
    prices = []
    volumes = []
    
    for _ in range(200):
        change = np.random.normal(0, 0.015)  # 1.5%の標準偏差
        price *= (1 + change)
        prices.append(max(price, 1000))  # 最低価格を設定
        volumes.append(np.random.uniform(500, 2000))
    
    data = {
        'Close': pd.Series(prices, index=dates),
        'High': pd.Series([p * (1 + np.random.uniform(0, 0.02)) for p in prices], index=dates),
        'Low': pd.Series([p * (1 - np.random.uniform(0, 0.02)) for p in prices], index=dates),
        'Open': pd.Series(prices, index=dates),
        'Volume': pd.Series(volumes, index=dates)
    }
    
    return data

def test_all_indicators_availability():
    """全指標の利用可能性をテスト"""
    print("📋 全指標利用可能性テスト")
    print("=" * 80)
    
    try:
        from app.core.services.indicators.constants import ALL_INDICATORS
        from app.core.services.auto_strategy.factories.indicator_calculator import IndicatorCalculator
        
        calculator = IndicatorCalculator()
        available_indicators = list(calculator.indicator_adapters.keys())
        
        print("定義されている指標 (constants.py):")
        for indicator in ALL_INDICATORS:
            print(f"  - {indicator}")
        
        print(f"\n利用可能な指標 (IndicatorCalculator):")
        for indicator in available_indicators:
            print(f"  - {indicator}")
        
        print(f"\n比較結果:")
        missing_indicators = []
        for indicator in ALL_INDICATORS:
            if indicator in available_indicators:
                print(f"  ✅ {indicator}")
            else:
                print(f"  ❌ {indicator} (利用不可)")
                missing_indicators.append(indicator)
        
        if missing_indicators:
            print(f"\n⚠️ 利用できない指標: {missing_indicators}")
        else:
            print(f"\n✅ 全ての指標が利用可能です")
        
        return len(missing_indicators) == 0, missing_indicators
        
    except Exception as e:
        print(f"❌ 指標利用可能性テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False, []

def test_all_indicators_calculation():
    """全指標の計算テスト"""
    print("\n🧮 全指標計算テスト")
    print("=" * 80)
    
    try:
        from app.core.services.auto_strategy.factories.indicator_calculator import IndicatorCalculator
        
        calculator = IndicatorCalculator()
        test_data = create_comprehensive_test_data()
        
        # 各指標のデフォルトパラメータ
        default_parameters = {
            "SMA": {"period": 20},
            "EMA": {"period": 20},
            "RSI": {"period": 14},
            "STOCH": {"period": 14},
            "CCI": {"period": 14},
            "ADX": {"period": 14},
            "MACD": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
            "ATR": {"period": 14},
            "BB": {"period": 20, "std_dev": 2},
            "OBV": {},
        }
        
        calculation_results = {}
        failed_indicators = []
        
        for indicator_name in calculator.indicator_adapters.keys():
            print(f"\n{indicator_name}の計算テスト:")
            
            try:
                parameters = default_parameters.get(indicator_name, {"period": 14})
                print(f"  パラメータ: {parameters}")
                
                result, result_name = calculator.calculate_indicator(
                    indicator_name,
                    parameters,
                    test_data['Close'],
                    test_data['High'],
                    test_data['Low'],
                    test_data['Volume'],
                    test_data['Open']
                )
                
                if result is not None:
                    print(f"  ✅ 計算成功")
                    print(f"    結果名: {result_name}")
                    print(f"    結果タイプ: {type(result)}")
                    
                    if hasattr(result, 'columns'):
                        print(f"    カラム: {list(result.columns)}")
                    elif isinstance(result, dict):
                        print(f"    キー: {list(result.keys())}")
                    
                    if hasattr(result, '__len__'):
                        print(f"    データ数: {len(result)}")
                    
                    calculation_results[indicator_name] = True
                else:
                    print(f"  ❌ 計算失敗 (結果がNone)")
                    calculation_results[indicator_name] = False
                    failed_indicators.append(indicator_name)
                    
            except Exception as e:
                print(f"  ❌ 計算エラー: {e}")
                calculation_results[indicator_name] = False
                failed_indicators.append(indicator_name)
        
        print(f"\n" + "=" * 60)
        print("計算結果サマリー:")
        for indicator_name, success in calculation_results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {indicator_name}")
        
        if failed_indicators:
            print(f"\n⚠️ 計算に失敗した指標: {failed_indicators}")
        
        return len(failed_indicators) == 0, failed_indicators
        
    except Exception as e:
        print(f"❌ 指標計算テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False, []

def test_all_indicators_initialization():
    """全指標の初期化テスト"""
    print("\n🔧 全指標初期化テスト")
    print("=" * 80)
    
    try:
        from app.core.services.auto_strategy.factories.indicator_initializer import IndicatorInitializer
        from app.core.services.auto_strategy.models.strategy_gene import IndicatorGene
        from app.core.services.auto_strategy.factories.indicator_calculator import IndicatorCalculator
        
        initializer = IndicatorInitializer()
        calculator = IndicatorCalculator()
        test_data_dict = create_comprehensive_test_data()
        
        # バックテストライブラリのDataクラスをシミュレート
        mock_bt_data = Mock()
        mock_bt_data.Close = test_data_dict['Close'].values
        mock_bt_data.High = test_data_dict['High'].values
        mock_bt_data.Low = test_data_dict['Low'].values
        mock_bt_data.Open = test_data_dict['Open'].values
        mock_bt_data.Volume = test_data_dict['Volume'].values
        
        # モック戦略インスタンス
        mock_strategy = Mock()
        mock_strategy.indicators = {}
        mock_strategy.I = Mock(return_value=Mock())
        
        # 各指標のデフォルトパラメータ
        default_parameters = {
            "SMA": {"period": 20},
            "EMA": {"period": 20},
            "RSI": {"period": 14},
            "STOCH": {"period": 14},
            "CCI": {"period": 14},
            "ADX": {"period": 14},
            "MACD": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
            "ATR": {"period": 14},
            "BB": {"period": 20, "std_dev": 2},
            "OBV": {},
        }
        
        initialization_results = {}
        failed_initializations = []
        
        for indicator_name in calculator.indicator_adapters.keys():
            print(f"\n{indicator_name}の初期化テスト:")
            
            try:
                parameters = default_parameters.get(indicator_name, {"period": 14})
                print(f"  パラメータ: {parameters}")
                
                # 指標遺伝子を作成
                indicator_gene = IndicatorGene(
                    type=indicator_name,
                    parameters=parameters,
                    enabled=True
                )
                
                # 初期化実行
                result = initializer.initialize_indicator(
                    indicator_gene, mock_bt_data, mock_strategy
                )
                
                if result:
                    print(f"  ✅ 初期化成功")
                    print(f"    返された指標名: {result}")
                    initialization_results[indicator_name] = True
                else:
                    print(f"  ❌ 初期化失敗 (結果がNone)")
                    initialization_results[indicator_name] = False
                    failed_initializations.append(indicator_name)
                    
            except Exception as e:
                print(f"  ❌ 初期化エラー: {e}")
                initialization_results[indicator_name] = False
                failed_initializations.append(indicator_name)
        
        print(f"\n" + "=" * 60)
        print("初期化結果サマリー:")
        for indicator_name, success in initialization_results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {indicator_name}")
        
        print(f"\n登録された指標: {list(mock_strategy.indicators.keys())}")
        
        if failed_initializations:
            print(f"\n⚠️ 初期化に失敗した指標: {failed_initializations}")
        
        return len(failed_initializations) == 0, failed_initializations
        
    except Exception as e:
        print(f"❌ 指標初期化テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False, []

def main():
    """メインテスト実行"""
    print("🎯 全テクニカル指標の包括的初期化テスト")
    print("=" * 100)
    print("目的: どの指標が初期化に失敗しているかを特定")
    print("=" * 100)
    
    tests = [
        ("指標利用可能性", test_all_indicators_availability),
        ("指標計算", test_all_indicators_calculation),
        ("指標初期化", test_all_indicators_initialization),
    ]
    
    all_results = {}
    overall_failed = []
    
    for test_name, test_func in tests:
        try:
            success, failed_list = test_func()
            all_results[test_name] = (success, failed_list)
            if failed_list:
                overall_failed.extend(failed_list)
        except Exception as e:
            print(f"\n❌ {test_name}テスト実行エラー: {e}")
            all_results[test_name] = (False, [])
    
    print("\n" + "=" * 100)
    print("📊 最終結果サマリー")
    print("=" * 100)
    
    for test_name, (success, failed_list) in all_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if failed_list:
            print(f"    失敗した指標: {failed_list}")
    
    # 全体的な問題のある指標を特定
    unique_failed = list(set(overall_failed))
    
    print("\n" + "=" * 100)
    if unique_failed:
        print("⚠️ 問題のある指標:")
        for indicator in unique_failed:
            print(f"  - {indicator}")
        print(f"\n修正が必要な指標数: {len(unique_failed)}")
    else:
        print("🎉 全ての指標が正常に動作しています！")
    
    return 0 if len(unique_failed) == 0 else 1

if __name__ == "__main__":
    exit(main())
