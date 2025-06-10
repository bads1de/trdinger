"""
OI/FR修正版テスト

修正されたOI/FR機能が問題を解決しているかテストします。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from unittest.mock import Mock

from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene, IndicatorGene, Condition
)
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory

def test_nan_value_handling():
    """NaN値処理テスト"""
    print("=== NaN値処理テスト ===")
    
    factory = StrategyFactory()
    
    # NaN値を含むデータ
    mock_data = Mock()
    mock_data.Close = pd.Series([100, 101, 102])
    mock_data.OpenInterest = pd.Series([1000000, np.nan, 2000000])
    mock_data.FundingRate = pd.Series([0.001, np.nan, 0.002])
    
    gene = StrategyGene(
        indicators=[],
        entry_conditions=[Condition("FundingRate", ">", 0.001)],
        exit_conditions=[Condition("close", "<", 95)]
    )
    
    try:
        strategy_class = factory.create_strategy_class(gene)
        strategy = strategy_class(data=mock_data, params={})
        
        # NaN値でのアクセス（最後の値は有効）
        fr_value = strategy._get_oi_fr_value("FundingRate")
        oi_value = strategy._get_oi_fr_value("OpenInterest")
        
        print(f"FundingRate値: {fr_value} (期待: 0.002)")
        print(f"OpenInterest値: {oi_value} (期待: 2000000)")
        
        success = fr_value == 0.002 and oi_value == 2000000
        print(f"✅ NaN値処理: {'成功' if success else '失敗'}")
        return success
        
    except Exception as e:
        print(f"❌ NaN値処理テスト失敗: {e}")
        return False

def test_all_nan_values():
    """全NaN値処理テスト"""
    print("\n=== 全NaN値処理テスト ===")
    
    factory = StrategyFactory()
    
    # 全てNaN値のデータ
    mock_data = Mock()
    mock_data.Close = pd.Series([100, 101, 102])
    mock_data.OpenInterest = pd.Series([np.nan, np.nan, np.nan])
    mock_data.FundingRate = pd.Series([np.nan, np.nan, np.nan])
    
    gene = StrategyGene(
        indicators=[],
        entry_conditions=[Condition("FundingRate", ">", 0.001)],
        exit_conditions=[Condition("close", "<", 95)]
    )
    
    try:
        strategy_class = factory.create_strategy_class(gene)
        strategy = strategy_class(data=mock_data, params={})
        
        # 全NaN値でのアクセス
        fr_value = strategy._get_oi_fr_value("FundingRate")
        oi_value = strategy._get_oi_fr_value("OpenInterest")
        
        print(f"FundingRate値: {fr_value} (期待: 0.0)")
        print(f"OpenInterest値: {oi_value} (期待: 0.0)")
        
        success = fr_value == 0.0 and oi_value == 0.0
        print(f"✅ 全NaN値処理: {'成功' if success else '失敗'}")
        return success
        
    except Exception as e:
        print(f"❌ 全NaN値処理テスト失敗: {e}")
        return False

def test_mixed_nan_values():
    """混合NaN値処理テスト"""
    print("\n=== 混合NaN値処理テスト ===")
    
    factory = StrategyFactory()
    
    # 最後がNaN、途中に有効値があるデータ
    mock_data = Mock()
    mock_data.Close = pd.Series([100, 101, 102])
    mock_data.OpenInterest = pd.Series([1000000, 1500000, np.nan])  # 最後がNaN
    mock_data.FundingRate = pd.Series([0.001, np.nan, np.nan])      # 最初だけ有効
    
    gene = StrategyGene(
        indicators=[],
        entry_conditions=[Condition("FundingRate", ">", 0.0005)],
        exit_conditions=[Condition("close", "<", 95)]
    )
    
    try:
        strategy_class = factory.create_strategy_class(gene)
        strategy = strategy_class(data=mock_data, params={})
        
        # 混合NaN値でのアクセス
        fr_value = strategy._get_oi_fr_value("FundingRate")
        oi_value = strategy._get_oi_fr_value("OpenInterest")
        
        print(f"FundingRate値: {fr_value} (期待: 0.001)")
        print(f"OpenInterest値: {oi_value} (期待: 1500000)")
        
        success = fr_value == 0.001 and oi_value == 1500000
        print(f"✅ 混合NaN値処理: {'成功' if success else '失敗'}")
        return success
        
    except Exception as e:
        print(f"❌ 混合NaN値処理テスト失敗: {e}")
        return False

def test_extreme_values():
    """極端値処理テスト"""
    print("\n=== 極端値処理テスト ===")
    
    factory = StrategyFactory()
    
    # 極端な値
    mock_data = Mock()
    mock_data.Close = pd.Series([100, 101, 102])
    mock_data.OpenInterest = pd.Series([1e15, 1e16, 1e17])  # 極大値
    mock_data.FundingRate = pd.Series([-1.0, -0.5, 0.5])   # 極端な負値から正値
    
    gene = StrategyGene(
        indicators=[],
        entry_conditions=[Condition("FundingRate", ">", 0)],
        exit_conditions=[Condition("OpenInterest", ">", 1e16)]
    )
    
    try:
        strategy_class = factory.create_strategy_class(gene)
        strategy = strategy_class(data=mock_data, params={})
        
        # 条件評価
        entry_result = strategy._check_entry_conditions()
        exit_result = strategy._check_exit_conditions()
        
        print(f"エントリー条件 (FR > 0): {entry_result} (期待: True)")
        print(f"イグジット条件 (OI > 1e16): {exit_result} (期待: True)")
        
        success = entry_result and exit_result
        print(f"✅ 極端値処理: {'成功' if success else '失敗'}")
        return success
        
    except Exception as e:
        print(f"❌ 極端値処理テスト失敗: {e}")
        return False

def test_data_type_compatibility():
    """データ型互換性テスト"""
    print("\n=== データ型互換性テスト ===")
    
    factory = StrategyFactory()
    
    # 異なるデータ型
    test_cases = [
        ("list", [100, 101, 102]),
        ("numpy", np.array([100, 101, 102])),
        ("pandas", pd.Series([100, 101, 102])),
    ]
    
    results = []
    
    for dtype_name, data_values in test_cases:
        try:
            mock_data = Mock()
            mock_data.Close = pd.Series([100, 101, 102])
            setattr(mock_data, "OpenInterest", data_values)
            setattr(mock_data, "FundingRate", data_values)
            
            gene = StrategyGene(
                indicators=[],
                entry_conditions=[Condition("FundingRate", ">", 50)],
                exit_conditions=[Condition("close", "<", 95)]
            )
            
            strategy_class = factory.create_strategy_class(gene)
            strategy = strategy_class(data=mock_data, params={})
            
            # データアクセステスト
            fr_value = strategy._get_oi_fr_value("FundingRate")
            oi_value = strategy._get_oi_fr_value("OpenInterest")
            
            print(f"{dtype_name}型: FR={fr_value}, OI={oi_value}")
            
            success = fr_value == 102 and oi_value == 102
            results.append(success)
            print(f"✅ {dtype_name}型互換性: {'成功' if success else '失敗'}")
            
        except Exception as e:
            print(f"❌ {dtype_name}型互換性テスト失敗: {e}")
            results.append(False)
    
    return all(results)

def test_condition_evaluation_robustness():
    """条件評価堅牢性テスト"""
    print("\n=== 条件評価堅牢性テスト ===")
    
    factory = StrategyFactory()
    
    # 複雑な条件セット
    mock_data = Mock()
    mock_data.Close = pd.Series([100, 101, 102])
    mock_data.OpenInterest = pd.Series([1000000, 1100000, 1200000])
    mock_data.FundingRate = pd.Series([0.001, 0.002, 0.003])
    
    gene = StrategyGene(
        indicators=[IndicatorGene("SMA", {"period": 20})],
        entry_conditions=[
            Condition("close", ">", "SMA_20"),
            Condition("FundingRate", ">", 0.002),
            Condition("OpenInterest", ">", 1100000),
        ],
        exit_conditions=[
            Condition("FundingRate", "<", 0.001),
            Condition("OpenInterest", "<", 1000000),
        ]
    )
    
    try:
        strategy_class = factory.create_strategy_class(gene)
        strategy = strategy_class(data=mock_data, params={})
        
        # 指標を手動で設定（テスト用）
        strategy.indicators = {"SMA_20": Mock()}
        strategy.indicators["SMA_20"].__getitem__ = lambda x: 101  # SMA値を101に設定
        strategy.indicators["SMA_20"].__len__ = lambda: 3
        
        # 条件評価
        entry_result = strategy._check_entry_conditions()
        exit_result = strategy._check_exit_conditions()
        
        print(f"エントリー条件評価: {entry_result}")
        print(f"イグジット条件評価: {exit_result}")
        
        # エントリー条件: close(102) > SMA(101) AND FR(0.003) > 0.002 AND OI(1200000) > 1100000 = True
        # イグジット条件: FR(0.003) < 0.001 OR OI(1200000) < 1000000 = False
        
        success = entry_result == True and exit_result == False
        print(f"✅ 条件評価堅牢性: {'成功' if success else '失敗'}")
        return success
        
    except Exception as e:
        print(f"❌ 条件評価堅牢性テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("🔧 OI/FR修正版テスト開始\n")
    
    results = []
    
    # 修正版テスト実行
    results.append(test_nan_value_handling())
    results.append(test_all_nan_values())
    results.append(test_mixed_nan_values())
    results.append(test_extreme_values())
    results.append(test_data_type_compatibility())
    results.append(test_condition_evaluation_robustness())
    
    # 結果サマリー
    total_tests = len(results)
    successful_tests = sum(results)
    failed_tests = total_tests - successful_tests
    
    print(f"\n📊 修正版テスト結果サマリー:")
    print(f"  総テスト数: {total_tests}")
    print(f"  成功: {successful_tests}")
    print(f"  失敗: {failed_tests}")
    print(f"  成功率: {successful_tests/total_tests*100:.1f}%")
    
    if failed_tests == 0:
        print("\n🎉 全ての修正版テストが成功しました！")
        print("✅ OI/FR機能の問題が修正されました。")
    else:
        print(f"\n⚠️ {failed_tests}個のテストが失敗しました。")
    
    return failed_tests == 0

if __name__ == "__main__":
    main()
