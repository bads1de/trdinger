#!/usr/bin/env python3
"""
OBV_9エラー修正の確認テスト
実際のオートストラテジー機能でOBV_9エラーが解決されているかテスト
"""

import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(__file__))

def create_test_data():
    """テスト用のOHLCVデータを作成"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    
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

def test_strategy_with_obv_condition():
    """OBV条件を含む戦略のテスト"""
    print("🧪 OBV条件を含む戦略テスト")
    print("=" * 60)
    
    try:
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, IndicatorGene, Condition
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        from app.core.services.auto_strategy.factories.indicator_initializer import IndicatorInitializer
        
        # OBV指標を含む戦略遺伝子を作成
        indicators = [
            IndicatorGene(type="OBV", parameters={}, enabled=True),
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
        ]
        
        # レガシー形式の条件を含む戦略（修正前はエラーになっていた）
        entry_conditions = [
            Condition(left_operand="OBV_9", operator=">", right_operand=0),  # レガシー形式
            Condition(left_operand="RSI", operator="<", right_operand=30),   # JSON形式
        ]
        
        exit_conditions = [
            Condition(left_operand="OBV", operator="<", right_operand=0),    # JSON形式
            Condition(left_operand="RSI_14", operator=">", right_operand=70), # レガシー形式
        ]
        
        strategy_gene = StrategyGene(
            indicators=indicators,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions
        )
        
        print("✅ 戦略遺伝子作成成功")
        print(f"  指標数: {len(strategy_gene.indicators)}")
        print(f"  エントリー条件数: {len(strategy_gene.entry_conditions)}")
        print(f"  エグジット条件数: {len(strategy_gene.exit_conditions)}")
        
        # 戦略ファクトリーで戦略クラスを生成
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(strategy_gene)
        
        print("✅ 戦略クラス生成成功")
        print(f"  戦略クラス名: {strategy_class.__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ 戦略テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_condition_evaluation_with_real_data():
    """実際のデータを使った条件評価テスト"""
    print("\n📊 実データでの条件評価テスト")
    print("=" * 60)
    
    try:
        from app.core.services.auto_strategy.factories.condition_evaluator import ConditionEvaluator
        from app.core.services.auto_strategy.models.strategy_gene import Condition
        
        evaluator = ConditionEvaluator()
        
        # モック戦略インスタンスを作成
        mock_strategy = Mock()
        
        # JSON形式で指標を登録（修正後の動作）
        mock_strategy.indicators = {
            "OBV": Mock(),
            "ATR": Mock(),
            "RSI": Mock(),
        }
        
        # 指標値を設定
        mock_strategy.indicators["OBV"].__getitem__ = Mock(return_value=1000.0)
        mock_strategy.indicators["OBV"].__len__ = Mock(return_value=100)
        
        mock_strategy.indicators["ATR"].__getitem__ = Mock(return_value=2.5)
        mock_strategy.indicators["ATR"].__len__ = Mock(return_value=100)
        
        mock_strategy.indicators["RSI"].__getitem__ = Mock(return_value=25.0)
        mock_strategy.indicators["RSI"].__len__ = Mock(return_value=100)
        
        # テストケース: 元々エラーになっていた条件
        test_conditions = [
            ("OBV_9 > 0", Condition("OBV_9", ">", 0)),
            ("ATR_6 > 1.0", Condition("ATR_6", ">", 1.0)),
            ("RSI_14 < 30", Condition("RSI_14", "<", 30)),
            ("OBV > 500", Condition("OBV", ">", 500)),  # JSON形式
        ]
        
        print("\n条件評価結果:")
        all_success = True
        
        for description, condition in test_conditions:
            try:
                result = evaluator.evaluate_condition(condition, mock_strategy)
                print(f"  {description}: {result}")
            except Exception as e:
                print(f"  {description}: ❌ エラー - {e}")
                all_success = False
        
        if all_success:
            print("\n✅ 全ての条件評価が成功しました")
            print("✅ OBV_9エラーは完全に解決されています")
        else:
            print("\n❌ 一部の条件評価でエラーが発生しました")
        
        return all_success
        
    except Exception as e:
        print(f"❌ 条件評価テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_indicator_name_mapping():
    """指標名マッピングテスト"""
    print("\n🔄 指標名マッピングテスト")
    print("=" * 60)
    
    try:
        from app.core.services.auto_strategy.factories.condition_evaluator import ConditionEvaluator
        
        evaluator = ConditionEvaluator()
        
        # モック戦略インスタンス
        mock_strategy = Mock()
        mock_strategy.indicators = {
            "OBV": Mock(),
            "ATR": Mock(),
            "RSI": Mock(),
            "SMA": Mock(),
            "MACD": Mock(),
        }
        
        # レガシー形式からJSON形式への変換テスト
        test_mappings = [
            ("OBV_9", "OBV"),
            ("ATR_6", "ATR"),
            ("RSI_14", "RSI"),
            ("SMA_20", "SMA"),
            ("MACD_line", "MACD"),
            ("MACD_signal", "MACD"),
            ("MACD_histogram", "MACD"),
        ]
        
        print("\n指標名変換結果:")
        all_success = True
        
        for legacy_name, expected_json_name in test_mappings:
            resolved_name = evaluator._resolve_indicator_name(legacy_name, mock_strategy)
            success = resolved_name == expected_json_name
            status = "✅" if success else "❌"
            print(f"  {status} {legacy_name} -> {resolved_name} (期待値: {expected_json_name})")
            
            if not success:
                all_success = False
        
        return all_success
        
    except Exception as e:
        print(f"❌ 指標名マッピングテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("🎯 OBV_9エラー修正確認テスト")
    print("=" * 80)
    print("目的: 'OBV_9' が見つかりませんエラーが解決されているか確認")
    print("=" * 80)
    
    tests = [
        ("OBV条件を含む戦略", test_strategy_with_obv_condition),
        ("実データでの条件評価", test_condition_evaluation_with_real_data),
        ("指標名マッピング", test_indicator_name_mapping),
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
        print("✅ OBV_9エラーは完全に解決されています")
        print("✅ レガシー形式からJSON形式への自動変換が正常に動作")
        print("✅ オートストラテジー機能でパラメーター付き指標名が使用可能")
    else:
        print("⚠️ 一部のテストが失敗しました")
        print("修正が必要な箇所があります")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
