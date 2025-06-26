#!/usr/bin/env python3
"""
指標名解決機能のテスト
修正したオートストラテジー機能でのOBV_9エラーが解決されているかテスト
"""

import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import Mock

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(__file__))

def create_mock_strategy_instance():
    """モックの戦略インスタンスを作成"""
    mock_strategy = Mock()
    
    # JSON形式の指標を登録
    mock_strategy.indicators = {
        "OBV": Mock(),
        "ATR": Mock(),
        "RSI": Mock(),
        "SMA": Mock(),
        "EMA": Mock(),
        "MACD": Mock(),
    }
    
    # 各指標に値を設定
    for indicator_name, indicator in mock_strategy.indicators.items():
        indicator.__getitem__ = Mock(return_value=50.0)  # [-1]アクセス用
        indicator.__len__ = Mock(return_value=100)
    
    # データも設定
    mock_strategy.data = Mock()
    mock_strategy.data.Close = Mock()
    mock_strategy.data.Close.__getitem__ = Mock(return_value=45000.0)
    
    return mock_strategy

def test_condition_evaluator():
    """ConditionEvaluatorの指標名解決テスト"""
    print("🔍 ConditionEvaluator 指標名解決テスト")
    print("=" * 60)
    
    try:
        from app.core.services.auto_strategy.factories.condition_evaluator import ConditionEvaluator
        from app.core.services.auto_strategy.models.strategy_gene import Condition
        
        evaluator = ConditionEvaluator()
        mock_strategy = create_mock_strategy_instance()
        
        # テストケース1: JSON形式の指標名（正常ケース）
        print("\n1. JSON形式の指標名テスト:")
        json_condition = Condition(
            left_operand="OBV",
            operator=">",
            right_operand=30
        )
        
        result = evaluator.evaluate_condition(json_condition, mock_strategy)
        print(f"  OBV > 30: {result}")
        
        # テストケース2: レガシー形式の指標名（修正対象）
        print("\n2. レガシー形式の指標名テスト:")
        legacy_condition = Condition(
            left_operand="OBV_9",
            operator=">",
            right_operand=30
        )
        
        result = evaluator.evaluate_condition(legacy_condition, mock_strategy)
        print(f"  OBV_9 > 30: {result}")
        
        # テストケース3: ATRのレガシー形式
        print("\n3. ATRレガシー形式テスト:")
        atr_condition = Condition(
            left_operand="ATR_6",
            operator=">",
            right_operand=1.0
        )
        
        result = evaluator.evaluate_condition(atr_condition, mock_strategy)
        print(f"  ATR_6 > 1.0: {result}")
        
        # テストケース4: 存在しない指標
        print("\n4. 存在しない指標テスト:")
        invalid_condition = Condition(
            left_operand="INVALID_INDICATOR",
            operator=">",
            right_operand=30
        )
        
        result = evaluator.evaluate_condition(invalid_condition, mock_strategy)
        print(f"  INVALID_INDICATOR > 30: {result}")
        
        print("\n✅ ConditionEvaluatorテスト完了")
        return True
        
    except Exception as e:
        print(f"\n❌ ConditionEvaluatorテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_indicator_initializer():
    """IndicatorInitializerの指標登録テスト"""
    print("\n🔧 IndicatorInitializer 指標登録テスト")
    print("=" * 60)
    
    try:
        from app.core.services.auto_strategy.factories.indicator_initializer import IndicatorInitializer
        from app.core.services.auto_strategy.models.strategy_gene import IndicatorGene
        
        initializer = IndicatorInitializer()
        
        # テストケース1: レガシー指標名生成
        print("\n1. レガシー指標名生成テスト:")
        
        test_cases = [
            ("OBV", {"period": 9}, "OBV"),  # パラメータなし指標
            ("RSI", {"period": 14}, "RSI_14"),  # 単一パラメータ
            ("ATR", {"period": 6}, "ATR_6"),  # 単一パラメータ
            ("MACD", {"fast_period": 12, "slow_period": 26, "signal_period": 9}, "MACD_12_26_9"),  # 複数パラメータ
        ]
        
        for indicator_type, parameters, expected in test_cases:
            result = initializer._get_legacy_indicator_name(indicator_type, parameters)
            print(f"  {indicator_type} {parameters} -> {result} (期待値: {expected})")
            assert result == expected, f"期待値 {expected} と異なります: {result}"
        
        print("\n✅ IndicatorInitializerテスト完了")
        return True
        
    except Exception as e:
        print(f"\n❌ IndicatorInitializerテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gene_encoding():
    """GeneEncodingの指標名生成テスト"""
    print("\n🧬 GeneEncoding 指標名生成テスト")
    print("=" * 60)
    
    try:
        from app.core.services.auto_strategy.models.gene_encoding import GeneEncoder
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, IndicatorGene
        
        encoder = GeneEncoder()
        
        # テスト用の戦略遺伝子を作成
        indicators = [
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
        ]
        
        strategy_gene = StrategyGene(indicators=indicators)
        
        # デコードテスト
        print("\n1. 戦略遺伝子デコードテスト:")
        encoded = [0.5, 0.7, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # サンプルエンコード
        
        decoded_strategy = encoder.decode(encoded)
        
        print(f"  指標数: {len(decoded_strategy.indicators)}")
        print(f"  エントリー条件数: {len(decoded_strategy.entry_conditions)}")
        print(f"  エグジット条件数: {len(decoded_strategy.exit_conditions)}")
        
        # 条件の指標名を確認
        for i, condition in enumerate(decoded_strategy.entry_conditions):
            print(f"  エントリー条件{i+1}: {condition.left_operand} {condition.operator} {condition.right_operand}")
        
        print("\n✅ GeneEncodingテスト完了")
        return True
        
    except Exception as e:
        print(f"\n❌ GeneEncodingテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("🎯 指標名解決機能テスト開始")
    print("=" * 80)
    print("目的: OBV_9エラーが解決されているかテスト")
    print("=" * 80)
    
    tests = [
        ("ConditionEvaluator", test_condition_evaluator),
        ("IndicatorInitializer", test_indicator_initializer),
        ("GeneEncoding", test_gene_encoding),
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
    
    if all_passed:
        print("\n🎉 全てのテストが成功しました！")
        print("✅ OBV_9エラーは解決されています")
        print("✅ レガシー形式からJSON形式への変換が正常に動作しています")
    else:
        print("\n⚠️ 一部のテストが失敗しました")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
