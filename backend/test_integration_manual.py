#!/usr/bin/env python3
"""
手動統合テスト

新しいパラメータ管理システムの動作確認を行います。
"""

import sys
import os

# パスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """基本機能のテスト"""
    print("=== 基本機能テスト ===")
    
    try:
        from app.core.services.indicators.parameter_manager import IndicatorParameterManager
        from app.core.services.indicators.config.indicator_config import (
            IndicatorConfig,
            ParameterConfig,
            IndicatorResultType,
        )
        
        # IndicatorParameterManagerの作成
        manager = IndicatorParameterManager()
        print("✓ IndicatorParameterManager作成成功")
        
        # RSI設定の作成
        rsi_config = IndicatorConfig(
            indicator_name="RSI",
            required_data=["close"],
            result_type=IndicatorResultType.SINGLE,
        )
        rsi_config.add_parameter(
            ParameterConfig(
                name="period",
                default_value=14,
                min_value=2,
                max_value=100,
                description="RSI計算期間",
            )
        )
        print("✓ RSI設定作成成功")
        
        # パラメータ生成
        params = manager.generate_parameters("RSI", rsi_config)
        print(f"✓ パラメータ生成成功: {params}")
        
        # バリデーション
        is_valid = manager.validate_parameters("RSI", params, rsi_config)
        print(f"✓ バリデーション成功: {is_valid}")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本機能テスト失敗: {e}")
        return False

def test_parameter_generators_integration():
    """parameter_generators統合テスト"""
    print("\n=== parameter_generators統合テスト ===")
    
    try:
        from app.core.services.auto_strategy.utils.parameter_generators import (
            generate_indicator_parameters,
        )
        
        # 各指標タイプでのパラメータ生成
        test_indicators = ["SMA", "EMA", "RSI", "MACD", "BB", "OBV"]
        
        for indicator_type in test_indicators:
            params = generate_indicator_parameters(indicator_type)
            print(f"✓ {indicator_type}: {params}")
            
        return True
        
    except Exception as e:
        print(f"✗ parameter_generators統合テスト失敗: {e}")
        return False

def test_gene_encoding_integration():
    """gene_encoding統合テスト"""
    print("\n=== gene_encoding統合テスト ===")
    
    try:
        from app.core.services.auto_strategy.models.gene_encoding import GeneEncoder
        
        encoder = GeneEncoder()
        
        # 各指標タイプでのパラメータ生成
        test_indicators = ["RSI", "MACD", "BB"]
        
        for indicator_type in test_indicators:
            params = encoder._generate_indicator_parameters(indicator_type, 0.5)
            print(f"✓ {indicator_type}: {params}")
            
        return True
        
    except Exception as e:
        print(f"✗ gene_encoding統合テスト失敗: {e}")
        return False

def test_indicator_config_methods():
    """IndicatorConfigの新メソッドテスト"""
    print("\n=== IndicatorConfig新メソッドテスト ===")
    
    try:
        from app.core.services.indicators.config.indicator_config import (
            IndicatorConfig,
            ParameterConfig,
            IndicatorResultType,
        )
        
        # RSI設定の作成
        rsi_config = IndicatorConfig(
            indicator_name="RSI",
            required_data=["close"],
            result_type=IndicatorResultType.SINGLE,
        )
        rsi_config.add_parameter(
            ParameterConfig(
                name="period",
                default_value=14,
                min_value=2,
                max_value=100,
                description="RSI計算期間",
            )
        )
        
        # 新メソッドのテスト
        has_params = rsi_config.has_parameters()
        print(f"✓ has_parameters(): {has_params}")
        
        ranges = rsi_config.get_parameter_ranges()
        print(f"✓ get_parameter_ranges(): {ranges}")
        
        random_params = rsi_config.generate_random_parameters()
        print(f"✓ generate_random_parameters(): {random_params}")
        
        return True
        
    except Exception as e:
        print(f"✗ IndicatorConfig新メソッドテスト失敗: {e}")
        return False

def main():
    """メイン関数"""
    print("パラメータ管理システム統合テスト開始\n")
    
    tests = [
        test_basic_functionality,
        test_parameter_generators_integration,
        test_gene_encoding_integration,
        test_indicator_config_methods,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== テスト結果 ===")
    print(f"成功: {passed}/{total}")
    
    if passed == total:
        print("✓ 全てのテストが成功しました！")
        return 0
    else:
        print("✗ 一部のテストが失敗しました。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
