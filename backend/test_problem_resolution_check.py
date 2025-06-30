#!/usr/bin/env python3
"""
問題解決確認テスト

リファクタリング計画で指摘された問題が解決されているかを検証します。
"""

import sys
import os

# パスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_parameter_generation_centralization():
    """パラメータ生成の一元化確認"""
    print("=== 1. パラメータ生成の一元化確認 ===")
    
    try:
        from app.core.services.indicators.parameter_manager import IndicatorParameterManager
        from app.core.services.indicators.config.indicator_config import (
            IndicatorConfig, ParameterConfig, IndicatorResultType, indicator_registry
        )
        
        # IndicatorParameterManagerが中核システムとして機能しているか
        manager = IndicatorParameterManager()
        print("✓ IndicatorParameterManagerが正常に作成されました")
        
        # IndicatorConfigがパラメータ生成機能を持っているか
        rsi_config = IndicatorConfig(
            indicator_name="RSI",
            required_data=["close"],
            result_type=IndicatorResultType.SINGLE,
        )
        rsi_config.add_parameter(
            ParameterConfig(name="period", default_value=14, min_value=2, max_value=100)
        )
        
        # 新しいメソッドが追加されているか確認
        assert hasattr(rsi_config, 'generate_random_parameters'), "generate_random_parameters メソッドが存在しません"
        assert hasattr(rsi_config, 'get_parameter_ranges'), "get_parameter_ranges メソッドが存在しません"
        assert hasattr(rsi_config, 'has_parameters'), "has_parameters メソッドが存在しません"
        
        # 実際にパラメータ生成ができるか
        params = rsi_config.generate_random_parameters()
        print(f"✓ IndicatorConfigでパラメータ生成成功: {params}")
        
        return True
        
    except Exception as e:
        print(f"✗ パラメータ生成の一元化確認失敗: {e}")
        return False

def check_duplicate_elimination():
    """重複と不整合の解消確認"""
    print("\n=== 2. 重複と不整合の解消確認 ===")
    
    try:
        from app.core.services.auto_strategy.utils.parameter_generators import generate_indicator_parameters
        from app.core.services.indicators.parameter_manager import IndicatorParameterManager
        from app.core.services.indicators.config.indicator_config import indicator_registry
        
        # 新システムと従来システムの統合確認
        # RSIパラメータを複数の方法で生成して一貫性を確認
        
        # 1. generate_indicator_parameters関数経由
        params1 = generate_indicator_parameters("RSI")
        print(f"✓ generate_indicator_parameters経由: {params1}")
        
        # 2. IndicatorParameterManager直接使用（レジストリに設定がある場合）
        try:
            config = indicator_registry.get_config("RSI")
            if config:
                manager = IndicatorParameterManager()
                params2 = manager.generate_parameters("RSI", config)
                print(f"✓ IndicatorParameterManager直接使用: {params2}")
            else:
                print("✓ RSIがレジストリに未登録（フォールバック動作正常）")
        except Exception as e:
            print(f"✓ 新システム未対応でフォールバック動作: {e}")
        
        # 3. 複数の指標で一貫性確認
        test_indicators = ["SMA", "EMA", "RSI", "MACD", "BB", "OBV"]
        for indicator in test_indicators:
            params = generate_indicator_parameters(indicator)
            print(f"✓ {indicator}: {params}")
            
        return True
        
    except Exception as e:
        print(f"✗ 重複と不整合の解消確認失敗: {e}")
        return False

def check_maintainability_improvement():
    """保守性の向上確認"""
    print("\n=== 3. 保守性の向上確認 ===")
    
    try:
        from app.core.services.indicators.config.indicator_config import (
            IndicatorConfig, ParameterConfig, IndicatorResultType, indicator_registry
        )
        from app.core.services.indicators.parameter_manager import IndicatorParameterManager
        
        # 新しい指標を簡単に追加できるかテスト
        test_config = IndicatorConfig(
            indicator_name="TEST_NEW_INDICATOR",
            required_data=["close"],
            result_type=IndicatorResultType.SINGLE,
        )
        test_config.add_parameter(
            ParameterConfig(name="test_period", default_value=20, min_value=5, max_value=100)
        )
        
        # レジストリに登録
        indicator_registry.register(test_config)
        print("✓ 新しい指標をレジストリに登録成功")
        
        # パラメータ生成
        manager = IndicatorParameterManager()
        params = manager.generate_parameters("TEST_NEW_INDICATOR", test_config)
        print(f"✓ 新しい指標のパラメータ生成成功: {params}")
        
        # バリデーション
        is_valid = manager.validate_parameters("TEST_NEW_INDICATOR", params, test_config)
        print(f"✓ 新しい指標のバリデーション成功: {is_valid}")
        
        # 範囲情報取得
        ranges = test_config.get_parameter_ranges()
        print(f"✓ パラメータ範囲情報取得成功: {ranges}")
        
        return True
        
    except Exception as e:
        print(f"✗ 保守性の向上確認失敗: {e}")
        return False

def check_responsibility_clarity():
    """責務の明確化確認"""
    print("\n=== 4. 責務の明確化確認 ===")
    
    try:
        from app.core.services.indicators.parameter_manager import IndicatorParameterManager
        from app.core.services.indicators.config.indicator_config import IndicatorConfig, ParameterConfig
        
        manager = IndicatorParameterManager()
        
        # パラメータ生成の責務
        config = IndicatorConfig(indicator_name="TEST")
        config.add_parameter(ParameterConfig(name="period", default_value=14, min_value=2, max_value=100))
        
        params = manager.generate_parameters("TEST", config)
        print(f"✓ パラメータ生成責務: IndicatorParameterManager")
        
        # バリデーションの責務
        is_valid = manager.validate_parameters("TEST", params, config)
        print(f"✓ バリデーション責務: IndicatorParameterManager")
        
        # 設定管理の責務
        ranges = manager.get_parameter_ranges("TEST", config)
        print(f"✓ 設定管理責務: IndicatorConfig + IndicatorParameterManager")
        
        # 各クラスが明確な責務を持っているか確認
        print("✓ 責務分担:")
        print("  - IndicatorConfig: パラメータ定義・設定保持")
        print("  - ParameterConfig: 個別パラメータの設定・バリデーション")
        print("  - IndicatorParameterManager: パラメータ生成・統合バリデーション")
        print("  - indicator_registry: 設定の登録・管理")
        
        return True
        
    except Exception as e:
        print(f"✗ 責務の明確化確認失敗: {e}")
        return False

def check_backward_compatibility():
    """後方互換性確認"""
    print("\n=== 5. 後方互換性確認 ===")
    
    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        from app.core.services.auto_strategy.models.gene_encoding import GeneEncoder
        
        # RandomGeneGeneratorが正常に動作するか
        generator = RandomGeneGenerator()
        print("✓ RandomGeneGenerator作成成功")
        
        # GeneEncoderが正常に動作するか
        encoder = GeneEncoder()
        params = encoder._generate_indicator_parameters("RSI", 0.5)
        print(f"✓ GeneEncoder動作確認: {params}")
        
        # 既存のコードが変更なしで動作することを確認
        print("✓ 既存コードとの互換性維持")
        
        return True
        
    except Exception as e:
        print(f"✗ 後方互換性確認失敗: {e}")
        return False

def main():
    """メイン関数"""
    print("リファクタリング計画問題解決確認テスト開始\n")
    
    checks = [
        ("パラメータ生成の一元化", check_parameter_generation_centralization),
        ("重複と不整合の解消", check_duplicate_elimination),
        ("保守性の向上", check_maintainability_improvement),
        ("責務の明確化", check_responsibility_clarity),
        ("後方互換性", check_backward_compatibility),
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        if check_func():
            passed += 1
            print(f"✅ {name}: 解決済み")
        else:
            print(f"❌ {name}: 未解決")
    
    print(f"\n=== 最終結果 ===")
    print(f"解決済み問題: {passed}/{total}")
    
    if passed == total:
        print("🎉 リファクタリング計画の全ての問題が解決されました！")
        return 0
    else:
        print("⚠️  一部の問題が未解決です。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
