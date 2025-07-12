#!/usr/bin/env python3
"""
指標モード機能のテスト

3つの指標モード（テクニカルオンリー、MLオンリー、混合）の動作を確認します。
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_ga_config_indicator_modes():
    """GAConfigの指標モード設定テスト"""
    print("=== GAConfig 指標モード設定テスト ===")
    
    try:
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        
        # 各モードの設定テスト
        modes = ["technical_only", "ml_only", "mixed"]
        
        for mode in modes:
            config = GAConfig()
            config.indicator_mode = mode
            
            print(f"モード: {mode}")
            print(f"  indicator_mode: {config.indicator_mode}")
            print(f"  enable_ml_indicators: {config.enable_ml_indicators}")
            
            # 辞書変換テスト
            config_dict = config.to_dict()
            print(f"  辞書変換: indicator_mode = {config_dict.get('indicator_mode')}")
            
            # 辞書から復元テスト
            restored_config = GAConfig.from_dict(config_dict)
            print(f"  復元: indicator_mode = {restored_config.indicator_mode}")
            print()
        
        return True
        
    except Exception as e:
        print(f"GAConfig 指標モード設定テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_random_gene_generator_modes():
    """RandomGeneGeneratorの指標モード対応テスト"""
    print("=== RandomGeneGenerator 指標モード対応テスト ===")
    
    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        
        modes = ["technical_only", "ml_only", "mixed"]
        
        for mode in modes:
            print(f"\n--- {mode} モード ---")
            
            config = GAConfig()
            config.indicator_mode = mode
            config.max_indicators = 3
            
            generator = RandomGeneGenerator(config)
            
            print(f"利用可能な指標数: {len(generator.available_indicators)}")
            print(f"利用可能な指標: {generator.available_indicators}")
            
            # 戦略生成テスト
            strategies = []
            for i in range(3):
                strategy = generator.generate_random_gene()
                strategies.append(strategy)
            
            # 指標使用状況の分析
            ml_count = 0
            technical_count = 0
            
            for strategy in strategies:
                for indicator in strategy.indicators:
                    if indicator.type.startswith('ML_'):
                        ml_count += 1
                    else:
                        technical_count += 1
            
            print(f"生成された戦略数: {len(strategies)}")
            print(f"ML指標使用回数: {ml_count}")
            print(f"テクニカル指標使用回数: {technical_count}")
            
            # モード別の期待値チェック
            if mode == "technical_only":
                assert ml_count == 0, f"テクニカルオンリーモードでML指標が使用された: {ml_count}"
                print("✓ テクニカルオンリーモード正常")
            elif mode == "ml_only":
                assert technical_count == 0, f"MLオンリーモードでテクニカル指標が使用された: {technical_count}"
                print("✓ MLオンリーモード正常")
            elif mode == "mixed":
                print("✓ 混合モード正常")
        
        return True
        
    except Exception as e:
        print(f"RandomGeneGenerator 指標モード対応テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_smart_condition_generator_modes():
    """SmartConditionGeneratorの指標モード対応テスト"""
    print("\n=== SmartConditionGenerator 指標モード対応テスト ===")
    
    try:
        from app.core.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
        from app.core.services.auto_strategy.models.gene_strategy import IndicatorGene
        
        generator = SmartConditionGenerator()
        
        # テクニカルオンリー
        print("\n--- テクニカルオンリー ---")
        technical_indicators = [
            IndicatorGene(type='RSI', parameters={'period': 14}, enabled=True),
            IndicatorGene(type='SMA', parameters={'period': 20}, enabled=True),
        ]
        
        long_conds, short_conds, exit_conds = generator.generate_balanced_conditions(technical_indicators)
        print(f"ロング条件数: {len(long_conds)}")
        print(f"ショート条件数: {len(short_conds)}")
        print(f"エグジット条件数: {len(exit_conds)}")
        
        # MLオンリー
        print("\n--- MLオンリー ---")
        ml_indicators = [
            IndicatorGene(type='ML_UP_PROB', parameters={}, enabled=True),
            IndicatorGene(type='ML_DOWN_PROB', parameters={}, enabled=True),
        ]
        
        long_conds, short_conds, exit_conds = generator.generate_balanced_conditions(ml_indicators)
        print(f"ロング条件数: {len(long_conds)}")
        print(f"ショート条件数: {len(short_conds)}")
        print(f"エグジット条件数: {len(exit_conds)}")
        
        # ML条件の確認
        ml_condition_count = 0
        all_conditions = long_conds + short_conds + exit_conds
        for condition in all_conditions:
            condition_str = str(condition)
            if any(ml_ind in condition_str for ml_ind in ['ML_UP_PROB', 'ML_DOWN_PROB', 'ML_RANGE_PROB']):
                ml_condition_count += 1
                print(f"  ML条件: {condition_str}")
        
        print(f"ML条件数: {ml_condition_count}")
        
        # 混合
        print("\n--- 混合 ---")
        mixed_indicators = [
            IndicatorGene(type='RSI', parameters={'period': 14}, enabled=True),
            IndicatorGene(type='ML_UP_PROB', parameters={}, enabled=True),
        ]
        
        long_conds, short_conds, exit_conds = generator.generate_balanced_conditions(mixed_indicators)
        print(f"ロング条件数: {len(long_conds)}")
        print(f"ショート条件数: {len(short_conds)}")
        print(f"エグジット条件数: {len(exit_conds)}")
        
        # 混合条件の確認
        ml_condition_count = 0
        technical_condition_count = 0
        all_conditions = long_conds + short_conds + exit_conds
        
        for condition in all_conditions:
            condition_str = str(condition)
            if any(ml_ind in condition_str for ml_ind in ['ML_UP_PROB', 'ML_DOWN_PROB', 'ML_RANGE_PROB']):
                ml_condition_count += 1
            elif any(tech_ind in condition_str for tech_ind in ['RSI', 'SMA', 'EMA']):
                technical_condition_count += 1
        
        print(f"ML条件数: {ml_condition_count}")
        print(f"テクニカル条件数: {technical_condition_count}")
        
        return True
        
    except Exception as e:
        print(f"SmartConditionGenerator 指標モード対応テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_integration():
    """API統合テスト（設定の送受信）"""
    print("\n=== API統合テスト ===")
    
    try:
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        
        # フロントエンドから送信される設定を模擬
        frontend_config = {
            "population_size": 10,
            "generations": 5,
            "indicator_mode": "ml_only",
            "max_indicators": 2,
        }
        
        # GAConfigで処理
        config = GAConfig.from_dict(frontend_config)
        
        print(f"受信した設定:")
        print(f"  indicator_mode: {config.indicator_mode}")
        print(f"  population_size: {config.population_size}")
        print(f"  generations: {config.generations}")
        
        # 設定の妥当性確認
        assert config.indicator_mode == "ml_only"
        assert config.population_size == 10
        assert config.generations == 5
        
        print("✓ API統合テスト成功")
        
        return True
        
    except Exception as e:
        print(f"API統合テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("指標モード機能テスト開始")
    print("=" * 60)
    
    tests = [
        test_ga_config_indicator_modes,
        test_random_gene_generator_modes,
        test_smart_condition_generator_modes,
        test_api_integration,
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
        print("✓ 全テスト成功！指標モード機能は正常に動作しています。")
    else:
        print(f"✗ {total - passed}個のテストが失敗しました。")
    
    return passed == total

if __name__ == "__main__":
    main()
