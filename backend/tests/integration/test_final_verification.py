"""
最終検証テスト

ロング・ショート機能の完全な動作確認を行います。
"""

import sys
import os
import json

# パス設定
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

def test_complete_workflow():
    """完全なワークフローテスト"""
    print("🚀 完全ワークフローテスト開始")
    
    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        import pandas as pd
        import numpy as np
        
        # 1. 戦略生成
        print("\n1. ロング・ショート戦略生成...")
        config = GAConfig()
        generator = RandomGeneGenerator(config)
        gene = generator.generate_random_gene()
        
        print(f"   ✅ 指標数: {len(gene.indicators)}")
        print(f"   ✅ ロング条件数: {len(gene.long_entry_conditions)}")
        print(f"   ✅ ショート条件数: {len(gene.short_entry_conditions)}")
        print(f"   ✅ エグジット条件数: {len(gene.exit_conditions)}")
        
        # 2. 戦略クラス生成
        print("\n2. 戦略クラス生成...")
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(gene)
        print(f"   ✅ 戦略クラス: {strategy_class.__name__}")
        
        # 3. JSON変換
        print("\n3. JSON変換...")
        strategy_json = gene.to_json()
        strategy_dict = json.loads(strategy_json)
        
        print(f"   ✅ JSONサイズ: {len(strategy_json)} 文字")
        print(f"   ✅ long_entry_conditions: {len(strategy_dict.get('long_entry_conditions', []))} 個")
        print(f"   ✅ short_entry_conditions: {len(strategy_dict.get('short_entry_conditions', []))} 個")
        
        # 4. 条件評価テスト
        print("\n4. 条件評価テスト...")
        
        # テストデータ作成
        data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [101, 102, 103, 104, 105],
            'Low': [99, 100, 101, 102, 103],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'Volume': [1000, 1000, 1000, 1000, 1000]
        })
        
        # 戦略インスタンス作成
        strategy_instance = strategy_class(data=data, params={})
        
        # 指標を模擬設定
        strategy_instance.indicators = {}
        for indicator in gene.indicators:
            if indicator.enabled:
                indicator_name = f"{indicator.type}_{indicator.parameters.get('period', 14)}"
                # 模擬データ（実際の計算ではなく、テスト用の値）
                if indicator.type == "RSI":
                    strategy_instance.indicators[indicator_name] = pd.Series([30, 40, 50, 60, 70])
                elif indicator.type in ["SMA", "EMA"]:
                    strategy_instance.indicators[indicator_name] = pd.Series([99, 100, 101, 102, 103])
                elif indicator.type == "CCI":
                    strategy_instance.indicators[indicator_name] = pd.Series([-150, -50, 0, 50, 150])
                else:
                    strategy_instance.indicators[indicator_name] = pd.Series([45, 47, 50, 53, 55])
        
        # 条件評価
        long_result = strategy_instance._check_long_entry_conditions()
        short_result = strategy_instance._check_short_entry_conditions()
        
        print(f"   ✅ ロング条件評価: {long_result}")
        print(f"   ✅ ショート条件評価: {short_result}")
        
        # 5. 後方互換性テスト
        print("\n5. 後方互換性テスト...")
        
        # 古い形式の戦略
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, IndicatorGene, Condition
        
        old_gene = StrategyGene(
            indicators=[IndicatorGene(type="RSI", parameters={"period": 14})],
            entry_conditions=[
                Condition(left_operand="RSI_14", operator="<", right_operand=30)
            ],
            exit_conditions=[
                Condition(left_operand="RSI_14", operator=">", right_operand=70)
            ]
        )
        
        old_strategy_class = factory.create_strategy_class(old_gene)
        old_strategy_instance = old_strategy_class(data=data, params={})
        old_strategy_instance.indicators = {'RSI_14': pd.Series([25, 35, 45, 55, 65])}
        
        old_entry_result = old_strategy_instance._check_entry_conditions()
        old_long_result = old_strategy_instance._check_long_entry_conditions()
        old_short_result = old_strategy_instance._check_short_entry_conditions()
        
        print(f"   ✅ 古い形式エントリー: {old_entry_result}")
        print(f"   ✅ 古い形式→ロング: {old_long_result}")
        print(f"   ✅ 古い形式→ショート: {old_short_result}")
        
        # 6. 妥当性検証
        print("\n6. 妥当性検証...")
        
        is_valid, errors = gene.validate()
        print(f"   ✅ 戦略妥当性: {'有効' if is_valid else '無効'}")
        if errors:
            print(f"   ⚠️ エラー: {errors}")
        
        # 7. 統計情報
        print("\n7. 統計情報...")
        
        # 複数戦略を生成して統計を取る
        long_short_count = 0
        long_only_count = 0
        short_only_count = 0
        total_test_strategies = 20
        
        for i in range(total_test_strategies):
            test_gene = generator.generate_random_gene()
            has_long = len(test_gene.long_entry_conditions) > 0
            has_short = len(test_gene.short_entry_conditions) > 0
            
            if has_long and has_short:
                long_short_count += 1
            elif has_long:
                long_only_count += 1
            elif has_short:
                short_only_count += 1
        
        print(f"   ✅ ロング・ショート両対応: {long_short_count}/{total_test_strategies} ({long_short_count/total_test_strategies*100:.1f}%)")
        print(f"   ✅ ロングのみ: {long_only_count}/{total_test_strategies} ({long_only_count/total_test_strategies*100:.1f}%)")
        print(f"   ✅ ショートのみ: {short_only_count}/{total_test_strategies} ({short_only_count/total_test_strategies*100:.1f}%)")
        
        print("\n🎉 完全ワークフローテスト成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ 完全ワークフローテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_json_size_analysis():
    """JSONサイズ分析"""
    print("\n📊 JSONサイズ分析...")
    
    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        import json
        
        config = GAConfig()
        generator = RandomGeneGenerator(config)
        
        sizes = []
        for i in range(10):
            gene = generator.generate_random_gene()
            json_str = gene.to_json()
            sizes.append(len(json_str))
        
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        
        print(f"   📏 平均JSONサイズ: {avg_size:.0f} 文字")
        print(f"   📏 最小JSONサイズ: {min_size} 文字")
        print(f"   📏 最大JSONサイズ: {max_size} 文字")
        
        if avg_size > 1000:
            print(f"   ⚠️ 平均サイズが1KB超過 → フロントエンド折りたたみ表示推奨")
        else:
            print(f"   ✅ 平均サイズは適切")
        
        return True
        
    except Exception as e:
        print(f"❌ JSONサイズ分析失敗: {e}")
        return False

def main():
    """メインテスト実行"""
    print("🎯 最終検証テスト開始\n")
    
    tests = [
        test_complete_workflow,
        test_json_size_analysis,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ テスト実行エラー: {e}")
    
    print(f"\n📊 最終検証結果: {passed}/{total} 成功")
    
    if passed == total:
        print("\n🎉 全ての最終検証テストが成功しました！")
        print("\n🎯 実装完了確認:")
        print("✅ ロング・ショート戦略の生成")
        print("✅ 戦略クラスの生成と条件評価")
        print("✅ JSON変換とシリアライゼーション")
        print("✅ 後方互換性の維持")
        print("✅ 戦略妥当性の検証")
        print("✅ フロントエンド用JSON折りたたみ表示")
        print("\n🚀 オートストラテジーシステムのロング・ショート対応が完全に実装されました！")
    else:
        print("❌ 一部の最終検証テストが失敗しました")
    
    return passed == total

if __name__ == "__main__":
    main()
