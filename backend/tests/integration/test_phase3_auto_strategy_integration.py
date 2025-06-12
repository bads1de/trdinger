#!/usr/bin/env python3
"""
Phase 3 新規指標のオートストラテジー統合テスト
BOP, PPO, MIDPOINT, MIDPRICE, TRIMA指標の統合確認
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_phase3_auto_strategy_integration():
    """Phase 3 新規指標のオートストラテジー統合テスト"""
    print("🧪 Phase 3 新規指標オートストラテジー統合テスト")
    print("=" * 70)
    
    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        
        # 新規指標リスト
        phase3_indicators = ["BOP", "PPO", "MIDPOINT", "MIDPRICE", "TRIMA"]
        
        print(f"📊 テスト対象指標: {', '.join(phase3_indicators)}")
        
        # 1. RandomGeneGeneratorでの統合確認
        print("\n1️⃣ RandomGeneGenerator統合テスト")
        print("-" * 50)
        
        generator = RandomGeneGenerator()
        total_indicators = len(generator.available_indicators)
        print(f"📊 利用可能指標数: {total_indicators}")
        
        success_count = 0
        for indicator in phase3_indicators:
            if indicator in generator.available_indicators:
                print(f"✅ {indicator}: 利用可能リストに含まれています")
                success_count += 1
            else:
                print(f"❌ {indicator}: 利用可能リストに含まれていません")
        
        print(f"\n📊 統合結果: {success_count}/{len(phase3_indicators)} 統合済み")
        
        # 2. パラメータ生成テスト
        print("\n2️⃣ パラメータ生成テスト")
        print("-" * 50)
        
        param_success = 0
        for indicator in phase3_indicators:
            try:
                params = generator._generate_indicator_parameters(indicator)
                assert isinstance(params, dict)
                assert "period" in params
                print(f"✅ {indicator}: パラメータ生成成功 - {params}")
                param_success += 1
            except Exception as e:
                print(f"❌ {indicator}: パラメータ生成失敗 - {e}")
        
        print(f"\n📊 パラメータ生成結果: {param_success}/{len(phase3_indicators)} 成功")
        
        # 3. StrategyFactoryでの統合確認
        print("\n3️⃣ StrategyFactory統合テスト")
        print("-" * 50)
        
        factory = StrategyFactory()
        factory_success = 0
        
        for indicator in phase3_indicators:
            if indicator in factory.indicator_adapters:
                print(f"✅ {indicator}: StrategyFactoryに統合済み")
                factory_success += 1
            else:
                print(f"❌ {indicator}: StrategyFactoryに未統合")
        
        print(f"\n📊 StrategyFactory統合結果: {factory_success}/{len(phase3_indicators)} 統合済み")
        
        # 4. 戦略遺伝子生成テスト
        print("\n4️⃣ 戦略遺伝子生成テスト")
        print("-" * 50)
        
        gene_success = 0
        for i in range(5):  # 5回試行
            try:
                gene = generator.generate_random_gene()
                
                # 新規指標が含まれているかチェック
                used_indicators = [ind.type for ind in gene.indicators if ind.enabled]
                phase3_used = [ind for ind in used_indicators if ind in phase3_indicators]
                
                if phase3_used:
                    print(f"✅ 試行{i+1}: Phase3指標使用 - {', '.join(phase3_used)}")
                    gene_success += 1
                else:
                    print(f"⚪ 試行{i+1}: Phase3指標未使用 - {', '.join(used_indicators)}")
                
                # 遺伝子の妥当性チェック
                is_valid, errors = factory.validate_gene(gene)
                if not is_valid:
                    print(f"⚠️  試行{i+1}: 遺伝子検証エラー - {errors}")
                
            except Exception as e:
                print(f"❌ 試行{i+1}: 遺伝子生成失敗 - {e}")
        
        print(f"\n📊 戦略遺伝子生成結果: {gene_success}/5 でPhase3指標使用")
        
        # 5. 総合評価
        print("\n5️⃣ 総合評価")
        print("-" * 50)
        
        total_tests = 4
        passed_tests = 0
        
        if success_count == len(phase3_indicators):
            print("✅ RandomGeneGenerator統合: 合格")
            passed_tests += 1
        else:
            print("❌ RandomGeneGenerator統合: 不合格")
        
        if param_success == len(phase3_indicators):
            print("✅ パラメータ生成: 合格")
            passed_tests += 1
        else:
            print("❌ パラメータ生成: 不合格")
        
        if factory_success == len(phase3_indicators):
            print("✅ StrategyFactory統合: 合格")
            passed_tests += 1
        else:
            print("❌ StrategyFactory統合: 不合格")
        
        if gene_success > 0:
            print("✅ 戦略遺伝子生成: 合格")
            passed_tests += 1
        else:
            print("❌ 戦略遺伝子生成: 不合格")
        
        print(f"\n🎯 総合結果: {passed_tests}/{total_tests} テスト合格")
        
        if passed_tests == total_tests:
            print("\n🎉 Phase 3 新規指標のオートストラテジー統合が完了しました！")
            return True
        else:
            print("\n⚠️  一部のテストが失敗しました。修正が必要です。")
            return False
        
    except Exception as e:
        print(f"❌ 統合テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_indicator_count():
    """指標数の確認"""
    print("\n🧪 指標数確認テスト")
    print("=" * 70)
    
    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        
        generator = RandomGeneGenerator()
        total_indicators = len(generator.available_indicators)
        
        print(f"📊 現在の利用可能指標数: {total_indicators}")
        
        # 期待される指標数（前回39 + 今回3 = 42）
        expected_count = 42
        
        if total_indicators >= expected_count:
            print(f"✅ 指標数確認成功: {total_indicators}種類（期待値: {expected_count}以上）")
            
            # 全指標リストを表示
            print("\n📋 利用可能指標一覧:")
            for i, indicator in enumerate(sorted(generator.available_indicators), 1):
                print(f"  {i:2d}. {indicator}")
            
            return True
        else:
            print(f"❌ 指標数不足: {total_indicators}種類（期待値: {expected_count}以上）")
            return False
        
    except Exception as e:
        print(f"❌ 指標数確認テスト失敗: {e}")
        return False

def main():
    """メイン実行関数"""
    print("🚀 Phase 3 新規指標統合テスト開始")
    print("=" * 70)
    
    # テスト実行
    test1_result = test_phase3_auto_strategy_integration()
    test2_result = test_indicator_count()
    
    print("\n" + "=" * 70)
    print("📊 最終結果")
    print("=" * 70)
    
    if test1_result and test2_result:
        print("🎉 全てのテストが成功しました！")
        print("✅ Phase 3 新規指標がオートストラテジーで使用可能です")
        return True
    else:
        print("❌ 一部のテストが失敗しました")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
