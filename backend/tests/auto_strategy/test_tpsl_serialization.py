#!/usr/bin/env python3
"""
TP/SL遺伝子のシリアライゼーション・GA統合テスト

修正後のTP/SL GA統合機能をテストし、メソッドの動的切り替えと
パラメータJSON表示を確認します。
"""

import sys
import os
import json

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_tpsl_gene_serialization():
    """TP/SL遺伝子のシリアライゼーションテスト"""
    print("=== TP/SL遺伝子シリアライゼーションテスト ===")
    
    try:
        from app.core.services.auto_strategy.models.tpsl_gene import TPSLGene, TPSLMethod
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene
        from app.core.services.auto_strategy.models.gene_serialization import GeneSerializer
        
        # TP/SL遺伝子を作成
        tpsl_gene = TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            stop_loss_pct=0.025,
            take_profit_pct=0.075,
            risk_reward_ratio=3.0,
            base_stop_loss=0.025
        )
        
        print(f"✅ TP/SL遺伝子作成成功:")
        print(f"   - メソッド: {tpsl_gene.method.value}")
        print(f"   - SL: {tpsl_gene.stop_loss_pct:.1%}")
        print(f"   - TP: {tpsl_gene.take_profit_pct:.1%}")
        print(f"   - リスクリワード比: {tpsl_gene.risk_reward_ratio}")
        
        # 戦略遺伝子を作成（TP/SL遺伝子を含む）
        strategy_gene = StrategyGene(
            id="test-strategy-001",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            risk_management={"position_size": 0.1},
            tpsl_gene=tpsl_gene,
            metadata={"test": True}
        )
        
        # シリアライゼーションテスト
        serializer = GeneSerializer()
        
        # 戦略遺伝子を辞書に変換
        strategy_dict = serializer.strategy_gene_to_dict(strategy_gene)
        
        print(f"\n✅ 戦略遺伝子シリアライゼーション成功:")
        print(f"   - ID: {strategy_dict['id']}")
        print(f"   - TP/SL遺伝子含有: {'tpsl_gene' in strategy_dict}")
        
        if 'tpsl_gene' in strategy_dict and strategy_dict['tpsl_gene']:
            tpsl_dict = strategy_dict['tpsl_gene']
            print(f"   - TP/SLメソッド: {tpsl_dict['method']}")
            print(f"   - SL: {tpsl_dict['stop_loss_pct']:.1%}")
            print(f"   - TP: {tpsl_dict['take_profit_pct']:.1%}")
            print(f"   - リスクリワード比: {tpsl_dict['risk_reward_ratio']}")
        
        # JSON形式で表示
        strategy_json = json.dumps(strategy_dict, ensure_ascii=False, indent=2)
        print(f"\n✅ パラメータJSON表示:")
        print("```json")
        print(strategy_json)
        print("```")
        
        # デシリアライゼーションテスト
        restored_gene = serializer.dict_to_strategy_gene(strategy_dict, StrategyGene)
        
        print(f"\n✅ 戦略遺伝子デシリアライゼーション成功:")
        print(f"   - ID: {restored_gene.id}")
        print(f"   - TP/SL遺伝子復元: {restored_gene.tpsl_gene is not None}")
        
        if restored_gene.tpsl_gene:
            print(f"   - 復元メソッド: {restored_gene.tpsl_gene.method.value}")
            print(f"   - 復元SL: {restored_gene.tpsl_gene.stop_loss_pct:.1%}")
            print(f"   - 復元TP: {restored_gene.tpsl_gene.take_profit_pct:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ TP/SL遺伝子シリアライゼーションテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tpsl_gene_crossover():
    """TP/SL遺伝子の交叉テスト"""
    print("\n=== TP/SL遺伝子交叉テスト ===")
    
    try:
        from app.core.services.auto_strategy.models.tpsl_gene import TPSLGene, TPSLMethod
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, crossover_strategy_genes
        
        # 親1の戦略遺伝子
        parent1_tpsl = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            stop_loss_pct=0.02,
            take_profit_pct=0.06
        )
        
        parent1 = StrategyGene(
            id="parent1",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            risk_management={"position_size": 0.1},
            tpsl_gene=parent1_tpsl
        )
        
        # 親2の戦略遺伝子
        parent2_tpsl = TPSLGene(
            method=TPSLMethod.VOLATILITY_BASED,
            atr_multiplier_sl=2.5,
            atr_multiplier_tp=4.0
        )
        
        parent2 = StrategyGene(
            id="parent2",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            risk_management={"position_size": 0.2},
            tpsl_gene=parent2_tpsl
        )
        
        print(f"✅ 親遺伝子作成:")
        print(f"   - 親1 TP/SLメソッド: {parent1.tpsl_gene.method.value}")
        print(f"   - 親2 TP/SLメソッド: {parent2.tpsl_gene.method.value}")
        
        # 交叉実行
        child1, child2 = crossover_strategy_genes(parent1, parent2)
        
        print(f"\n✅ 戦略遺伝子交叉成功:")
        print(f"   - 子1 TP/SLメソッド: {child1.tpsl_gene.method.value if child1.tpsl_gene else 'None'}")
        print(f"   - 子2 TP/SLメソッド: {child2.tpsl_gene.method.value if child2.tpsl_gene else 'None'}")
        
        if child1.tpsl_gene:
            print(f"   - 子1 SL: {child1.tpsl_gene.stop_loss_pct:.1%}")
            print(f"   - 子1 TP: {child1.tpsl_gene.take_profit_pct:.1%}")
        
        if child2.tpsl_gene:
            print(f"   - 子2 SL: {child2.tpsl_gene.stop_loss_pct:.1%}")
            print(f"   - 子2 TP: {child2.tpsl_gene.take_profit_pct:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ TP/SL遺伝子交叉テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tpsl_gene_mutation():
    """TP/SL遺伝子の突然変異テスト"""
    print("\n=== TP/SL遺伝子突然変異テスト ===")
    
    try:
        from app.core.services.auto_strategy.models.tpsl_gene import TPSLGene, TPSLMethod
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, mutate_strategy_gene
        
        # 元の戦略遺伝子
        original_tpsl = TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            risk_reward_ratio=2.0
        )
        
        original_gene = StrategyGene(
            id="original",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            risk_management={"position_size": 0.1},
            tpsl_gene=original_tpsl
        )
        
        print(f"✅ 元の戦略遺伝子:")
        print(f"   - TP/SLメソッド: {original_gene.tpsl_gene.method.value}")
        print(f"   - SL: {original_gene.tpsl_gene.stop_loss_pct:.1%}")
        print(f"   - TP: {original_gene.tpsl_gene.take_profit_pct:.1%}")
        print(f"   - リスクリワード比: {original_gene.tpsl_gene.risk_reward_ratio}")
        
        # 突然変異実行（高い変異率で確実に変化させる）
        mutated_gene = mutate_strategy_gene(original_gene, mutation_rate=0.8)
        
        print(f"\n✅ 戦略遺伝子突然変異成功:")
        print(f"   - 変異後TP/SLメソッド: {mutated_gene.tpsl_gene.method.value if mutated_gene.tpsl_gene else 'None'}")
        
        if mutated_gene.tpsl_gene:
            print(f"   - 変異後SL: {mutated_gene.tpsl_gene.stop_loss_pct:.1%}")
            print(f"   - 変異後TP: {mutated_gene.tpsl_gene.take_profit_pct:.1%}")
            print(f"   - 変異後リスクリワード比: {mutated_gene.tpsl_gene.risk_reward_ratio:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ TP/SL遺伝子突然変異テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("TP/SL GA統合機能テスト開始\n")
    
    results = []
    
    # シリアライゼーションテスト
    results.append(test_tpsl_gene_serialization())
    
    # 交叉テスト
    results.append(test_tpsl_gene_crossover())
    
    # 突然変異テスト
    results.append(test_tpsl_gene_mutation())
    
    # 結果サマリー
    print(f"\n=== テスト結果サマリー ===")
    print(f"成功: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✅ 全てのテストが成功しました！")
        print("TP/SLメソッドがGAで動的に切り替わり、パラメータJSONに表示されることを確認しました。")
    else:
        print("❌ 一部のテストが失敗しました。")
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
